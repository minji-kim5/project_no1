# ============================================
# HDBSCAN One-Class + 라벨기반 튜닝
# + τ 캘리브레이션(정밀도 하한 & 알람률 캡 동시 만족)
# + OHE(handle_unknown="ignore") 강제 적용
# + (옵션) 세그먼트별 모델 저장 + 글로벌 폴백 항상 저장
# + PCA 차원 자동 클리핑(세그먼트별 max_dim 계산)
# ============================================
import os, re, joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- HDBSCAN 안전 임포트 (로컬 파일명 충돌 방지) ---
def _import_hdbscan_safely():
    import importlib, os as _os
    m = importlib.import_module("hdbscan")  # pip install hdbscan
    mod_path = getattr(m, "__file__", "") or ""
    if _os.path.basename(mod_path).lower() == "hdbscan.py":
        raise ImportError(
            f"'hdbscan' 패키지 대신 로컬 파일이 임포트되었습니다: {mod_path}\n"
            f"→ 로컬 파일명을 변경(예: hdbscan_local.py)하거나, 디렉터리를 옮기세요."
        )
    from hdbscan import HDBSCAN
    from hdbscan.prediction import approximate_predict
    return HDBSCAN, approximate_predict, mod_path

HDBSCAN, approximate_predict, _hdbscan_path = _import_hdbscan_safely()
print(f"[info] using hdbscan from: {_hdbscan_path}")

# ----------------
# 설정
# ----------------
CLF_PKL     = "final_model.pkl"          # 전처리 파이프라인 포함 분류 pkl(이미 학습)
TRAIN_XLSX  = "data/train.xlsx"
SAVE_DIR    = "./hdbscan_models"         # 번들 저장 폴더
MODEL_NAME  = "hdbscan_oneclass"         # 파일 접두
SEED        = 42

# τ 캘리브레이션 제약 (둘 다 적용)
MIN_PRECISION   = 0.30                   # 정밀도 하한 (예: ≥30%)
MAX_ALERT_RATE  = 0.10                   # 알람률 상한 (예: ≤10%)

# 튜닝 그리드(정수 차원 후보) — 실제 사용 전 세그먼트별로 자동 클리핑
PCA_GRID_BASE = [3,5,7,9,11,13,15,17]
MCS_GRID      = [10,20,30,40, 50,60,70, 80, 120]        # min_cluster_size
MS_GRID       = [5, 10, 15]              # min_samples
METHODS       = ["eom"]                  # eom이 대체로 노이즈율 낮음

# (옵션) 세그먼트 분리: None이면 전체 하나, 컬럼명을 주면 그룹별 모델
SEGMENT_COL = "mold_code"                # 예: "mold_code" / None

# ----------------
# 데이터 로드 & 정리 (안전하게)
# ----------------
df = pd.read_excel(TRAIN_XLSX)

# 인덱스 기반 드랍(있을 때만) + 센티넬 값 기반 필터
drop_idx = [19327, 6000, 11811, 17598, 46546]
df = df.drop(index=[i for i in drop_idx if i in df.index], errors="ignore")

for col, bad_val in [
    ("physical_strength", 65535),
    ("low_section_speed", 65535),
    ("Coolant_temperature", 1449),
    ("upper_mold_temp1", 1449),
    ("upper_mold_temp2", 4232),
]:
    if col in df.columns:
        df = df[df[col] != bad_val]

# 파생/형변환
if "registration_time" in df.columns:
    df["registration_time"] = pd.to_datetime(df["registration_time"], errors="coerce")
    df["hour"] = df["registration_time"].dt.hour.astype(object)

# 결측 보정
if "tryshot_signal" in df.columns:
    df["tryshot_signal"] = df["tryshot_signal"].fillna("A")
if "molten_volume" in df.columns:
    df["molten_volume"] = df["molten_volume"].fillna(0)
if ("molten_volume" in df.columns) and ("heating_furnace" in df.columns):
    cond = df["molten_volume"].notna() & df["heating_furnace"].isna()
    df.loc[cond, "heating_furnace"] = "C"

# 범주형 캐스팅(존재할 때만)
for c in ["mold_code", "EMS_operation_time", "hour"]:
    if c in df.columns:
        df[c] = df[c].astype(object)

# 센서 하한 → NaN
if "molten_temp" in df.columns:
    df.loc[df["molten_temp"] <= 80, "molten_temp"] = np.nan
if "physical_strength" in df.columns:
    df.loc[df["physical_strength"] <= 5, "physical_strength"] = np.nan

# 필요없는 열 안전 제거(있는 것만)
cols_to_drop = [
    'id','line','name','mold_name','emergency_stop','time','date','registration_time',
    'upper_mold_temp3','lower_mold_temp3','working'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

train_df = df.copy()

# ----------------
# 유틸
# ----------------
def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

def _sanitize_filename(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))

def _force_ohe_handle_unknown(prep, mode="ignore"):
    """ColumnTransformer/ Pipeline 안쪽의 OneHotEncoder에 handle_unknown=mode 강제."""
    def _visit(obj):
        cnt = 0
        if isinstance(obj, OneHotEncoder):
            obj.handle_unknown = mode
            cnt += 1
        elif isinstance(obj, Pipeline):
            for _, step in obj.steps:
                cnt += _visit(step)
        elif isinstance(obj, ColumnTransformer):
            for _, trans, _ in obj.transformers:
                if trans in ("drop", "passthrough"):
                    continue
                cnt += _visit(trans)
        return cnt
    return _visit(prep)

def _prepare_xy(prep, feat, df_):
    X = prep.transform(df_[feat])
    X = _to_dense(X)
    y = df_["passorfail"].astype(int).values  # 1=불량
    return X, y

def _anomaly_score(labels, strengths):
    """연속 이상점수: 노이즈=1.0, 배정점=1-strength (0~1, 클수록 이상)"""
    return np.where(labels == -1, 1.0, 1.0 - strengths)

def _alert_rate_at_threshold(labels_val, score_val, thr):
    """임계 thr(score 기준)에서의 전체 알람률 = P(label=-1 or score>=thr)"""
    return float(np.mean((labels_val == -1) | (score_val >= thr)))

def _fit_eval_candidate(X_norm, X_val, y_val, p_dim, mcs, ms, method):
    # 1) PCA: 정상 참조로 fit
    pca = PCA(n_components=p_dim, random_state=SEED).fit(X_norm)
    Z_norm = pca.transform(X_norm)
    Z_val  = pca.transform(X_val)

    # 2) HDBSCAN: 정상만 적합
    hdb = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_method=method,
        prediction_data=True
    ).fit(Z_norm)

    # 3) 검증셋 연속점수/지표
    lab_va, str_va = approximate_predict(hdb, Z_val)
    lab_va = lab_va.astype(int); str_va = str_va.astype(float)
    score_va = _anomaly_score(lab_va, str_va)

    pr_auc = average_precision_score(y_val, score_va)
    roc_auc = roc_auc_score(y_val, score_va)
    base_noise_on_norm = float((hdb.labels_ == -1).mean())

    return {
        "pca": pca, "hdb": hdb,
        "PR_AUC": float(pr_auc), "ROC_AUC": float(roc_auc),
        "base_noise_on_norm": base_noise_on_norm,
        "labels_val": lab_va, "score_val": score_va
    }

def calibrate_tau_both(labels_val, score_val, y_true_val,
                       min_precision=0.30, max_alert_rate=0.10):
    """
    두 제약(precision 하한, alert rate 상한)을 동시에 만족하는 τ 선택.
    - 둘 다 만족하는 임계 중 recall 최대 선택
    - 없으면: (1) alert만 만족 중 최대 recall → (2) precision만 만족 중 최대 recall → (3) 최대 recall
    """
    prec, rec, thr = precision_recall_curve(y_true_val, score_val)  # thr aligns with prec[1:], rec[1:]
    if len(thr) == 0:
        return 0.0, {"precision": float(prec[-1]), "recall": float(rec[-1]),
                     "alert_rate": _alert_rate_at_threshold(labels_val, score_val, 1.0),
                     "score_threshold": 0.0}

    alert_rates = np.array([_alert_rate_at_threshold(labels_val, score_val, t) for t in thr])

    ok_both = (prec[1:] >= min_precision) & (alert_rates <= max_alert_rate)
    if np.any(ok_both):
        idx = np.argmax(rec[1:][ok_both]); best = np.where(ok_both)[0][idx]
    else:
        ok_alert = (alert_rates <= max_alert_rate)
        if np.any(ok_alert):
            best = np.where(ok_alert)[0][np.argmax(rec[1:][ok_alert])]
        else:
            ok_prec = (prec[1:] >= min_precision)
            if np.any(ok_prec):
                best = np.where(ok_prec)[0][np.argmax(rec[1:][ok_prec])]
            else:
                best = int(np.argmax(rec[1:]))

    th_star = float(thr[best])            # score 임계
    tau_star = 1.0 - th_star              # τ = 1 - score_threshold (배정점 멤버십 임계)
    info = {
        "precision": float(prec[1:][best]),
        "recall": float(rec[1:][best]),
        "alert_rate": float(alert_rates[best]),
        "score_threshold": th_star
    }
    return tau_star, info

def _make_pca_grid_for_segment(X_tr_norm, base_grid):
    """세그먼트별 사용 가능한 최대 차원으로 PCA 후보 자동 클리핑."""
    n, d = X_tr_norm.shape
    # PCA 이론상 최대 성분 수: min(n_samples, n_features)
    max_dim = max(1, min(n, d) - 1)  # -1은 여유(수치/랭크 문제 방지)
    grid = sorted({p for p in base_grid if isinstance(p, int) and 1 <= p <= max_dim})
    if not grid:
        grid = [max_dim]
    return grid, max_dim

# ----------------
# 학습 한 세그먼트 처리
# ----------------
def train_one_segment(df_seg, tag="global"):
    print(f"\n[Segment: {tag}] samples={len(df_seg)}  prevalence={df_seg['passorfail'].mean():.3%}")

    # 0) 전처리/피처 로드
    clf_bundle = joblib.load(CLF_PKL)
    clf_pipe   = clf_bundle["model"]
    feat_order = list(clf_bundle["feature_names"])
    prep       = clf_pipe.named_steps["preprocessor"]

    # --- OHE(handle_unknown="ignore") 강제 적용 ---
    ohe_changed = _force_ohe_handle_unknown(prep, mode="ignore")
    print(f"[{tag}] OneHotEncoder handle_unknown set to 'ignore' on {ohe_changed} encoder(s).")

    # passorfail이 feature_names에 있을 가능성 방어
    feat = [c for c in feat_order if c != "passorfail"]

    # feature 일치 검사
    miss = set(feat) - set(df_seg.columns)
    extra = set(df_seg.columns) - set(feat) - {"passorfail"}
    if miss:
        raise ValueError(f"[{tag}] 누락 컬럼: {sorted(miss)}")
    if extra:
        print(f"[{tag}] 참고: feature에 포함되지 않는 추가 컬럼 {len(extra)}개는 무시됩니다.")

    # 1) 전체 전처리 → stratified split
    X_all, y_all = _prepare_xy(prep, feat, df_seg)
    if len(np.unique(y_all)) < 2:
        raise ValueError(f"[{tag}] passorfail에 클래스가 1개뿐입니다. (검증 불가)")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X_all, y_all))
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]

    # 정상만으로 학습
    X_tr_norm = X_tr[y_tr == 0]
    if len(X_tr_norm) == 0:
        raise ValueError(f"[{tag}] 학습용 정상 표본이 없습니다.")

    # --- 여기서 PCA 후보를 자동 클리핑 ---
    pca_grid, max_dim = _make_pca_grid_for_segment(X_tr_norm, PCA_GRID_BASE)
    print(f"[{tag}] PCA dim candidates (clipped by max_dim={max_dim}): {pca_grid}")

    # 2) 그리드 탐색 (PR-AUC 1순위, ROC-AUC 2순위)
    best = None
    for p in pca_grid:
        for mcs in MCS_GRID:
            for ms in MS_GRID:
                for md in METHODS:
                    try:
                        r = _fit_eval_candidate(X_tr_norm, X_va, y_va, p, mcs, ms, md)
                    except ValueError as e:
                        # 혹시 수치/랭크 문제로 PCA/HDBSCAN 실패하면 해당 후보 스킵
                        print(f"[{tag}] skip PCA={p}, mcs={mcs}, ms={ms}, method={md} -> {e}")
                        continue
                    key = (r["PR_AUC"], r["ROC_AUC"])
                    if (best is None) or (key > (best["PR_AUC"], best["ROC_AUC"])):
                        best = {"pca": r["pca"], "hdb": r["hdb"],
                                "PR_AUC": r["PR_AUC"], "ROC_AUC": r["ROC_AUC"],
                                "p": p, "mcs": mcs, "ms": ms, "md": md,
                                "labels_val": r["labels_val"], "score_val": r["score_val"],
                                "base_noise_on_norm": r["base_noise_on_norm"]}

    if best is None:
        raise RuntimeError(f"[{tag}] 모든 후보가 실패했습니다. (표본수/특징수 점검)")

    print(f"[{tag}] best -> PCA={best['p']}, mcs={best['mcs']}, ms={best['ms']}, method={best['md']}, "
          f"PR-AUC={best['PR_AUC']:.3f}, ROC-AUC={best['ROC_AUC']:.3f}, "
          f"base_noise_on_norm={best['base_noise_on_norm']:.3%}")

    # 3) τ 캘리브레이션 (정밀도 하한 & 알람률 캡 동시 적용)
    tau_star, info = calibrate_tau_both(
        best["labels_val"], best["score_val"], y_va,
        min_precision=MIN_PRECISION, max_alert_rate=MAX_ALERT_RATE
    )
    print(f"[{tag}] tau*={tau_star:.3f} | precision={info['precision']:.3f} "
          f"recall={info['recall']:.3f} alert_rate={info['alert_rate']:.3f}")

    # 4) 번들 저장
    os.makedirs(SAVE_DIR, exist_ok=True)
    suffix = "" if tag == "global" else f"_{_sanitize_filename(tag)}"
    out_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}{suffix}.pkl")

    bundle = {
        "preprocessor": prep,            # OHE handle_unknown='ignore' 반영된 객체
        "pca": best["pca"],
        "hdb": best["hdb"],
        "tau": float(tau_star),
        "feature_order": feat,
        "meta": {
            "segment": tag,
            "pca_dim": best["p"],
            "min_cluster_size": best["mcs"],
            "min_samples": best["ms"],
            "method": best["md"],
            "PR_AUC_val": best["PR_AUC"],
            "ROC_AUC_val": best["ROC_AUC"],
            "base_noise_on_norm": best["base_noise_on_norm"],
            "calibration": {
                "min_precision": MIN_PRECISION,
                "max_alert_rate": MAX_ALERT_RATE,
                **info
            }
        }
    }
    joblib.dump(bundle, out_path)
    print(f"[{tag}] Saved: {out_path}")
    return out_path

# ----------------
# 실행 (글로벌 먼저 저장 → 세그먼트별 저장)
# ----------------
os.makedirs(SAVE_DIR, exist_ok=True)

# 0) 전처리 pkl 존재/구조 점검 및 feature_names-데이터 일치 확인
clf_bundle = joblib.load(CLF_PKL)
assert "model" in clf_bundle and "feature_names" in clf_bundle, \
    "final_model.pkl에는 {'model': Pipeline(...), 'feature_names': array([...])} 구조가 있어야 합니다."
feat_order = list(clf_bundle["feature_names"])
missing_global = set(feat_order) - set(train_df.columns)
if missing_global:
    raise ValueError(f"[global] train_df에 누락된 feature 컬럼이 있습니다: {sorted(missing_global)}")

# 1) 항상 글로벌 모델 하나 저장(폴백용)
train_one_segment(train_df, tag="global")

# 2) 세그먼트 분리 옵션
if SEGMENT_COL is not None:
    if SEGMENT_COL not in train_df.columns:
        print(f"[경고] SEGMENT_COL='{SEGMENT_COL}' 컬럼이 train_df에 없어 글로벌만 저장합니다.")
    else:
        for seg, df_seg in train_df.groupby(SEGMENT_COL, dropna=False):
            seg_name = "NA" if pd.isna(seg) else str(seg)
            try:
                train_one_segment(df_seg, tag=f"{SEGMENT_COL}={seg_name}")
            except Exception as e:
                print(f"[SKIP] {SEGMENT_COL}={seg_name} -> {e}")




import os, glob, joblib, datetime, numpy as np

# HDBSCAN predict에 필요
from hdbscan import HDBSCAN
from hdbscan.prediction import approximate_predict
from scipy import sparse

SAVE_DIR = "./hdbscan_models"            # pkl들이 있는 폴더
MODEL_BASENAME = "hdbscan_oneclass"      # 파일 접두
MASTER_PATH = os.path.join(SAVE_DIR, f"{MODEL_BASENAME}_MASTER.pkl")

def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

def _load_all_bundles(src_dir, pattern):
    paths = sorted(glob.glob(os.path.join(src_dir, pattern)))
    bundles = []
    for p in paths:
        try:
            b = joblib.load(p)
            b["_source_path"] = p
            bundles.append(b)
        except Exception as e:
            print(f"[WARN] load fail: {p} -> {e}")
    return bundles

def _pick_global(bundles):
    # meta.segment == "global" 을 최우선
    for b in bundles:
        if "meta" in b and b["meta"].get("segment") == "global":
            return b
    # 파일명이 접두만 있는 경우도 지원 (예: hdbscan_oneclass.pkl)
    for b in bundles:
        if os.path.basename(b.get("_source_path","")).startswith(MODEL_BASENAME + ".pkl"):
            return b
    # 마지막 대안: 첫 번째
    return bundles[0]

def build_and_save_master(src_dir=SAVE_DIR, basename=MODEL_BASENAME, master_path=MASTER_PATH):
    bundles = _load_all_bundles(src_dir, f"{basename}*.pkl")
    if not bundles:
        raise FileNotFoundError(f"No bundles matched: {os.path.join(src_dir, basename + '*.pkl')}")

    global_bundle = _pick_global(bundles)
    preprocessor = global_bundle["preprocessor"]
    feature_order = global_bundle["feature_order"]

    segments = {}
    for b in bundles:
        seg_key = b.get("meta", {}).get("segment", "unknown")
        # 세그먼트 항목에는 preprocessor는 제외해 용량 절약
        segments[seg_key] = {
            "pca": b["pca"],
            "hdb": b["hdb"],
            "tau": b["tau"],
            "meta": b.get("meta", {}),
        }

    master = {
        "model_name": basename,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "preprocessor": preprocessor,      # 공용
        "feature_order": feature_order,    # 공용
        "segments": segments,              # 각 세그먼트별 컴포넌트
        "routing": {
            "segment_col": "mold_code",    # 필요 시 바꾸세요
            "fallback": "global"
        },
        "version": 1,
    }

    # 압축 저장 (3은 적당한 속도/용량 절충)
    joblib.dump(master, master_path, compress=3)
    print(f"[OK] MASTER saved -> {master_path}  (segments={list(segments.keys())})")

if __name__ == "__main__":
    build_and_save_master()


# ============================================
# MASTER(HDBSCAN Router)로 train_df 평가
# - 양성(1) = "이상탐지"
# - MASTER 구조:
#   { 'preprocessor', 'feature_order',
#     'segments': {'global': {...}, 'mold_code=...': {...}, ...},
#     'routing': {'segment_col': 'mold_code', 'fallback': 'global'}
#   }
# ============================================
import joblib, numpy as np, pandas as pd
from scipy import sparse as sp
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from hdbscan.prediction import approximate_predict

MASTER_PKL = "./hdbscan_models/hdbscan_oneclass_MASTER.pkl"  # 경로 맞게

# ---------- helpers ----------
def _to_dense(X):
    return X.toarray() if sp.issparse(X) else X

def _seg_key(seg_col, val):
    if seg_col is None:
        return None
    return f"{seg_col}={'NA' if pd.isna(val) else str(val)}"

def _get_fallback_bundle(router: dict):
    """MASTER 구조에서 폴백 번들을 찾아 반환."""
    segments = router.get("segments", {})
    routing  = router.get("routing", {}) or {}
    fb_key   = routing.get("fallback")
    # 우선순위: routing.fallback -> segments['global'] -> router['global'](구버전) -> 첫 세그먼트
    if fb_key and fb_key in segments:
        return segments[fb_key]
    if "global" in segments:
        return segments["global"]
    if "global" in router:  # 혹시 옛 구조를 썼다면
        return router["global"]
    if len(segments) > 0:
        first_key = next(iter(segments))
        print(f"[WARN] fallback이 없어 '{first_key}' 세그먼트를 폴백으로 사용합니다.")
        return segments[first_key]
    raise KeyError("MASTER에서 폴백 번들을 찾을 수 없습니다.")

def route_predict_master(df_pred: pd.DataFrame, router: dict) -> pd.DataFrame:
    """
    MASTER 라우터(글로벌+세그먼트 한 파일)로 배치 예측.
    반환: DataFrame['anomaly_flag','anomaly_score']
    """
    prep    = router["preprocessor"]
    feat    = router["feature_order"]            # ★ top-level feature_order 사용
    segments = router.get("segments", {})
    seg_col  = (router.get("routing") or {}).get("segment_col", None)
    fallback_bundle = _get_fallback_bundle(router)

    # 컬럼 체크
    missing = set(feat) - set(df_pred.columns)
    if missing:
        raise ValueError(f"입력 데이터에 누락된 feature: {sorted(missing)}")

    out = pd.DataFrame(index=df_pred.index, columns=["anomaly_flag","anomaly_score"], dtype=float)
    out.loc[:, :] = 0.0

    # 그룹핑(세그먼트 라우팅)
    if seg_col and (seg_col in df_pred.columns):
        groups = df_pred.groupby(seg_col, dropna=False)
    else:
        groups = [(None, df_pred)]

    for seg_val, g in groups:
        key = _seg_key(seg_col, seg_val) if seg_col else None
        bundle = segments.get(key, fallback_bundle)

        Xt = _to_dense(prep.transform(g[feat]))
        Zt = bundle["pca"].transform(Xt)
        labels, strengths = approximate_predict(bundle["hdb"], Zt)
        labels = labels.astype(int); strengths = strengths.astype(float)

        score = np.where(labels == -1, 1.0, 1.0 - strengths)                  # 연속 이상점수(↑=이상)
        flag  = ((labels == -1) | (strengths < float(bundle["tau"]))).astype(int)  # 이진 이상 플래그

        out.loc[g.index, "anomaly_flag"]  = flag
        out.loc[g.index, "anomaly_score"] = score

    out["anomaly_flag"] = out["anomaly_flag"].astype(int)
    return out

def print_anomaly_metrics(y_true_defect, y_pred_anomaly, y_score, set_name="TRAIN"):
    """
    y_true_defect : 정답(불량 1, 정상 0)
    y_pred_anomaly: 모델의 '이상탐지' 플래그(이상 1, 정상 0) ← 양성(1) = 이상
    y_score       : 연속 이상점수 (클수록 이상)
    """
    y_true = np.asarray(y_true_defect).astype(int)
    y_pred = np.asarray(y_pred_anomaly).astype(int)
    score  = np.asarray(y_score).astype(float)

    # 매크로 지표 & 정확도
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)

    # 연속 점수 기반
    try: roc = roc_auc_score(y_true, score)
    except ValueError: roc = float("nan")
    try: pr_auc = average_precision_score(y_true, score)
    except ValueError: pr_auc = float("nan")

    # 양성=이상(1) 기준
    prec1, rec1, f1_1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # 혼동행렬 (tn, fp, fn, tp)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

    # ----- 출력 (캡처 포맷) -----
    print(f"[{set_name} 성능(전체 macro 기준)]")
    print(f"Precision-macro : {prec_macro}")
    print(f"Recall-macro    : {rec_macro}")
    print(f"F1-macro        : {f1_macro}")
    print(f"Balanced Acc    : {bacc}")
    print(f"ROC-AUC         : {roc}")
    print(f"Accuracy        : {acc}\n")

    print(f"[참고: 이상(1) 기준 - {set_name}]")
    print(f"Precision(1)    : {prec1}")
    print(f"Recall(1)       : {rec1}")
    print(f"F1(1)           : {f1_1}")
    print(f"Confusion Matrix {set_name} (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})\n")

    print(f"[Classification Report - {set_name}]")
    print(classification_report(
        y_true, y_pred, labels=[0,1],
        target_names=["정상판정(0)", "이상탐지(1)"], digits=3, zero_division=0
    ))

# ---------- run ----------
# 전제: train_df 가 메모리에 있고, 정답 라벨은 'passorfail'(불량=1, 정상=0)
router = joblib.load(MASTER_PKL)

pred = route_predict_master(train_df, router)

# 성능 출력 (양성=이상)
print_anomaly_metrics(
    y_true_defect=train_df["passorfail"].astype(int).values,
    y_pred_anomaly=pred["anomaly_flag"].astype(int).values,
    y_score=pred["anomaly_score"].astype(float).values,
    set_name="TRAIN"
)





model = joblib.load('data/hdbscan_router.pkl')


df = pd.read_csv("data/test.csv", encoding="utf-8")
df["timestamp"] = pd.to_datetime(df["registration_time"], errors="coerce")
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['hour'] = df['registration_time'].dt.hour.astype(object)

# ===== 0) 준비 =====
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from hdbscan.prediction import approximate_predict

PKL_PATH = "hdbscan_router.pkl"   # <- pkl 경로
TIME_COL = "timestamp"            # <- 시계열 오버레이에 쓸 시간 컬럼명 (예: 'timestamp')
Y_COL    = "molten_volume"        # <- 시계열 오버레이에 쓸 값 컬럼명 (예: 'cast_pressure')

# ===== 1) 로더 & 라우팅 =====
router = joblib.load(PKL_PATH)

SEG_COL = router.get("segment_col", None)
FEATURES = router["global"]["feature_order"]
GLOBAL_PRE  = router["preprocessor"]
GLOBAL_PCA  = router["global"]["pca"]          # n_components=20 로 학습됨
GLOBAL_HDB  = router["global"]["hdb"]

SEG_MODELS = router.get("segments", {})        # 세그먼트별 모델이 있으면 활용, 없으면 global 사용

def _get_models(seg_value):
    """세그먼트 값에 맞는 (preprocessor, pca, hdb) 반환. 없으면 global."""
    m = SEG_MODELS.get(seg_value, None)
    if m is None:
        return GLOBAL_PRE, GLOBAL_PCA, GLOBAL_HDB
    return m["preprocessor"], m["pca"], m["hdb"]

# ===== 2) 변환 → 예측 =====
def hdbscan_infer(df: pd.DataFrame):
    """
    df: 새 샘플 포함 데이터프레임(학습과 동일 스키마)
    반환: df에 예측 라벨/확률 컬럼 추가, 그리고 2D 임베딩(Z2)도 반환
    """
    if SEG_COL is None:
        pre, pca, hdb = GLOBAL_PRE, GLOBAL_PCA, GLOBAL_HDB
        X = pre.transform(df[FEATURES])
        Xp = pca.transform(X)                 # (n, 20)
        labels, strengths = approximate_predict(hdb, Xp)  # 라벨 & membership probability
        Z2 = Xp[:, :2]                        # 2D 시각화는 PCA20 중 앞 2축 사용
        out = df.copy()
        out["_hdb_label"] = labels
        out["_hdb_strength"] = strengths
        return out, Z2, hdb
    else:
        # 세그먼트별로 처리
        outs, Z2s = [], []
        hdb_used = None
        for seg_value, g in df.groupby(SEG_COL, sort=False):
            pre, pca, hdb = _get_models(seg_value)
            X = pre.transform(g[FEATURES])
            Xp = pca.transform(X)
            labels, strengths = approximate_predict(hdb, Xp)
            z2 = Xp[:, :2]
            gg = g.copy()
            gg["_hdb_label"] = labels
            gg["_hdb_strength"] = strengths
            outs.append(gg)
            Z2s.append(pd.DataFrame(z2, index=gg.index, columns=["z1","z2"]))
            hdb_used = hdb  # 임의 반환용
        out = pd.concat(outs).loc[df.index]        # 원래 순서 유지
        Z2 = pd.concat(Z2s).loc[df.index].values
        return out, Z2, hdb_used

# ===== 3) 플롯 유틸 =====
def plot_embedding(Z2, labels, strengths, title="HDBSCAN embedding (PCA-2D)"):
    plt.figure(figsize=(7,6))
    # 노이즈(-1) 먼저
    noise = labels == -1
    plt.scatter(Z2[noise,0], Z2[noise,1], s=(10+60*strengths[noise]),
                alpha=0.35, c="lightgray", label="Noise (-1)")
    # 각 클러스터
    for k in np.unique(labels[~noise]):
        idx = labels == k
        plt.scatter(Z2[idx,0], Z2[idx,1], s=(10+60*strengths[idx]),
                    alpha=0.8, label=f"Cluster {k}")
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout(); plt.show()

def plot_condensed_tree(hdb):
    if getattr(hdb, "condensed_tree_", None) is None:
        print("⚠️ condensed_tree_가 없습니다. 학습 시 prediction_data=True & store_condensed_tree=True 권장.")
        return
    ax = hdb.condensed_tree_.plot(select_clusters=True, label_clusters=True)
    ax.set_title("HDBSCAN Condensed Tree (stability)")

def plot_outlier_hist_training(hdb, quantiles=(0.95,0.98,0.995)):
    """
    학습 데이터에 대한 outlier_scores_ 분포(새 데이터 아님).
    """
    try:
        scores = hdb.outlier_scores_  # 학습셋 기준 (접근 시 계산)
    except Exception as e:
        print("⚠️ outlier_scores_ 계산 불가:", e)
        return
    qs = np.quantile(scores, quantiles)
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=40, alpha=0.85)
    for qv, q in zip(qs, quantiles):
        plt.axvline(qv, linestyle="--", label=f"q{int(q*100)}={qv:.3f}")
    plt.title("HDBSCAN Outlier Scores (train set)")
    plt.xlabel("outlier score"); plt.ylabel("count")
    plt.legend(); plt.tight_layout(); plt.show()
    print({f"q{int(q*100)}": float(v) for q, v in zip(quantiles, qs)})

def plot_timeseries_with_anomalies(df, time_col, y_col,
                                   labels_col="_hdb_label",
                                   strength_col="_hdb_strength",
                                   strength_thr=0.15):
    """
    라벨 -1을 1차 이상치로, membership prob < strength_thr도 보조 표식으로.
    """
    if (time_col not in df.columns) or (y_col not in df.columns):
        print(f"⚠️ '{time_col}' 또는 '{y_col}' 컬럼이 없어 시계열 플롯을 건너뜁니다.")
        return
    t = pd.to_datetime(df[time_col])
    y = df[y_col].values
    labels = df[labels_col].values
    strengths = df[strength_col].values

    plt.figure(figsize=(11,4))
    plt.plot(t, y, linewidth=1.2, label=y_col)

    # 노이즈(-1)
    noise = labels == -1
    plt.scatter(t[noise], y[noise], s=36, marker="o", label="Noise (-1)")

    # 낮은 membership (보조 이상표식)
    weak = strengths < strength_thr
    plt.scatter(t[weak], y[weak], s=48, marker="x", label=f"strength<{strength_thr:.2f}")

    plt.title(f"{y_col} vs Time — HDBSCAN anomalies overlay")
    plt.xlabel("time"); plt.ylabel(y_col); plt.legend()
    plt.ylim(0,400)
    plt.tight_layout(); plt.show()


out, Z2, hdb_used = hdbscan_infer(df)
plot_embedding(Z2, out["_hdb_label"].values, out["_hdb_strength"].values,
                title="HDBSCAN (PCA-2D): label & membership")
plot_condensed_tree(hdb_used)
plot_outlier_hist_training(hdb_used, quantiles=(0.97,0.99,0.999))
plot_timeseries_with_anomalies(out, TIME_COL, Y_COL, strength_thr=0.15)

noise = (out["_hdb_label"] == -1).values
weak  = (out["_hdb_strength"] < 0.15).values  # 너의 임계 사용

def visible_rate(col, mask):
    vis = np.isfinite(out[col].values) & mask
    return (weak & vis).sum() / vis.sum(), (noise & vis).sum() / vis.sum()

# 1) 각 변수 “자기 분모”로 본 비율
r_w_mv, r_n_mv = visible_rate("molten_volume", np.ones(len(out), dtype=bool))
r_w_cp, r_n_cp = visible_rate("cast_pressure", np.ones(len(out), dtype=bool))
print("molten_volume  weak/noise:", r_w_mv, r_n_mv)
print("cast_pressure  weak/noise:", r_w_cp, r_n_cp)

# 2) 두 변수 모두 유효한 “공통 분모”로 본 비율(정답처럼 같아야 함)
common = np.isfinite(out["molten_volume"].values) & np.isfinite(out["cast_pressure"].values)
r_w_common_mv, r_n_common_mv = visible_rate("molten_volume", common)
r_w_common_cp, r_n_common_cp = visible_rate("cast_pressure", common)
print("COMMON weak/noise (mv):", r_w_common_mv, r_n_common_mv)
print("COMMON weak/noise (cp):", r_w_common_cp, r_n_common_cp)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_multi_ts_anomalies(
    df: pd.DataFrame,
    time_col: str,
    y_cols: list[str],
    label_col: str = "_hdb_label",
    ncols: int = 2,
    figsize_per_ax=(6.0, 2.6),
    trim_frac: float = 0.001,   # 각 변수별 하위/상위 이 만큼(0.1%=0.001) 잘라냄
    savepath: str | None = None,
):
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' 컬럼이 없습니다.")
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' 컬럼이 없습니다.")
    if not y_cols:
        raise ValueError("y_cols에 최소 1개 이상의 변수명을 넣어주세요.")

    # 공통 시간/라벨
    t_all   = pd.to_datetime(df[time_col], errors="coerce")
    lab_all = df[label_col].values
    anom_all = (lab_all == -1)

    n = len(y_cols)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True,
        figsize=(figsize_per_ax[0]*ncols, figsize_per_ax[1]*nrows)
    )
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for i, y_col in enumerate(y_cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        if y_col not in df.columns:
            ax.text(0.5, 0.5, f"Missing column: {y_col}", ha="center", va="center")
            ax.axis("off")
            continue

        y_all = pd.to_numeric(df[y_col], errors="coerce")
        valid = t_all.notna() & np.isfinite(y_all)

        if valid.sum() < 5:
            ax.text(0.5, 0.5, f"Not enough valid data: {y_col}", ha="center", va="center")
            ax.axis("off")
            continue

        # 분위수 기반 트림 경계 (양쪽 0.1% 기본)
        q_lo, q_hi = np.nanquantile(y_all[valid], [trim_frac, 1.0 - trim_frac])
        in_rng = valid & (y_all >= q_lo) & (y_all <= q_hi)

        # 트림된 데이터만 플롯
        t = t_all[in_rng]
        y = y_all[in_rng]
        anom = anom_all[in_rng]

        ax.plot(t, y, lw=1.2, label=y_col)
        # 확정 이상(라벨 -1)만 빨간 X로
        ax.scatter(t[anom], y[anom], marker='x', s=46, linewidths=1.6,
                   color="red", label="Anomaly (label = -1)")

        ax.set_ylabel(y_col)
        ax.set_ylim(q_lo, q_hi)  # 축도 트림 범위로 고정
        if r == 0 and c == 0:
            ax.legend(loc="upper left")

    # 남는 빈 축 제거
    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    axes[-1, 0].set_xlabel("time")
    fig.autofmt_xdate()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

plot_multi_ts_anomalies(
    out, time_col="timestamp",
    y_cols=["molten_volume", "cast_pressure", "upper_mold_temp1",'physical_strength','sleeve_temperature',
            'biscuit_thickness','Coolant_temperature'],
    trim_frac=0.01  # 필요시 0.002(0.2%) 등으로 조절
)

out.columns
out['_hdb_strength']







import numpy as np
import pandas as pd
import joblib

# optional: 카이제곱 꼬리확률(없으면 min-max로 대체)
try:
    from scipy.stats import chi2
    _HAS_CHI2 = True
except Exception:
    _HAS_CHI2 = False

try:
    from hdbscan.prediction import approximate_predict
    _HAS_APPROX = True
except Exception:
    _HAS_APPROX = False


def attach_hdbscan_anomaly_proba(df: pd.DataFrame, model_dict: dict,
                                 proba_col="_anom_proba",
                                 score_col="_anom_score",
                                 label_col="_hdb_label",
                                 strength_col="_hdb_strength"):
    """
    df 에 다음 컬럼을 추가해 반환:
      - _hdb_label    : HDBSCAN 라벨 (−1이면 확정 이상)
      - _hdb_strength : 멤버십 강도(0~1, 클수록 정상)
      - _anom_score   : 0~1 이상도 점수(클수록 이상)
      - _anom_proba   : 0~1 '이상확률' (보정/꼬리확률 기반)

    * prediction_data_ 가 없는 HDBSCAN이면 approximate_predict 대신
      '로버스트 거리 -> 카이제곱 꼬리확률' 대체 경로 사용.
    """

    # ---- 공통 자원 꺼내기 ----
    FEATURES = model_dict.get("feature_order") or model_dict.get("global", {}).get("feature_order")
    if FEATURES is None:
        raise ValueError("feature_order 를 pkl에서 찾을 수 없습니다.")
    PRE = model_dict.get("preprocessor") or model_dict.get("global", {}).get("preprocessor")
    if PRE is None:
        raise ValueError("preprocessor 를 pkl에서 찾을 수 없습니다.")

    routing = model_dict.get("routing", {})
    seg_col = routing.get("segment_col")  # 예: 'mold_code'
    seg_models = model_dict.get("segments") or model_dict.get("models") or {}
    glob = model_dict.get("global", {})
    PCA_G = glob.get("pca")
    HDB_G = glob.get("hdb")

    def _fetch(seg_value):
        m = seg_models.get(seg_value, {}) if seg_models else {}
        pre = m.get("preprocessor", PRE)
        pca = m.get("pca", PCA_G)
        hdb = m.get("hdb", HDB_G)
        return pre, pca, hdb

    def _ecdf_tail(ref, vals):
        ref_sorted = np.sort(ref)
        idx = np.searchsorted(ref_sorted, vals, side="right")
        return 1.0 - idx / (len(ref_sorted) + 1.0)

    out_chunks = []

    # ---- 세그먼트별 처리 또는 글로벌 ----
    if seg_col and seg_col in df.columns and len(seg_models) > 0:
        groups = df.groupby(seg_col, sort=False)
    else:
        # 세그먼트 없음 → 전체를 한 번에
        groups = [(None, df)]

    for seg_value, g in groups:
        pre, pca, hdb = _fetch(seg_value)

        # 전처리 → (선택)PCA
        X = pre.transform(g[FEATURES])
        Xp = pca.transform(X) if pca is not None else X

        # 초기화
        labels = np.full(len(g), -1, dtype=int)
        strengths = np.zeros(len(g), dtype=float)
        s_raw = np.zeros(len(g), dtype=float)
        p_like = np.zeros(len(g), dtype=float)

        # ===== 경로 1: prediction_data_ 있고 approx 가능 → 정식 예측 =====
        if (hdb is not None) and _HAS_APPROX and (getattr(hdb, "prediction_data_", None) is not None):
            lbl, stren = approximate_predict(hdb, Xp)
            labels[:] = lbl.astype(int)
            strengths[:] = stren.astype(float)
            s_raw[:] = np.where(labels == -1, 1.0, 1.0 - strengths)

            # 학습 분포 기반 eCDF 꼬리확률(라벨이 없을 때도 쓸 수 있음)
            lab_tr = getattr(hdb, "labels_", None)
            prob_tr = getattr(hdb, "probabilities_", None)
            if lab_tr is not None and prob_tr is not None:
                s_ref = np.where(lab_tr == -1, 1.0, 1.0 - prob_tr)
                ref = s_ref[lab_tr != -1] if np.any(lab_tr != -1) else s_ref
                p_like[:] = _ecdf_tail(ref, s_raw)
            else:
                # fallback: min-max
                smin, sptp = float(np.min(s_raw)), float(np.ptp(s_raw) + 1e-9)
                p_like[:] = (s_raw - smin) / sptp

        else:
            # ===== 경로 2: prediction_data_ 없음 → 대체 점수 =====
            # ColumnTransformer 첫 블록이 'num'이라고 가정(니 pkl 구조가 그랬음)
            num_cols = list(pre.transformers_[0][2])  # 수치형 원컬럼 목록
            n_num = len(num_cols)
            # 전처리 출력에서 수치 파트 추출(보통 앞부분)
            X_num = X[:, :n_num] if n_num > 0 else X

            # 로버스트 스케일 공간의 L2 거리로 이상도 점수
            # (RobustScaler -> 중앙=0, IQR=1: 표준정규 가정시 제곱합은 ~카이제곱)
            rd2 = np.sum(np.square(X_num), axis=1)  # 거리^2
            # 점수(0~1): 값 클수록 이상
            if _HAS_CHI2 and n_num >= 1:
                p_like[:] = chi2.sf(rd2, df=n_num)   # 카이제곱 꼬리확률
                s_raw[:] = p_like
            else:
                # scipy 없으면 rank→0~1
                order = rd2.argsort().argsort().astype(float)
                p_like[:] = order / (len(rd2) - 1 + 1e-9)
                s_raw[:] = p_like

            # label/strength은 모름 → 보수적으로 strength=0, label은 규칙으로 부여 가능
            # 여기선 label은 건드리지 않고(전부 0으로 두지 않음), 확률만 제공

        chunk = g.copy()
        chunk[label_col] = labels
        chunk[strength_col] = strengths
        chunk[score_col] = s_raw
        chunk[proba_col] = p_like
        out_chunks.append(chunk)

    out = pd.concat(out_chunks).loc[df.index]
    return out

# 1) 모델 로드
model = joblib.load("hdbscan_router.pkl")  # 너의 pkl 경로

# 2) df에 확률 붙이기
df_with_proba = attach_hdbscan_anomaly_proba(df, model)

# 3) 확인
df_with_proba[["_hdb_label", "_hdb_strength", "_anom_score", "_anom_proba"]].head()








# ============================================
# HDBSCAN 성능평가 (vs passorfail) - train_df 대상
# - 글로벌 모델만 평가
# - 세그먼트 모델 있으면 라우팅 평가(없으면 글로벌 폴백)
# ============================================
import os, glob, joblib, numpy as np, pandas as pd
from scipy import sparse
import hdbscan
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)

# ----------------
# 설정
# ----------------
SAVE_DIR     = "./hdbscan_models"   # 앞서 저장한 폴더
MODEL_NAME   = "hdbscan_oneclass"   # 파일 접두
SEGMENT_COL  = "mold_code"          # 세그먼트 기준 컬럼 (없으면 None)

assert "train_df" in globals(), "train_df가 메모리에 있어야 합니다."
assert "passorfail" in train_df.columns, "train_df에 passorfail(1=불량,0=정상) 컬럼이 필요합니다."

# ----------------
# 유틸
# ----------------
def _to_dense(X):
    from scipy import sparse
    return X.toarray() if sparse.issparse(X) else X

def _anom_score_and_pred(prep, pca, hdb, tau, df_part, feature_order):
    # 동일 전처리
    X = prep.transform(df_part[feature_order])
    X = _to_dense(X)
    Z = pca.transform(X) if pca is not None else X

    # 예측
    labels, strengths = hdbscan.approximate_predict(hdb, Z)
    labels    = labels.astype(int)
    strengths = strengths.astype(float)

    # 연속 점수(클수록 이상)
    score = np.where(labels == -1, 1.0, 1.0 - strengths)
    # 이진 판정(현재 τ)
    y_pred = ((labels == -1) | ((labels != -1) & (strengths < float(tau)))).astype(int)

    return y_pred, score, labels, strengths

def _eval_metrics(y_true, y_pred, score):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])   # [[TP,FN],[FP,TN]]
    TP, FN = int(cm[0,0]), int(cm[0,1])
    FP, TN = int(cm[1,0]), int(cm[1,1])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    roc  = roc_auc_score(y_true, score)
    pr   = average_precision_score(y_true, score)  # PR-AUC (불균형 데이터에 유리)
    prev = float(np.mean(y_true))
    pred_rate = float(np.mean(y_pred))

    return {
        "confusion_matrix [[TP,FN],[FP,TN]]": [[TP, FN], [FP, TN]],
        "precision@τ": float(prec),
        "recall@τ": float(rec),
        "f1@τ": float(f1),
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr),
        "prevalence(실제 이상 비율)": prev,
        "predicted_rate@τ(예측 이상 비율)": pred_rate,
    }

def _print_report(title, y_true, y_pred, score):
    print(f"\n=== {title} ===")
    metrics = _eval_metrics(y_true, y_pred, score)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

# ----------------
# 모델 로딩(글로벌 + 세그먼트)
# ----------------
# 글로벌
global_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
assert os.path.exists(global_path), f"글로벌 모델이 없습니다: {global_path}"
global_bundle = joblib.load(global_path)

# 세그먼트 맵(있으면)
seg_bundles = {}  # {"mold_code=A123": bundle, ...}
for p in glob.glob(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{SEGMENT_COL}=*.pkl")):
    tag = os.path.splitext(os.path.basename(p))[0].replace(MODEL_NAME + "_", "")
    seg_bundles[tag] = joblib.load(p)

# ----------------
# 1) 글로벌 모델만 평가
# ----------------
gb = global_bundle
feat = list(gb["feature_order"])

missing = set(feat) - set(train_df.columns)
if missing:
    raise ValueError(f"[global] train_df에 누락된 feature 컬럼: {sorted(missing)}")

y_true = train_df["passorfail"].astype(int).values
y_pred, score, labels, strengths = _anom_score_and_pred(
    gb["preprocessor"], gb["pca"], gb["hdb"], gb["tau"], train_df, feat
)
_print_report("Train 성능 (글로벌 모델, τ 기준)", y_true, y_pred, score)

# ----------------
# 2) 세그먼트 라우팅 평가 (세그먼트 모델 있으면 사용, 없으면 글로벌 폴백)
# ----------------
if (SEGMENT_COL is not None) and (SEGMENT_COL in train_df.columns) and (len(seg_bundles) > 0):
    y_true_all = []
    y_pred_all = []
    score_all  = []

    for seg, df_seg in train_df.groupby(SEGMENT_COL, dropna=False):
        seg_name = "NA" if pd.isna(seg) else str(seg)
        key = f"{SEGMENT_COL}={seg_name}"
        bundle = seg_bundles.get(key, global_bundle)  # 없으면 글로벌 폴백

        feat = list(bundle["feature_order"])
        miss = set(feat) - set(df_seg.columns)
        if miss:
            print(f"[SKIP] {key} -> 누락 컬럼 {sorted(miss)} (글로벌로 대체)")
            bundle = global_bundle
            feat   = list(bundle["feature_order"])

        yp, sc, _, _ = _anom_score_and_pred(
            bundle["preprocessor"], bundle["pca"], bundle["hdb"], bundle["tau"], df_seg, feat
        )
        y_true_all.append(df_seg["passorfail"].astype(int).values)
        y_pred_all.append(yp)
        score_all.append(sc)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    score_all  = np.concatenate(score_all)

    _print_report("Train 성능 (세그먼트 라우팅, τ 기준)", y_true_all, y_pred_all, score_all)
else:
    print("\n[알림] 세그먼트 모델이 없거나 SEGMENT_COL이 없어 라우팅 평가는 생략했습니다.")




# ============================================
# Classification report (기준: "hdbscan 유무")
#  - y_true_anom: HDBSCAN hdbscan 플래그(1=이상)
#  - y_pred_anom: passorfail(1=불량)을 이용해 "이상을 맞췄는가"로 간주
#  - 출력 형식은 요청하신 포맷으로 맞춤
# ============================================
import os, joblib, numpy as np, pandas as pd
from scipy import sparse
import hdbscan
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)

SAVE_DIR   = "./hdbscan_models"
MODEL_NAME = "hdbscan_oneclass"
FALLBACK_BUNDLE = "hdbscan_bundle.pkl"  # (구버전 단일 파일일 때)

assert "train_df" in globals(), "train_df가 메모리에 있어야 합니다."
assert "passorfail" in train_df.columns, "train_df에 passorfail(1=불량,0=정상) 필요."

def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

# --- 글로벌 번들 로드 ---
global_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
if os.path.exists(global_path):
    bundle = joblib.load(global_path)
elif os.path.exists(FALLBACK_BUNDLE):
    bundle = joblib.load(FALLBACK_BUNDLE)
else:
    raise FileNotFoundError(f"모델 파일이 없습니다: {global_path} 또는 {FALLBACK_BUNDLE}")

prep   = bundle["preprocessor"]
pca    = bundle.get("pca", None)
hdb    = bundle["hdb"]
tau    = float(bundle["tau"])
feat   = list(bundle["feature_order"])

# --- 전처리 & HDBSCAN 점수/플래그 ---
missing = set(feat) - set(train_df.columns)
if missing:
    raise ValueError(f"[global] 누락 feature: {sorted(missing)}")

X = prep.transform(train_df[feat])
X = _to_dense(X)
Z = pca.transform(X) if pca is not None else X

labels, strengths = hdbscan.approximate_predict(hdb, Z)
labels    = labels.astype(int)
strengths = strengths.astype(float)

# 연속 이상점수(클수록 이상)
score_anom = np.where(labels == -1, 1.0, 1.0 - strengths)
# 이진 hdbscan 플래그(τ 적용) → 이것을 "정답(y_true_anom)"으로 사용
y_true_anom = ((labels == -1) | ((labels != -1) & (strengths < tau))).astype(int)

# passorfail(불량)을 이용해 '이상을 맞췄는가'를 예측으로 간주
y_pred_anom = train_df["passorfail"].astype(int).values  # 1=불량

# --- 지표 계산 (양성=이상=1) ---
cm = confusion_matrix(y_true_anom, y_pred_anom, labels=[1, 0])  # [[TP,FN],[FP,TN]] (truth=anomaly)
TP, FN = int(cm[0, 0]), int(cm[0, 1])
FP, TN = int(cm[1, 0]), int(cm[1, 1])

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true_anom, y_pred_anom, average="binary", pos_label=1, zero_division=0
)

# 연속 점수는 "이상 확률"에 해당하므로, AUC도 이상 기준으로 계산
roc = roc_auc_score(y_true_anom, score_anom)
pr  = average_precision_score(y_true_anom, score_anom)

prevalence_anom = float(np.mean(y_true_anom))      # 실제 "이상" 비율(=알람률)
predicted_rate  = float(np.mean(y_pred_anom))      # 여기서는 불량률(=예측측면)

print("=== Train 성능 (hdbscan 기준, τ 기준) ===")
print(f"confusion_matrix [[TP,FN],[FP,TN]]: [[{TP}, {FN}], [{FP}, {TN}]]")
print(f"precision@τ: {prec}")
print(f"recall@τ: {rec}")
print(f"f1@τ: {f1}")
print(f"ROC_AUC: {roc}")
print(f"PR_AUC: {pr}")
print(f"prevalence(실제 '이상' 비율): {prevalence_anom}")
print(f"predicted_rate(예측측 비율=불량률): {predicted_rate}\n")

# 분류 리포트도 '이상=1 / 정상=0' 클래스로 표시
print("Classification report (기준: hdbscan):")
print(classification_report(
    y_true_anom, y_pred_anom,
    labels=[0,1],
    target_names=["anomaly=0 (정상판정)", "anomaly=1 (hdbscan)"],
    digits=3
))






# 모든 세그먼트 모델을 robust하게 로딩(파일명에 '='→'_' 바뀐 경우 포함)
import os, glob, joblib, re

SAVE_DIR   = "./hdbscan_models"
MODEL_NAME = "hdbscan_oneclass"
SEGMENT_COL = "mold_code"  # 세그먼트 기준 컬럼

# 1) 글로벌 모델
global_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
global_bundle = joblib.load(global_path)

# 2) 세그먼트 모델들(파일명 패턴과 무관하게 meta.segment를 키로 사용)
seg_bundles = {}  # 예: {"mold_code=8412": bundle, "mold_code=8573": bundle, ...}

for p in glob.glob(os.path.join(SAVE_DIR, f"{MODEL_NAME}_*.pkl")):
    if os.path.basename(p) == f"{MODEL_NAME}.pkl":
        continue  # 글로벌은 스킵
    b = joblib.load(p)
    seg_tag = b.get("meta", {}).get("segment")  # 학습 때 저장된 원래 태그("mold_code=8412")
    if not seg_tag:  # 메타가 없다면 파일명에서 최대한 유추
        base = os.path.splitext(os.path.basename(p))[0]
        # hdbscan_oneclass_mold_code_8412 → mold_code=8412 로 복원 시도
        m = re.match(rf"{re.escape(MODEL_NAME)}_(.+)", base)
        seg_tag = m.group(1).replace("_", "=", 1) if m else base
    seg_bundles[seg_tag] = b

print(f"Loaded global + {len(seg_bundles)} segment models")
print("examples:", list(seg_bundles.keys())[:5])

# 세그먼트 라우팅 시 선택 로직
def pick_bundle_for_row(seg_value):
    seg_name = "NA" if pd.isna(seg_value) else str(seg_value)
    key_eq = f"{SEGMENT_COL}={seg_name}"           # 학습 당시 meta.segment 키
    key_us = f"{SEGMENT_COL}_{seg_name}"           # 혹시 모를 대체 키
    return seg_bundles.get(key_eq) or seg_bundles.get(key_us) or global_bundle





# ============================================
# HDBSCAN 성능평가 (세그먼트 라우팅, train_df) - 인덱스 리셋 버전
# - 저장된 모든 세그먼트 모델 사용
# - 평가 2종:
#   A) 불량 기준  : y_true = passorfail, y_pred = anomaly_flag
#   B) 이상탐지 기준: y_true = anomaly_flag, y_pred = passorfail
# ============================================
import os, glob, re, joblib, numpy as np, pandas as pd
from scipy import sparse
import 이상탐지
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)

# ----------------
# 설정
# ----------------
SAVE_DIR     = "./hdbscan_models"
MODEL_NAME   = "hdbscan_oneclass"
SEGMENT_COL  = "mold_code"
FALLBACK_BUNDLE = "hdbscan_bundle.pkl"  # (단일 번들만 있을 때 폴백)

assert "train_df" in globals(), "train_df가 메모리에 있어야 합니다."
assert "passorfail" in train_df.columns, "train_df에 passorfail(1=불량,0=정상) 컬럼이 필요합니다."

# ★ 인덱스 리셋(중요) — 여기서부터 이 df만 사용
base_df = train_df.reset_index(drop=True).copy()

# ----------------
# 유틸
# ----------------
def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

def _load_all_bundles():
    # 글로벌
    global_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
    if os.path.exists(global_path):
        gb = joblib.load(global_path)
    elif os.path.exists(FALLBACK_BUNDLE):
        gb = joblib.load(FALLBACK_BUNDLE)
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {global_path} 또는 {FALLBACK_BUNDLE}")

    # 세그먼트(파일명 무엇이든 meta.segment로 키)
    seg_map = {}
    for p in glob.glob(os.path.join(SAVE_DIR, f"{MODEL_NAME}_*.pkl")):
        if os.path.basename(p) == f"{MODEL_NAME}.pkl":
            continue
        b = joblib.load(p)
        seg_tag = b.get("meta", {}).get("segment")
        if not seg_tag:
            base = os.path.splitext(os.path.basename(p))[0]
            m = re.match(rf"{re.escape(MODEL_NAME)}_(.+)", base)
            seg_tag = m.group(1).replace("_", "=", 1) if m else base
        seg_map[seg_tag] = b
    return gb, seg_map

def _apply_bundle(bundle, df_part):
    feat = list(bundle["feature_order"])
    missing = set(feat) - set(df_part.columns)
    if missing:
        # feature가 안 맞으면 상위에서 글로벌 폴백하도록 예외 던짐
        raise ValueError(f"누락 feature: {sorted(missing)}")
    X = bundle["preprocessor"].transform(df_part[feat])
    X = _to_dense(X)
    pca = bundle.get("pca", None)
    Z = pca.transform(X) if pca is not None else X
    lab, st = 이상탐지.approximate_predict(bundle["hdb"], Z)
    lab = lab.astype(int); st = st.astype(float)
    tau = float(bundle["tau"])
    # 이상 플래그(1=이상)
    y_anom = ((lab == -1) | ((lab != -1) & (st < tau))).astype(int)
    # 연속 이상점수(클수록 이상)
    score  = np.where(lab == -1, 1.0, 1.0 - st)
    return y_anom, score

def _route_predict(df, gb, seg_map):
    n = len(df)
    y_anom = np.zeros(n, dtype=int)
    score  = np.zeros(n, dtype=float)

    # 세그먼트 없으면 전체 글로벌
    if SEGMENT_COL not in df.columns or len(seg_map) == 0:
        ya, sc = _apply_bundle(gb, df)
        y_anom[:] = ya; score[:] = sc
        return y_anom, score

    # 세그먼트별 처리 — ★ df는 reset_index 상태라 pos 인덱스가 0..n-1
    for seg, df_seg in df.groupby(SEGMENT_COL, dropna=False):
        pos = df_seg.index.to_numpy()  # 0..n-1 (연속)
        seg_name = "NA" if pd.isna(seg) else str(seg)
        key_eq = f"{SEGMENT_COL}={seg_name}"
        key_us = f"{SEGMENT_COL}_{seg_name}"
        b = seg_map.get(key_eq) or seg_map.get(key_us) or gb
        try:
            ya, sc = _apply_bundle(b, df_seg)
        except Exception:
            ya, sc = _apply_bundle(gb, df_seg)
        y_anom[pos] = ya
        score[pos]  = sc
    return y_anom, score

def _print_metrics(title, y_true, y_pred, score):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # [[TP,FN],[FP,TN]]
    TP, FN = int(cm[0,0]), int(cm[0,1])
    FP, TN = int(cm[1,0]), int(cm[1,1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    roc = roc_auc_score(y_true, score)
    pr  = average_precision_score(y_true, score)

    print(f"=== {title} ===")
    print(f"confusion_matrix [[TP,FN],[FP,TN]]: [[{TP}, {FN}], [{FP}, {TN}]]")
    print(f"precision@τ: {prec}")
    print(f"recall@τ: {rec}")
    print(f"f1@τ: {f1}")
    print(f"ROC_AUC: {roc}")
    print(f"PR_AUC: {pr}")
    print(f"prevalence(실제 양성 비율): {float(np.mean(y_true))}")
    print(f"predicted_rate@τ(예측 양성 비율): {float(np.mean(y_pred))}\n")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=3))

# ----------------
# 실행
# ----------------
global_bundle, seg_bundles = _load_all_bundles()

# 세그먼트 라우팅으로 이상탐지 플래그/점수 생성 (base_df 사용!)
y_anom_flag, anom_score = _route_predict(base_df, global_bundle, seg_bundles)

# A) 불량 기준: y_true=passorfail, y_pred=이상탐지
y_true_defect = base_df["passorfail"].astype(int).values
_print_metrics("Train 성능 (세그먼트 라우팅, 불량 기준 vs 이상탐지, τ)", y_true_defect, y_anom_flag, anom_score)

# B) 이상탐지 기준: y_true=이상탐지, y_pred=passorfail
y_pred_defect = y_true_defect  # 이름만 바꿔서 재사용
_print_metrics("Train 성능 (세그먼트 라우팅, 이상탐지 기준 vs 불량, τ)", y_anom_flag, y_pred_defect, anom_score)


# ====================================================================================================================
# ============================================
# 여러 HDBSCAN 세그먼트 모델을 하나의 "라우터" pkl로 병합
# - 글로벌 + mold_code별 서브모델을 한 파일에 저장
# - 예측 시 mold_code로 자동 라우팅, 없으면 글로벌로 폴백
# ============================================
import os, glob, re, joblib, json
from datetime import datetime

SAVE_DIR     = "./hdbscan_models"
MODEL_NAME   = "hdbscan_oneclass"
SEGMENT_COL  = "mold_code"                 # 라우팅 기준
ROUTER_PKL   = os.path.join(SAVE_DIR, "hdbscan_router.pkl")

# 1) 글로벌 로드
global_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.pkl")
assert os.path.exists(global_path), f"글로벌 모델이 없습니다: {global_path}"
gb = joblib.load(global_path)

# 2) 세그먼트 로드 (파일명 패턴과 무관하게 meta.segment 우선)
seg_map = {}
for p in glob.glob(os.path.join(SAVE_DIR, f"{MODEL_NAME}_*.pkl")):
    if os.path.basename(p) == f"{MODEL_NAME}.pkl":
        continue
    b = joblib.load(p)
    seg_tag = b.get("meta", {}).get("segment")
    if not seg_tag:
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.match(rf"{re.escape(MODEL_NAME)}_(.+)", base)
        seg_tag = m.group(1).replace("_", "=", 1) if m else base
    seg_map[seg_tag] = {
        "pca": b.get("pca", None),
        "hdb": b["hdb"],
        "tau": float(b["tau"]),
        "feature_order": list(b["feature_order"]),
    }

# 3) feature/preprocessor 정합성 체크 (글로벌 기준)
feat_global = list(gb["feature_order"])
for k, sb in seg_map.items():
    if sb["feature_order"] != feat_global:
        raise ValueError(f"[{k}] feature_order가 글로벌과 다릅니다. 동일 전처리로 재학습 필요.")

# 4) 라우터 객체 구성 (preprocessor는 글로벌 것 하나만 사용)
router = {
    "segment_col": SEGMENT_COL,
    "preprocessor": gb["preprocessor"],
    "global": {
        "pca": gb.get("pca", None),
        "hdb": gb["hdb"],
        "tau": float(gb["tau"]),
        "feature_order": feat_global,
    },
    "segments": seg_map,  # e.g. {"mold_code=8412": {...}, ...}
    "meta": {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_segments": len(seg_map),
        "model_name": MODEL_NAME,
    }
}
joblib.dump(router, ROUTER_PKL)
print(f"[OK] 라우터 저장: {ROUTER_PKL} (세그먼트 {len(seg_map)}개 포함)")

# ============================================
# 라우터 pkl로 배치/단건 예측
# ============================================
import numpy as np, pandas as pd
from scipy import sparse
import 이상탐지, joblib

ROUTER_PKL = "./hdbscan_models/hdbscan_router.pkl"

def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

def _apply_bundle(prep, bundle, df_part):
    feat = bundle["feature_order"]
    missing = set(feat) - set(df_part.columns)
    if missing:
        raise ValueError(f"누락 feature: {sorted(missing)}")
    X = prep.transform(df_part[feat]); X = _to_dense(X)
    pca = bundle.get("pca", None)
    Z = pca.transform(X) if pca is not None else X
    labels, strengths = 이상탐지.approximate_predict(bundle["hdb"], Z)
    labels    = labels.astype(int)
    strengths = strengths.astype(float)
    tau = float(bundle["tau"])
    # 이상 플래그/점수
    anom_flag  = ((labels == -1) | ((labels != -1) & (strengths < tau))).astype(int)
    anom_score = np.where(labels == -1, 1.0, 1.0 - strengths)  # 클수록 이상
    return anom_flag, anom_score, labels, strengths, tau

def predict_batch_with_router(df):
    router = joblib.load(ROUTER_PKL)
    seg_col = router["segment_col"]
    prep    = router["preprocessor"]
    out = {
        "anom_flag":  np.zeros(len(df), dtype=int),
        "anom_score": np.zeros(len(df), dtype=float),
        "label":      np.zeros(len(df), dtype=int),
        "strength":   np.zeros(len(df), dtype=float),
        "tau_used":   np.zeros(len(df), dtype=float),
        "bundle_key": np.array([""]*len(df), dtype=object),
    }
    df_ = df.reset_index(drop=True).copy()
    # 세그먼트 없거나 값이 비어있으면 전부 글로벌
    if seg_col not in df_.columns or len(router["segments"]) == 0:
        af, sc, lb, st, tau = _apply_bundle(prep, router["global"], df_)
        out["anom_flag"][:]  = af
        out["anom_score"][:] = sc
        out["label"][:]      = lb
        out["strength"][:]   = st
        out["tau_used"][:]   = tau
        out["bundle_key"][:] = "global"
        return out

    for seg_value, sub in df_.groupby(seg_col, dropna=False):
        pos = sub.index.to_numpy()
        seg_name = "NA" if pd.isna(seg_value) else str(seg_value)
        key_eq = f"{seg_col}={seg_name}"
        key_us = f"{seg_col}_{seg_name}"
        bundle = router["segments"].get(key_eq) or router["segments"].get(key_us) or router["global"]
        af, sc, lb, st, tau = _apply_bundle(prep, bundle, sub)
        out["anom_flag"][pos]  = af
        out["anom_score"][pos] = sc
        out["label"][pos]      = lb
        out["strength"][pos]   = st
        out["tau_used"][pos]   = tau
        out["bundle_key"][pos] = key_eq if key_eq in router["segments"] else ("global" if bundle is router["global"] else key_us)
    return out

def detect_one(row_dict):
    """단건 판정(실시간). row_dict에는 mold_code 포함 권장."""
    router = joblib.load(ROUTER_PKL)
    seg_col = router["segment_col"]
    seg_val = row_dict.get(seg_col, None)
    seg_name = "NA" if pd.isna(seg_val) else str(seg_val)
    key_eq = f"{seg_col}={seg_name}"
    key_us = f"{seg_col}_{seg_name}"
    bundle = router["segments"].get(key_eq) or router["segments"].get(key_us) or router["global"]

    # 1-row DataFrame으로 전처리
    import pandas as pd
    feat = bundle["feature_order"]
    df1 = pd.DataFrame([row_dict], columns=feat)  # feature 누락 시 KeyError
    prep = router["preprocessor"]
    X = prep.transform(df1); X = _to_dense(X)
    pca = bundle.get("pca", None)
    Z = pca.transform(X) if pca is not None else X
    label, strength = 이상탐지.approximate_predict(bundle["hdb"], Z)
    label, strength = int(label[0]), float(strength[0])
    tau = float(bundle["tau"])
    is_anom = int((label == -1) or ((label != -1) and (strength < tau)))
    score   = 1.0 if label == -1 else 1.0 - strength
    return {
        "segment_used": key_eq if key_eq in router["segments"] else ("global" if bundle is router["global"] else key_us),
        "anomaly": is_anom, "score": score, "label": label, "strength": strength, "tau": tau
    }

# ============================================
# 성능평가: 불량 기준 & 이상탐지 기준 (둘 다)
# ============================================
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)

assert "train_df" in globals(), "train_df가 필요합니다."
assert "passorfail" in train_df.columns, "passorfail(1=불량,0=정상)이 필요합니다."

pred = predict_batch_with_router(train_df)

# A) 불량 기준: y_true = passorfail, y_pred = anom_flag
y_true_def = train_df["passorfail"].astype(int).values
y_pred_anom = pred["anom_flag"]
score = pred["anom_score"]

def print_metrics(title, y_true, y_pred, score):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # [[TP,FN],[FP,TN]]
    TP, FN = int(cm[0,0]), int(cm[0,1])
    FP, TN = int(cm[1,0]), int(cm[1,1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    roc = roc_auc_score(y_true, score)
    pr  = average_precision_score(y_true, score)
    print(f"=== {title} ===")
    print(f"confusion_matrix [[TP,FN],[FP,TN]]: [[{TP}, {FN}], [{FP}, {TN}]]")
    print(f"precision@τ: {prec}")
    print(f"recall@τ: {rec}")
    print(f"f1@τ: {f1}")
    print(f"ROC_AUC: {roc}")
    print(f"PR_AUC: {pr}")
    print(f"prevalence(실제 양성 비율): {float(np.mean(y_true))}")
    print(f"predicted_rate@τ(예측 양성 비율): {float(np.mean(y_pred))}\n")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=3))

print_metrics("Train 성능 (라우터, 불량 기준 vs 이상탐지, τ)", y_true_def, y_pred_anom, score)

# B) 이상탐지 기준: y_true = anom_flag, y_pred = passorfail
y_true_anom = y_pred_anom
y_pred_def  = y_true_def
print_metrics("Train 성능 (라우터, 이상탐지 기준 vs 불량, τ)", y_true_anom, y_pred_def, score)

