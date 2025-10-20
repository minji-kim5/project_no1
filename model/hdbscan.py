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
PCA_GRID_BASE = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
MCS_GRID      = [30, 50, 80, 120]        # min_cluster_size
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
