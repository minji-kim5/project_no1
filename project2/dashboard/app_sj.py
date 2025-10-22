import pandas as pd
import numpy as np
import joblib
import shiny
from shiny import App, ui, render, reactive
from pathlib import Path
import datetime
from datetime import datetime, timedelta
import os
import asyncio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
import re
import google.generativeai as genai
from scipy.stats import ks_2samp
import seaborn as sns
import pickle

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- shared에서 모델과 동일 전처리/피처 순서/예측함수 가져오기 ---
from shared import ANOM_PREPROCESSOR, ANOM_FEATURE_ORDER, anomaly_transform, get_preproc_feature_names_out
from shared import (
    streaming_df, RealTimeStreamer, defect_model, feature_cols,
    train_df, test_label_df, test_df, predict_anomaly, defect_threshold, model_dict,
    ANOMALY_PROBA_THRESHOLD
)

# ARIMA 통계 로드
arima_stats = {}
arima_pkl_path = "./data/arima_stats.pkl"
if os.path.exists(arima_pkl_path):
    try:
        with open(arima_pkl_path, 'rb') as f:
            arima_stats = pickle.load(f)
        print(f"✅ ARIMA 통계 로드 완료: {len(arima_stats)}개 변수")
    except Exception as e:
        print(f"⚠️ ARIMA 통계 로드 실패: {e}")
else:
    print(f"⚠️ {arima_pkl_path} 파일을 찾을 수 없습니다.")

# 드리프트 제외 컬럼
excluded_drift_cols = [
    'count', 'hour', 'EMS_operation_time', 'tryshot_signal',
    'mold_code', 'heating_furnace'
]
drift_feature_choices = [col for col in feature_cols if col not in excluded_drift_cols]

# 한글 폰트 설정
import platform
from matplotlib import font_manager, rc
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from collections import defaultdict

plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    try:
        if platform.system() == 'Linux':
            font_paths = ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 
                          '/usr/share/fonts/nanum/NanumGothic.ttf',
                          '/usr/local/share/fonts/NanumGothic.ttf']
            font_path = None
            for path in font_paths:
                if os.path.exists(path):
                    font_path = path
                    break
            if font_path:
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                rc('font', family=font_name)
            else:
                rc('font', family='NanumBarunGothic')
        else:
            rc('font', family='NanumBarunGothic')
    except Exception:
        rc('font', family='NanumBarunGothic')

TARGET_COL = 'passorfail'
PREDICTION_THRESHOLD = defect_threshold
CHUNK_SIZE = 200
DRIFT_CHUNK_SIZE = 100
CAL_K = 80
_MAX_CAL_SAMPLES = 500
_RIDGE = 1e-6
startup_error = ""

# 검증 성능
validation_recall = 0.0
validation_precision = 0.0
recall_lcl = 0.0
precision_lcl = 0.0

# ==================== 이상치 설명 준비 ====================
INPUT_FEATURES = ANOM_FEATURE_ORDER if len(ANOM_FEATURE_ORDER) else feature_cols

try:
    PREPROC_FEATURES = list(get_preproc_feature_names_out())
except Exception:
    PREPROC_FEATURES = [f"feat_{i}" for i in range(anomaly_transform(train_df[INPUT_FEATURES]).shape[1])]

def _base_from_preproc_name(name: str) -> str:
    n = str(name).split("__")[-1]
    if "=" in n:
        return n.split("=")[0]
    cand = n.rsplit("_", 1)[0]
    return cand if cand in INPUT_FEATURES else n

_GROUP_IDX = defaultdict(list)
_PREPROC_BASES = []
for j, pname in enumerate(PREPROC_FEATURES):
    base = _base_from_preproc_name(pname)
    if base == pname and pname.startswith("feat_"):
        if j < len(INPUT_FEATURES):
            base = INPUT_FEATURES[j]
    _PREPROC_BASES.append(base)
    _GROUP_IDX[base].append(j)

# 숫자형 결측 대치값 추출
_NUM_IMPUTE_STATS = {}
try:
    if hasattr(ANOM_PREPROCESSOR, "transformers_"):
        for _, trans, cols in ANOM_PREPROCESSOR.transformers_:
            if hasattr(trans, "named_steps") and "imputer" in trans.named_steps:
                imp = trans.named_steps["imputer"]
                if isinstance(imp, SimpleImputer) and hasattr(imp, "statistics_"):
                    for c, v in zip(cols, imp.statistics_):
                        _NUM_IMPUTE_STATS[c] = v
            elif isinstance(trans, SimpleImputer) and hasattr(trans, "statistics_"):
                for c, v in zip(cols, trans.statistics_):
                    _NUM_IMPUTE_STATS[c] = v
except Exception:
    pass

def _imputed_raw(row: pd.Series, base: str):
    v = row.get(base, np.nan)
    if pd.isna(v) and base in _NUM_IMPUTE_STATS:
        return _NUM_IMPUTE_STATS[base]
    return v

# 참조 행렬
_Z_ref = anomaly_transform(train_df[INPUT_FEATURES])
_nn_model = NearestNeighbors(n_neighbors=min(CAL_K, len(_Z_ref)), metric="euclidean")
_nn_model.fit(_Z_ref)

def _xi_from_row(row: pd.Series) -> np.ndarray:
    X1 = pd.DataFrame([row]).reindex(columns=INPUT_FEATURES)
    Zi = anomaly_transform(X1)[0]
    return Zi

def _local_mahalanobis_parts(xi: np.ndarray):
    _, idx = _nn_model.kneighbors([xi], n_neighbors=min(CAL_K, len(_Z_ref)))
    neigh = _Z_ref[idx[0]]
    mu = neigh.mean(axis=0)
    Sigma = np.cov(neigh, rowvar=False)
    Sigma = Sigma + np.eye(Sigma.shape[0]) * _RIDGE
    invS = np.linalg.pinv(Sigma)
    diff = xi - mu
    parts = diff * (invS @ diff)
    D2 = float(diff @ invS @ diff)
    return parts, D2, mu

def _effective_anom_proba(ana_res: pd.DataFrame):
    aprob_raw = float(ana_res.get("anom_proba", [np.nan])[0]) if "anom_proba" in ana_res.columns else np.nan
    strength = float(ana_res.get("prob", [np.nan])[0]) if "prob" in ana_res.columns else np.nan
    if (not np.isfinite(aprob_raw)) or (aprob_raw <= 0):
        aprob_eff = 1.0 - (strength if np.isfinite(strength) else 0.0)
    else:
        aprob_eff = aprob_raw
    return aprob_eff, strength

# Isotonic 보정
_iso = None
_cal_bounds = None

def _collect_calibration_pairs(max_samples=_MAX_CAL_SAMPLES):
    if len(test_df) == 0:
        return None, None
    idxs_eq = np.linspace(0, len(test_df)-1, num=min(max_samples, len(test_df)), dtype=int)
    idxs_rd = np.random.RandomState(42).choice(len(test_df), size=min(max_samples//2, len(test_df)), replace=False)
    idxs = np.unique(np.concatenate([idxs_eq, idxs_rd]))
    
    d2_list, p_list = [], []
    for i in idxs:
        r = test_df.iloc[i]
        try:
            xi = _xi_from_row(r)
            parts, D2, _ = _local_mahalanobis_parts(xi)
            ana_res = predict_anomaly(pd.DataFrame([r]))
            aprob_eff, _ = _effective_anom_proba(ana_res)
            if np.isfinite(D2) and np.isfinite(aprob_eff):
                d2_list.append(D2)
                p_list.append(aprob_eff)
        except Exception:
            continue
    if len(d2_list) < 20:
        return None, None
    
    d2 = np.asarray(d2_list)
    p = np.asarray(p_list)
    order = np.argsort(d2)
    return d2[order], p[order]

_cal_d2, _cal_p = _collect_calibration_pairs()
if _cal_d2 is not None:
    _iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    _iso.fit(_cal_d2, _cal_p)
    _cal_bounds = (float(np.min(_cal_d2)), float(np.max(_cal_d2)))

def _p_of_D2(D2: float) -> float:
    if _iso is None:
        return float("nan")
    if _cal_bounds is None:
        return float(_iso.predict([D2])[0])
    lo, hi = _cal_bounds
    x = min(max(float(D2), lo), hi)
    return float(_iso.predict([x])[0])

def _dp_dD2(D2: float) -> float:
    if _iso is None:
        return 0.0
    lo, hi = _cal_bounds if _cal_bounds is not None else (D2-1.0, D2+1.0)
    width = max(hi - lo, 1.0)
    eps = 1e-3 * width
    x = lo + eps if D2 <= lo else (hi - eps if D2 >= hi else float(D2))
    p_plus = float(_iso.predict([x + eps])[0])
    p_minus = float(_iso.predict([x - eps])[0])
    return (p_plus - p_minus) / (2.0 * eps)

def explain_hdbscan_local_influence_prob(row: pd.Series, topn: int = 5):
    try:
        xi = _xi_from_row(row)
        parts, D2, mu = _local_mahalanobis_parts(xi)
        slope = _dp_dD2(D2)
        deltas = slope * parts
        
        items = []
        for base, idxs in _GROUP_IDX.items():
            dp_sum = float(np.sum(deltas[idxs]))
            raw_val = _imputed_raw(row, base)
            mu_base = float(np.mean(mu[idxs]))
            items.append((base, dp_sum, raw_val, mu_base))
        
        items.sort(key=lambda t: abs(t[1]), reverse=True)
        return items[:topn], D2, _p_of_D2(D2), slope
    except Exception:
        return [], float("nan"), float("nan"), 0.0

def _fmt_raw(v):
    if pd.isna(v):
        return "nan"
    try:
        return f"{float(v):.3g}"
    except Exception:
        return str(v)

def topk_prob_attribution(row: pd.Series, topk: int = 3):
    items_all, D2, p_hat, slope = explain_hdbscan_local_influence_prob(row, topn=10_000)
    
    model_prob = row.get("_anom_proba", np.nan)
    if not np.isfinite(model_prob):
        st = row.get("_hdb_strength", np.nan)
        model_prob = (1.0 - float(st)) if np.isfinite(st) else np.nan
    
    pos = [(f, max(0.0, dp), raw_val, mu, dp) for (f, dp, raw_val, mu) in items_all]
    S = sum(v for _, v, _, _, _ in pos)
    if S == 0:
        pos = [(f, abs(dp), raw_val, mu, dp) for (f, dp, raw_val, mu) in items_all]
        S = sum(v for _, v, _, _, _ in pos)
    
    out = []
    if np.isfinite(model_prob) and S > 0:
        for f, v, raw_val, mu, dp in pos:
            share = model_prob * (v / S)
            out.append((f, float(share), _fmt_raw(raw_val), mu, float(dp)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out[:topk], float(model_prob)
    
    return [], float(model_prob) if np.isfinite(model_prob) else float("nan")

def _fmt_time(ts):
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

# ==================== 성능 계산 ====================
try:
    if defect_model is None:
        raise ValueError("shared.py에서 모델을 로드하지 못했습니다.")
    
    split_index = int(len(train_df) * 0.8)
    valid_df = train_df.iloc[split_index:].copy().reset_index(drop=True)
    
    if TARGET_COL not in valid_df.columns:
        print(f"Warning: Validation 데이터에 '{TARGET_COL}' 컬럼이 없어 성능 계산을 건너뜁니다.")
    else:
        X_valid = valid_df[feature_cols]
        y_valid = valid_df[TARGET_COL]
        y_pred_proba = defect_model.predict_proba(X_valid)[:, 1]
        y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()
        validation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        validation_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recalls_per_chunk, precisions_per_chunk = [], []
        for i in range(0, len(valid_df), CHUNK_SIZE):
            chunk = valid_df.iloc[i: i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE or chunk[TARGET_COL].sum() == 0:
                continue
            
            X_chunk = chunk[feature_cols]
            y_true_chunk = chunk[TARGET_COL]
            y_pred_proba_chunk = defect_model.predict_proba(X_chunk)[:, 1]
            y_pred_chunk = (y_pred_proba_chunk >= PREDICTION_THRESHOLD).astype(int)
            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true_chunk, y_pred_chunk, labels=[0, 1]).ravel()
            
            recalls_per_chunk.append(tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0)
            precisions_per_chunk.append(tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0)
        
        if len(recalls_per_chunk) > 1:
            mean_recall = np.mean(recalls_per_chunk)
            recall_lcl = max(0, mean_recall - 3 * np.sqrt(mean_recall * (1 - mean_recall) / CHUNK_SIZE))
        if len(precisions_per_chunk) > 1:
            mean_precision = np.mean(precisions_per_chunk)
            precision_lcl = max(0, mean_precision - 3 * np.sqrt(mean_precision * (1 - mean_precision) / CHUNK_SIZE))

except Exception as e:
    startup_error = f"초기화 오류: {e}"

# ==================== P-관리도 준비 ====================
monitoring_vars = [
    'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
    'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature'
]

var_stats = {}
for var in monitoring_vars:
    if var in train_df.columns:
        values = train_df[var].dropna()
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            var_stats[var] = {
                'mean': mean, 'std': std,
                'ucl': mean + 3 * std, 'lcl': mean - 3 * std
            }

if arima_stats:
    monitoring_vars = [var for var in monitoring_vars if var in arima_stats]

def calculate_p_values(df, var_stats):
    p_values = []
    for idx, row in df.iterrows():
        abnormal_count = 0
        valid_var_count = 0
        for var in var_stats.keys():
            if var in row and pd.notna(row[var]):
                valid_var_count += 1
                value = row[var]
                ucl = var_stats[var]['ucl']
                lcl = var_stats[var]['lcl']
                if value > ucl or value < lcl:
                    abnormal_count += 1
        p = abnormal_count / valid_var_count if valid_var_count > 0 else 0
        p_values.append(p)
    return np.array(p_values)

all_p_values = calculate_p_values(test_df, var_stats)
p_bar = all_p_values.mean()
n = len(var_stats)
CL = p_bar
UCL = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
LCL = max(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

def check_nelson_rules(p_values, cl, ucl, lcl):
    violations = {'rule1': [], 'rule4': [], 'rule8': []}
    sigma = (ucl - cl) / 3 if (ucl-cl) > 0 else 0
    n = len(p_values)
    
    for i in range(n):
        if p_values[i] > ucl or p_values[i] < lcl:
            violations['rule1'].append(i)
        
        if i >= 13:
            alternating = True
            for j in range(i - 12, i):
                if j > 0:
                    diff1 = p_values[j + 1] - p_values[j]
                    diff2 = p_values[j] - p_values[j - 1]
                    if diff1 * diff2 >= 0:
                        alternating = False
                        break
            if alternating:
                violations['rule4'].append(i)
        
        if i >= 7 and sigma > 0:
            all_outside = True
            for j in range(i - 7, i + 1):
                if abs(p_values[j] - cl) <= sigma:
                    all_outside = False
                    break
            if all_outside:
                violations['rule8'].append(i)
    
    return violations

# ==================== Reactive 변수 ====================
streamer = reactive.Value(RealTimeStreamer(streaming_df))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)
was_reset = reactive.Value(False)

defect_logs = reactive.Value(pd.DataFrame(columns=["Time", "ID", "Prob"]))
anom_logs = reactive.Value(pd.DataFrame(columns=["Time", "ID", "Proba"]))

latest_anomaly_status = reactive.Value(0)
latest_defect_status = reactive.Value(0)

r_feedback_data = reactive.Value(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"]))
r_correct_status = reactive.Value(None)

realtime_performance = reactive.Value(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
latest_performance_metrics = reactive.Value({"recall": 0.0, "precision": 0.0})
last_processed_count = reactive.Value(0)

performance_degradation_status = reactive.Value({"degraded": False})
cumulative_cm_components = reactive.Value({"tp": 0, "fn": 0, "fp": 0})
cumulative_performance = reactive.Value({"recall": 0.0, "precision": 0.0})

recall_tooltip = reactive.Value(None)
precision_tooltip = reactive.Value(None)

ks_test_results = reactive.Value(pd.DataFrame(columns=["Count", "Feature", "PValue"]))
chunk_snapshot_data = reactive.Value(pd.DataFrame())
data_drift_status = reactive.Value({"degraded": False, "feature": None})
last_drift_processed_count = reactive.Value(0)

p_chart_streaming = reactive.Value(False)
p_chart_current_index = reactive.Value(0)

chatbot_visible = reactive.Value(False)
r_ai_answer = reactive.Value("질문을 입력해주세요.")
r_is_loading = reactive.Value(False)

# Gemini API 설정
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
try:
    API_KEY = "AIzaSyAJbO4gJXKf8HetBy6TKwD5fEqAllgX-nc"
    if API_KEY == "YOUR_API_KEY_HERE":
        raise KeyError("API 키가 설정되지 않았습니다.")
    genai.configure(api_key=API_KEY)
except KeyError:
    startup_error = "GEMINI_API_KEY가 설정되지 않았습니다. 챗봇을 사용할 수 없습니다."
    print(f"ERROR: {startup_error}")
except Exception as e:
    startup_error = f"Gemini API 키 설정 오류: {e}"
    print(f"ERROR: {startup_error}")

# ==================== UI ====================
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { overflow-y: auto !important; }
        .card-body { overflow-y: visible !important; }
        .plot-tooltip {
            position: absolute; background: rgba(0, 0, 0, 0.8); color: white;
            padding: 5px 10px; border-radius: 5px; pointer-events: none;
            z-index: 1000; font-size: 0.9rem;
        }
        .plot-tooltip table { color: white; border-collapse: collapse; }
        .plot-tooltip th, .plot-tooltip td { border: 1px solid #555; padding: 4px 8px; text-align: center; }
        .plot-tooltip th { background-color: #333; }
        .violation-item {
            padding: 12px; margin: 8px 0; border-left: 4px solid #dc3545;
            background-color: #fff5f5; border-radius: 4px;
        }
        .violation-header { font-weight: bold; color: #dc3545; margin-bottom: 6px; font-size: 14px; }
        .violation-detail { font-size: 13px; color: #666; margin: 4px 0; }
        .violation-rule {
            display: inline-block; padding: 2px 8px; margin: 2px;
            background-color: #dc3545; color: white; border-radius: 3px; font-size: 11px;
        }
        .btn-cause {
            margin-top: 8px; padding: 6px 12px; font-size: 12px;
            background-color: #007bff; color: white; border: none;
            border-radius: 4px; cursor: pointer;
        }
        .btn-cause:hover { background-color: #0056b3; }
        .violations-container { max-height: 1220px; overflow-y: auto; padding-right: 10px; }
        
        #chatbot_response .card-body { padding: 1.5rem; }
        #chatbot_response pre { 
            background-color: #f7f7f7; 
            padding: 10px; 
            border-radius: 5px; 
            overflow-x: auto;
        }
        
        table.custom-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
            word-wrap: break-word;
        }
        .custom-table th {
            background-color: #f5f5f5;
        }
        .custom-table td:nth-child(1) { width: 10%; }
        .custom-table td:nth-child(2) { width: 20%; }
        .custom-table td:nth-child(3) { width: 20%; }
        .custom-table td:nth-child(4) { width: 50%; text-align: left; }
    """),
    ui.tags.script("""
        function scrollToArima() {
            setTimeout(function() {
                const arimaCard = document.getElementById('arima_chart_card');
                if (arimaCard) {
                    arimaCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);
        }
    """),

    ui.h2("실시간 품질 모니터링 대시보드", class_="text-center fw-bold my-3"),
    
    ui.navset_card_tab(
        # ==================== 탭 1: 실시간 모니터링 ====================
        ui.nav_panel("실시간 모니터링",
            ui.div(
                {"class": "d-flex align-items-center gap-3 mb-3 sticky-top bg-light p-2 shadow-sm"},
                ui.input_action_button("start", "▶ 시작", class_="btn btn-success"),
                ui.input_action_button("pause", "⏸ 일시정지", class_="btn btn-warning"),
                ui.input_action_button("reset", "🔄 리셋", class_="btn btn-secondary"),
                ui.output_ui("stream_status"),
            ),
            ui.div(ui.p(f"⚠️ {startup_error}", style="color:red; font-weight:bold;") if startup_error else ""),

            ui.card(
                ui.card_header("변수 선택"),
                ui.h5("확인할 변수 선택"),
                ui.input_checkbox_group(
                    "selected_sensors", None,
                    choices={
                       "cast_pressure": "주조압력(cast_pressure)",
                        "upper_mold_temp1": "상부금형온도1(upper_mold_temp1)",
                        "lower_mold_temp2": "하부금형온도2(lower_mold_temp2)",
                        "low_section_speed": "저속구간속도(low_section_speed)",
                        "lower_mold_temp1": "하부금형온도1(lower_mold_temp1)",
                        "sleeve_temperature": "슬리브온도(sleeve_temperature)",
                        "high_section_speed": "고속구간속도(high_section_speed)",
                        "upper_mold_temp2": "상부금형온도2(upper_mold_temp2)",
                        "biscuit_thickness": "비스킷두께(biscuit_thickness)",
                        "facility_operation_cycleTime": "설비작동사이클시간(facility_operation_cycleTime)",
                        "Coolant_temperature": "냉각수온도(Coolant_temperature)",
                        "production_cycletime": "생산사이클시간(production_cycletime)",
                        "molten_temp": "용탕온도(molten_temp)",
                        "physical_strength": "물리적강도(physical_strength)"

                    },
                    selected=["molten_temp", "cast_pressure"],
                    inline=True
                ),
                ui.h5("몰드코드 선택"),
                ui.input_checkbox_group(
                    "selected_molds", None,
                    choices={"ALL":"ALL","8412":"8412","8573":"8573","8600":"8600","8722":"8722","8917":"8917","8413":"8413","8576":"8576"},
                    selected=["ALL"], inline=True
                ),
            ),

            ui.div(
                {"class": "d-flex justify-content-around align-items-center flex-wrap mt-3"},
                ui.div([ui.span("최신 수신 시각: "), ui.output_text("latest_timestamp_text")],
                       class_="text-center my-2", style="font-size: 16px; font-weight: bold;"),
                ui.div([ui.div("이상치 상태", class_="fw-bold text-center mb-1"), ui.output_ui("anomaly_status_ui")],
                       class_="text-center mx-3"),
                ui.div([ui.div("불량 판정", class_="fw-bold text-center mb-1"), ui.output_ui("defect_status_ui")],
                       class_="text-center mx-3"),
            ),

            ui.output_ui("realtime_graphs"),
            ui.card(ui.output_ui("defect_stats_ui")),

            ui.hr(),
            ui.card(
                ui.card_header("모델 예측 불량 확인 및 피드백"),
                ui.row(
                    ui.column(6,
                            ui.h4("불량 제품"),
                            ui.output_ui("prediction_output_ui"),
                            ),
                    ui.column(6,
                            ui.h4("누적 피드백"),
                            ui.output_ui("feedback_table"),
                            ),
                ),
            ),
        ),

        # ==================== 탭 2: 이상 탐지 (통합) ====================
        ui.nav_panel("이상 탐지",
            ui.div(
                {"style": "padding: 20px;"},
                ui.h4("공정 이상 탐지 P-관리도 및 개별 변수 관리도"),
                ui.p(f"모니터링 변수: {len(var_stats)}개 | 총 데이터: {len(test_df):,}건",
                     style="color: #666; margin-bottom: 20px;")
            ),

            ui.row(
                ui.column(7,
                    ui.card(
                        ui.card_header(ui.h4("HDBSCAN 이상 예측 확률 (실시간)", style="margin: 0;")),
                        ui.output_plot("anom_proba_plot", height="220px")
                    )
                ),
                ui.column(5,
                    ui.card(
                        ui.card_header(ui.h4(f"이상치 로그 (th ≥ {ANOMALY_PROBA_THRESHOLD:.2f})", style="margin: 0;")),
                        ui.output_ui("anom_logs_ui")
                    )
                )
            ),

            ui.row(
                ui.column(8,
                    ui.card(
                        ui.card_header(
                            ui.div(
                                {"class": "d-flex justify-content-between align-items-center"},
                                ui.h4("P-관리도 (공정 이상 비율)", style="margin: 0;"),
                                ui.div(
                                    {"style": "width: 250px;"},
                                    ui.input_slider("lot_size", "로트(서브그룹) 크기:",
                                                   min=10, max=500, value=200, step=10, animate=False)
                                )
                            )
                        ),
                        ui.output_plot("control_chart", height="500px")
                    ),
                    ui.card(
                        {"id": "arima_chart_card"},
                        ui.card_header(ui.h5("개별 변수 관리도 (Auto ARIMA - 실제값)")),
                        ui.input_select(
                            "selected_variable",
                            "모니터링할 변수 선택:",
                            choices=monitoring_vars if monitoring_vars else ["없음"],
                            selected=monitoring_vars[0] if monitoring_vars else "없음"
                        ),
                        ui.output_plot("arima_control_chart", height="500px")
                    )
                ),
                ui.column(4,
                    ui.card(
                        ui.card_header(ui.h4("관리도이상패턴탐지", style="margin: 0;")),
                        ui.div({"class": "violations-container"}, ui.output_ui("violations_list"))
                    )
                )
            ),
        ),

        # ==================== 탭 3: 모델 성능 평가 ====================
        ui.nav_panel("모델 성능 평가",
            ui.layout_columns(
                ui.card(
                    ui.card_header("실시간 성능 (Chunk=200)", class_="text-center fw-bold"),
                    ui.layout_columns(
                        ui.div(
                            ui.p("최신 Recall"),
                            ui.h4(ui.output_text("latest_recall_text")),
                            style="background-color: #fff0f5; padding: 1rem; border-radius: 8px; text-align: center;"
                        ),
                        ui.div(
                            ui.p("최신 Precision"),
                            ui.h4(ui.output_text("latest_precision_text")),
                            style="background-color: #fff8f0; padding: 1rem; border-radius: 8px; text-align: center;"
                        ),
                        col_widths=[6, 6]
                    )
                ),
                ui.card(
                    ui.card_header("누적 성능 지표", class_="text-center fw-bold"),
                    ui.layout_columns(
                        ui.div(
                            ui.p(f"누적 Recall (Valid = {validation_recall:.2%})"),
                            ui.h5(ui.output_text("cumulative_recall_text"), class_="text-center text-primary mt-1")
                        ),
                        ui.div(
                            ui.p(f"누적 Precision (Valid = {validation_precision:.2%})"),
                            ui.h5(ui.output_text("cumulative_precision_text"), class_="text-center text-success mt-1")
                        ),
                        col_widths=[6, 6]
                    ),
                ),
                col_widths=[6, 6]
            ),

            ui.layout_columns(
                 ui.card(
                     ui.card_header("모델 성능 상태"),
                     ui.output_ui("model_performance_status_ui")
                 ),
                 ui.card(
                     ui.card_header("데이터 드리프트 상태"),
                     ui.output_ui("data_drift_status_ui")
                 ),
                 col_widths=[6, 6]
            ),

            ui.hr(),

            ui.layout_columns(
                ui.div(
                    ui.card(
                        ui.card_header(
                            ui.div("실시간 재현율(Recall) 추이",
                                 ui.tags.small("※ p관리도 기준, n=200", class_="text-muted ms-2 fw-normal"),
                                 class_="d-flex align-items-baseline")
                        ),
                        ui.div(
                            ui.output_plot("realtime_recall_plot", height="230px"),
                            ui.output_ui("recall_tooltip_ui"),
                            style="position: relative;"
                        )
                    ),
                    ui.card(
                        ui.card_header("실시간 정밀도(Precision) 추이"),
                        ui.div(
                            ui.output_plot("realtime_precision_plot", height="230px"),
                            ui.output_ui("precision_tooltip_ui"),
                            style="position: relative;"
                        )
                    )
                ),

                ui.div(
                    ui.card(
                        ui.card_header("실시간 데이터 분포 (KDE)"),
                        ui.layout_columns(
                            ui.input_select(
                                "drift_feature_select",
                                "특성(Feature) 선택:",
                                choices=drift_feature_choices,
                                selected=drift_feature_choices[0] if len(drift_feature_choices) > 0 else None
                            ),
                            ui.div(
                                {"style": "display: flex; align-items: flex-end;"},
                                ui.p("학습 vs 실시간(100개) 데이터 분포 비교.",
                                     class_="text-muted small", style="margin-bottom: 0.5rem;")
                            ),
                            col_widths=[7, 5]
                        ),
                        ui.output_plot("drift_plot", height="230px")
                    ),
                    ui.card(
                        ui.card_header("데이터 분포 변화 (KS 검정 P-value)"),
                         ui.layout_columns(
                            ui.input_select(
                                "ks_feature_select",
                                "특성(Feature) 선택:",
                                choices=drift_feature_choices,
                                selected=drift_feature_choices[0] if len(drift_feature_choices) > 0 else None
                            ),
                             ui.div(
                                 {"style": "display: flex; align-items: flex-end;"},
                                 ui.p("100개 chunk 단위 KS 검정 p-value 추이.",
                                      class_="text-muted small", style="margin-bottom: 0.5rem;")
                             ),
                            col_widths=[7, 5]
                        ),
                        ui.output_plot("ks_test_plot", height="230px")
                    ),
                ),
                col_widths=[6, 6]
            )
        )
    ),

    # ================== 챗봇 ====================
    ui.TagList(
        ui.div(
            ui.input_action_button("toggle_chatbot", "🤖",
                                 style=("position: fixed; bottom: 20px; right: 20px; width: 50px; height: 50px; "
                                        "border-radius: 25px; font-size: 24px; background-color: #4CAF50; color: white; "
                                        "border: none; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.3); z-index: 1000;")
                                 )
        ),
        ui.div(
            ui.output_ui("chatbot_popup"),
            id="chatbot_popup_wrapper"
        )
    )
)

# ==================== SERVER ====================
def server(input, output, session):
    # ==================== 공통 제어 ====================
    @reactive.effect
    @reactive.event(input.start)
    def _():
        is_streaming.set(True)
        was_reset.set(False)
        p_chart_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def _():
        is_streaming.set(False)
        p_chart_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        streamer().reset_stream()
        current_data.set(pd.DataFrame())
        defect_logs.set(pd.DataFrame(columns=["Time", "ID", "Prob"]))
        anom_logs.set(pd.DataFrame(columns=["Time", "ID", "Proba"]))
        latest_anomaly_status.set(0)
        latest_defect_status.set(0)
        r_feedback_data.set(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"]))
        r_correct_status.set(None)
        realtime_performance.set(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
        latest_performance_metrics.set({"recall": 0.0, "precision": 0.0})
        last_processed_count.set(0)
        is_streaming.set(False)
        was_reset.set(True)
        performance_degradation_status.set({"degraded": False})
        cumulative_cm_components.set({"tp": 0, "fn": 0, "fp": 0})
        cumulative_performance.set({"recall": 0.0, "precision": 0.0})
        ks_test_results.set(pd.DataFrame(columns=["Count", "Feature", "PValue"]))
        chunk_snapshot_data.set(pd.DataFrame())
        data_drift_status.set({"degraded": False, "feature": None})
        last_drift_processed_count.set(0)
        p_chart_streaming.set(False)
        p_chart_current_index.set(0)
        r_ai_answer.set("질문을 입력해주세요.")

    @output
    @render.ui
    def stream_status():
        status, color = ("🔴 일시 정지됨", "red")
        mold_text = "전체 몰드코드 표시 중"

        if was_reset():
            status, color = ("🟡 리셋됨", "orange")
        elif is_streaming():
            status, color = ("🟢 공정 진행 중", "green")

        molds = input.selected_molds()
        if molds:
            mold_text = f"선택된 몰드코드: {', '.join(molds)}"

        return ui.div(
            f"{status} | {mold_text}",
            style=f"font-weight:bold; color:{color}; margin-left:15px;"
        )

    # ==================== 실시간 스트리밍 ====================
    @reactive.effect
    def _():
        try:
            if not is_streaming():
                return

            reactive.invalidate_later(0.5)
            s = streamer()
            next_batch = s.get_next_batch(1)

            if next_batch is not None:
                current_stream_idx = s.current_index - 1
                original_df_idx = s.full_data.index[current_stream_idx]

                # ----- HDBSCAN 이상치 예측 -----
                try:
                    single_row_df = s.full_data.iloc[[current_stream_idx]]
                    ana_res = predict_anomaly(single_row_df)

                    if ana_res is not None and not ana_res.empty:
                        aprob_eff, strength = _effective_anom_proba(ana_res)
                        sev = 1 if aprob_eff >= ANOMALY_PROBA_THRESHOLD else 0

                        s.full_data.loc[original_df_idx, "_anom_proba"] = aprob_eff
                        s.full_data.loc[original_df_idx, "_hdb_strength"] = strength
                        s.full_data.loc[original_df_idx, "anomaly_status"] = sev
                        latest_anomaly_status.set(sev)

                        if sev == 1:
                            logs = anom_logs()
                            if logs.empty or (original_df_idx not in logs["ID"].tolist()):
                                ts = s.full_data.loc[original_df_idx, "registration_time"] \
                                    if "registration_time" in s.full_data.columns else pd.NaT
                                new_row = pd.DataFrame({"Time":[ts], "ID":[original_df_idx], "Proba":[aprob_eff]})
                                anom_logs.set(pd.concat([logs, new_row], ignore_index=True))
                except Exception as e:
                    print(f"⚠️ 이상치 예측 오류: {e}")

                # ----- 불량 예측 -----
                if defect_model is not None:
                    latest_row = s.full_data.iloc[[current_stream_idx]].copy()
                    for col in feature_cols:
                        if col not in latest_row.columns:
                            latest_row[col] = 0
                    try:
                        prob = defect_model.predict_proba(latest_row[feature_cols])[0, 1]
                        pred = 1 if prob >= PREDICTION_THRESHOLD else 0
                    except Exception as e:
                        print(f"⚠️ 모델 예측 오류: {e}")
                        prob, pred = 0.0, 0

                    s.full_data.loc[original_df_idx, "defect_status"] = int(pred)
                    latest_defect_status.set(int(pred))

                    if int(pred) == 1:
                        logs = defect_logs()
                        if logs.empty or original_df_idx not in logs['ID'].values:
                            new_log = pd.DataFrame({
                                "Time": [s.full_data.loc[original_df_idx, "registration_time"]],
                                "ID": [original_df_idx],
                                "Prob": [prob]
                            })
                            defect_logs.set(pd.concat([logs, new_log], ignore_index=True))

                # 화면 데이터 업데이트
                current_data.set(s.get_current_data())

                # ----- 성능 집계 -----
                df_now = current_data()
                current_count = len(df_now)
                last_count = last_processed_count()
                
                if current_count // CHUNK_SIZE > last_count // CHUNK_SIZE:
                    chunk_number = current_count // CHUNK_SIZE
                    start_idx, end_idx = (chunk_number - 1) * CHUNK_SIZE, chunk_number * CHUNK_SIZE
                    
                    if len(test_label_df) >= end_idx:
                        chunk_data = df_now.iloc[start_idx:end_idx]
                        y_true_chunk = test_label_df.iloc[start_idx:end_idx][TARGET_COL].values
                        X_chunk = chunk_data[feature_cols]
                        y_pred_proba_chunk = defect_model.predict_proba(X_chunk)[:, 1]
                        y_pred_chunk = (y_pred_proba_chunk >= PREDICTION_THRESHOLD).astype(int)
                        
                        if len(np.unique(y_true_chunk)) > 1:
                            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true_chunk, y_pred_chunk, labels=[0, 1]).ravel()
                        else:
                            if np.unique(y_true_chunk)[0] == 0:
                                tn_c = (y_pred_chunk == 0).sum()
                                fp_c = (y_pred_chunk == 1).sum()
                                fn_c = 0
                                tp_c = 0
                            else:
                                tn_c = 0
                                fp_c = 0
                                fn_c = (y_pred_chunk == 0).sum()
                                tp_c = (y_pred_chunk == 1).sum()

                        chunk_recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                        chunk_precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0

                        new_perf = pd.DataFrame({
                            "Chunk": [chunk_number], "Recall": [chunk_recall], "Precision": [chunk_precision],
                            "TN": [tn_c], "FP": [fp_c], "FN": [fn_c], "TP": [tp_c]
                        })
                        updated_perf = pd.concat([realtime_performance(), new_perf], ignore_index=True)
                        realtime_performance.set(updated_perf)
                        latest_performance_metrics.set({"recall": chunk_recall, "precision": chunk_precision})

                        cum_comps = cumulative_cm_components()
                        new_comps = {"tp": cum_comps["tp"] + tp_c, "fn": cum_comps["fn"] + fn_c, "fp": cum_comps["fp"] + fp_c}
                        cumulative_cm_components.set(new_comps)

                        cum_recall = new_comps["tp"] / (new_comps["tp"] + new_comps["fn"]) if (new_comps["tp"] + new_comps["fn"]) > 0 else 0.0
                        cum_precision = new_comps["tp"] / (new_comps["tp"] + new_comps["fp"]) if (new_comps["tp"] + new_comps["fp"]) > 0 else 0.0
                        cumulative_performance.set({"recall": cum_recall, "precision": cum_precision})

                        if len(updated_perf) >= 3:
                            last_three_recalls = updated_perf["Recall"].tail(3)
                            last_three_precisions = updated_perf["Precision"].tail(3)

                            recall_degraded = (last_three_recalls < recall_lcl).all()
                            precision_degraded = (last_three_precisions < precision_lcl).all()

                            performance_degradation_status.set({"degraded": recall_degraded or precision_degraded})
                    
                    last_processed_count.set(current_count)

                # 데이터 드리프트 평가
                last_drift_count = last_drift_processed_count()
                if current_count // DRIFT_CHUNK_SIZE > last_drift_count // DRIFT_CHUNK_SIZE:
                    drift_chunk_number = current_count // DRIFT_CHUNK_SIZE
                    start_idx = (drift_chunk_number - 1) * DRIFT_CHUNK_SIZE
                    end_idx = drift_chunk_number * DRIFT_CHUNK_SIZE

                    current_drift_chunk = df_now.iloc[start_idx:end_idx].copy()

                    if not current_drift_chunk.empty:
                        new_ks_results = []
                        for feature in drift_feature_choices:
                            if feature in train_df.columns and feature in current_drift_chunk.columns:
                                train_vals = train_df[feature].dropna()
                                rt_vals = current_drift_chunk[feature].dropna() 

                                if len(train_vals) > 1 and len(rt_vals) > 1:
                                    try:
                                        ks_stat, p_value = ks_2samp(train_vals, rt_vals)
                                        new_ks_results.append({
                                            "Count": end_idx,
                                            "Feature": feature,
                                            "PValue": p_value
                                        })
                                    except Exception as ks_e:
                                        print(f"⚠️ KS 검정 오류 ({feature}): {ks_e}")

                        if new_ks_results:
                            ks_df = ks_test_results()
                            ks_test_results.set(pd.concat([ks_df, pd.DataFrame(new_ks_results)], ignore_index=True))

                        drift_detected = False
                        drifting_feature = None
                        
                        if current_count >= 300: 
                            all_ks_results = ks_test_results()
                            if not all_ks_results.empty:
                                for feature in drift_feature_choices:
                                    feature_history = all_ks_results[
                                        all_ks_results["Feature"] == feature
                                    ].sort_values(by="Count")

                                    if len(feature_history) >= 3:
                                        last_three_pvalues = feature_history["PValue"].tail(3)
                                        if (last_three_pvalues < 0.05).all():
                                            drift_detected = True
                                            drifting_feature = feature
                                            break
                        
                        data_drift_status.set({"degraded": drift_detected, "feature": drifting_feature})
                        chunk_snapshot_data.set(current_drift_chunk)
                    
                    last_drift_processed_count.set(current_count)

        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️ Streaming loop error: {e}")

    # ==================== 탭 1: 실시간 모니터링 UI ====================
    @output
    @render.text
    def latest_timestamp_text():
        df = current_data()
        if df.empty or "registration_time" not in df.columns:
            return "⏳ 아직 데이터 없음"
        latest_time = pd.to_datetime(df["registration_time"], errors='coerce').max()
        if pd.isna(latest_time):
             return "⏳ 유효한 시간 없음"
        return latest_time.strftime("%Y-%m-%d %H:%M:%S")

    @output
    @render.ui
    def anomaly_status_ui():
        _ = is_streaming()
        _ = current_data()
        st = latest_anomaly_status()
        label, color = {0: ("양호", "#28a745"), 1: ("경고", "#ffc107")}.get(st, ("-", "gray"))
        return ui.div(label, class_="text-white fw-bold text-center",
                      style=f"background:{color}; padding:8px 18px; border-radius:10px;")

    @output
    @render.ui
    def defect_status_ui():
        _ = is_streaming()
        _ = current_data()
        st = latest_defect_status()
        label, color = {0: ("양품", "#28a745"), 1: ("불량", "#dc3545")}.get(st, ("-", "gray"))
        return ui.div(label, class_="text-white fw-bold text-center",
                      style=f"background:{color}; padding:8px 18px; border-radius:10px;")

    def get_realtime_stats(df: pd.DataFrame):
        if df.empty:
            return {
                "total": 0, "anomaly_rate": 0.0, "anomaly_count": 0,
                "defect_rate": 0.0, "today_defect_rate": 0.0, 
                "defect_accuracy": 0.0, "goal_progress": 0.0, "goal_current": 0, "goal_target": 0
            }

        total = len(df)

        # 이상치 카운트 (anom_proba 기반)
        ap = None
        if "_anom_proba" in df.columns:
            ap_try = pd.to_numeric(df["_anom_proba"], errors="coerce")
            if np.isfinite(ap_try).any() and float(np.nanmax(ap_try)) > 0.0:
                ap = ap_try

        if ap is None and "_hdb_strength" in df.columns:
            st = pd.to_numeric(df["_hdb_strength"], errors="coerce")
            if st.notna().any():
                ap = 1.0 - st

        if ap is None:
            anomaly_count = int((pd.to_numeric(df.get("anomaly_status"), errors="coerce").fillna(0) == 1).sum())
        else:
            anomaly_count = int((ap >= ANOMALY_PROBA_THRESHOLD).sum())

        anomaly_rate = (anomaly_count / total) * 100.0 if total > 0 else 0.0

        defect_rate = (
            pd.to_numeric(df.get("defect_status", 0), errors="coerce").fillna(0).astype(int).eq(1).mean() * 100
            if "defect_status" in df.columns else 0.0
        )

        today_defect_rate = 0.0
        if "registration_time" in df.columns:
            try:
                times_coerced = pd.to_datetime(df["registration_time"], errors="coerce")
                today = pd.Timestamp.now().normalize()
                df_today = df[times_coerced >= today]
                if not df_today.empty:
                    today_defect_rate = (
                        pd.to_numeric(df_today.get("defect_status", 0), errors="coerce").fillna(0).astype(int).eq(1).mean() * 100
                    )
            except Exception as e:
                print(f"⚠️ today_defect_rate 계산 오류: {e}")
                today_defect_rate = 0.0

        defect_accuracy = 0.0
        try:
            if not df.empty and not test_label_df.empty:
                current_indices = df.index
                if len(test_label_df) >= len(current_indices):
                    valid_indices = test_label_df.index.intersection(current_indices)
                    if not valid_indices.empty:
                        relevant_labels = test_label_df.loc[valid_indices, [TARGET_COL]]
                        merged = df.loc[valid_indices].join(relevant_labels, how="inner")
                    
                        if "defect_status" in merged.columns and TARGET_COL in merged.columns:
                            y_true = merged[TARGET_COL].astype(int)
                            y_pred = merged["defect_status"].astype(int)
                            correct = (y_true == y_pred).sum()
                            if len(merged) > 0:
                                defect_accuracy = (correct / len(merged)) * 100
        except Exception as e:
            print(f"⚠️ defect_accuracy 계산 오류: {e}")
            defect_accuracy = 0.0

        goal_progress = 0.0
        goal_target = 0
        try:
            if "hour" in train_df.columns:
                total_len = len(train_df)
                daily_counts = total_len / 24  
                goal_target = int(round(daily_counts))
            else:
                goal_target = 100

            if goal_target > 0:
                goal_progress = (len(df) / goal_target) * 100
                goal_progress = min(goal_progress, 100.0)
        except Exception as e:
            print(f"⚠️ 목표 달성률 계산 오류: {e}")
            goal_progress = 0.0
            goal_target = 0

        return {
            "total": total,
            "anomaly_rate": anomaly_rate,
            "anomaly_count": anomaly_count,
            "defect_rate": defect_rate,
            "today_defect_rate": today_defect_rate,
            "defect_accuracy": defect_accuracy,
            "goal_progress": goal_progress,
            "goal_current": len(df),
            "goal_target": goal_target
        }
        
    @output
    @render.ui
    def defect_stats_ui():
        df = current_data()
        stats = get_realtime_stats(df)

        total_count = stats.get("total", 0)
        anomaly_count = stats.get("anomaly_count", 0)
        correct_count = int(total_count * stats["defect_accuracy"] / 100) if total_count > 0 else 0

        return ui.layout_columns(
            ui.div(
                ui.h5("이상치 탐지"),
                ui.h2(f"{stats['anomaly_rate']:.2f}%"),
                ui.p(f"(총 {total_count}개 중 {anomaly_count}개 이상)"),
                class_="card text-white bg-primary text-center p-3",
                style="border-radius: 5px;"
            ),
            ui.div(
                ui.h5("불량 탐지"),
                ui.h2(f"{stats['defect_rate']:.2f}%"),
                ui.p(f"(총 {total_count}개 중 {int(total_count * stats['defect_rate'] / 100)}개 불량)"),
                class_="card text-white bg-success text-center p-3",
                style="border-radius: 5px;"
            ),
            ui.div(
                ui.h5("모델 예측 정확도"),
                ui.h2(f"{stats['defect_accuracy']:.2f}%"),
                ui.p(f"(총 {total_count}개 중 {correct_count}개 일치)"),
                class_="card text-white bg-danger text-center p-3",
                style="border-radius: 5px;"
            ),
            ui.div(
                ui.h5("목표 달성률"),
                ui.h2(f"{stats['goal_progress']:.2f}%"),
                ui.p(f"(총 {stats['goal_target']}개 중 {stats['goal_current']}개 완료)"),
                class_="card bg-warning text-dark text-center p-3",
                style="border-radius: 5px;"
            ),
        )

    @output
    @render.ui
    def realtime_graphs():
        selected = input.selected_sensors()
        if not selected:
            return ui.div("표시할 센서를 선택하세요.", class_="text-warning text-center p-3")

        return ui.div(
            {"class": "d-flex flex-column gap-2"},
            *[ui.card(
                ui.card_header(f"{col}"),
                ui.output_plot(f"plot_{col}", width="100%", height="150px")
            ) for col in selected]
        )

    def make_plot_output(col):
        @output(id=f"plot_{col}")
        @render.plot
        def _plot():
            df = current_data()
            fig, ax = plt.subplots(figsize=(5, 1.6))

            if df.empty or col not in df.columns or df[col].isnull().all():
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                y = df[col].dropna().values
                if len(y) == 0:
                    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    x = np.arange(len(y))
                    window_size = 50
                    if len(y) > window_size:
                        x_window = x[-window_size:]
                        y_window = y[-window_size:]
                    else:
                        x_window = x
                        y_window = y
                    
                    ax.plot(x_window, y_window, linewidth=1.5, color="#007bff", marker="o", markersize=3)
                    if len(x_window) > 0:
                        ax.scatter(x_window[-1], y_window[-1], color="red", s=25, zorder=5)
                        ax.set_xlim(x_window[0], x_window[-1])

                    ax.set_title(f"{col}", fontsize=9, pad=2)
                    ax.tick_params(axis="x", labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.grid(True, linewidth=0.4, alpha=0.4)
            
            plt.tight_layout(pad=0.3)
            return fig
        return _plot

    for col in [
    "molten_temp", "cast_pressure", "low_section_speed",
    "high_section_speed", "biscuit_thickness", "Coolant_temperature", "upper_mold_temp1", "lower_mold_temp2", "lower_mold_temp1",
    "sleeve_temperature", "upper_mold_temp2", "facility_operation_cycleTime",
    "production_cycletime", "physical_strength",
   ]:
        make_plot_output(col)

    @output
    @render.ui
    def prediction_output_ui():
        logs = defect_logs()
        if logs.empty:
            return ui.div("현재 불량 제품이 없습니다.", class_="text-muted text-center p-3")

        display_logs = logs.iloc[::-1].copy()

        if "Time" in display_logs.columns:
            display_logs["시간"] = pd.to_datetime(display_logs["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_logs = display_logs.drop(columns=["Time"])

        if "Prob" in display_logs.columns:
            display_logs["확률"] = (display_logs["Prob"] * 100).round(2).astype(str) + "%"
            display_logs = display_logs.drop(columns=["Prob"])

        rows_html = ""
        for _, row in display_logs.iterrows():
            id_val = row["ID"]
            time_val = row["시간"]
            prob_val = row["확률"]
            rows_html += f"""
                <tr onclick="Shiny.setInputValue('clicked_log_id', {id_val}, {{priority: 'event'}})" style="cursor:pointer;">
                    <td>{id_val}</td><td>{time_val}</td><td>{prob_val}</td>
                </tr>
            """

        table_html = f"""
            <table class="table table-sm table-striped table-hover text-center align-middle">
                <thead><tr><th>ID</th><th>시간</th><th>확률</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        """

        return ui.div(
            ui.HTML(table_html),
            style="max-height: 300px; overflow-y: auto; overflow-x: auto;"
        )

    @reactive.effect
    @reactive.event(input.clicked_log_id)
    def show_log_detail_modal():
        log_id = input.clicked_log_id()
        logs = defect_logs()

        if logs.empty or log_id not in logs["ID"].values:
            ui.notification_show("⚠️ 해당 ID 정보를 찾을 수 없습니다.", duration=3, type="warning")
            return

        row = logs[logs["ID"] == log_id].iloc[0]
        time_val = pd.to_datetime(row["Time"]).strftime("%Y-%m-%d %H:%M:%S")
        prob_val = f"{row['Prob']*100:.2f}%"

        true_label = "데이터 없음"
        if not test_label_df.empty and "id" in test_label_df.columns:
            match = test_label_df[test_label_df["id"] == log_id]
            if not match.empty:
                val = int(match.iloc[0]["passorfail"])
                true_label = "불량" if val == 1 else "양품"

        ui.modal_show(
            ui.modal(
                ui.h4(f"📄 불량 제품 상세 (ID: {log_id})"),
                ui.p(f"시간: {time_val}"),
                ui.p(f"예측 확률: {prob_val}"),
                ui.hr(),
                ui.h5(f"🔍 실제 라벨: {true_label}",
                       class_="fw-bold text-center",
                       style="color:#007bff; font-size:18px;"),
                ui.hr(),

                ui.div(
                    {"class": "d-flex justify-content-center gap-3 mt-3"},
                    ui.input_action_button("correct_btn", "✅ 불량 맞음 (Correct)", class_="btn btn-success px-4 py-2"),
                    ui.input_action_button("incorrect_btn", "❌ 불량 아님 (Incorrect)", class_="btn btn-danger px-4 py-2"),
                ),

                ui.input_text(f"feedback_note_{log_id}", "", placeholder="예: 냉각수온도 급변", width="100%"),
                ui.input_action_button("submit_btn", "💾 피드백 저장", class_="btn btn-primary w-100 mt-3"),

                title="불량 상세 확인 및 피드백",
                easy_close=True,
                footer=None
            )
        )
    
    @reactive.Effect
    @reactive.event(input.correct_btn)
    def set_correct():
        r_correct_status.set("✅ 불량 맞음")
        ui.notification_show("'불량 맞음' 선택됨", duration=2, type="success")

    @reactive.Effect
    @reactive.event(input.incorrect_btn)
    def set_incorrect():
        r_correct_status.set("❌ 불량 아님")
        ui.notification_show("'불량 아님' 선택됨", duration=2, type="error")

    @reactive.Effect
    @reactive.event(input.submit_btn)
    def save_feedback():
        correct_status = r_correct_status()
        log_id = input.clicked_log_id()

        feedback_input_id = f"feedback_note_{log_id}"
        
        feedback_text = ""
        try:
            feedback_text = getattr(input, feedback_input_id)()
        except Exception as e:
            print(f"피드백 텍스트 가져오기 오류: {e}")

        if correct_status is None:
            ui.notification_show("🚨 실제 불량 여부를 먼저 선택해야 합니다.", duration=3, type="warning")
            return

        if not feedback_text:
            ui.notification_show("⚠️ 피드백 내용을 입력해주세요.", duration=3, type="warning")
            return

        new_feedback = pd.DataFrame({
            "ID": [log_id],
            "Prediction": ["불량"],
            "Correct": [correct_status],
            "Feedback": [feedback_text]
        })

        df_old = r_feedback_data()
        df_new = pd.concat([df_old[df_old["ID"] != log_id], new_feedback], ignore_index=True)
        r_feedback_data.set(df_new)
        
        r_correct_status.set(None)

        ui.notification_show("✅ 피드백이 성공적으로 저장되었습니다.", duration=3, type="success")
        ui.modal_remove()

    @output
    @render.ui
    def feedback_table():
        df_feedback = r_feedback_data()
        if df_feedback.empty:
            return ui.div("아직 저장된 피드백이 없습니다.", class_="text-muted text-center p-3")

        if "ID" in df_feedback.columns:
            df_feedback = df_feedback.sort_values(by="ID", ascending=False)

        col_map = {
            "ID": "ID", "Prediction": "예측", "Correct": "정답", "Feedback": "피드백"
        }
        df_feedback = df_feedback.rename(columns=col_map)
        df_feedback = df_feedback[col_map.values()]

        header = ui.tags.tr(*[ui.tags.th(col) for col in df_feedback.columns])
        rows = []
        for _, row in df_feedback.iterrows():
            correct_text = str(row.get("정답", ""))
            correct_style = ""
            if "맞음" in correct_text:
                correct_style = "background-color: #d4edda; color: #155724;"
            elif "아님" in correct_text:
                correct_style = "background-color: #f8d7da; color: #721c24; font-weight: bold;"
            tds = [
                ui.tags.td(str(row.get("ID", ""))),
                ui.tags.td(str(row.get("예측", ""))),
                ui.tags.td(correct_text, style=correct_style),
                ui.tags.td(str(row.get("피드백", "")))
            ]
            rows.append(ui.tags.tr(*tds))

        return ui.tags.div(
            ui.tags.table({"class": "custom-table"}, ui.tags.thead(header), ui.tags.tbody(*rows)),
            style="max-height: 300px; overflow-y: auto;"
        )

    # ==================== 탭 2: P-관리도 UI ====================
    
    @reactive.effect
    def _():
        if not p_chart_streaming():
            return
        
        reactive.invalidate_later(0.5)
        
        df = current_data()
        if df.empty:
            return
        
        current_idx = p_chart_current_index()
        n_points = 50
        max_idx = max(0, len(df) - n_points)
        
        if current_idx < max_idx:
            # lot_size만큼 건너뛰어서 스트리밍 (ARIMA와 동일한 속도)
            lot_size = input.lot_size()
            new_idx = min(current_idx + lot_size, max_idx)
            p_chart_current_index.set(new_idx)
    
    @reactive.Calc
    def get_current_p_data():
        df = current_data()
        if df.empty:
            return [], 0, 0
        
        start = p_chart_current_index()
        n_points = 50
        end = min(start + n_points, len(df))
        
        lot_size = input.lot_size()
        
        # lot_size 미만이면 빈 리스트 반환 (최소 1개 샘플 필요)
        if end - start < lot_size:
            return [], start, end
        p_values = []
        
        # lot_size 단위로 이상치 비율 계산
        for i in range(start, end, lot_size):
            chunk_end = min(i + lot_size, end)
            # 마지막 청크가 lot_size보다 작으면 포함
            if chunk_end - i < lot_size * 0.5:  # 절반 미만이면 제외
                break
            
            chunk = df.iloc[i:chunk_end]
            
            # anomaly_status 컬럼 확인
            if "anomaly_status" in chunk.columns:
                p = chunk["anomaly_status"].mean()
            elif "_anom_proba" in chunk.columns:
                # 대체: _anom_proba가 threshold 이상이면 이상치로 간주
                p = (chunk["_anom_proba"] >= 0.7).mean()
            else:
                p = 0.0
            
            p_values.append(p)
        
        return p_values, start, end

    @reactive.Calc
    def get_violations():
        current_p, start, end = get_current_p_data()
        violations = check_nelson_rules(current_p, CL, UCL, LCL)
        violations_absolute = {
            'rule1': [idx + start for idx in violations['rule1']],
            'rule4': [idx + start for idx in violations['rule4']],
            'rule8': [idx + start for idx in violations['rule8']]
        }
        return violations_absolute, current_p

    @output
    @render.plot(alt="HDBSCAN Anomaly Probability")
    def anom_proba_plot():
        df = current_data()
        fig, ax = plt.subplots(figsize=(12, 3.2))

        if df.empty:
            ax.text(0.5, 0.5, "⏳ 데이터 수신 대기 중...", ha="center", va="center", color="gray")
            ax.axis("off")
            return fig

        y = pd.to_numeric(df.get("_anom_proba"), errors="coerce")
        x = pd.to_datetime(df["registration_time"], errors="coerce") if "registration_time" in df.columns else pd.Series(np.arange(len(df)))

        m = y.notna() & x.notna()
        x, y = x[m], y[m]

        use_fallback = (len(y) == 0) or ((np.nanmax(y.fillna(0)) == 0.0) and ("_hdb_strength" in df.columns))
        if use_fallback:
            y_strength = pd.to_numeric(df.get("_hdb_strength"), errors="coerce")
            x_time = pd.to_datetime(df.get("registration_time"), errors="coerce") if "registration_time" in df.columns else pd.Series(np.arange(len(df)))
            mk = y_strength.notna() & x_time.notna()
            x = x_time[mk]
            y = (1.0 - y_strength[mk]).fillna(0.0)

        window = 400
        if len(y) > window:
            x, y = x.iloc[-window:], y.iloc[-window:]

        if not x.empty:
            ax.plot(x, y, lw=1.2, label="_anom_proba")
            ax.axhline(ANOMALY_PROBA_THRESHOLD, linestyle="--", linewidth=1.0,
                       color="purple", label=f"th={ANOMALY_PROBA_THRESHOLD:.2f}")

            above = y >= ANOMALY_PROBA_THRESHOLD
            if above.any():
                ax.scatter(x[above], y[above], marker='x', s=48, linewidths=1.5,
                           color='red', label="Anomaly(>th)")

            ax.legend(loc="upper right", fontsize=8)
            fig.autofmt_xdate()
        else:
            ax.text(0.5, 0.5, "표시할 데이터 없음", ha="center", va="center", color="gray")

        ax.set_title(f"이상치 확률 (마지막 {len(y)}개)", fontsize=10)
        ax.set_xlabel("시간", fontsize=9)
        ax.set_ylabel("확률", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def anom_logs_ui():
        logs_df = anom_logs()
        if logs_df is None or getattr(logs_df, "empty", True):
            return ui.div("현재 이상치 없음", class_="text-muted text-center p-3")
        
        logs_df = logs_df.copy().iloc[::-1]
        if "Time" in logs_df.columns:
            logs_df["시간"] = pd.to_datetime(logs_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "Proba" in logs_df.columns:
            logs_df["확률"] = (logs_df["Proba"] * 100).round(2).astype(str) + "%"
        
        rows = []
        for _, r in logs_df.iterrows():
            rid = int(r["ID"])
            ts = r.get("시간", "-")
            pr = r.get("확률", "-")
            rows.append(
                ui.tags.tr(
                    {"style": "cursor:pointer;", "onclick": f"Shiny.setInputValue('anom_row_click', {rid}, {{priority: 'event'}})"},
                    ui.tags.td(str(rid)), ui.tags.td(str(ts)), ui.tags.td(str(pr)),
                )
            )
        
        table = ui.tags.table(
            {"class": "table table-sm table-hover table-striped mb-0"},
            ui.tags.thead(ui.tags.tr(ui.tags.th("ID"), ui.tags.th("시간"), ui.tags.th("확률"))),
            ui.tags.tbody(*rows),
        )
        return ui.div(table, style="max-height: 260px; overflow-y:auto; overflow-x:auto;")

    @reactive.effect
    @reactive.event(input.anom_row_click)
    def _show_anom_detail_modal():
        sel_id = input.anom_row_click()
        if sel_id is None:
            return
        s = streamer()
        fd = s.full_data
        if sel_id not in fd.index:
            return
        row = fd.loc[sel_id]
        proba = row.get("_anom_proba", np.nan)
        strength = row.get("_hdb_strength", np.nan)
        ts = _fmt_time(row.get("registration_time", None))

        try:
            top3, model_prob_eff = topk_prob_attribution(row, topk=3)
        except Exception:
            top3, model_prob_eff = [], row.get("_anom_proba", np.nan)

        if not top3:
            modal = ui.modal(
                ui.h4(f"이상치 상세 (ID: {sel_id})"),
                ui.p(f"시간: {ts}"),
                ui.p(
                    f"모델 이상확률: {float(proba):.2%}" if np.isfinite(proba)
                    else (f"모델 이상확률: {(1.0 - float(strength)):.2%}" if np.isfinite(strength)
                        else "모델 이상확률: -")
                ),
                ui.hr(),
                ui.h5("가능한 원인 분석 중..."),
                ui.p("근처 이웃 기준으로 뚜렷한 편차 없음"),
                title="이상치 원인 분석",
                easy_close=True,
                footer=ui.input_action_button("anom_modal_close", "닫기", class_="btn btn-secondary"),
                size="l",
            )
            ui.modal_show(modal)
            return

        pretty = []
        for f, share, raw_val, mu, dp in top3:
            arrow = "↑" if dp > 0 else "↓"
            pretty.append(f"{f} {arrow} (기여={share*100:.2f}%, 값={raw_val})")

        modal = ui.modal(
            ui.h4(f"이상치 상세 (ID: {sel_id})"),
            ui.p(f"시간: {ts}"),
            ui.p(f"모델 이상확률: {model_prob_eff:.2%}" if np.isfinite(model_prob_eff) else "모델 이상확률: -"),
            ui.hr(),
            ui.h5("가능한 원인 (TOP 3) — 현재 확률 기여"),
            ui.tags.ul(*[ui.tags.li(line) for line in pretty]),
            title="이상치 원인 분석",
            easy_close=True,
            footer=ui.input_action_button("anom_modal_close", "닫기", class_="btn btn-secondary"),
            size="l",
        )
        ui.modal_show(modal)

    @reactive.effect
    @reactive.event(input.anom_modal_close)
    def _close_anom_modal():
        ui.modal_remove()

    @output
    @render.plot(alt="P-Control Chart")
    def control_chart():
        current_p, start, end = get_current_p_data()
        violations_abs, _ = get_violations()
        
        lot_size = input.lot_size()

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 데이터가 없는 경우
        if len(current_p) == 0:
            df = current_data()
            if df.empty:
                msg = "데이터 수집 대기 중..."
            else:
                msg = f"데이터 수집 중... ({len(df)}/{lot_size}개, 최소 1개 샘플 = {lot_size}개 데이터 필요)"
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color="gray")
            ax.axis('off')
            plt.close(fig)
            return fig
        
        # x축은 실제 데이터 시점 (ARIMA와 동일)
        x_values = np.arange(start, start + len(current_p) * lot_size, lot_size)

        ax.plot(x_values, current_p, 'o-', color='#1f77b4',
                linewidth=2, markersize=6, label='이상 비율 (p)', zorder=3)

        ax.axhline(y=CL, color='green', linewidth=2, linestyle='-', label=f'CL ({CL:.4f})')
        ax.axhline(y=UCL, color='red', linewidth=2, linestyle='--', label=f'UCL ({UCL:.4f})')
        ax.axhline(y=LCL, color='red', linewidth=2, linestyle='--', label=f'LCL ({LCL:.4f})')

        if (UCL - CL) > 0:
            sigma = (UCL - CL) / 3
            ax.axhline(y=CL + sigma, color='orange', linewidth=1, linestyle=':', alpha=0.5, label='+1σ')
            ax.axhline(y=CL - sigma, color='orange', linewidth=1, linestyle=':', alpha=0.5, label='-1σ')

        all_violations_set = set()
        for rule_violations in violations_abs.values():
            all_violations_set.update(rule_violations)

        # violation은 실제 데이터 시점으로 표시
        if all_violations_set and len(current_p) > 0:
            for sample_idx in sorted(all_violations_set):
                if 0 <= sample_idx < len(current_p):
                    marker, color, size = 'x', 'red', 150

                    if sample_idx in violations_abs['rule1']:
                        color, marker, size = 'red', 'x', 200
                    elif sample_idx in violations_abs['rule8']:
                        color, marker, size = 'orange', 'D', 120
                    elif sample_idx in violations_abs['rule4']:
                        color, marker, size = 'purple', '*', 200

                    # 실제 데이터 시점으로 변환
                    actual_x = start + sample_idx * lot_size
                    ax.scatter([actual_x], [current_p[sample_idx]], marker=marker, s=size, c=color,
                                edgecolors='white', linewidths=2, zorder=5)

        ax.set_xlabel('시점 (데이터 포인트)', fontsize=12, fontweight='bold')
        ax.set_ylabel('이상 비율 (p)', fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        y_margin = (UCL - LCL) * 0.1 if (UCL - LCL) > 0 else 0.01
        ax.set_ylim([max(0, LCL - y_margin), max(UCL + y_margin, np.max(current_p) * 1.1 if len(current_p) > 0 else UCL + y_margin)])

        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.ui
    def violations_list():
        violations_abs, current_p = get_violations()
        start, end = get_current_p_data()[1:]

        all_violations = {}
        for rule, indices in violations_abs.items():
            for idx in indices:
                if start <= idx < end:
                    if idx not in all_violations:
                        all_violations[idx] = []
                    all_violations[idx].append(rule)

        if not all_violations:
            return ui.div(
                ui.p("✅ 현재 범위에서 관리도 이상패턴 탐지 위반이 없습니다.",
                     style="color: #28a745; padding: 20px; text-align: center;")
            )

        sorted_violations = sorted(all_violations.items(), key=lambda x: x[0], reverse=True)
        violation_items = []

        rule_names = {
            'rule1': 'Rule 1: 3σ 초과',
            'rule4': 'Rule 4: 14개 연속 교대',
            'rule8': 'Rule 8: 8개 연속 ±1σ 밖'
        }

        rule_descriptions = {
            'rule1': '관리한계선(UCL/LCL)을 벗어남',
            'rule4': '14개 이상 점이 연속해서 상승-하락이 번갈아 나타남',
            'rule8': '8개 연속 점이 모두 중심선에서 ±1σ 밖에 위치'
        }

        for idx, rules in sorted_violations:
            p_value = all_p_values[idx]

            abnormal_vars = []
            df = current_data()
            if not df.empty and idx < len(df):
                row = df.iloc[idx]
                for var in var_stats.keys():
                    if var in row and pd.notna(row[var]):
                        value = row[var]
                        ucl = var_stats[var]['ucl']
                        lcl = var_stats[var]['lcl']
                        if value > ucl or value < lcl:
                            abnormal_vars.append(var)

            rules_badges = [ui.span(rule_names[rule], class_="violation-rule") for rule in rules]
            rules_desc = [rule_descriptions[rule] for rule in rules]

            first_abnormal_var = abnormal_vars[0] if abnormal_vars else ""
            abnormal_vars_str = ', '.join(abnormal_vars) if abnormal_vars else '없음'
            
            onclick_js = f"""
                if ('{first_abnormal_var}') {{
                    Shiny.setInputValue('selected_variable', '{first_abnormal_var}', {{priority: 'event'}});
                    var dropdown = document.getElementById('selected_variable');
                    if (dropdown) {{ dropdown.value = '{first_abnormal_var}'; }}
                    scrollToArima();
                }}
                alert('시점 {idx}의 상세 분석\\n\\n이상 비율: {p_value:.2%}\\n위반 규칙: {', '.join([rule_names[r] for r in rules])}\\n이상 변수: {abnormal_vars_str}');
            """

            violation_items.append(
                ui.div(
                    {"class": "violation-item"},
                    ui.div(f"⚠️ 시점 {idx} (이상 비율: {p_value:.2%})", class_="violation-header"),
                    ui.div(*rules_badges, style="margin: 6px 0;"),
                    ui.div(
                        ui.tags.ul(
                            *[ui.tags.li(desc, style="font-size: 12px; color: #666;") for desc in rules_desc],
                            style="margin: 8px 0; padding-left: 20px;"
                        )
                    ),
                    ui.div(
                        f"이상 변수: {', '.join(abnormal_vars[:5])}" + ("..." if len(abnormal_vars) > 5 else ""),
                        class_="violation-detail"
                    ) if abnormal_vars else None,
                    ui.tags.button(
                        "🔍 이상원인 분석",
                        class_="btn-cause",
                        onclick=onclick_js
                    )
                )
            )

        total_violations = len(sorted_violations)

        return ui.div(
            ui.div(
                f"총 {total_violations}건의 위반이 발견되었습니다.",
                style="padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 15px; font-weight: bold;"
            ),
            *violation_items
        )

    @output
    @render.plot(alt="ARIMA Control Chart")
    def arima_control_chart():
        var_name = input.selected_variable()
        
        if not var_name or var_name not in arima_stats:
            fig, ax = plt.subplots(figsize=(12, 6))
            msg = "ARIMA 통계가 없습니다.\ntrain_arima_models.py를 실행하여\narima_stats.pkl 파일을 생성하세요."
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="gray")
            ax.axis('off')
            plt.close(fig)
            return fig
        
        df = current_data()
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "데이터 수신 대기 중...", ha="center", va="center", fontsize=12, color="gray")
            ax.axis('off')
            plt.close(fig)
            return fig
        
        start = p_chart_current_index()
        n_points = 50
        end = min(start + n_points, len(df))
        
        if var_name not in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"'{var_name}' 데이터를 찾을 수 없습니다.", 
                    ha="center", va="center", fontsize=12, color="red")
            ax.axis('off')
            plt.close(fig)
            return fig
        
        actual_data = df[var_name].iloc[start:end].dropna()
        
        if len(actual_data) < 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "데이터 수집 대기 중...", 
                    ha="center", va="center", fontsize=12, color="gray")
            ax.axis('off')
            plt.close(fig)
            return fig
        
        try:
            stats = arima_stats[var_name]
            order = stats['order']
            aic = stats['aic']
            
            data_mean = stats['data_mean']
            ucl_data = stats['ucl_data']
            lcl_data = stats['lcl_data']
            data_std = stats['data_std']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x_values = np.arange(start, start + len(actual_data))
            
            ax.plot(x_values, actual_data.values, 'o-', color='#1f77b4', 
                    linewidth=2, markersize=4, label='실제값', alpha=0.8)
            ax.axhline(y=data_mean, color='green', linewidth=2.5, linestyle='-', 
                       label=f'평균 (CL: {data_mean:.2f})')
            ax.axhline(y=ucl_data, color='red', linewidth=2.5, linestyle='--', 
                       label=f'UCL: {ucl_data:.2f}')
            ax.axhline(y=lcl_data, color='red', linewidth=2.5, linestyle='--', 
                       label=f'LCL: {lcl_data:.2f}')
            
            ax.axhline(y=data_mean + data_std, color='orange', linewidth=1, 
                       linestyle=':', alpha=0.5, label='+1σ')
            ax.axhline(y=data_mean - data_std, color='orange', linewidth=1, 
                       linestyle=':', alpha=0.5, label='-1σ')
            
            ooc_data = (actual_data.values > ucl_data) | (actual_data.values < lcl_data)
            if ooc_data.any():
                ax.scatter(x_values[ooc_data], actual_data.values[ooc_data], 
                           marker='x', s=200, c='red', edgecolors='white', 
                           linewidths=3, zorder=5, label='🚨 관리 이탈')
            
            ax.set_xlabel('시점 (데이터 포인트)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{var_name} 측정값', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            y_min = min(lcl_data, actual_data.min())
            y_max = max(ucl_data, actual_data.max())
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim([y_min - y_margin, y_max + y_margin])
            
            plt.tight_layout()
            plt.close(fig)
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"차트 생성 오류: {str(e)}", 
                    ha="center", va="center", fontsize=12, color="red")
            ax.axis('off')
            plt.close(fig)
            return fig

    # ==================== 탭 3: 모델 성능 평가 ====================
    @output
    @render.text
    def latest_recall_text():
        return f"{latest_performance_metrics.get()['recall']:.2%}"

    @output
    @render.text
    def latest_precision_text():
        return f"{latest_performance_metrics.get()['precision']:.2%}"

    @output
    @render.text
    def cumulative_recall_text():
        return f"{cumulative_performance.get()['recall']:.2%}"

    @output
    @render.text
    def cumulative_precision_text():
        return f"{cumulative_performance.get()['precision']:.2%}"

    @output
    @render.ui
    def model_performance_status_ui():
        status = performance_degradation_status.get()
        if status["degraded"]:
            bg_color = "#dc3545"
            title = "⚠️ 모델 성능 저하"
            body = "최근 성능 지표가 관리 하한선을 연속 이탈했습니다. 모델 재학습 또는 점검이 필요합니다."
        else:
            bg_color = "#28a745"
            title = "✅ 모델 성능 양호"
            body = "정상 작동 중입니다."

        return ui.div(
            ui.div(
                ui.h5(title, class_="card-title text-center text-white"),
                ui.hr(style="border-top: 1px solid white; opacity: 0.5; margin: 10px 0;"),
                ui.p(body, class_="card-text text-center text-white", style="font-size: 0.9rem;"),
                style=f"background-color: {bg_color}; padding: 15px; border-radius: 8px; min-height: 160px;",
                class_="d-flex flex-column justify-content-center"
            ),
            ui.p(
                "※ 최근 3개 청크(n=200)의 Recall 또는 Precision이 연속으로 LCL 미만일 경우 '성능 저하'로 표시됩니다.",
                class_="text-muted text-center",
                style="font-size: 0.75rem; margin-top: 8px;"
            )
        )

    @output
    @render.ui
    def data_drift_status_ui():
        status = data_drift_status.get()
        current_count = last_processed_count()

        note = f"※ {DRIFT_CHUNK_SIZE * 3}개 데이터 누적 후, 100개 단위 P-value가 3회 연속 0.05 미만일 경우 '드리프트 의심'으로 표시됩니다."

        if current_count < (DRIFT_CHUNK_SIZE * 3):
            bg_color = "#6c757d"
            title = "🔍 데이터 수집 중"
            body = f"드리프트 모니터링은 {DRIFT_CHUNK_SIZE * 3}개 데이터 수집 후 시작됩니다. (현재 {current_count}개)"
        elif status["degraded"]:
            bg_color = "#ffc107"
            title = "⚠️ 데이터 드리프트 의심"
            body = f"'{status.get('feature', 'N/A')}' 변수 분포 변화 의심. 점검 필요."
        else:
            bg_color = "#28a745"
            title = "✅ 데이터 분포 양호"
            body = "데이터 드리프트 징후가 없습니다."

        return ui.div(
            ui.div(
                ui.h5(title, class_="card-title text-center text-white"),
                ui.hr(style="border-top: 1px solid white; opacity: 0.5; margin: 10px 0;"),
                ui.p(body, class_="card-text text-center text-white", style="font-size: 0.9rem;"),
                style=f"background-color: {bg_color}; padding: 15px; border-radius: 8px; min-height: 160px;",
                class_="d-flex flex-column justify-content-center"
            ),
            ui.p(
                note,
                class_="text-muted text-center",
                style="font-size: 0.75rem; margin-top: 8px;"
            )
        )

    @output
    @render.plot(alt="Data Drift KDE Plot")
    def drift_plot():
        selected_col = input.drift_feature_select()
        rt_df = chunk_snapshot_data()
        fig, ax = plt.subplots()

        if rt_df.empty:
            ax.text(0.5, 0.5, f"데이터 수집 중... ({DRIFT_CHUNK_SIZE}개 도달 시 시작)", ha="center", va="center", color="gray", fontsize=10)
            ax.axis('off')
        elif not selected_col or selected_col not in drift_feature_choices:
            ax.text(0.5, 0.5, "표시할 유효한 특성을 선택하세요.", ha="center", va="center", color="gray", fontsize=10)
            ax.axis('off')
        elif selected_col not in train_df.columns:
            ax.text(0.5, 0.5, f"'{selected_col}'는 학습 데이터에 없습니다.", ha="center", va="center", color="orange", fontsize=10)
            ax.axis('off')
        else:
            try:
                train_series = train_df[selected_col].dropna()
                if not train_series.empty:
                    sns.kdeplot(train_series, ax=ax, label="학습 데이터 (Train)", color="blue", fill=True, alpha=0.2, linewidth=1.5)
                else:
                    ax.text(0.5, 0.6, "학습 데이터 없음", ha="center", va="center", color="blue", alpha=0.5, fontsize=9)

                if selected_col in rt_df.columns:
                    rt_series = rt_df[selected_col].dropna()
                    if len(rt_series) > 1:
                        sns.kdeplot(rt_series, ax=ax, label=f"실시간 (최근 {len(rt_series)}개)", color="red", linewidth=2, linestyle='-')
                    elif len(rt_series) == 1:
                        ax.axvline(rt_series.iloc[0], color="red", linestyle='--', linewidth=1.5, label="실시간 (1개)")

                ax.set_title(f"'{selected_col}' 분포 비교 (KDE)", fontsize=11, pad=10)
                ax.set_xlabel(selected_col, fontsize=9)
                ax.set_ylabel("밀도 (Density)", fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.tick_params(axis='both', which='major', labelsize=8)

            except Exception as e:
                print(f"Drift Plot Error for {selected_col}: {e}")
                ax.text(0.5, 0.5, f"플롯 생성 오류 발생", ha="center", va="center", color="red", fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        return fig

    @output
    @render.plot(alt="KS Test P-value Trend Plot")
    def ks_test_plot():
        selected_ks_col = input.ks_feature_select()
        results_df = ks_test_results()
        fig, ax = plt.subplots()

        if not selected_ks_col or selected_ks_col not in drift_feature_choices:
            ax.text(0.5, 0.5, "P-value 추이를 볼 특성을 선택하세요.", ha="center", va="center", color="gray", fontsize=10)
            ax.axis('off')
        elif results_df.empty or results_df[results_df["Feature"] == selected_ks_col].empty:
            ax.text(0.5, 0.5, f"아직 KS 검정 결과가 없습니다.\n(데이터 {DRIFT_CHUNK_SIZE}개 도달 시 시작)", ha="center", va="center", color="gray", fontsize=10)
            ax.axis('off')
            ax.set_xlim(0, DRIFT_CHUNK_SIZE * 2)
            ax.set_ylim(0, 0.2)
        else:
            try:
                feature_results = results_df[results_df["Feature"] == selected_ks_col].copy()
                feature_results = feature_results.sort_values(by="Count")

                ax.plot(feature_results["Count"], feature_results["PValue"], marker='o', linestyle='-', markersize=5, label='P-value')
                ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='유의수준 (0.05)')

                below_threshold = feature_results[feature_results["PValue"] < 0.05]
                if not below_threshold.empty:
                    ax.scatter(below_threshold["Count"], below_threshold["PValue"], color='red', s=50, zorder=5, label='P < 0.05')

                ax.set_title(f"'{selected_ks_col}' KS 검정 P-value 추이", fontsize=11, pad=10)
                ax.set_xlabel("데이터 수집 시점 (개수)", fontsize=9)
                ax.set_ylabel("P-value", fontsize=9)
                ax.set_ylim(0, 0.2)

                min_x, max_x = feature_results["Count"].min(), feature_results["Count"].max()
                x_margin = max(DRIFT_CHUNK_SIZE * 0.5, (max_x - min_x) * 0.05)
                ax.set_xlim(max(0, min_x - x_margin), max_x + x_margin)

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=8)

                ax.grid(True, alpha=0.3, linestyle=':')
                ax.tick_params(axis='both', which='major', labelsize=8)

            except Exception as e:
                print(f"KS Plot Error for {selected_ks_col}: {e}")
                ax.text(0.5, 0.5, f"플롯 생성 오류 발생", ha="center", va="center", color="red", fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        return fig

    @output
    @render.plot(alt="Real-time Recall Trend Plot")
    def realtime_recall_plot():
        perf_df = realtime_performance()
        fig, ax = plt.subplots()
        if perf_df.empty:
            ax.text(0.5, 0.5, "데이터 수집 중...", ha="center", va="center", color="gray", fontsize=9)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.05)
            ax.axis('off')
        else:
            ax.plot(perf_df["Chunk"], perf_df["Recall"], marker='o', linestyle='-', markersize=4,
                    label='Recall', color='#007bff', zorder=2)
            ax.axhline(y=recall_lcl, color='#6495ED', linestyle='--', linewidth=1.5,
                       label=f'LCL ({recall_lcl:.2%})', zorder=1)
            below_lcl_points = perf_df[perf_df['Recall'] < recall_lcl]
            if not below_lcl_points.empty:
                ax.scatter(below_lcl_points['Chunk'], below_lcl_points['Recall'],
                           color='red', s=40, zorder=3, label='LCL 미만', marker='v')

            ax.set_xlabel("청크 번호 (n=200)", fontsize=9)
            ax.set_ylabel("재현율", fontsize=9)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_ylim(-0.05, 1.05)
            min_x, max_x = perf_df["Chunk"].min(), perf_df["Chunk"].max()
            x_margin = max(1, (max_x - min_x) * 0.05)
            ax.set_xlim(max(0, min_x - x_margin), max_x + x_margin)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout(pad=0.5)
        return fig

    @output
    @render.plot(alt="Real-time Precision Trend Plot")
    def realtime_precision_plot():
        perf_df = realtime_performance()
        fig, ax = plt.subplots()
        if perf_df.empty:
            ax.text(0.5, 0.5, "데이터 수집 중...", ha="center", va="center", color="gray", fontsize=9)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.05)
            ax.axis('off')
        else:
            ax.plot(perf_df["Chunk"], perf_df["Precision"], marker='s', linestyle='-', markersize=4,
                    label='Precision', color='#28a745', zorder=2)
            ax.axhline(y=precision_lcl, color='#3CB371', linestyle='--', linewidth=1.5,
                       label=f'LCL ({precision_lcl:.2%})', zorder=1)
            below_lcl_points = perf_df[perf_df['Precision'] < precision_lcl]
            if not below_lcl_points.empty:
                ax.scatter(below_lcl_points['Chunk'], below_lcl_points['Precision'],
                           color='red', s=40, zorder=3, label='LCL 미만', marker='v')

            ax.set_xlabel("청크 번호 (n=200)", fontsize=9)
            ax.set_ylabel("정밀도", fontsize=9)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_ylim(-0.05, 1.05)
            min_x, max_x = perf_df["Chunk"].min(), perf_df["Chunk"].max()
            x_margin = max(1, (max_x - min_x) * 0.05)
            ax.set_xlim(max(0, min_x - x_margin), max_x + x_margin)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout(pad=0.5)
        return fig

    def create_tooltip_ui(hover_info, perf_data, lcl_value, metric_name):
        if not hover_info or perf_data.empty:
            return None
        x_hover = hover_info['x']
        if perf_data.empty:
            return None

        distances = (perf_data['Chunk'] - x_hover).abs()
        if distances.empty:
            return None

        try:
            nearest_chunk_idx = distances.idxmin()
            point = perf_data.loc[nearest_chunk_idx]

            if abs(point['Chunk'] - x_hover) > 0.5:
                return None

            if point[metric_name] < lcl_value:
                cm_html = f"""
                <table style='margin: 0;'>
                    <tr><th colspan='2' style='font-size: 0.85rem; padding: 3px 6px;'>Chunk {int(point['Chunk'])}</th></tr>
                    <tr><td style='padding: 3px 6px;'>TP: {int(point['TP'])}</td><td style='padding: 3px 6px;'>FP: {int(point['FP'])}</td></tr>
                    <tr><td style='padding: 3px 6px;'>FN: {int(point['FN'])}</td><td style='padding: 3px 6px;'>TN: {int(point['TN'])}</td></tr>
                </table>
                <div style='font-size: 0.8rem; text-align: center; margin-top: 3px;'>
                    {metric_name}: {point[metric_name]:.2%} (LCL: {lcl_value:.2%})
                </div>
                """
                left = hover_info['coords_css']['x'] + 10
                top = hover_info['coords_css']['y'] + 10
                return ui.div(ui.HTML(cm_html), class_="plot-tooltip",
                                style=f"left: {left}px; top: {top}px; border: 1px solid red;")
        except KeyError:
            return None
        return None

    @reactive.effect
    def _():
        recall_tooltip.set(create_tooltip_ui(
            input.realtime_recall_plot_hover(), realtime_performance(), recall_lcl, 'Recall'
        ))

    @output
    @render.ui
    def recall_tooltip_ui():
        return recall_tooltip.get()

    @reactive.effect
    def _():
        precision_tooltip.set(create_tooltip_ui(
            input.realtime_precision_plot_hover(), realtime_performance(), precision_lcl, 'Precision'
        ))

    @output
    @render.ui
    def precision_tooltip_ui():
        return precision_tooltip.get()

    # ===================== 챗봇 =====================
    @output
    @render.ui
    def chatbot_popup():
        if not chatbot_visible.get():
            return None
    
        return ui.div(
            ui.div(
                style=(
                    "position: fixed; top: 0; left: 0; width: 100%; height: 100%; "
                    "background-color: rgba(0, 0, 0, 0.5); z-index: 1050;"
                )
            ),
            ui.div(
                ui.div("🤖 AI 챗봇", class_="fw-bold mb-2", style="font-size: 22px; text-align:center;"),
                ui.div(
                    ui.output_ui("chatbot_response"),
                    style=(
                        "height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 10px; "
                        "padding: 15px; background-color: #f0f4f8; margin-bottom: 12px; font-size: 14px; line-height: 1.4;"
                    )
                ),
                ui.div(
                    ui.input_text("chat_input", "", placeholder="메시지를 입력하세요...", width="80%"),
                    ui.input_action_button("send_chat", "전송", class_="btn btn-primary", style="width: 18%; margin-left: 2%;"),
                    style="display: flex; align-items: center;"
                ),
                ui.input_action_button("close_chatbot", "닫기 ✖", class_="btn btn-secondary mt-3 w-100"),
                style=(
                    "position: fixed; bottom: 90px; right: 20px; width: 800px; background-color: white; "
                    "border-radius: 15px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); "
                    "z-index: 1100; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;"
                )
            )
        )  
    
    @reactive.effect
    @reactive.event(input.toggle_chatbot)
    def _():
        chatbot_visible.set(not chatbot_visible.get())
    
    @reactive.Effect
    @reactive.event(input.close_chatbot)
    def _():
        chatbot_visible.set(False)
      
    @reactive.Effect
    @reactive.event(input.send_chat)
    async def handle_chat_send():
        query = input.chat_input().strip()
        if not query:
            ui.notification_show("질문을 입력해주세요.", duration=3, type="warning")
            return
        
        ui.update_text("chat_input", value="")
        process_chat_query(query)

    def process_chat_query(query: str):
        if not API_KEY:
            r_ai_answer.set("❌ Gemini API 키가 설정되지 않아 챗봇을 사용할 수 없습니다.")
            return

        r_is_loading.set(True)
        r_ai_answer.set("")

        df = current_data()
        if df.empty:
            r_ai_answer.set("❗ 데이터가 없습니다. 스트리밍을 시작해주세요.")
            r_is_loading.set(False)
            return

        dashboard_summary = get_dashboard_summary(df)
        df_filtered, analyze_type = filter_df_by_question(df, query)

        if df_filtered.empty and analyze_type != "No Match":
            r_ai_answer.set(f"❗ '{analyze_type}'에 대한 데이터는 찾을 수 없습니다.")
            r_is_loading.set(False)
            return

        date_range_info = dashboard_summary.get("최신_시간", "N/A")
        defect_count_info = "불량 예측 결과 없음"
        if not df_filtered.empty and 'defect_status' in df_filtered.columns:
            label_counts = df_filtered['defect_status'].value_counts()
            defect_count = label_counts.get(1, 0)
            good_count = label_counts.get(0, 0)
            total_count_filtered = label_counts.sum()
            defect_rate_filtered = (defect_count / total_count_filtered) * 100 if total_count_filtered > 0 else 0
            defect_count_info = f"필터링된 {total_count_filtered}건 분석 중 (불량: {defect_count}건, 양품: {good_count}건, 불량률: {defect_rate_filtered:.2f}%)"
            
            if 'registration_time' in df_filtered.columns:
                try:
                    min_date = df_filtered['registration_time'].min().strftime('%Y-%m-%d %H:%M')
                    max_date = df_filtered['registration_time'].max().strftime('%Y-%m-%d %H:%M')
                    date_range_info = f"기간: {min_date} ~ {max_date}"
                except Exception:
                    date_range_info = "기간 정보 오류"

        latest_defect_id_info = "불량 제품 ID 정보 없음."
        defect_log_df = defect_logs.get()
        if not defect_log_df.empty and 'ID' in defect_log_df.columns:
            latest_ids_raw = defect_log_df['ID'].tail(20).tolist()
            latest_ids = list(map(str, latest_ids_raw))
            latest_defect_id_info = f"최근 불량 제품 20건의 ID: {', '.join(latest_ids)}"

        summary_text = generate_summary_for_gemini(dashboard_summary, query)
        prompt = f"""
        당신은 공정 모니터링 대시보드의 AI 챗봇입니다.
        아래 [대시보드 핵심 정보]와 [데이터 분석 결과]를 참고하여, 사용자의 질문에 대해 명확하고 간결하게 답변해 주세요. 답변은 한국어로 작성해주세요.

        ---
        **[대시보드 핵심 정보 (탭 1 & 3)]**
        {summary_text}
        
        **[데이터 분석 결과 (질문 기반 필터링)]**
        - 분석 대상: {analyze_type}
        - 분석 대상 기간/시점: {date_range_info}
        - {defect_count_info}
        - {latest_defect_id_info}

        ---
        사용자의 질문: "{query}"

        **답변 가이드:**
        1. 질문의 핵심 키워드를 파악하세요.
        2. 질문에 해당하는 정보가 [대시보드 핵심 정보]에 있다면, 해당 정보를 중심으로 답변하세요.
        3. 질문이 특정 기간이나 건수를 명시했다면, [데이터 분석 결과]를 우선적으로 사용하세요.
        4. 수치에는 단위를 명확히 표시하고, 중요한 정보는 **굵게** 표시해 주세요.
        5. 답변은 친절하고 전문적인 톤을 유지하세요.
        """
        
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            # ✅ 타임아웃을 포함한 동기식 호출 (30초)
            import threading
            result = {"text": None, "error": None}
            
            def api_call():
                try:
                    response = model.generate_content(prompt)
                    result["text"] = response.text.strip()
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=api_call, daemon=True)
            thread.start()
            thread.join(timeout=30)  # 30초 타임아웃
            
            if result["text"]:
                r_ai_answer.set(result["text"])
            elif result["error"]:
                error_msg = result["error"]
                if "API_KEY" in error_msg or "401" in error_msg:
                    r_ai_answer.set("❌ Gemini API 키가 유효하지 않습니다. 환경 변수를 확인하세요.")
                else:
                    r_ai_answer.set(f"❌ AI 응답 오류: {error_msg}")
            else:
                r_ai_answer.set("❌ AI 응답 타임아웃 (30초 초과). 나중에 다시 시도해주세요.")
                
        except Exception as e:
            r_ai_answer.set(f"❌ 예상치 못한 오류: {str(e)}")
            print(f"ERROR: {e}")
        
        finally:
            # ✅ 항상 로딩 상태 해제
            r_is_loading.set(False)

    @output
    @render.ui
    def chatbot_response():
        if r_is_loading.get():
            return ui.div(
                 ui.div({"class": "spinner-border text-primary", "role": "status"}, 
                        ui.span({"class": "visually-hidden"}, "Loading...")),
                 ui.p("AI가 답변을 생성 중입니다...", style="margin-left: 10px; color: #555;"),
                 style="display: flex; align-items: center; justify-content: center; height: 100%;"
            )

        return ui.markdown(r_ai_answer.get())

    def filter_df_by_question(df, query):
        df_filtered = pd.DataFrame()
        analyze_type = "No Match" 

        if 'registration_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['registration_time']):
                try:
                    df = df.copy()
                    df['registration_time'] = pd.to_datetime(df['registration_time'], errors='coerce')
                    df = df.dropna(subset=['registration_time'])
                except Exception as e:
                    print(f"시간 데이터 변환 오류: {e}")
                    return pd.DataFrame(), "시간 데이터 변환 오류"
        else:
             return pd.DataFrame(), "시간 컬럼('registration_time') 없음"

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        count_pattern = re.compile(r'(?:최근|가장 최근)\s*(\d+)\s*(?:개|건)')
        count_match = count_pattern.search(query)

        if count_match:
            count = int(count_match.group(1))
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted.tail(count).copy()
            analyze_type = f"필터링된 건수: 최근 {len(df_filtered)}건"
            return df_filtered, analyze_type

        start_date, end_date = None, None
        query_lower = query.lower().replace(" ", "")

        if '오늘' in query_lower or '당일' in query_lower:
            start_date = today
            end_date = today + timedelta(days=1)
            analyze_type = "필터링된 기간: 오늘"
        elif '어제' in query_lower or '전일' in query_lower:
            start_date = today - timedelta(days=1)
            end_date = today
            analyze_type = "필터링된 기간: 어제"
        elif '이번주' in query_lower or '금주' in query_lower:
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(weeks=1)
            analyze_type = "필터링된 기간: 이번 주"
        elif '지난주' in query_lower or '전주' in query_lower:
            start_of_this_week = today - timedelta(days=today.weekday())
            start_date = start_of_this_week - timedelta(weeks=1)
            end_date = start_of_this_week
            analyze_type = "필터링된 기간: 지난 주"

        if start_date is not None:
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted[
                (df_sorted['registration_time'] >= start_date) &
                (df_sorted['registration_time'] < end_date)
            ].copy()

            if df_filtered.empty:
                return pd.DataFrame(), f"{analyze_type} (데이터 없음)"

            return df_filtered, f"{analyze_type} (총 {len(df_filtered)}건)"
        
        return pd.DataFrame(), "No Match"

    def get_dashboard_summary(current_data_df: pd.DataFrame):
        status_text = "🟢 공정 진행 중"
        if was_reset():
            status_text = "🟡 리셋됨"
        elif not is_streaming():
            status_text = "🔴 일시 정지됨"

        anomaly_label = {0: "양호", 1: "경고"}.get(latest_anomaly_status(), "N/A")
        defect_label = {0: "양품", 1: "불량"}.get(latest_defect_status(), "N/A")
        
        latest_time_str = "데이터 없음"
        if not current_data_df.empty and "registration_time" in current_data_df.columns:
             latest_time = pd.to_datetime(current_data_df["registration_time"], errors='coerce').max()
             if not pd.isna(latest_time):
                 latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")

        stats = get_realtime_stats(current_data_df)
        total_count = stats.get("total", 0)
        anomaly_rate = stats.get("anomaly_rate", 0.0)
        anomaly_count = stats.get("anomaly_count", 0)
        defect_rate = stats.get("defect_rate", 0.0)
        today_defect_rate = stats.get("today_defect_rate", 0.0)
        accuracy = stats.get("defect_accuracy", 0.0)
        goal_progress = stats.get("goal_progress", 0.0)
        goal_target = stats.get("goal_target", 'N/A')

        cum_perf = cumulative_performance()
        cum_recall = f"{cum_perf['recall'] * 100:.2f}%"
        cum_precision = f"{cum_perf['precision'] * 100:.2f}%"
    
        latest_perf = latest_performance_metrics()
        latest_recall = f"{latest_perf['recall'] * 100:.2f}%"
        latest_precision = f"{latest_perf['precision'] * 100:.2f}%"

        perf_status = performance_degradation_status()
        perf_status_text = "🚨 성능 저하 감지" if perf_status["degraded"] else "✅ 성능 양호"

        drift_stat = data_drift_status()
        drift_status_text = f"🚨 드리프트 의심 ({drift_stat.get('feature', 'N/A')})" if drift_stat["degraded"] else "✅ 분포 양호"
        
        defect_log_count = len(defect_logs())
        feedback_count = len(r_feedback_data())

        summary = {
            "공정_상태": status_text, "최신_시간": latest_time_str,
            "최근_이상치_상태": anomaly_label, "최근_불량_상태": defect_label,
            "총_처리_건수": total_count, 
            "이상치_탐지율": f"{anomaly_rate:.2f}%",
            "이상치_탐지_건수": anomaly_count,
            "불량_탐지율_전체": f"{defect_rate:.2f}%", "오늘_불량률": f"{today_defect_rate:.2f}%",
            "모델_예측_정확도": f"{accuracy:.2f}%",
            "목표_달성률": f"{goal_progress:.2f}% (목표: {goal_target}개)",
            "누적_재현율": cum_recall, "누적_정밀도": cum_precision,
            "최근_청크_재현율": latest_recall, "최근_청크_정밀도": latest_precision,
            "모델_성능_상태": perf_status_text,
            "데이터_분포_상태": drift_status_text,
            "불량_로그_건수": defect_log_count, "피드백_총_건수": feedback_count,
        }
        return summary

    KEYWORD_TO_INFO = {
        "상태": ["공정_상태", "총_처리_건수", "최신_시간"],
        "현재": ["공정_상태", "불량_탐지율_전체", "최신_시간", "오늘_불량률"],
        "지금": ["공정_상태", "최신_시간"],
        "오늘": ["오늘_불량률", "총_처리_건수"],
        "멈췄": ["공정_상태"], "리셋": ["공정_상태"],
        "이상치": ["최근_이상치_상태", "이상치_탐지율", "이상치_탐지_건수"],
        "불량": ["최근_불량_상태", "불량_탐지율_전체", "오늘_불량률", "불량_로그_건수"],
        "불량률": ["불량_탐지율_전체", "오늘_불량률"],
        "정확도": ["모델_예측_정확도"],
        "재현율": ["누적_재현율", "최근_청크_재현율"],
        "정밀도": ["누적_정밀도", "최근_청크_정밀도"],
        "성능": ["모델_성능_상태", "누적_재현율", "누적_정밀도"],
        "드리프트": ["데이터_분포_상태"],
        "분포": ["데이터_분포_상태"],
        "목표": ["목표_달성률"],
        "피드백": ["피드백_총_건수"],
        "총": ["총_처리_건수"], "최신": ["최신_시간"],
    }

    def generate_summary_for_gemini(summary, query):
        query_lower = query.lower().replace(" ", "")
        required_keys = set()
        for keyword, keys in KEYWORD_TO_INFO.items():
            if keyword in query_lower:
                required_keys.update(keys)

        if not required_keys or any(k in query_lower for k in ["전체", "요약", "모든", "현황", "알려줘"]):
            required_keys = {
                "공정_상태", "총_처리_건수", "이상치_탐지율", "이상치_탐지_건수",
                "불량_탐지율_전체", "모델_예측_정확도",
                "누적_재현율", "누적_정밀도", "모델_성능_상태", "데이터_분포_상태"
            }

        info_parts = []
        base_keys = ["공정_상태", "총_처리_건수", "최신_시간"]
        for key in base_keys:
             if key in summary:
                 info_parts.append(f"[{key.replace('_', ' ')}]: {summary[key]}")
        
        for key, value in summary.items():
            if key in required_keys and key not in base_keys:
                info_parts.append(f"[{key.replace('_', ' ')}]: {value}")

        return "\n".join(info_parts)

# ==================== APP 실행 ====================
app = App(app_ui, server)
