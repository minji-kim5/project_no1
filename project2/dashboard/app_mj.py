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
import re  # 1번 코드 (챗봇)
import google.generativeai as genai  # 1번 코드 (챗봇)
from scipy.stats import ks_2samp  # 2번 코드 (KS 검정)
import seaborn as sns  # 2번 코드 (KDE 플롯)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# shared.py에서 필요한 모든 것을 가져옵니다.
from shared import (
    streaming_df, RealTimeStreamer, defect_model, feature_cols,
    train_df, test_label_df, test_df, predict_anomaly, defect_threshold, model_dict
)

# 2번 코드 (데이터 드리프트 모니터링)
excluded_drift_cols = [
    'count', 'hour', 'EMS_operation_time', 'tryshot_signal',
    'mold_code', 'heating furnace'
]
# 드리프트 플롯/KS 검정에서 선택할 수 있는 연속형 변수 리스트
drift_feature_choices = [
    col for col in feature_cols
    if col not in excluded_drift_cols
]


# ------------------------------
# Matplotlib 한글 폰트 설정
# ------------------------------
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    rc('font', family='NanumBarunGothic')

# ------------------------------
# 기본 설정 및 데이터 준비
# ------------------------------
TARGET_COL = 'passorfail'
PREDICTION_THRESHOLD = defect_threshold
CHUNK_SIZE = 200 # 성능 평가용 청크 사이즈
DRIFT_CHUNK_SIZE = 100 # ⭐ (요청 1) 드리프트 탐지용 청크 사이즈
startup_error = ""

# Validation 성능 지표 계산
validation_recall = 0.0
validation_precision = 0.0
recall_lcl = 0.0
precision_lcl = 0.0

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
        # LCL/UCL 계산은 성능 평가 기준(CHUNK_SIZE=200)을 따름
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

# ==================== P-관리도 데이터 준비 ====================
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
    sigma = (ucl - cl) / 3 if (ucl-cl) > 0 else 0 # 0으로 나눠지는 오류 방지
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

        if i >= 7 and sigma > 0: # sigma가 0일때는 룰8 검사 무의미
            all_outside = True
            for j in range(i - 7, i + 1):
                if abs(p_values[j] - cl) <= sigma:
                    all_outside = False
                    break
            if all_outside:
                violations['rule8'].append(i)

    return violations

# ------------------------------
# Reactive 변수 선언 (1번 + 2번 통합)
# ------------------------------
streamer = reactive.Value(RealTimeStreamer(streaming_df))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)
was_reset = reactive.Value(False)
defect_logs = reactive.Value(pd.DataFrame(columns=["Time", "ID", "Prob"]))

# 1페이지용 reactive 변수
latest_anomaly_status = reactive.Value(0)
latest_defect_status = reactive.Value(0)
r_feedback_data = reactive.Value(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"]))  # 1번 코드 (개선된 피드백)
r_correct_status = reactive.Value(None)  # 1번 코드 (개선된 피드백)

# 3페이지 (성능)
realtime_performance = reactive.Value(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
latest_performance_metrics = reactive.Value({"recall": 0.0, "precision": 0.0})
last_processed_count = reactive.Value(0) # 성능 평가(200개) 기준
performance_degradation_status = reactive.Value({"degraded": False})
cumulative_cm_components = reactive.Value({"tp": 0, "fn": 0, "fp": 0})
cumulative_performance = reactive.Value({"recall": 0.0, "precision": 0.0})
recall_tooltip = reactive.Value(None)
precision_tooltip = reactive.Value(None)

# 3페이지 (데이터 드리프트 - 2번 코드에서 추가)
ks_test_results = reactive.Value(pd.DataFrame(columns=["Count", "Feature", "PValue"]))
chunk_snapshot_data = reactive.Value(pd.DataFrame())
data_drift_status = reactive.Value({"degraded": False, "feature": None})
last_drift_processed_count = reactive.Value(0) # ⭐ (요청 1) 드리프트(100개) 기준

# 챗봇 (1번 코드에서 추가)
chatbot_visible = reactive.value(False)
r_test_df = reactive.Value(pd.DataFrame())
r_ai_answer = reactive.Value("질문을 입력해주세요.")
r_is_loading = reactive.Value(False)

# --- 챗봇 설정값 (1번 코드) ---
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


# — UI 정의 (1번 + 2번 통합) —
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
        .violations-container { height: 700px; overflow-y: auto; padding-right: 10px; }
        .tooltip-icon { cursor: help; font-size: 0.8em; }
        
        /* 1번 코드 (챗봇) 스타일 */
        #chatbot_response .card-body { padding: 1.5rem; }
        #chatbot_response pre { 
            background-color: #f7f7f7; 
            padding: 10px; 
            border-radius: 5px; 
            overflow-x: auto;
        }
        
        /* 1번 코드 (피드백 테이블) 스타일 */
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

    ui.h2("🚀 실시간 품질 모니터링 대시보드", class_="text-center fw-bold my-3"),
    ui.navset_card_tab(
        # ==================== 탭 1: 실시간 모니터링 (1번 코드 기준) ====================
        ui.nav_panel("실시간 모니터링",
                     ui.div(
                         {"class": "d-flex align-items-center gap-3 mb-3 sticky-top bg-light p-2 shadow-sm"},
                         ui.input_action_button("start", "▶ 시작", class_="btn btn-success"),
                         ui.input_action_button("pause", "⏸ 일시정지", class_="btn btn-warning"),
                         ui.input_action_button("reset", "🔄 리셋", class_="btn btn-secondary"),
                         ui.output_ui("stream_status"),
                     ),
                     ui.div(ui.p(f"⚠️ {startup_error}", style="color:red; font-weight:bold;") if startup_error else ""),

                     # 센서 및 몰드 선택
                     ui.card(
                         ui.card_header("🧭 변수 선택"),
                         ui.h5("확인할 변수 선택"),
                         ui.input_checkbox_group(
                             "selected_sensors",
                             None,
                             choices={
                                 "molten_temp": "용탕온도",
                                 "facility_operation_cycleTime": "설비작동사이클시간",
                                 "production_cycletime": "생산사이클시간",
                                 "low_section_speed": "저속구간속도",
                                 "high_section_speed": "고속구간속도",
                                 "cast_pressure": "주조압력",
                                 "biscuit_thickness": "비스킷두께",
                                 "upper_mold_temp1": "상부금형온도1",
                                 "upper_mold_temp2": "상부금형온도2",
                                 "lower_mold_temp1": "하부금형온도1",
                                 "lower_mold_temp2": "하부금형온도2",
                                 "sleeve_temperature": "슬리브온도",
                                 "physical_strength": "물리적강도",
                                 "Coolant_temperature": "냉각수온도",
                             },
                             selected=["molten_temp", "cast_pressure"],
                             inline=True
                         ),
                         ui.h5("몰드코드 선택"),
                         ui.input_checkbox_group(
                             "selected_molds",
                             None,
                             choices={
                                 "ALL": "ALL",
                                 "8412": "8412",
                                 "8573": "8573",
                                 "8600": "8600",
                                 "8722": "8722",
                                 "8917": "8917",
                                 "8413": "8413",
                                 "8576": "8576"
                             },
                             selected=["ALL"],
                             inline=True
                         ),
                     ),

                     # 최신 데이터 및 상태 표시
                     ui.div(
                         {"class": "d-flex justify-content-around align-items-center flex-wrap mt-3"},
                         ui.div([
                             ui.span("📅 최신 수신 시각: "),
                             ui.output_text("latest_timestamp_text")
                         ], class_="text-center my-2", style="font-size: 16px; font-weight: bold;"),
                         ui.div([
                             ui.div("이상치 상태", class_="fw-bold text-center mb-1"),
                             ui.output_ui("anomaly_status_ui")
                         ], class_="text-center mx-3"),
                         ui.div([
                             ui.div("불량 판정", class_="fw-bold text-center mb-1"),
                             ui.output_ui("defect_status_ui")
                         ], class_="text-center mx-3"),
                     ),

                     # 실시간 그래프
                     ui.output_ui("realtime_graphs"),

                     # 공정 이상·불량 현황 (1번 코드 기준)
                     ui.card(
                         ui.output_ui("defect_stats_ui")
                     ),

                     # 모델 예측 불량 확인 및 피드백 (1번 코드 기준 - 모달)
                     ui.hr(),
                     ui.card(
                         ui.card_header("🤖 모델 예측 불량 확인 및 피드백"),
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
        ), # 탭 1 종료

        # ==================== 탭 2: P-관리도 (1, 2번 공통) ====================
        ui.nav_panel("P-관리도 이상 탐지",
                     ui.div(
                         {"style": "padding: 20px;"},
                         ui.h4("🔍 공정 이상 탐지 P-관리도"),
                         ui.p(f"모니터링 변수: {len(var_stats)}개 | 총 데이터: {len(test_df):,}건",
                              style="color: #666; margin-bottom: 20px;")
                     ),
                     ui.row(
                         ui.column(
                             8,
                             ui.card(
                                 ui.card_header(ui.h4("P-관리도 (공정 이상 비율)", style="margin: 0;")),
                                 ui.output_plot("control_chart", height="650px")
                             )
                         ),
                         ui.column(
                             4,
                             ui.card(
                                 ui.card_header(ui.h4("Nelson Rules 위반 목록", style="margin: 0;")),
                                 ui.div({"class": "violations-container"}, ui.output_ui("violations_list"))
                             )
                         )
                     ),
                     ui.row(
                         ui.column(
                             12,
                             ui.card(
                                 ui.card_header("📊 데이터 범위 설정"),
                                 ui.row(
                                     ui.column(
                                         6,
                                         ui.input_slider("data_points", "표시할 데이터 포인트 수:",
                                                         min=50, max=min(1000, len(test_df)), value=200, step=10, animate=True)
                                     ),
                                     ui.column(
                                         6,
                                         ui.input_slider("start_point", "시작 포인트:",
                                                         min=0, max=len(test_df) - 50, value=0, step=10, animate=True)
                                     )
                                 )
                             )
                         )
                     )
        ), # 탭 2 종료
        
        # ==================== 탭 3: 모델 성능 평가 (2번 코드 기준) ====================
        ui.nav_panel("모델 성능 평가",
            # --- 상단 레이아웃 (실시간 성능 / 누적 성능) ---
            ui.layout_columns(
                # 왼쪽: 실시간 성능 카드
                ui.card(
                    ui.card_header("실시간 성능 (Chunk=200)", class_="text-center fw-bold"), # 청크 크기 명시
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
                # 오른쪽: 누적 성능 지표 카드 (Valid 성능 괄호 추가)
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
            ), # 상단 layout_columns 종료

            # --- 상태 카드 레이아웃 (모델 성능 / 데이터 드리프트) ---
            ui.layout_columns(
                 # 왼쪽: 모델 성능 상태 카드
                 ui.card(
                     ui.card_header("모델 성능 상태"),
                     ui.output_ui("model_performance_status_ui")
                 ),
                 # 오른쪽: 데이터 드리프트 상태 카드
                 ui.card(
                     ui.card_header("데이터 드리프트 상태"),
                     ui.output_ui("data_drift_status_ui")
                 ),
                 col_widths=[6, 6] # 좌우 50% 비율
            ), # 상태 카드 layout_columns 종료

            ui.hr(), # 구분선

            # --- 그래프 레이아웃 (성능 추이 / 데이터 분포) ---
            ui.layout_columns(
                # 왼쪽: 성능 추이 그래프 (Recall, Precision) - 상/하 배치
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
                ), # 왼쪽 div 종료

                # 오른쪽: 데이터 분포 그래프 (KDE, KS P-value) - 상/하 배치
                ui.div(
                    # KDE 분포 비교
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
                                ui.p("학습 vs 실시간(100개) 데이터 분포 비교.", # ⭐ 문구 수정
                                     class_="text-muted small", style="margin-bottom: 0.5rem;")
                            ),
                            col_widths=[7, 5]
                        ),
                        ui.output_plot("drift_plot", height="230px")
                    ),
                    # KS 검정 P-value 추이
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
                                 ui.p("100개 chunk 단위 KS 검정 p-value 추이.", # ⭐ 문구 수정
                                      class_="text-muted small", style="margin-bottom: 0.5rem;")
                             ),
                            col_widths=[7, 5]
                        ),
                        ui.output_plot("ks_test_plot", height="230px")
                    ),
                ), # 오른쪽 div 종료
                col_widths=[6, 6]
            ), # 그래프 layout_columns 종료
        )
    )   , # Navset_card_tab 종료
    
    # ================== 챗봇 (1번 코드) =================
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
    ) # 챗봇 종료
) # page_fluid 종료


# ------------------------------
# SERVER
# ------------------------------
def server(input, output, session):
    # ==================== 공통 제어 ====================
    @reactive.effect
    @reactive.event(input.start)
    def _():
        is_streaming.set(True)
        was_reset.set(False)

    @reactive.effect
    @reactive.event(input.pause)
    def _():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        streamer().reset_stream()
        current_data.set(pd.DataFrame())
        defect_logs.set(pd.DataFrame(columns=["Time", "ID", "Prob"]))
        latest_anomaly_status.set(0)
        latest_defect_status.set(0)
        r_feedback_data.set(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"])) # 1번
        r_correct_status.set(None) # 1번
        realtime_performance.set(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
        latest_performance_metrics.set({"recall": 0.0, "precision": 0.0})
        last_processed_count.set(0)
        is_streaming.set(False)
        was_reset.set(True)
        performance_degradation_status.set({"degraded": False})
        cumulative_cm_components.set({"tp": 0, "fn": 0, "fp": 0})
        cumulative_performance.set({"recall": 0.0, "precision": 0.0})
        ks_test_results.set(pd.DataFrame(columns=["Count", "Feature", "PValue"]))  # 2번
        chunk_snapshot_data.set(pd.DataFrame())  # 2번
        data_drift_status.set({"degraded": False, "feature": None}) # 2번
        last_drift_processed_count.set(0) # ⭐ (요청 1) 드리프트 카운터 리셋
        r_ai_answer.set("질문을 입력해주세요.") # 1번

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

    # ==================== 실시간 스트리밍 (1번 + 2번 통합) ====================
    @reactive.effect
    def _():
        try:
            if not is_streaming():
                return

            reactive.invalidate_later(0.5)  # 2번 코드 기준 (0.5초)
            s = streamer()
            next_batch = s.get_next_batch(1)

            if next_batch is not None:
                df_now = s.get_current_data().copy()
                last_idx = df_now.index[-1] if not df_now.empty else None

                # HDBSCAN 이상치 예측
                try:
                    if last_idx is not None:
                        ana_res = predict_anomaly(df_now.iloc[[-1]])
                        if ana_res is not None and not ana_res.empty:
                            pred01 = int(ana_res.get("anomaly_status", [0])[0])
                            sev = 1 if pred01 == 1 else 0
                            df_now.loc[last_idx, "anomaly_status"] = int(sev)
                            latest_anomaly_status.set(int(sev))
                            try:
                                s.full_data.loc[last_idx, "anomaly_status"] = int(sev)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"⚠️ 이상치 예측 오류: {e}")

                # 불량 예측
                if defect_model is not None and not df_now.empty:
                    latest_row = df_now.iloc[[-1]].copy()
                    for col in feature_cols:
                        if col not in latest_row.columns:
                            latest_row[col] = 0
                    latest_row = latest_row[feature_cols]

                    try:
                        prob = defect_model.predict_proba(latest_row)[0, 1]
                        pred = 1 if prob >= PREDICTION_THRESHOLD else 0
                    except Exception as e:
                        print(f"⚠️ 모델 예측 오류: {e}")
                        pred = 0
                        prob = 0.0

                    pred_int = int(pred)
                    df_now.loc[last_idx, "defect_status"] = pred_int
                    latest_defect_status.set(pred_int)
                    try:
                        s.full_data.loc[last_idx, "defect_status"] = pred_int
                    except Exception:
                        pass

                    try:
                        df_now["defect_status"] = pd.to_numeric(df_now["defect_status"], errors="coerce").fillna(0).astype(int)
                    except Exception:
                        pass

                    if int(pred) == 1:
                        actual_time = df_now.loc[last_idx, "registration_time"]
                        
                        # 1번 코드 수정 (test_df에서 id 가져오기)
                        if "id" in test_df.columns and last_idx < len(test_df):
                            actual_id = int(test_df.iloc[last_idx]["id"])
                        else:
                            actual_id = last_idx # fallback

                        new_log = pd.DataFrame({
                            "Time": [actual_time],
                            "ID": [actual_id],
                            "Prob": [prob]
                        })
                        logs = defect_logs()
                        defect_logs.set(pd.concat([logs, new_log], ignore_index=True))

                current_data.set(df_now)

                # 성능 평가 & KS 검정 수행
                current_count = len(df_now)


                # --- 청크 단위 성능 평가 (CHUNK_SIZE = 200) ---
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
                        
                        # y_true_chunk에 0과 1이 모두 있는지 확인
                        if len(np.unique(y_true_chunk)) > 1:
                            tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true_chunk, y_pred_chunk, labels=[0, 1]).ravel()
                        else:
                            # (예외 처리) 실제 값이 한 종류만 있을 경우
                            if np.unique(y_true_chunk)[0] == 0: # 모두 0 (양품)
                                tn_c = (y_pred_chunk == 0).sum()
                                fp_c = (y_pred_chunk == 1).sum()
                                fn_c = 0
                                tp_c = 0
                            else: # 모두 1 (불량)
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
                    
                    # 마지막 처리 카운트 업데이트 (성능용)
                    last_processed_count.set(current_count)


                # --- ⭐ (요청 1) 청크 단위 데이터 드리프트 평가 (DRIFT_CHUNK_SIZE = 100) ---
                last_drift_count = last_drift_processed_count()
                if current_count // DRIFT_CHUNK_SIZE > last_drift_count // DRIFT_CHUNK_SIZE:
                    drift_chunk_number = current_count // DRIFT_CHUNK_SIZE
                    start_idx = (drift_chunk_number - 1) * DRIFT_CHUNK_SIZE
                    end_idx = drift_chunk_number * DRIFT_CHUNK_SIZE

                    # ⭐ (요청 2) 누적이 아닌 100개 청크만 사용
                    current_drift_chunk = df_now.iloc[start_idx:end_idx].copy()

                    if not current_drift_chunk.empty:
                        new_ks_results = []
                        for feature in drift_feature_choices:
                            if feature in train_df.columns and feature in current_drift_chunk.columns:
                                train_vals = train_df[feature].dropna()
                                # ⭐ (요청 2) 100개 청크의 값만 사용
                                rt_vals = current_drift_chunk[feature].dropna() 

                                if len(train_vals) > 1 and len(rt_vals) > 1:
                                    try:
                                        ks_stat, p_value = ks_2samp(train_vals, rt_vals)
                                        new_ks_results.append({
                                            "Count": end_idx,  # X축을 시점(100, 200...)으로
                                            "Feature": feature,
                                            "PValue": p_value
                                        })
                                    except Exception as ks_e:
                                        print(f"⚠️ KS 검정 오류 ({feature}): {ks_e}")
                                else:
                                    print(f"ℹ️ KS 검정 건너뜀 ({feature}): 데이터 부족 (Train: {len(train_vals)}, Realtime: {len(rt_vals)})")

                        if new_ks_results:
                            ks_df = ks_test_results()
                            ks_test_results.set(pd.concat([ks_df, pd.DataFrame(new_ks_results)], ignore_index=True))

                        # 드리프트 상태 점검 로직 (P-value 3회 연속 < 0.05)
                        drift_detected = False
                        drifting_feature = None
                        
                        # 1000개 누적 데이터 *이후부터* 검사 시작 (이 로직은 유지)
                        if current_count >= 1000: 
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

                        # ⭐ (요청 2) 3페이지 KDE 플롯용 데이터 스냅샷 (100개 chunk)
                        chunk_snapshot_data.set(current_drift_chunk)
                    
                    # 드리프트 처리 카운트 업데이트
                    last_drift_processed_count.set(current_count)

        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️ Streaming loop error: {e}")

    # ==================== 탭 1: 실시간 모니터링 UI (1번 코드 기준) ====================
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

    # 1번 코드의 get_realtime_stats 사용
    def get_realtime_stats(df: pd.DataFrame):
        if df.empty:
            return {
                "total": 0, "anomaly_rate": 0.0, "defect_rate": 0.0,
                "today_defect_rate": 0.0, "defect_accuracy": 0.0,
                "goal_progress": 0.0, "goal_current": 0, "goal_target": 0
            }

        total = len(df)

        # 🔹 이상치 탐지율
        anomaly_rate = (
            pd.to_numeric(df.get("anomaly_status", 0), errors="coerce").fillna(0).astype(int).ne(0).mean() * 100
            if "anomaly_status" in df.columns else 0.0
        )

        # 🔹 불량 탐지율
        defect_rate = (
            pd.to_numeric(df.get("defect_status", 0), errors="coerce").fillna(0).astype(int).eq(1).mean() * 100
            if "defect_status" in df.columns else 0.0
        )

        # 🔹 오늘 불량률
        today_defect_rate = 0.0
        if "registration_time" in df.columns:
            try:
                # df["registration_time"] = pd.to_datetime(df["registration_time"], errors="coerce") # 원본 수정 방지
                times_coerced = pd.to_datetime(df["registration_time"], errors="coerce")
                today = pd.Timestamp.now().normalize()
                df_today = df[times_coerced >= today] # 변환된 시간으로 필터링
                if not df_today.empty:
                    today_defect_rate = (
                        pd.to_numeric(df_today.get("defect_status", 0), errors="coerce").fillna(0).astype(int).eq(1).mean() * 100
                    )
            except Exception as e:
                print(f"⚠️ today_defect_rate 계산 오류: {e}")
                today_defect_rate = 0.0 # 오류 시 0

        # 🔹 모델 예측 정확도 (실제 라벨 join 방식)
        defect_accuracy = 0.0
        try:
            if not df.empty and not test_label_df.empty:
                # current_data의 인덱스를 test_label_df와 맞추기 위해 reset
                current_indices = df.index
                if len(test_label_df) >= len(current_indices):
                    # loc을 사용하여 안전하게 인덱싱
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
            defect_accuracy = 0.0 # 오류 시 0

        # ✅ 목표 달성률 계산 (train_df 하루 평균 대비)
        goal_progress = 0.0
        goal_target = 0
        try:
            if "hour" in train_df.columns:
                total_len = len(train_df)
                # 하루 단위 묶기 (0~23시까지가 하루이므로)
                daily_counts = total_len / 24  
                goal_target = int(round(daily_counts))
            else:
                goal_target = 100  # fallback

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
        correct_count = int(total_count * stats["defect_accuracy"] / 100) if total_count > 0 else 0

        return ui.layout_columns(
            ui.div(
                ui.h5("이상치 탐지"),
                ui.h2(f"{stats['anomaly_rate']:.2f}%"),
                ui.p(f"(총 {total_count}개 중 {int(total_count * stats['anomaly_rate'] / 100)}개 이상)"),
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
                ui.card_header(f"📈 {col}"),
                ui.output_plot(f"plot_{col}", width="100%", height="150px")
            ) for col in selected]
        )

    # 1번 코드의 make_plot_output (모든 변수 리스트 포함)
    def make_plot_output(col):
        @output(id=f"plot_{col}")
        @render.plot
        def _plot():
            df = current_data()
            fig, ax = plt.subplots(figsize=(5, 1.6)) # 플롯 초기화 먼저

            if df.empty or col not in df.columns or df[col].isnull().all():
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                y = df[col].dropna().values # NaN 제거
                if len(y) == 0:
                    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    x = np.arange(len(y)) # NaN 제거된 데이터 기준 x축
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
                        ax.set_xlim(x_window[0], x_window[-1]) # x축 범위

                    ax.set_title(f"{col}", fontsize=9, pad=2)
                    ax.tick_params(axis="x", labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.grid(True, linewidth=0.4, alpha=0.4)
            
            plt.tight_layout(pad=0.3)
            return fig
        return _plot

    # 1번 코드의 전체 변수 리스트 적용
    for col in [ 'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
    'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature']:
        make_plot_output(col)

    # 1번 코드의 prediction_output_ui (모달 클릭 기능 포함)
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

        # 🔥 ID 클릭 시 JS 이벤트 추가
        rows_html = ""
        for _, row in display_logs.iterrows():
            id_val = row["ID"]
            time_val = row["시간"]
            prob_val = row["확률"]
            # JS 클릭 이벤트: Shiny.setInputValue('clicked_log_id', ID값)
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

    # 1번 코드의 모달 표시 로직
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

        # ✅ 실제 라벨 가져오기
        true_label = "데이터 없음"
        if not test_label_df.empty and "id" in test_label_df.columns:
            match = test_label_df[test_label_df["id"] == log_id]
            if not match.empty:
                val = int(match.iloc[0]["passorfail"])
                true_label = "불량" if val == 1 else "양품"

        # ✅ 모달 표시
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
                footer=None # 닫기 버튼 제거
            )
        )
    
    # ==================== 1번 코드: 피드백 저장 로직 ====================
    @reactive.Effect
    @reactive.event(input.correct_btn)
    def set_correct():
        r_correct_status.set("✅ 불량 맞음")
        ui.notification_show(" '불량 맞음' 선택됨", duration=2, type="success")


    @reactive.Effect
    @reactive.event(input.incorrect_btn)
    def set_incorrect():
        r_correct_status.set("❌ 불량 아님")
        ui.notification_show(" '불량 아님' 선택됨", duration=2, type="error")

    @reactive.Effect
    @reactive.event(input.submit_btn)
    def save_feedback():
        correct_status = r_correct_status()
        log_id = input.clicked_log_id()

        feedback_input_id = f"feedback_note_{log_id}"
        
        # input.feedback_note_...() 를 동적으로 호출
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
        # ID가 중복되면 최신 것으로 덮어쓰기
        df_new = pd.concat([df_old[df_old["ID"] != log_id], new_feedback], ignore_index=True)
        r_feedback_data.set(df_new)
        
        r_correct_status.set(None) # 상태 초기화

        ui.notification_show("✅ 피드백이 성공적으로 저장되었습니다.", duration=3, type="success")
        ui.modal_remove() # 모달 닫기

    # 1번 코드의 feedback_table
    @output
    @render.ui
    def feedback_table():
        df_feedback = r_feedback_data()
        if df_feedback.empty:
            return ui.div("아직 저장된 피드백이 없습니다.", class_="text-muted text-center p-3")

        # ✅ 최신순 정렬 (가장 최근 피드백이 위로)
        if "ID" in df_feedback.columns:
            df_feedback = df_feedback.sort_values(by="ID", ascending=False)

        col_map = {
            "ID": "ID", "Prediction": "예측", "Correct": "정답", "Feedback": "피드백"
        }
        df_feedback = df_feedback.rename(columns=col_map)
        df_feedback = df_feedback[col_map.values()] # 순서 고정

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
            # 스타일은 app_ui의 <style> 태그로 이동시킴
            ui.tags.table({"class": "custom-table"}, ui.tags.thead(header), ui.tags.tbody(*rows)),
            style="max-height: 300px; overflow-y: auto;"
        )

    # ==================== 탭 2: P-관리도 (2번 코드 기준 개선) ====================
    @reactive.Calc
    def get_current_p_data():
        start = input.start_point()
        n_points = input.data_points()
        end = min(start + n_points, len(test_df))
        current_p = all_p_values[start:end]
        return current_p, start, end

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
    @render.plot(alt="P-Control Chart")
    def control_chart():
        current_p, start, end = get_current_p_data()
        violations, _ = get_violations()

        fig, ax = plt.subplots(figsize=(12, 7))
        x_values = np.arange(start, end)

        ax.plot(x_values, current_p, 'o-', color='#1f77b4',
                linewidth=1.5, markersize=3, label='이상 비율 (p)')

        ax.axhline(y=CL, color='green', linewidth=1.5, linestyle='-', label=f'CL ({CL:.4f})')
        ax.axhline(y=UCL, color='red', linewidth=1.5, linestyle='--', label=f'UCL ({UCL:.4f})')
        ax.axhline(y=LCL, color='red', linewidth=1.5, linestyle='--', label=f'LCL ({LCL:.4f})')

        if UCL > CL:
            sigma = (UCL - CL) / 3
            if sigma > 1e-9:
                ax.axhline(y=CL + sigma, color='orange', linewidth=1, linestyle=':', alpha=0.7, label='±1σ')
                ax.axhline(y=CL - sigma, color='orange', linewidth=1, linestyle=':', alpha=0.7)

        all_violations_set = set()
        for rule_indices in violations.values():
            all_violations_set.update(rule_indices)

        violation_points = {}
        if all_violations_set:
            for idx in sorted(list(all_violations_set)):
                if start <= idx < end:
                    point_p_value = all_p_values[idx]
                    rules_violated = [rule for rule, indices in violations.items() if idx in indices]

                    marker, color, size = 'o', 'gray', 80
                    if 'rule1' in rules_violated:
                        marker, color, size = 'X', 'red', 150
                    elif 'rule8' in rules_violated:
                        marker, color, size = 'D', 'darkorange', 100
                    elif 'rule4' in rules_violated:
                        marker, color, size = 's', 'purple', 100

                    violation_points[idx] = {'p': point_p_value, 'marker': marker, 'color': color, 'size': size}

        for idx, attrs in violation_points.items():
            ax.scatter([idx], [attrs['p']], marker=attrs['marker'], s=attrs['size'], c=attrs['color'],
                       edgecolors='black', linewidths=0.5, zorder=5, label=f'Rule Violation (at {idx})')

        ax.set_xlabel('데이터 포인트 인덱스', fontsize=11)
        ax.set_ylabel('이상 비율 (p)', fontsize=11)
        ax.set_title('P-관리도 (공정 이상 비율)', fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.4, linestyle=':')

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        violation_label_added = False
        for handle, label in zip(handles, labels):
            if "Rule Violation" in label:
                if not violation_label_added:
                    unique_labels[label.split(' (')[0]] = handle
                    violation_label_added = True
            elif label not in unique_labels:
                unique_labels[label] = handle

        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=8.5, framealpha=0.95)

        min_p = min(current_p) if len(current_p) > 0 else 0
        max_p = max(current_p) if len(current_p) > 0 else 0
        upper_limit = max(UCL, max_p)
        lower_limit = min(LCL if LCL > 0 else 0, min_p)
        y_margin = (upper_limit - lower_limit) * 0.15
        y_margin = max(y_margin, 0.005)

        ax.set_ylim([max(0, lower_limit - y_margin), upper_limit + y_margin])
        ax.set_xlim([start - 1, end])

        plt.tight_layout()
        return fig

    @output
    @render.ui
    def violations_list():
        violations, current_p = get_violations()
        start = input.start_point()
        n_points = input.data_points()
        end = start + n_points

        all_violations = {}
        for rule, indices in violations.items():
            for idx in indices:
                if start <= idx < end:
                    if idx not in all_violations:
                        all_violations[idx] = []
                    all_violations[idx].append(rule)

        if not all_violations:
            return ui.div(
                ui.p("✅ 현재 선택된 범위에서 Nelson Rules 위반이 없습니다.",
                     style="color: #28a745; padding: 20px; text-align: center; font-weight: bold;")
            )
        # violations_list UI 렌더링 (탭 2)
        sorted_violations = sorted(all_violations.items(), key=lambda item: item[0], reverse=True)
        violation_items = []

        rule_names = {
            'rule1': 'Rule 1: 3σ 초과',
            'rule4': 'Rule 4: 14개 연속 교대',
            'rule8': 'Rule 8: 8개 연속 ±1σ 밖'
        }

        rule_descriptions = {
            'rule1': '관리 한계선(UCL/LCL) 벗어남',
            'rule4': '14개 이상 점이 연속 상승/하강 교대',
            'rule8': '8개 연속 점이 중심선 ±1σ 밖에 위치'
        }

        for idx, rules in sorted_violations:
            p_value = all_p_values[idx]

            abnormal_vars = []
            if idx < len(test_df):
                row = test_df.iloc[idx]
                for var, stats in var_stats.items():
                    if var in row and pd.notna(row[var]):
                        value = row[var]
                        if value > stats['ucl'] or value < stats['lcl']:
                            direction = "↑" if value > stats['ucl'] else "↓"
                            abnormal_vars.append(f"{var} ({value:.2f} {direction})")

            rules_badges = [ui.span(rule_names[rule], class_="violation-rule", style="margin-right: 4px;") for rule in rules]
            rules_desc_items = [ui.tags.li(f"{rule_names[rule]}: {rule_descriptions[rule]}", style="font-size: 12px; color: #555;") for rule in rules]

            violation_items.append(
                ui.div(
                    {"class": "violation-item", "style": "margin-bottom: 10px;"},
                    ui.div(f"🚨 시점 {idx} (이상 비율: {p_value:.3f})", class_="violation-header"),
                    ui.div(*rules_badges, style="margin-top: 5px; margin-bottom: 8px;"),
                    ui.tags.ul(*rules_desc_items, style="margin: 0; padding-left: 18px;"),
                    ui.div(
                        f"관련 변수: {', '.join(abnormal_vars[:5])}" + ("..." if len(abnormal_vars) > 5 else ""),
                        class_="violation-detail",
                        style="margin-top: 8px; font-style: italic;"
                    ) if abnormal_vars else ui.div("관련 변수 없음", class_="violation-detail", style="margin-top: 8px; font-style: italic; color: #888;"),
                    ui.tags.button(
                        "🔍 상세 분석", class_="btn-cause",
                        onclick=f"alert('시점 {idx} 상세 분석\\n\\n이상 비율: {p_value:.3f}\\n위반 규칙: {', '.join([rule_names[r] for r in rules])}\\n\\n관련 변수:\\n{chr(10).join(abnormal_vars) if abnormal_vars else '없음'}')"
                    )
                )
            )

        total_violations_in_view = len(sorted_violations)

        return ui.div(
            ui.div(
                f"현재 범위 내 총 {total_violations_in_view}건의 위반 감지됨 (최신순 정렬)",
                style="padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 15px; font-weight: bold; font-size: 14px;"
            ),
            *violation_items
        )

    # ==================== 탭 3: 모델 성능 평가 (2번 코드 기준 UI) ====================
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
            bg_color = "#dc3545"; title = "⚠️ 모델 성능 저하"; body = "최근 성능 지표가 관리 하한선을 연속 이탈했습니다. 모델 재학습 또는 점검이 필요합니다."
        else:
            bg_color = "#28a745"; title = "✅ 모델 성능 양호"; body = "정상 작동 중입니다."

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
    def cumulative_performance_ui():
        return ui.div(
            ui.layout_columns(
                ui.div(
                    ui.p("누적 Recall", class_="text-center fw-bold mb-0", style="font-size: 0.85rem;"),
                    ui.h5(ui.output_text("cumulative_recall_text"), class_="text-center text-primary mt-1")
                ),
                ui.div(
                    ui.p("누적 Precision", class_="text-center fw-bold mb-0", style="font-size: 0.85rem;"),
                    ui.h5(ui.output_text("cumulative_precision_text"), class_="text-center text-success mt-1")
                ),
                col_widths=[6, 6]
            )
        )

    @output
    @render.ui
    def data_drift_status_ui():
        status = data_drift_status.get()
        current_count = last_processed_count() # 성능 카운트를 따라감

        note = f"※ {DRIFT_CHUNK_SIZE * 3}개 데이터 누적 후, 100개 단위 P-value가 3회 연속 0.05 미만일 경우 '드리프트 의심'으로 표시됩니다."

        if current_count < (DRIFT_CHUNK_SIZE * 3):
            bg_color = "#6c757d"; title = "🔍 데이터 수집 중"; body = f"드리프트 모니터링은 {DRIFT_CHUNK_SIZE * 3}개 데이터 수집 후 시작됩니다. (현재 {current_count}개)"
        elif status["degraded"]:
            bg_color = "#ffc107"; title = "⚠️ 데이터 드리프트 의심"; body = f"'{status.get('feature', 'N/A')}' 변수 분포 변화 의심. 점검 필요." # 경고 색상으로 변경
        else:
            bg_color = "#28a745"; title = "✅ 데이터 분포 양호"; body = "데이터 드리프트 징후가 없습니다."

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
        rt_df = chunk_snapshot_data() # ⭐ 이제 100개 chunk 데이터임
        fig, ax = plt.subplots()

        if rt_df.empty:
            # ⭐ (요청 1) 문구 수정
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
                    sns.kdeplot(train_series, ax=ax, label="학습 데이터 (Train)", color="blue", fill=True, alpha=0.2, linewidth=1.5, cut=0)
                else:
                    ax.text(0.5, 0.6, "학습 데이터 없음", ha="center", va="center", color="blue", alpha=0.5, fontsize=9)

                if selected_col in rt_df.columns:
                    rt_series = rt_df[selected_col].dropna() # ⭐ 100개 chunk 데이터
                    if len(rt_series) > 1:
                        # ⭐ (요청 2) 범례(label) 수정
                        sns.kdeplot(rt_series, ax=ax, label=f"실시간 (최근 {len(rt_series)}개)", color="red", linewidth=2, linestyle='-', cut=0)
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
        results_df = ks_test_results() # ⭐ chunk 단위 p-value 기록
        fig, ax = plt.subplots()

        if not selected_ks_col or selected_ks_col not in drift_feature_choices:
            ax.text(0.5, 0.5, "P-value 추이를 볼 특성을 선택하세요.", ha="center", va="center", color="gray", fontsize=10)
            ax.axis('off')
        elif results_df.empty or results_df[results_df["Feature"] == selected_ks_col].empty:
            # ⭐ (요청 1) 문구 수정
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
                # ⭐ (요청 2) X축 레이블 수정
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

            ax.set_xlabel("청크 번호 (n=200)", fontsize=9) # n=200 명시
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

            ax.set_xlabel("청크 번호 (n=200)", fontsize=9) # n=200 명시
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

    # 툴팁 생성 함수 (공통)
    def create_tooltip_ui(hover_info, perf_data, lcl_value, metric_name):
        if not hover_info or perf_data.empty: return None
        x_hover = hover_info['x']
        if perf_data.empty: return None

        distances = (perf_data['Chunk'] - x_hover).abs()
        if distances.empty: return None

        try:
            nearest_chunk_idx = distances.idxmin()
            point = perf_data.loc[nearest_chunk_idx]

            if abs(point['Chunk'] - x_hover) > 0.5: return None

            # LCL 미만일 때만 툴팁 표시
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


    # ===================== 챗봇 (1번 코드) =====================
    @output
    @render.ui
    def chatbot_popup():
        if not chatbot_visible.get():
            return None
    
        return ui.div(
            ui.div(  # 오버레이
                style=(
                    "position: fixed; top: 0; left: 0; width: 100%; height: 100%; "
                    "background-color: rgba(0, 0, 0, 0.5); z-index: 1050;"
                )
            ),
            ui.div(  # 팝업 카드
                ui.div("🤖 AI 챗봇", class_="fw-bold mb-2", style="font-size: 22px; text-align:center;"), # 폰트 크기 수정
                ui.div(  # 메시지 출력 영역
                    ui.output_ui("chatbot_response"),
                    style=(
                        "height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 10px; "
                        "padding: 15px; background-color: #f0f4f8; margin-bottom: 12px; font-size: 14px; line-height: 1.4;"
                    )
                ),
                ui.div(  # 입력 + 전송 버튼
                    ui.input_text("chat_input", "", placeholder="메시지를 입력하세요...", width="80%"),
                    ui.input_action_button("send_chat", "전송", class_="btn btn-primary", style="width: 18%; margin-left: 2%;"),
                    style="display: flex; align-items: center;"
                ),
                ui.input_action_button("close_chatbot", "닫기 ✖", class_="btn btn-secondary mt-3 w-100"), # 닫기 버튼 스타일 변경
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
      
    # 챗봇 질문 처리 로직 분리 (on_send_chat 함수 호출)
    @reactive.Effect
    @reactive.event(input.send_chat)
    async def handle_chat_send():
        query = input.chat_input().strip()
        if not query:
            ui.notification_show("질문을 입력해주세요.", duration=3, type="warning")
            return
        
        # 입력창 비우기
        ui.update_text("chat_input", value="") 
        
        # 비동기로 AI 응답 처리 실행
        await process_chat_query(query)
  
    async def process_chat_query(query: str):
        """AI 모델을 호출하고 결과를 업데이트하는 비동기 함수"""
        if not API_KEY: # API 키 없으면 실행 중단
            r_ai_answer.set("❌ Gemini API 키가 설정되지 않아 챗봇을 사용할 수 없습니다.")
            return

        r_is_loading.set(True)
        r_ai_answer.set("") # 이전 답변 지우기

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
        if not df_filtered.empty and 'defect_status' in df_filtered.columns: # 'predicted_label' -> 'defect_status'
            label_counts = df_filtered['defect_status'].value_counts()
            defect_count = label_counts.get(1, 0) # 불량은 1
            good_count = label_counts.get(0, 0)   # 양품은 0
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
        1. 질문의 핵심 키워드(예: 불량률, 재현율, 상태, 오늘, 최근 N건 등)를 파악하세요.
        2. 질문에 해당하는 정보가 [대시보드 핵심 정보]에 있다면, 해당 정보를 중심으로 답변을 시작하세요.
        3. 질문이 특정 기간('오늘', '어제', '이번 주')이나 건수('최근 N건')를 명시했다면, [데이터 분석 결과]를 우선적으로 사용하여 답변하세요. 
        4. 질문이 **'현재 불량률'** 또는 단순히 **'불량률'**을 묻는 경우, **[불량 탐지율 전체]** 값을 **주요 답변**으로 사용하고, 이것이 **지금까지 누적된 전체 불량률**임을 명시하세요. 추가적으로 [오늘 불량률] 정보를 제공할 수 있습니다.
        5. 수치에는 단위를 명확히 표시하고 (예: 95.50%), 중요한 정보는 **굵게** 표시해 주세요.
        6. 만약 질문과 관련된 정보가 없다면, "해당 정보는 현재 제공되지 않습니다." 또는 "질문을 좀 더 명확하게 해주시겠어요?" 와 같이 답변하세요.
        7. 답변은 친절하고 전문적인 톤을 유지하세요.
        """
        
        # AI 모델 호출 (try-except 추가)
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = await model.generate_content_async( # 비동기 호출로 변경
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config=genai.GenerationConfig(temperature=0.3)
            )
            r_ai_answer.set(response.text.strip())
        except Exception as e:
            error_message = f"❌ AI 응답 오류: {str(e)}"
            # API 키 관련 오류 메시지 개선
            if "API_KEY" in str(e):
                 error_message = "❌ Gemini API 키가 유효하지 않거나 설정되지 않았습니다. 환경 변수를 확인해주세요."
            r_ai_answer.set(error_message)
            print(f"ERROR: Gemini API 호출 중 오류 발생 - {e}") # 서버 로그에 상세 오류 출력

        finally:
            r_is_loading.set(False)


    # 챗봇 응답 UI 렌더링
    @output
    @render.ui
    def chatbot_response():
        if r_is_loading.get():
            return ui.div( # 로딩 표시 개선
                 ui.div({"class": "spinner-border text-primary", "role": "status"}, 
                        ui.span({"class": "visually-hidden"}, "Loading...")),
                 ui.p("AI가 답변을 생성 중입니다...", style="margin-left: 10px; color: #555;"),
                 style="display: flex; align-items: center; justify-content: center; height: 100%;"
            )

        # Gemini 응답은 기본적으로 Markdown을 지원하므로 ui.markdown 사용
        return ui.markdown(r_ai_answer.get())


    # 💡 챗봇용 함수들 (1번 코드)
    # 날짜/기간/건수 파싱 및 필터링 함수
    def filter_df_by_question(df, query):
        df_filtered = pd.DataFrame()
        analyze_type = "No Match" 

        # 시간 컬럼 타입 확인 및 변환 (오류 방지 강화)
        if 'registration_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['registration_time']):
                try:
                    # 원본 DataFrame을 변경하지 않도록 copy() 사용
                    df = df.copy()
                    df['registration_time'] = pd.to_datetime(df['registration_time'], errors='coerce')
                    df = df.dropna(subset=['registration_time']) # 변환 실패한 행 제거
                except Exception as e:
                    print(f"시간 데이터 변환 오류: {e}")
                    return pd.DataFrame(), "시간 데이터 변환 오류"
            # 시간 순 정렬은 필터링 직전에 수행
        else:
             return pd.DataFrame(), "시간 컬럼('registration_time') 없음"

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # 1. 건수 기반 필터링
        count_pattern = re.compile(r'(?:최근|가장 최근)\s*(\d+)\s*(?:개|건)') # 정규식 간소화
        count_match = count_pattern.search(query)

        if count_match:
            count = int(count_match.group(1))
            # 시간 순 정렬 후 마지막 N개 선택
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted.tail(count).copy()
            analyze_type = f"필터링된 건수: 최근 {len(df_filtered)}건"
            return df_filtered, analyze_type

        # 2. 특정 날짜/기간 기반 필터링
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
        else:
            # 특정 날짜 패턴 (다양한 형식 지원)
            # 예: 2023-10-21, 23/10/21, 10월 21일, 10.21 등
            date_pattern = re.compile(
                r'(\d{4}[년\-/.\s]+)?(\d{1,2})[월\-/.\s]+(\d{1,2})일?'
            )
            date_match = date_pattern.search(query)
            if date_match:
                try:
                    year_str = date_match.group(1)
                    month_str = date_match.group(2)
                    day_str = date_match.group(3)

                    year = int(re.sub(r'\D', '', year_str)) if year_str else today.year
                    month = int(month_str)
                    day = int(day_str)

                    target_date = datetime(year, month, day)
                    start_date = target_date
                    end_date = target_date + timedelta(days=1)
                    analyze_type = f"필터링된 기간: {target_date.strftime('%Y-%m-%d')}"
                except ValueError: # 잘못된 날짜 형식 (e.g., 2월 30일)
                    analyze_type = "잘못된 날짜 형식 감지"
                    pass # 파싱 실패 시 무시
                except Exception as e:
                    print(f"날짜 파싱 중 예상치 못한 오류: {e}")
                    pass # 기타 파싱 오류 무시

        # 3. 실제 필터링 적용
        if start_date is not None:
            # 필터링 전에 시간순 정렬
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted[
                (df_sorted['registration_time'] >= start_date) &
                (df_sorted['registration_time'] < end_date)
            ].copy()

            if df_filtered.empty:
                return pd.DataFrame(), f"{analyze_type} (데이터 없음)" # 데이터 없음을 명시

            return df_filtered, f"{analyze_type} (총 {len(df_filtered)}건)"
        
        # 날짜/건수 키워드가 없으면 필터링 없이 No Match 반환
        return pd.DataFrame(), "No Match"


    # 대시보드 요약 정보 생성 함수
    def get_dashboard_summary(current_data_df: pd.DataFrame) -> dict[str, any]:
        status_text = "🟢 공정 진행 중"
        if was_reset(): status_text = "🟡 리셋됨"
        elif not is_streaming(): status_text = "🔴 일시 정지됨"

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

        drift_stat = data_drift_status() # 데이터 드리프트 상태 추가
        drift_status_text = f"🚨 드리프트 의심 ({drift_stat.get('feature', 'N/A')})" if drift_stat["degraded"] else "✅ 분포 양호"
        
        defect_log_count = len(defect_logs())
        feedback_count = len(r_feedback_data())

        summary = {
            "공정_상태": status_text, "최신_시간": latest_time_str,
            "최근_이상치_상태": anomaly_label, "최근_불량_상태": defect_label,
            "총_처리_건수": total_count, "이상치_탐지율": f"{anomaly_rate:.2f}%",
            "불량_탐지율_전체": f"{defect_rate:.2f}%", "오늘_불량률": f"{today_defect_rate:.2f}%",
            "모델_예측_정확도": f"{accuracy:.2f}%",
            "목표_달성률": f"{goal_progress:.2f}% (목표: {goal_target}개)",
            "누적_재현율": cum_recall, "누적_정밀도": cum_precision,
            "최근_청크_재현율": latest_recall, "최근_청크_정밀도": latest_precision,
            "모델_성능_상태": perf_status_text, # 키 이름 변경
            "데이터_분포_상태": drift_status_text, # 드리프트 상태 추가
            "불량_로그_건수": defect_log_count, "피드백_총_건수": feedback_count,
        }
        return summary

    # 키워드-정보 매핑 (업데이트)
    KEYWORD_TO_INFO = {
        "상태": ["공정_상태", "총_처리_건수", "최신_시간"],
        "현재": ["공정_상태", "불량_탐지율_전체", "최신_시간", "오늘_불량률"],
        "지금": ["공정_상태", "최신_시간"],
        "오늘": ["오늘_불량률", "총_처리_건수"],
        "멈췄": ["공정_상태"], "리셋": ["공정_상태"],
        "이상치": ["최근_이상치_상태", "이상치_탐지율"],
        "불량": ["최근_불량_상태", "불량_탐지율_전체", "오늘_불량률", "불량_로그_건수"],
        "불량률": ["불량_탐지율_전체", "오늘_불량률"],
        "정확도": ["모델_예측_정확도"],
        "재현율": ["누적_재현율", "최근_청크_재현율"],
        "정밀도": ["누적_정밀도", "최근_청크_정밀도"],
        "성능": ["모델_성능_상태", "누적_재현율", "누적_정밀도"],
        "드리프트": ["데이터_분포_상태"], # 드리프트 키워드 추가
        "분포": ["데이터_분포_상태"], # 분포 키워드 추가
        "목표": ["목표_달성률"],
        "피드백": ["피드백_총_건수"],
        "총": ["총_처리_건수"], "최신": ["최신_시간"],
    }

    # Gemini 프롬프트용 요약 생성 함수 (업데이트)
    def generate_summary_for_gemini(summary: dict[str, any], query: str) -> str:
        query_lower = query.lower().replace(" ", "")
        required_keys = set()
        for keyword, keys in KEYWORD_TO_INFO.items():
            if keyword in query_lower:
                required_keys.update(keys)

        if not required_keys or any(k in query_lower for k in ["전체", "요약", "모든", "현황", "알려줘"]):
            required_keys = {
                "공정_상태", "총_처리_건수", "불량_탐지율_전체", "모델_예측_정확도",
                "누적_재현율", "누적_정밀도", "모델_성능_상태", "데이터_분포_상태" # 기본 요약에 드리프트 포함
            }

        info_parts = []
        # 항상 포함할 기본 정보
        base_keys = ["공정_상태", "총_처리_건수", "최신_시간"]
        for key in base_keys:
             if key in summary: info_parts.append(f"[{key.replace('_', ' ')}]: {summary[key]}")
        
        # 질문 기반 또는 전체 요약 정보 추가 (중복 제외)
        for key, value in summary.items():
            if key in required_keys and key not in base_keys:
                info_parts.append(f"[{key.replace('_', ' ')}]: {value}")

        return "\n".join(info_parts)

# ------------------------------
# APP 실행
# ------------------------------
app = App(app_ui, server)