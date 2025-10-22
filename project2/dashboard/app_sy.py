from shiny import App, reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import asyncio
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# shared.py에서 필요한 모든 것을 가져옵니다.
from shared import (
    streaming_df, RealTimeStreamer, defect_model, feature_cols, 
    train_df, test_label_df, test_df, predict_anomaly, defect_threshold
)

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
CHUNK_SIZE = 200
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
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0,1]).ravel()
        validation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        validation_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recalls_per_chunk, precisions_per_chunk = [], []
        for i in range(0, len(valid_df), CHUNK_SIZE):
            chunk = valid_df.iloc[i : i + CHUNK_SIZE]
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
    sigma = (ucl - cl) / 3
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
        
        if i >= 7:
            all_outside = True
            for j in range(i - 7, i + 1):
                if abs(p_values[j] - cl) <= sigma:
                    all_outside = False
                    break
            if all_outside:
                violations['rule8'].append(i)
    
    return violations

# ------------------------------
# Reactive 변수 선언
# ------------------------------
streamer = reactive.Value(RealTimeStreamer(streaming_df))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)
was_reset = reactive.Value(False)
defect_logs = reactive.Value(pd.DataFrame(columns=["Time", "ID", "Prob"]))
r_feedback_data = reactive.Value(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"]))
r_correct_status = reactive.Value(None)




# 1페이지용 reactive 변수
latest_anomaly_status = reactive.Value(0)
latest_defect_status = reactive.Value(0)

realtime_performance = reactive.Value(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
latest_performance_metrics = reactive.Value({"recall": 0.0, "precision": 0.0})
last_processed_count = reactive.Value(0)
performance_degradation_status = reactive.Value({"degraded": False})
cumulative_cm_components = reactive.Value({"tp": 0, "fn": 0, "fp": 0})
cumulative_performance = reactive.Value({"recall": 0.0, "precision": 0.0})
recall_tooltip = reactive.Value(None)
precision_tooltip = reactive.Value(None)

# ------------------------------
# UI 구성
# ------------------------------
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
    """),
    ui.h2("🚀 실시간 품질 모니터링 대시보드", class_="text-center fw-bold my-3"),
    ui.navset_card_tab(
        # ==================== 탭 1: 실시간 모니터링 (업그레이드 버전) ====================
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
                        "facility_operation_cycleTime" : "설비작동사이클시간",
                        "production_cycletime" : "생산사이클시간",
                        "low_section_speed": "저속구간속도",
                        "high_section_speed": "고속구간속도",
                        "cast_pressure" : "주조압력",
                        "biscuit_thickness": "비스킷두께",
                        "upper_mold_temp1" : "상부금형온도1",
                        "upper_mold_temp2" : "상부금형온도2",
                        "lower_mold_temp1" : "하부금형온도1",
                        "lower_mold_temp2" : "하부금형온도2",
                        "sleeve_temperature" : "슬리브온도",
                        "physical_strength" : "물리적강도",
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
            
            # 공정 이상·불량 현황
            ui.card(
                ui.output_ui("defect_stats_ui")
            ),
            
            # 모델 예측 불량 확인 및 피드백
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
        ),
        
        # ==================== 탭 2: P-관리도 ====================
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
        ),
        
        # ==================== 탭 3: 모델 성능 평가 ====================
        ui.nav_panel("모델 성능 평가",
            ui.layout_columns(
                ui.card(
                    ui.card_header("모델 성능", class_="text-center fw-bold"),
                    ui.layout_columns(
                        ui.div(
                            ui.p("재현율 (Recall)"),
                            ui.h4(f"{validation_recall:.2%}"),
                            style="background-color: #f0f8ff; padding: 1rem; border-radius: 8px; text-align: center;"
                        ),
                        ui.div(
                            ui.p("정밀도 (Precision)"),
                            ui.h4(f"{validation_precision:.2%}"),
                            style="background-color: #f0fff0; padding: 1rem; border-radius: 8px; text-align: center;"
                        ),
                        col_widths=[6, 6]
                    )
                ),
                ui.card(
                    ui.card_header("실시간 성능", class_="text-center fw-bold"),
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
                            ui.output_plot("realtime_recall_plot", height="230px", 
                                         hover={"id": "recall_plot_hover"}),
                            ui.output_ui("recall_tooltip_ui"),
                            style="position: relative;"
                        )
                    ),
                    ui.card(
                        ui.card_header("실시간 정밀도(Precision) 추이"),
                        ui.div(
                            ui.output_plot("realtime_precision_plot", height="230px", 
                                         hover={"id": "precision_plot_hover"}),
                            ui.output_ui("precision_tooltip_ui"),
                            style="position: relative;"
                        )
                    )
                ),
                ui.card(
                    ui.card_header("실시간 성능 상태"),
                    ui.output_ui("performance_degradation_ui")
                ),
                col_widths=[8, 4]
            )
        )
    )
)

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
        r_feedback_data.set(pd.DataFrame(columns=["ID", "Prediction", "Correct", "Feedback"]))
        realtime_performance.set(pd.DataFrame(columns=["Chunk", "Recall", "Precision", "TN", "FP", "FN", "TP"]))
        latest_performance_metrics.set({"recall": 0.0, "precision": 0.0})
        last_processed_count.set(0)
        is_streaming.set(False)
        was_reset.set(True)
        performance_degradation_status.set({"degraded": False})
        cumulative_cm_components.set({"tp": 0, "fn": 0, "fp": 0})
        cumulative_performance.set({"recall": 0.0, "precision": 0.0})

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

            reactive.invalidate_later(2.0)
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
                    #if latest_defect_status() != pred_int:
                    #    latest_defect_status.set(pred_int)
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
                        
                        if "id" in test_df.columns and last_idx < len(test_df):
                            actual_id = int(test_df.iloc[last_idx]["id"])
                        else:
                            actual_id = last_idx


                        new_log = pd.DataFrame({
                            "Time": [actual_time],
                            "ID": [actual_id],
                            "Prob": [prob]
                        })
                        logs = defect_logs()
                        defect_logs.set(pd.concat([logs, new_log], ignore_index=True))

                current_data.set(df_now)

                # 성능 평가
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
                        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true_chunk, y_pred_chunk, labels=[0, 1]).ravel()
                        
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

        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️ Streaming loop error: {e}")

    # ==================== 탭 1: 실시간 모니터링 UI (업그레이드) ====================
    @output
    @render.text
    def latest_timestamp_text():
        df = current_data()
        if df.empty or "registration_time" not in df.columns:
            return "⏳ 아직 데이터 없음"
        latest_time = pd.to_datetime(df["registration_time"]).max()
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

    def make_plot_output(col):
        @output(id=f"plot_{col}")
        @render.plot
        def _plot():
            df = current_data()
            if df.empty or col not in df.columns:
                fig, ax = plt.subplots(figsize=(5, 1.6))
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=9)
                plt.close(fig)
                return fig
            y = df[col].values
            x = range(len(y))
            fig, ax = plt.subplots(figsize=(5, 1.6))
            window_size = 50
            if len(y) > window_size:
                x_window = x[-window_size:]
                y_window = y[-window_size:]
            else:
                x_window = x
                y_window = y
            ax.plot(x_window, y_window, linewidth=1.5, color="#007bff", marker="o", markersize=3)
            ax.scatter(x_window[-1], y_window[-1], color="red", s=25, zorder=5)
            ax.set_xlim(x_window[0], x_window[-1])
            ax.set_title(f"{col}", fontsize=9, pad=2)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.4)
            plt.tight_layout(pad=0.3)
            plt.close(fig)
            return fig
        return _plot

    for col in [ 'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
    'low_section_speed', 'high_section_speed', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature']:
        make_plot_output(col)

    def get_realtime_stats(df: pd.DataFrame):
        if df.empty:
            return {
                "total": 0,
                "anomaly_rate": 0.0,
                "defect_rate": 0.0,
                "today_defect_rate": 0.0,
                "defect_accuracy": 0.0,
                "goal_progress": 0.0,  # ✅ 추가
                "goal_current": 0,
                "goal_target": 0
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
        today = pd.Timestamp.now().normalize()
        if "registration_time" in df.columns:
            df["registration_time"] = pd.to_datetime(df["registration_time"], errors="coerce")
            df_today = df[df["registration_time"] >= today]
            today_defect_rate = (
                pd.to_numeric(df_today.get("defect_status", 0), errors="coerce").fillna(0).astype(int).eq(1).mean() * 100
                if not df_today.empty else defect_rate
            )
        else:
            today_defect_rate = defect_rate


        # 🔹 모델 예측 정확도 (실제 라벨 join 방식)
        defect_accuracy = 0.0
        try:
            from shared import test_label_df
            if not df.empty and not test_label_df.empty and "id" in test_label_df.columns:
                df = df.reset_index(drop=True)
                merged = df.join(test_label_df[[TARGET_COL]].reset_index(drop=True), how="inner")
                if "defect_status" in merged.columns and TARGET_COL in merged.columns:
                    y_true = merged[TARGET_COL].astype(int)
                    y_pred = merged["defect_status"].astype(int)
                    correct = (y_true == y_pred).sum()
                    defect_accuracy = (correct / len(merged)) * 100
        except Exception as e:
            print(f"⚠️ defect_accuracy 계산 오류: {e}")

        # ✅ 목표 달성률 계산 (train_df 하루 평균 대비)
        try:
            if "hour" in train_df.columns:
                # 전체 데이터 개수
                total_len = len(train_df)

                # 하루 단위 묶기 (0~23시까지가 하루이므로)
                # 예: 전체 데이터가 73,596개면 하루 약 3,067개 (73,596 / 24)
                daily_counts = total_len / 24  
                goal_target = int(round(daily_counts))
            else:
                goal_target = 100  # fallback

            goal_progress = (len(df) / goal_target) * 100 if goal_target > 0 else 0
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
            rows_html += f"""
                <tr onclick="Shiny.setInputValue('clicked_log_id', {id_val}, {{priority: 'event'}})" style="cursor:pointer;">
                    <td>{id_val}</td><td>{time_val}</td><td>{prob_val}</td>
                </tr>
            """

        table_html = f"""
            <table class="table table-sm table-striped text-center align-middle">
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
                easy_close=True
            )
        )

    # ==================== 피드백 저장 로직 ====================
    # ✅ 불량 여부 선택
    @reactive.Effect
    @reactive.event(input.correct_btn)
    def set_correct():
        r_correct_status.set("✅ 불량 맞음")

    @reactive.Effect
    @reactive.event(input.incorrect_btn)
    def set_incorrect():
        r_correct_status.set("❌ 불량 아님")

    # ✅ 피드백 저장
    @reactive.Effect
    @reactive.event(input.submit_btn)
    def save_feedback():
        correct_status = r_correct_status()
        log_id = input.clicked_log_id()

        
        feedback_input_id = f"feedback_note_{log_id}"
        feedback_text = getattr(input, feedback_input_id)()

        if correct_status is None:
            ui.notification_show("🚨 실제 불량 여부를 먼저 선택해야 합니다.", duration=3, type="warning")
            return

        if not feedback_text:
            ui.notification_show("⚠️ 피드백 내용을 입력해주세요.", duration=3, type="warning")
            return

        df_now = current_data()
        current_id = input.clicked_log_id()
        # if not df_now.empty and "id" in df_now.columns:
        #     current_id = int(df_now.iloc[-1]["id"])  # 🔥 실제 화면에 표시된 샘플 ID

        new_feedback = pd.DataFrame({
            "ID": current_id,
            "Prediction": ["불량"],
            "Correct": [correct_status],
            "Feedback": [feedback_text]
        })

        df_old = r_feedback_data()
        r_feedback_data.set(pd.concat([new_feedback,df_old], ignore_index=True))
        r_correct_status.set(None)

        # ✅ 입력창 초기화
        ui.update_text(feedback_input_id, value="")

        ui.notification_show("✅ 피드백이 성공적으로 저장되었습니다.", duration=3, type="success")



    @output
    @render.ui
    def feedback_table():
        df_feedback = r_feedback_data()
        if df_feedback.empty:
            return ui.div("아직 저장된 피드백이 없습니다.", class_="text-muted text-center p-3")

        # ✅ 최신순 정렬 (가장 최근 피드백이 위로)
        if "ID" in df_feedback.columns:
            df_feedback = df_feedback.sort_values(by="ID", ascending=False)

        # ✅ 컬럼명 한글화
        col_map = {
            "ID": "ID",
            "Prediction": "예측",
            "Correct": "정답",
            "Feedback": "피드백"
        }
        df_feedback = df_feedback.rename(columns=col_map)

        # ✅ 테이블 스타일 구성
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
            ui.tags.style("""
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
            ui.tags.table({"class": "custom-table"}, ui.tags.thead(header), ui.tags.tbody(*rows))
        )



    

    @output
    @render.text
    def selected_status():
        return ""