import pandas as pd
import joblib
import shiny
from shiny import App, ui, render, reactive
from pathlib import Path
import datetime
import warnings
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import traceback
import plotly.graph_objects as go

# Matplotlib 한글 폰트 설정 (한글 깨짐 해결)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 파일 경로 설정 ---
APP_DIR = Path(__file__).parent
DATA_RAW_FILE_NAME = "train.csv"
DATA_TS_FILE_NAME = "train_drop.csv" # df_ts용 파일
DATA_PRED_FILE_NAME = "train_df.csv"
MODEL_FILE_NAME = "final_model.pkl"

DATA_RAW_FILE_PATH = APP_DIR / DATA_RAW_FILE_NAME
DATA_TS_FILE_PATH = APP_DIR / DATA_TS_FILE_NAME
DATA_PRED_FILE_PATH = APP_DIR / DATA_PRED_FILE_NAME
MODEL_FILE_PATH = APP_DIR / MODEL_FILE_NAME

# --- 변수명 한글 매핑 딕셔너리 ---
FEATURE_NAME_KR = {
    "cast_pressure": "주조압력(cast_pressure)",
    "count": "카운트(count)",
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
    "molten_volume": "용탕량(molten_volume)",
    "physical_strength": "물리적강도(physical_strength)",
    "EMS_operation_time": "EMS작동시간(EMS_operation_time)",
    "hour": "시간(hour)",
    "heating_furnace": "가열로(heating_furnace)",
    "tryshot_signal": "트라이샷신호(tryshot_signal)",
    "mold_code": "금형코드(mold_code)",
    "working": "가동여부(working)"
}

def get_kr_name(eng_name):
    """영어 변수명을 한글(영어) 형식으로 반환"""
    return FEATURE_NAME_KR.get(eng_name, eng_name)

EDA_DESCRIPTIONS = {
    "low_section_speed": """값 65535는 이상치로 판단되어 해당 데이터 행을 제거했습니다.<br>
                            49 이하이면서 양품인 데이터 26개 중 6개를 KNN을 통해 불량으로 치환했습니다.<br>
                            이후 남은 결측치는 KNN Imputer를 통해 주변 값으로 보간되었습니다.""",
    "molten_temp": """용탕 온도가 80도 이하인 데이터는 센서 오류로 간주하여 결측치(NaN)로 처리했습니다.<br>
                      이후 결측치는 KNN Imputer를 통해 보간되었습니다.""",
    "physical_strength": """값 65535인 데이터는 이상치로 판단되어 해당 행을 제거했습니다.<br>
                            강도가 5 이하인 값은 결측치(NAN)로 처리했습니다.<br>
                            이후 결측치는 KNN Imputer를 통해 보간되었습니다.""",
    "Coolant_temperature": "냉각수 온도가 1449인 데이터는 이상치로 판단되어 해당 데이터 행들을 제거했습니다. (총 9개 행)",
    "upper_mold_temp1": "상부 금형 온도가 1449인 데이터는 이상치로 판단되어 해당 데이터 행을 제거했습니다 (1개 행).",
    "upper_mold_temp2": "상부 금형 온도가 4232인 데이터는 이상치로 판단되어 해당 데이터 행을 제거했습니다 (1개 행).",
    "tryshot_signal": "결측치가 존재하는 경우, 가장 일반적인 상태인 'A'로 대체했습니다.",
    "molten_volume": "용탕량이 기록되지 않은 경우(NaN), 값을 0으로 채웠습니다. 이는 용탕량을 측정하지 않은 상태로 나타낼 수 있습니다.",
    "heating_furnace": "용탕량이 기록되었으나 가열로 정보가 없는 경우, 'C' 가열로에서 작업한 것으로 간주하여 결측치를 채웠습니다.",
    'hour': "registration time에서 시간을 차용하여 범주형으로 처리했습니다.",
    'EMS_operation_time': "값이 5개(0, 3, 6, 23, 25) 뿐이라 범주형으로 처리했습니다.",
    'mold_code': '수치의 의미보다 각 코드를 구분하는 의미라고 판단해 범주형으로 처리했습니다.'
}

# --- 데이터 로드 및 슬라이더 범위 설정 ---
DATA_PRED_LOADED_SUCCESSFULLY = False

feature_stats = {}
numerical_features = []
mold_code_choices = ["N/A"]
mold_code_choices_top5 = ["8722"]
working_choices = ["N/A"]
df_raw = pd.DataFrame()
df_ts = pd.DataFrame()
df_pred = pd.DataFrame()

EXISTING_NUMERICAL_FEATURES = [
    "cast_pressure", "count", "upper_mold_temp1", "lower_mold_temp2",
    "low_section_speed", "lower_mold_temp1", "sleeve_temperature", "high_section_speed",
    "upper_mold_temp2", "biscuit_thickness", "facility_operation_cycleTime",
    "Coolant_temperature", "production_cycletime", "molten_temp", "molten_volume",
    "physical_strength", "EMS_operation_time"
]
REQUIRED_BINARY_FEATURES = ["heating_furnace", "tryshot_signal"]
REQUIRED_HOUR_FEATURE = ["hour"]

cpk_results = {}
min_date_str = None
max_date_str = None

try:
    try:
        df_raw = pd.read_csv(DATA_RAW_FILE_PATH, low_memory=False)
    except UnicodeDecodeError:
        df_raw = pd.read_csv(DATA_RAW_FILE_PATH, encoding='cp949', low_memory=False)
    try:
        df_ts = pd.read_csv(DATA_TS_FILE_PATH, low_memory=False)
    except UnicodeDecodeError:
        df_ts = pd.read_csv(DATA_TS_FILE_PATH, encoding='cp949', low_memory=False)
    try:
        df_pred = pd.read_csv(DATA_PRED_FILE_PATH, low_memory=False)
    except UnicodeDecodeError:
        df_pred = pd.read_csv(DATA_PRED_FILE_PATH, encoding='cp949', low_memory=False)
    
    date_column_name = None
    if "registration_time" in df_raw.columns:
        date_column_name = "registration_time"
    elif "date" in df_raw.columns:
        date_column_name = "date"
    elif "time" in df_raw.columns:
        date_column_name = "time"

    if date_column_name:
        df_raw['datetime_full'] = pd.to_datetime(df_raw[date_column_name], errors='coerce')
        df_raw['date_only'] = df_raw['datetime_full'].dt.date
    else:
        df_raw['date_only'] = pd.NaT 
        df_raw['datetime_full'] = pd.NaT


    total_failures = df_pred['passorfail'].sum()
    total_count = len(df_pred)
    overall_failure_rate = (total_failures / total_count * 100).round(2) if total_count > 0 else 0
    
    date_column_name = None
    if "registration_time" in df_raw.columns:
        date_column_name = "registration_time"
    elif "date" in df_raw.columns:
        date_column_name = "date"
    elif "time" in df_raw.columns:
        date_column_name = "time"

    if date_column_name:
        df_raw['datetime_full'] = pd.to_datetime(df_raw[date_column_name], errors='coerce')
        df_raw['date_only'] = df_raw['datetime_full'].dt.date
    else:
        df_raw['date_only'] = pd.NaT 
        df_raw['datetime_full'] = pd.NaT

    if date_column_name and date_column_name in df_pred.columns:
        df_raw[date_column_name] = pd.to_datetime(df_raw[date_column_name], errors='coerce')
        df_pred[date_column_name] = pd.to_datetime(df_pred[date_column_name], errors='coerce')
        
        df_pred = pd.merge(
            df_pred,
            df_raw[[date_column_name, 'date_only']].drop_duplicates(),
            on=date_column_name,
            how='left'
        )
    else:
        print(f"경고: '{date_column_name}'")

    daily_stats = df_raw.groupby('date_only')['passorfail'].agg(
        ['count', lambda x: (x == 1.0).sum()]
    ).rename(columns={'count': 'total', '<lambda_0>': 'failures'})
    
    daily_stats = daily_stats[pd.notna(daily_stats.index)] 
    daily_stats['failure_rate'] = (daily_stats['failures'] / daily_stats['total'] * 100).round(2)
    
    if len(daily_stats) > 0:
        min_date_str = daily_stats.index.min().strftime('%Y-%m-%d')
        max_date_str = daily_stats.index.max().strftime('%Y-%m-%d')
    else:
        min_date_str = "2024-01-01"
        max_date_str = "2024-12-31"

    if date_column_name:
        if date_column_name in df_ts.columns:
            df_ts['datetime_full'] = pd.to_datetime(df_ts[date_column_name], errors='coerce')
            df_ts.dropna(subset=['datetime_full'], inplace=True)
            df_ts['date_only'] = df_ts['datetime_full'].dt.date
            print("[INFO] df_ts에 날짜 정보 처리 완료.")
        else:
            print(f"[WARNING] df_ts에 '{date_column_name}' 컬럼이 없습니다.")
            df_ts['datetime_full'] = pd.NaT
            df_ts['date_only'] = pd.NaT
    else:
        print("[WARNING] 원본 데이터(df_raw)에서 날짜 컬럼을 찾을 수 없습니다.")

    numerical_features = [col for col in EXISTING_NUMERICAL_FEATURES if col in df_pred.columns]

    feature_stats = {}
    for col in numerical_features:
        clean_series = df_pred[col].dropna()
        if not clean_series.empty:
            feature_stats[col] = {'min': round(float(clean_series.min()), 2),
                                  'max': round(float(clean_series.max()), 2),
                                  'value': round(float(clean_series.median()), 2)}
        else:
            feature_stats[col] = {'min': 0, 'max': 100, 'value': 50}

    HOUR_COL = 'hour'
    if HOUR_COL in df_raw.columns:
        clean_series = df_raw[HOUR_COL].dropna()
        if not clean_series.empty:
            hour_min, hour_max = clean_series.min(), clean_series.max()
            feature_stats[HOUR_COL] = {'min': round(float(hour_min), 2),
                                       'max': round(float(hour_max), 2),
                                       'value': round(float(clean_series.median()), 2)}
        else:
            feature_stats[HOUR_COL] = {'min': 0, 'max': 23, 'value': 12}
    else:
        feature_stats[HOUR_COL] = {'min': 0, 'max': 23, 'value': 12} 
        
    if HOUR_COL not in numerical_features:
        numerical_features.append(HOUR_COL) 

    if 'mold_code' in df_pred.columns:
        mold_code_choices = sorted(df_pred['mold_code'].dropna().astype(str).unique().tolist())
    working_choices = ['가동', '정지']
    
    DATA_PRED_LOADED_SUCCESSFULLY = True

    cpk_analysis_vars = numerical_features
    spec_limits = {}
    for var in cpk_analysis_vars:
        if var in df_pred.columns:
            series = df_pred[var].dropna()
            if len(series) > 1:
                mean, std = series.mean(), series.std()
                lsl = round(mean - 3 * std, 2)
                usl = round(mean + 3 * std, 2)
                spec_limits[var] = {'lsl': lsl, 'usl': usl, 'estimated': True}

    def calculate_cpk(series, lsl, usl):
        series = series.dropna()
        if len(series) < 2: return {}
        mu, sigma = series.mean(), series.std()
        if sigma == 0: return {}
        cp = (usl - lsl) / (6 * sigma)
        cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
        ucl = round(mu + 3 * sigma, 2)
        lcl = round(mu - 3 * sigma, 2)
        return {'cp': round(cp, 2), 'cpk': round(cpk, 2), 'mean': round(mu, 2), 'std': round(sigma, 2), 'ucl': ucl, 'lcl': lcl}

    try:
        df_pred = pd.read_csv(DATA_PRED_FILE_PATH, low_memory=False)
        DATA_PRED_LOADED_SUCCESSFULLY = True
        
        if 'mold_code' in df_pred.columns:
            top5_mold_codes = df_pred['mold_code'].value_counts().head(5).index.astype(str).tolist()
            mold_code_choices_top5 = top5_mold_codes if top5_mold_codes else ["8722"]
        
    except UnicodeDecodeError:
        df_pred = pd.read_csv(DATA_PRED_FILE_PATH, encoding='cp949', low_memory=False)
        DATA_PRED_LOADED_SUCCESSFULLY = True
        
        if 'mold_code' in df_pred.columns:
            top5_mold_codes = df_pred['mold_code'].value_counts().head(5).index.astype(str).tolist()
            mold_code_choices_top5 = top5_mold_codes if top5_mold_codes else ["8722"]

except FileNotFoundError as e:
    print(f"데이터 파일 로드 중 오류: {e}")
    DATA_PRED_LOADED_SUCCESSFULLY = False
    DATA_PRED_LOADED_SUCCESSFULLY = False
    df_raw, df_pred, daily_stats, feature_stats, cpk_results = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}
    numerical_features, mold_code_choices, working_choices = [], ["N/A"], ["N/A"]
    mold_code_choices_top5 = ["8722"]
    overall_failure_rate, total_count, total_failures = 0, 0, 0
    min_date_str, max_date_str = "2024-01-01", "2024-12-31"

HOURLY_CHOICES = [
    "cast_pressure", "upper_mold_temp1", "lower_mold_temp2",
    "low_section_speed", "lower_mold_temp1", "sleeve_temperature", "high_section_speed",
    "upper_mold_temp2", "biscuit_thickness", "Coolant_temperature", "molten_temp", 
    "molten_volume", "physical_strength", "EMS_operation_time"
]

pipeline, feature_names = None, []
MODEL_LOADED_SUCCESSFULLY = False
try:
    final_model = joblib.load(MODEL_FILE_PATH)
    
    if isinstance(final_model, dict):
        pipeline = final_model.get("model") 
        feature_names = final_model.get("feature_names", [])
    else:
        pipeline = final_model
        
    if pipeline is not None and hasattr(pipeline, 'predict'):
        MODEL_LOADED_SUCCESSFULLY = True

except FileNotFoundError:
    pipeline, feature_names = None, []
    MODEL_LOADED_SUCCESSFULLY = False
except Exception as e:
    print(f"모델 로드 중 오류: {e}")
    pipeline, feature_names = None, []
    MODEL_LOADED_SUCCESSFULLY = False

initial_feedback_data = pd.DataFrame(columns=["Time", "Prediction", "Correct", "Feedback"])

groups = {
    "1. 온도 관련 그룹": ["molten_temp", "upper_mold_temp1", "lower_mold_temp1", "upper_mold_temp2", "lower_mold_temp2", "sleeve_temperature", "Coolant_temperature"],
    "2. 압력 및 속도 그룹": ["cast_pressure", "physical_strength", "low_section_speed", "high_section_speed"],
    "3. 계량 및 시간 그룹": ["molten_volume", "biscuit_thickness", "facility_operation_cycleTime", "production_cycletime", "hour"],
    "4. 설비 상태 및 식별 그룹": ["count","EMS_operation_time"]
}

def create_group_ui(group_name, features):
    ui_elements = [ui.h4(group_name, style="margin-bottom: 15px; font-weight: bold;")]
    if group_name == "4. 설비 상태 및 식별 그룹":
        ui_elements.extend([
            ui.input_select("mold_code", get_kr_name("mold_code"), choices=mold_code_choices, selected=mold_code_choices[0] if mold_code_choices else None),
            ui.input_select("working", get_kr_name("working"), choices=working_choices, selected='가동'),
            ])
        ui_elements.append(
            ui.input_select(
                "heating_furnace", 
                get_kr_name("heating_furnace"), 
                choices=["A", "B", "C"],
                selected='A', 
                selectize=True
            )
        )
        ui_elements.append(
            ui.input_select(
                "tryshot_signal", 
                get_kr_name("tryshot_signal"), 
                choices=["A", "D"],
                selected='A',
                selectize=True
            )
        )
    for col in features:
        if col in feature_stats:
            stats = feature_stats[col]
            ui_elements.append(
                ui.card(
                    ui.card_header(get_kr_name(col), style="padding: 5px 10px; font-size: 0.9rem; font-weight: bold;"),
                    ui.row(
                        ui.column(8, ui.input_slider(id=f"{col}_slider", label="", min=stats['min'], max=stats['max'], value=stats['value'],step=1), class_="slider-col-card"),
                        ui.column(4, ui.input_numeric(id=col, label="", min=stats['min'], max=stats['max'], value=stats['value']), class_="numeric-col-card"),
                        class_="input-row-card"
                    ), style="margin-bottom: 10px; padding: 0;"
                )
            )
        else:
            ui_elements.append(ui.div(f"**경고: {get_kr_name(col)} 데이터 누락**", class_="text-warning"))
    return ui.div(*ui_elements)

if DATA_PRED_LOADED_SUCCESSFULLY:
    group_uis = [create_group_ui(name, feats) for name, feats in groups.items()]
else:
    group_uis = []


# ----------PDP 모델 로드
pipeline, feature_names = None, []
MODEL_LOADED_SUCCESSFULLY = False
try:
    final_model = joblib.load(MODEL_FILE_PATH)
    if isinstance(final_model, dict):
        pipeline = final_model.get("model")
        feature_names = final_model.get("feature_names", [])
    else:
        pipeline = final_model
    if pipeline is not None and hasattr(pipeline, "predict"):
        MODEL_LOADED_SUCCESSFULLY = True
except Exception as e:
    print("모델 로드 오류:", e)

fallback_feats = [c for c in df_pred.columns if c not in ["passorfail", "mold_code"] and df_pred[c].dtype != "O"]
ui_feature_list = feature_names if len(feature_names) else fallback_feats


# --- 사용자 인터페이스 (UI) 정의 ---
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .tooltip-icon {
                cursor: help;
                font-size: 0.8em;
                vertical-align: middle;
                margin-left: 5px;
                color: #0d6efd;
            }
            .tooltip-icon:hover::after {
                content: attr(data-tooltip);
                position: absolute;
                background-color: #333;
                color: white;
                padding: 10px;
                border-radius: 5px;
                white-space: pre-wrap;
                max-width: 300px;
                font-size: 0.85em;
                z-index: 1000;
                margin-top: 5px;
                line-height: 1.4;
            }
            .table th {
                text-align: left !important;
            }
        """)
    ),
    ui.page_navbar(
        ui.nav_panel(
            "성과 모니터링",
            ui.div(
                ui.div(
                    ui.h5("Line: 전자교반 3라인 2호기", style="margin: 0; color: #555;"),
                    ui.h5("Name: TM Carrier RH", style="margin: 5px 0 0 0; color: #555;"),
                    style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 15px;"
                ),
                ui.h3("불량률 현황"),
                ui.row(
                    ui.column(3, ui.div(
                        ui.input_date("date_selector", "날짜 선택", 
                                      value=max_date_str if max_date_str else "2024-12-31",
                                      min=min_date_str if min_date_str else "2024-01-01",
                                      max=max_date_str if max_date_str else "2024-12-31"),
                        style="margin-bottom: 15px;")),
                    ui.column(9)
                ),
                ui.row(
                    ui.column(3, ui.output_ui("overall_failure_rate_card_4col")),
                    ui.column(3, ui.output_ui("daily_change_card")),
                    ui.column(3, ui.output_ui("daily_failure_rate_card_4col")),
                    ui.column(3, ui.output_ui("target_failure_rate_card")),
                ),
                style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px;"
            ),
            ui.hr(),
            ui.panel_conditional("false" if DATA_PRED_LOADED_SUCCESSFULLY else "true", 
                ui.div(
                    ui.h4(f"오류: '{DATA_PRED_FILE_NAME}' 파일을 찾을 수 없습니다."), 
                    ui.p(f"앱 경로: '{str(DATA_PRED_FILE_PATH)}'"), 
                    class_="alert alert-danger"
                )
            ),
            ui.panel_conditional("false" if MODEL_LOADED_SUCCESSFULLY else "true", 
                ui.div(
                    ui.h4(f"경고: '{MODEL_FILE_NAME}' 모델 파일을 찾을 수 없습니다."), 
                    ui.p("예측 기능을 사용할 수 없습니다."), 
                    class_="alert alert-warning"
                )
            ),
            ui.div(
                ui.h4("품질 추이 및 모델 성능 분석 (시계열)"),
                ui.p("선택한 날짜와 공정 변수에 대한 시간(Time)별 데이터를 시계열 그래프로 확인합니다."),
                ui.input_select("cpk_variable_selector", "분석할 변수를 선택하세요:", 
                              choices={k: get_kr_name(k) for k in cpk_analysis_vars} if cpk_analysis_vars else {"none": "데이터 없음"}),
                ui.output_ui("cpk_values_ui"),
                ui.output_plot("cpk_plot", height="300px"),
                ui.hr(),
                ui.row(
                    ui.column(6, ui.input_date_range(
                            "date_range_hourly", 
                            "분석 기간 선택",
                            start=min_date_str, 
                            end=max_date_str, 
                            min=min_date_str,
                            max=max_date_str
                        )
                    ),
                    ui.column(6, ui.input_select("variable_selector_hourly", "분석할 변수 선택", 
                                               choices={k: get_kr_name(k) for k in HOURLY_CHOICES})),
                ),
                ui.output_ui("hourly_timeseries_plot", style="height:400px;"),
                style="padding: 15px;"
            )
        ),
        
        ui.nav_panel(
            "불량 원인 예측",
            ui.h3("불량 원인 분석 및 변수 영향도"),
            ui.p("선택된 mold_code에 대해 SHAP 분석과 PDP를 시각화합니다."),
            
            ui.input_select("target_mold", "금형 코드 선택", 
                            choices={str(code): f"금형코드 {code}" for code in mold_code_choices_top5},
                            selected=str(mold_code_choices_top5[0])),
            
            ui.input_select("target_feature", "PDP 변수 선택", 
                            choices={k: get_kr_name(k) for k in numerical_features}, 
                            selected="sleeve_temperature"),
            
            ui.hr(),
            ui.row(
                ui.column(6, ui.output_plot("mold_defect_plot", height="420px")),
                ui.column(6, ui.output_plot("shap_summary_plot", height="420px")),
            ),
            ui.hr(),
            ui.output_plot("pdp_plot", height="420px"),
            ),
        
        ui.nav_panel(
            "예측&개선",
            ui.h3("주조 공정 데이터 기반 불량 예측 모델"),
            ui.hr(),
            ui.panel_conditional("false" if DATA_PRED_LOADED_SUCCESSFULLY else "true", 
                ui.div(
                    ui.h4(f"오류: '{DATA_PRED_FILE_NAME}' 파일을 찾을 수 없습니다."), 
                    ui.p(f"앱이 '{DATA_PRED_FILE_PATH}' 경로에서 파일을 찾으려 했으나 실패했습니다."), 
                    class_="alert alert-danger"
                    )
            ),
            ui.panel_conditional("false" if MODEL_LOADED_SUCCESSFULLY else "true", 
                ui.div(
                    ui.h4(f"경고: '{MODEL_FILE_NAME}' 모델 파일을 찾을 수 없습니다."), 
                    ui.p("예측 기능을 사용할 수 없습니다."), 
                    class_="alert alert-warning"
                )
            ),
            ui.layout_column_wrap(1 / 4, fill=False, *group_uis),
            ui.hr(style="margin-top: 20px; margin-bottom: 20px;"),
            ui.row(
                ui.column(4, 
                    ui.h4("예측 결과"), 
                    ui.output_ui("prediction_output_ui"), 
                    ui.input_action_button("predict_button", "예측하기", class_="btn-primary btn-lg mt-2 w-100")
                ),
                ui.column(8, 
                    ui.h4("실제 불량 여부 확인 및 피드백"), 
                    ui.row(
                        ui.column(4, 
                            ui.div("실제 상태:", style="font-weight: bold; margin-bottom: 5px;"), 
                            ui.input_action_button("correct_btn", "✅ 불량 맞음 (Correct)", class_="btn-success w-100"), 
                            ui.input_action_button("incorrect_btn", "❌ 불량 아님 (Incorrect)", class_="btn-danger mt-2 w-100")
                        ), 
                        ui.column(8, 
                            ui.div("원인 메모:", style="font-weight: bold; margin-bottom: 5px;"), 
                            ui.input_text("feedback", "", placeholder="예: 냉각수온도(Coolant_temperature) 급변", width="100%"), 
                            ui.input_action_button("submit_btn", "💾 피드백 저장", class_="btn-primary w-100 mt-2")
                        )
                    ), 
                    ui.div(ui.output_text("selected_status"), style="margin-top: 10px; font-weight: bold;")
                ),
            ),
            ui.hr(style="margin-top: 20px; margin-bottom: 20px;"),
            ui.h3("누적 피드백 데이터"),
            ui.output_ui("feedback_table"),
            ui.hr(style="margin-top: 20px; margin-bottom: 20px;"),
            ui.h4("SHAP Bar Plot - 개별 예측 설명 (상위 5개 변수)"),
            ui.p("입력된 변수값이 예측 결과에 어떻게 영향을 미치는지 시각화합니다."),
            ui.output_plot("shap_bar_plot", height="400px"),
            ui.output_ui("shap_interpretation"),
            ui.hr(style="margin-top: 20px; margin-bottom: 20px;"),
            ui.h4("PDP (Partial Dependence Plot) - 변수별 영향도 분석"),
            ui.p("특정 변수의 값 변화가 불량 예측에 미치는 영향을 분석합니다."),
            ui.input_select("pdp_variable_selector", "PDP 분석 변수 선택", 
                            choices={k: get_kr_name(k) for k in numerical_features}, 
                            selected="sleeve_temperature"),
            ui.output_plot("prediction_pdp_plot", height="400px"),
            ui.output_ui("pdp_recommendation"),
        ),
        
        ui.nav_panel(
            "데이터 분석 (EDA)", 
            ui.h3("탐색적 데이터 분석(EDA)"), 
            ui.hr(),
            ui.panel_conditional("false" if DATA_PRED_LOADED_SUCCESSFULLY and DATA_PRED_LOADED_SUCCESSFULLY else "true",
                ui.div(
                    ui.h4("오류: 데이터 파일을 찾을 수 없어 EDA 분석이 불가능합니다."),
                    ui.p(f"필요한 파일: '{DATA_PRED_FILE_NAME}', '{DATA_PRED_FILE_NAME}'"),
                    class_="alert alert-danger"
                )
            ),
            ui.panel_conditional("true" if DATA_PRED_LOADED_SUCCESSFULLY and DATA_PRED_LOADED_SUCCESSFULLY else "false",
                ui.input_select(
                    "eda_variable_selector",
                    "분석할 변수를 선택하세요:",
                    choices={k: get_kr_name(k) for k in numerical_features},
                    selected="molten_temp"
                ),
                ui.div(
                    ui.HTML(
                        "<b>참고:</b><br>"
                        "id, name, line, mold_name, emergency_stop, working, upper_mold_temp3, lower_mold_temp3 컬럼은 대부분의 값이 하나로 고정되어 있어 분석에서 제외했습니다.<br>"
                        "결측치가 1개인 열의 대부분의 결측치가 19327 행에 모여있어, 분석에 방해가 될 것으로 판단해 제거했습니다.<br>"
                        "tryshot은 D를 뺀 나머지 결측치를 전부 A로 치환했습니다.<br>"
                        "EMS와 mold_code는 범주형으로 변환했습니다.<br>"
                        "heating_furnace가 결측이면서 molten_volume이 채워진 행은 C로 치환했습니다."
                    ),
                    style="font-size: 1em; color: #888; margin-top: 5px; line-height: 1.6;"
                ),
                ui.hr(),
                ui.row(
                    ui.column(8,
                        ui.h4("전처리 전 데이터 분석", style="text-align: center;"),
                        ui.output_plot("eda_combined_plot_before", height="450px"),
                        ui.hr(style="margin-top: 2rem; margin-bottom: 2rem;"),
                        ui.h4("전처리 후 데이터 분석", style="text-align: center;"),
                        ui.output_plot("eda_combined_plot_after", height="450px")
                    ),
                    ui.column(4,
                        ui.output_ui("eda_stats_ui")
                    )
                )
            ),
            ui.hr(style="margin-top: 40px; border-top: 2px solid #ccc;"),
            ui.h3("모델링 요약", style="text-align: center; margin-bottom: 20px;"),
            ui.div(
                ui.card(
                    ui.card_header(ui.h5("최종 전처리", style="margin: 0;")),
                    ui.card_body(
                        ui.p("주요 변수(low_section_speed, molten_temp 등)의 이상치 및 오류 값을 제거하고, KNN Imputer를 사용하여 결측치를 보간했습니다. 이후 수치형 변수는 RobustScaler를, 범주형 변수는 Ordinal Encoding을 적용하여 모델 학습에 적합한 형태로 변환했습니다.")
                    )
                ),
                ui.card(
                    ui.card_header(ui.h5("최종 선정 모델", style="margin: 0;")),
                    ui.card_body(
                        ui.p("LightGBM (선정 이유: 다양한 모델 중 불량 케이스를 놓치지 않는 성능(Recall)이 가장 우수하여 선정되었습니다.)")
                    )
                ),
                style="display: flex; flex-direction: column; gap: 20px; margin-top: 20px;"
            ),
            ui.hr(style="margin-top: 40px; border-top: 2px solid #ccc;"),
            ui.output_ui("model_performance_table"),
        ),
        id="navbar", 
        title="다이캐스팅 공정 품질관리", 
        bg="#f8f9fa", 
        inverse=False
    )
)


# --- 서버 로직 정의 ---
def server(input, output, session):
    r_prediction_text = reactive.Value("예측 버튼을 눌러주세요.")
    r_correct_status = reactive.Value(None)
    r_feedback_data = reactive.Value(initial_feedback_data)
    r_prediction_result = reactive.Value(None)
    r_shap_data = reactive.Value(None)
    r_current_input = reactive.Value(None)
    r_pdp_rec_range = reactive.Value(None)
    r_top_shap_feature = reactive.Value(None)

    @output
    @render.ui
    def overall_failure_rate_card_4col():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return ui.div("데이터 없음", class_="alert alert-secondary text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
        card_color = "bg-danger" if overall_failure_rate > 5 else "bg-primary"
        return ui.div(
            ui.h5("전체 기간 불량률"),
            ui.h2(f"{overall_failure_rate:.2f}%"),
            ui.p(f"(총 {int(total_count):,}개 중 {int(total_failures):,}개 불량)"),
            class_=f"card text-white {card_color} text-center p-3",
            style="border-radius: 5px; height: 100%; display: flex; flex-direction: column; justify-content: center;"
        )

    @output
    @render.ui
    def daily_failure_rate_card_4col():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return ui.div("데이터 파일을 불러올 수 없습니다.", class_="alert alert-warning")
        selected_date_str = input.date_selector()
        if not selected_date_str:
            return ui.div("날짜를 선택해주세요.", class_="alert alert-info text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
        try:
            selected_date = pd.to_datetime(selected_date_str).date()
            rate, total, failures = daily_stats.loc[selected_date, ['failure_rate', 'total', 'failures']]
            card_color = "bg-danger" if rate > 5 else "bg-success"
            return ui.div(
                ui.h5("선택일 불량률"),
                ui.h2(f"{rate:.2f}%"),
                ui.p(f"(총 {int(total):,}개 중 {int(failures):,}개 불량)"),
                class_=f"card text-white {card_color} text-center p-3",
                style="border-radius: 5px; height: 100%; display: flex; flex-direction: column; justify-content: center;"
            )
        except Exception:
            return ui.div(f"{selected_date_str} 데이터 없음", class_="alert alert-warning text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")

    @output
    @render.ui
    def daily_change_card():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return ui.div("데이터 없음", class_="alert alert-secondary text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
        selected_date_str = input.date_selector()
        try:
            current_date = pd.to_datetime(selected_date_str)
            date_index = daily_stats.index.get_loc(current_date.date())
            if date_index == 0:
                return ui.div("전일 데이터 없음", class_="alert alert-secondary text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
            prev_date = daily_stats.index[date_index - 1]
            current_rate = daily_stats.loc[current_date.date(), 'failure_rate']
            prev_rate = daily_stats.loc[prev_date, 'failure_rate']
            change = current_rate - prev_rate
            if change < 0:
                change_text, card_class, icon = f"{change:+.2f}%p", "bg-info", "bi-arrow-down-right"
            elif change > 0:
                change_text, card_class, icon = f"{change:+.2f}%p", "bg-danger", "bi-arrow-up-right"
            else:
                change_text, card_class, icon = "변화 없음", "bg-secondary", "bi-dash"
            return ui.div(
                ui.h5("전일 대비 증감"),
                ui.h2(ui.span(class_=f"bi {icon}"), f" {change_text}"),
                ui.p(f"({prev_date.strftime('%Y-%m-%d')} {prev_rate:.2f}% → {current_rate:.2f}%)"),
                class_=f"card text-white {card_class} text-center p-3",
                style="border-radius: 5px; height: 100%; display: flex; flex-direction: column; justify-content: center;"
            )
        except Exception:
            return ui.div("전일 데이터 없음", class_="alert alert-secondary text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")

    @output
    @render.ui
    def target_failure_rate_card():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return ui.div("데이터 없음", class_="alert alert-secondary text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
    
        selected_date_str = input.date_selector()
        if not selected_date_str:
            return ui.div("날짜를 선택해주세요.", class_="alert alert-info text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
    
        try:
            selected_date = pd.to_datetime(selected_date_str).date()
            df = df_raw.copy()
            if "datetime_full" not in df.columns:
                return ui.div("시간 데이터 없음", class_="alert alert-warning text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
    
            df_day = df[df["datetime_full"].dt.date == selected_date]
            if df_day.empty:
                return ui.div(f"{selected_date} 데이터 없음", class_="alert alert-info text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
    
            df_day["hour"] = df_day["datetime_full"].dt.hour
            failure_by_hour = df_day.groupby("hour")["passorfail"].mean() * 100
    
            if failure_by_hour.empty:
                return ui.div(f"{selected_date} 불량 데이터 없음", class_="alert alert-info text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")
    
            top_hours = failure_by_hour.sort_values(ascending=False).head(2)
    
            content = []
            for hour, rate in top_hours.items():
                content.append(ui.h3(f"{hour}시 → {rate:.2f}%"))
    
            return ui.div(
                ui.h5("불량 집중 시간대", style="margin-bottom: 10px;"),
                ui.div(*content, style="text-align: center; margin: 10px 0;"),
                ui.p(f"(선택일: {selected_date})", style="margin-top: 5px;"),
                class_="card bg-warning text-dark text-center p-3", 
                style="border-radius: 5px; height: 100%; display: flex; flex-direction: column; justify-content: center;"
            )
        except Exception as e:
            return ui.div(f"에러: {e}", class_="alert alert-danger text-center p-3", style="height: 100%; display: flex; flex-direction: column; justify-content: center;")

    
    @reactive.Calc
    def daily_filtered_data_and_cpk():
        selected_date_str = input.date_selector()
        selected_var = input.cpk_variable_selector()
    
        if not selected_date_str or not selected_var or selected_var == "none":
            return {'data': pd.Series(dtype='float64'), 'stats': {}}
    
        try:
            selected_date = pd.to_datetime(selected_date_str).date()
            
            if 'date_only' not in df_ts.columns:
                 print("오류: df_ts에 'date_only' 컬럼이 없습니다.")
                 return {'data': pd.Series(dtype='float64'), 'stats': {}}
    
            filtered_df = df_ts[df_ts['date_only'] == selected_date]
    
            if filtered_df.empty:
                return {'data': pd.Series(dtype='float64'), 'stats': {}}
    
            series = filtered_df[selected_var].dropna()
            limits = spec_limits.get(selected_var, {})
    
            if not limits or series.empty:
                 return {'data': series, 'stats': {}}
    
            results = calculate_cpk(series, limits['lsl'], limits['usl'])
            results.update({'lsl': limits['lsl'], 'usl': limits['usl'], 'estimated': limits['estimated']})
            
            return {'data': series, 'stats': results}
    
        except Exception as e:
            print(f"Cpk 계산 중 오류 발생: {e}")
            return {'data': pd.Series(dtype='float64'), 'stats': {}}


    @output
    @render.ui
    def cpk_values_ui():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return ui.div("데이터가 없어 분석할 수 없습니다.", class_="alert alert-warning")
        
        analysis_results = daily_filtered_data_and_cpk()
        results = analysis_results['stats']

        if not results:
            return ui.p("선택한 날짜에 대한 분석 데이터가 없습니다.")

        cp, cpk, ucl, lcl = results.get('cp'), results.get('cpk'), results.get('ucl'), results.get('lcl')

        if isinstance(cpk, (int, float)):
            if cpk >= 1.33:
                interpretation, cls = "매우 양호", "text-success"
            elif cpk >= 1.0:
                interpretation, cls = "양호", "text-primary"
            elif cpk >= 0.67:
                interpretation, cls = "주의 필요", "text-warning"
            else:
                interpretation, cls = "개선 시급", "text-danger"
        else:
            interpretation, cls = "판단 불가", "text-muted"

        ucl_tooltip = "UCL (Upper Control Limit)\n상한 관리 한계선\n\n공정이 정상적으로 작동할 때 예상되는 최대값\n이 선을 초과하면 공정에 이상이 있다고 판단\n일반적으로 평균 + 3σ(표준편차)로 설정"
        lcl_tooltip = "LCL (Lower Control Limit)\n하한 관리 한계선\n\n공정이 정상적으로 작동할 때 예상되는 최소값\n이 선 아래로 떨어지면 공정에 이상이 있다고 판단\n일반적으로 평균 - 3σ(표준편차)로 설정"
        cp_tooltip = "Cp (Process Capability Index)\n공정 능력 지수\n\n규격 범위 대비 공정 산포의 비율\n계산식: (USL - LSL) / 6σ\n공정의 잠재적 능력을 나타냄 (평균 위치 무관)\n해석:\nCp < 1.0: 부적합 (공정 개선 필요)\nCp ≥ 1.33: 양호\nCp ≥ 1.67: 우수"
        cpk_tooltip = "Cpk (Process Capability Index with centering)\n공정 능력 지수 (중심 보정)\n\n규격 중심으로부터 공정 평균의 치우침을 고려한 지수\n계산식: min[(USL-μ)/3σ, (μ-LSL)/3σ]\n공정의 실제 능력을 나타냄\n해석:\nCpk < 1.0: 부적합 (공정 조정 필요)\nCpk ≥ 1.33: 양호\nCpk ≥ 1.67: 우수\nCp와 Cpk가 비슷하면 공정이 중앙에 잘 위치"

        def create_info_box(title, value, tooltip_text, alert_class):
            return ui.column(3,
                ui.div(
                    ui.h5(
                        title,
                        ui.tags.span("ⓘ", class_="tooltip-icon", **{"data-tooltip": tooltip_text})
                    ),
                    ui.p(f"{value}"),
                    class_=f"text-center alert {alert_class} p-2", style="height: 100%; position: relative; display: flex; flex-direction: column; justify-content: center;"
                )
            )

        return ui.div(
            ui.row(
                create_info_box("UCL", ucl, ucl_tooltip, "alert-secondary"),
                create_info_box("LCL", lcl, lcl_tooltip, "alert-secondary"),
                create_info_box("Cp", cp, cp_tooltip, "alert-info"),
                create_info_box("Cpk", cpk, cpk_tooltip, "alert-info"),
            ),
            ui.div(ui.h5("공정 능력 평가", class_="mt-3"), ui.p(interpretation, class_=f"fw-bold {cls}"))
        )

    @output
    @render.plot(alt="선택일 Cp/Cpk 공정 능력 분석 그래프")
    def cpk_plot():
        if not DATA_PRED_LOADED_SUCCESSFULLY:
            return
        
        analysis_results = daily_filtered_data_and_cpk()
        data = analysis_results['data']
        stats = analysis_results['stats']
        selected_var = input.cpk_variable_selector()
        selected_date = input.date_selector()

        if data.empty or not stats:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"{selected_date}에 대한 데이터가 없습니다.", ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f'{get_kr_name(selected_var)} 공정 능력 분석', fontsize=12)
            ax.axis('off')
            return fig

        mean, ucl, lcl = stats.get('mean'), stats.get('ucl'), stats.get('lcl')
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(data, bins=30, density=True, color='skyblue', alpha=0.7, label='데이터 분포')
        if mean is not None:
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'평균: {mean}')
        if lcl is not None:
            ax.axvline(lcl, color='gold', linestyle='-', linewidth=2, label=f'LCL: {lcl}')
        if ucl is not None:
            ax.axvline(ucl, color='gold', linestyle='-', linewidth=2, label=f'UCL: {ucl}')
            
        ax.set_title(f'[{selected_date}] {get_kr_name(selected_var)} 공정 능력 분석', fontsize=12)
        ax.set_xlabel('값', fontsize=10)
        ax.set_ylabel('밀도', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()
        return fig
    
    if DATA_PRED_LOADED_SUCCESSFULLY:
        for col in numerical_features:
            def make_sync_observer(feature_name):
                slider_id, numeric_id = f"{feature_name}_slider", feature_name
                @reactive.Effect
                @reactive.event(input[slider_id])
                def _(): 
                    ui.update_numeric(numeric_id, value=input[slider_id]())
                @reactive.Effect
                @reactive.event(input[numeric_id])
                def _():
                    if input[slider_id]() != input[numeric_id](): 
                        ui.update_slider(slider_id, value=input[numeric_id]())
            make_sync_observer(col)

    @output
    @render.ui
    def hourly_timeseries_plot():
        if not DATA_PRED_LOADED_SUCCESSFULLY: 
            return ui.HTML("<div>⚠️ 데이터가 로드되지 않았습니다.</div>")
        
        date_range = input.date_range_hourly() 
        KEY_FEATURE = input.variable_selector_hourly()
        
        TIME_COL_PROCESSED = 'datetime_full' 

        if not date_range or not KEY_FEATURE: 
            return ui.HTML("<div>날짜와 변수를 선택해주세요.</div>")
        
        if TIME_COL_PROCESSED not in df_ts.columns:
            return ui.HTML("<div>⚠️ df_ts에 전처리된 시간 컬럼('datetime_full')이 없습니다. 전역 로드 코드를 확인해주세요.</div>")

        try:
            plot_data = pd.DataFrame() 
            start_date_str = date_range[0]
            end_date_str = date_range[1]
            
            start_dt = pd.to_datetime(start_date_str).normalize()
            end_dt = pd.to_datetime(end_date_str).normalize() + pd.Timedelta(days=1) 
            
            if start_dt >= end_dt:
                return ui.HTML("<div>⚠️ 시작 날짜는 끝 날짜보다 이전이어야 합니다.</div>")

            plot_data = df_ts[
                (df_ts[TIME_COL_PROCESSED] >= start_dt) & 
                (df_ts[TIME_COL_PROCESSED] < end_dt)
            ].copy()

            KEY_FEATURE_KR = get_kr_name(KEY_FEATURE)

            if plot_data.empty or plot_data[KEY_FEATURE].isnull().all():
                title_text = f"기간: {start_date_str} ~ {end_date_str} ({KEY_FEATURE_KR})"
                fig = go.Figure()
                fig.update_layout(
                    title=title_text,
                    xaxis_title="날짜",
                    yaxis_title=f"{KEY_FEATURE_KR} 값",
                    annotations=[{
                        "text": f"⚠️ 선택된 기간에 데이터가 없습니다.",
                        "xref": "paper", "yref": "paper",
                        "x": 0.5, "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 16, "color": "gray"}
                    }]
                )
            
            else:
                WINDOW_SIZE = 3
                SMOOTHED_FEATURE = f'{KEY_FEATURE}_smoothed'
                
                plot_data = plot_data.set_index(TIME_COL_PROCESSED) 
                
                plot_data[SMOOTHED_FEATURE] = (
                    plot_data[KEY_FEATURE]
                    .rolling(window=WINDOW_SIZE, center=True, min_periods=1)
                    .mean()
                )
                plot_data = plot_data.reset_index()
                
                min_time = plot_data[TIME_COL_PROCESSED].min()
                max_time = plot_data[TIME_COL_PROCESSED].max()
                y_min = plot_data[KEY_FEATURE].min()
                y_max = plot_data[KEY_FEATURE].max()
                y_padding = (y_max - y_min) * 0.1 
                
                fail_data = plot_data[
                    (plot_data['passorfail'] == 1) & 
                    (plot_data[KEY_FEATURE].notna())].copy()

                title_text = f"기간: {start_date_str} ~ {end_date_str} ({KEY_FEATURE_KR}) 시계열 추이"
                
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=plot_data[TIME_COL_PROCESSED], 
                    y=plot_data[SMOOTHED_FEATURE],
                    name=KEY_FEATURE_KR,
                    mode="lines",
                    line={"width": 2, "smoothing": 0.8, "color": "blue"}
                ))
                
                if not fail_data.empty:
                    fig.add_trace(go.Scatter(
                        x=fail_data[TIME_COL_PROCESSED], 
                        y=fail_data[KEY_FEATURE],
                        name="불량 발생 (Fail)",
                        mode="markers",
                        marker=dict(
                            symbol='circle',
                            size=5,
                            color='red',
                        ),
                        hoverinfo='name+x+y'
                    ))

                fig.update_layout(
                    title=title_text,
                    height=600,
                    hovermode="x unified",
                    template="plotly_white",
                    xaxis=dict(
                        type="date",
                        title="날짜/시간",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1개월", step="month", stepmode="backward"),
                                dict(count=3, label="3개월", step="month", stepmode="backward"),
                                dict(count=6, label="6개월", step="month", stepmode="backward"),
                                dict(step="all", label="전체 기간")
                            ])
                        ),
                        rangeslider=dict(
                            visible=True,
                            range=[min_time, max_time],
                            yaxis=dict(
                                range=[y_min - y_padding, y_max + y_padding]
                            )
                        )
                    ),
                    yaxis=dict(
                        title=f"{KEY_FEATURE_KR} 값",
                        autorange=True,
                        rangemode='tozero',
                    )
                )

                html_out = fig.to_html(full_html=False, include_plotlyjs='cdn')
                return ui.HTML(html_out)

        except Exception as e:
            import traceback
            error_message = f"그래프 생성 중 오류 발생: {e}"
            print(f"❌ Plotly 그래프 생성 오류: {e}")
            traceback.print_exc()
            return ui.HTML(f"<div>{error_message}</div>")


    @output
    @render.plot
    @reactive.event(input.target_mold)
    def mold_defect_plot():
        if df_pred.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            return fig

        mold_defect = (
            df_pred.groupby("mold_code")["passorfail"]
            .mean()
            .reset_index()
            .rename(columns={"passorfail": "불량률"})
            .sort_values("mold_code", ascending=True)
        )

        selected_mold = str(input.target_mold())

        base_color = 'tab:blue'
        colors = [
            'red' if str(code) == selected_mold else base_color
            for code in mold_defect["mold_code"]
        ]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(mold_defect["mold_code"].astype(str), mold_defect["불량률"].values,
            color=colors, edgecolor='black', alpha=0.9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel("불량률")
        ax.set_title("금형코드별 불량률")
        fig.tight_layout()
        return fig



    @output
    @render.plot
    @reactive.event(input.target_mold, input.target_feature)
    def shap_summary_plot():
        if not MODEL_LOADED_SUCCESSFULLY or df_pred.empty:
            return None

        target_mold = str(input.target_mold())
        target_feature_original = input.target_feature()

        df_seg = df_pred[df_pred["mold_code"].astype(str) == target_mold].copy()
        if df_seg.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"mold_code {target_mold} 데이터 없음", ha="center", va="center")
            return fig

        X_seg = df_seg[feature_names]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_transformed = preprocessor.transform(X_seg)
        feature_names_trans = list(preprocessor.get_feature_names_out())

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap_mean_abs = np.abs(shap_values_plot).mean(axis=0)
        top_idx = np.argsort(shap_mean_abs)[-10:][::-1]

        feature_names_top_raw = [feature_names_trans[i] for i in top_idx]
        feature_basenames_top = [name.split('__')[-1] for name in feature_names_top_raw]
        feature_names_kr = [get_kr_name(b) for b in feature_basenames_top]

        colors = ['tab:blue'] * len(feature_basenames_top)
        if target_feature_original in feature_basenames_top:
            colors[feature_basenames_top.index(target_feature_original)] = 'red'

        fig = plt.figure(figsize=(8, 3.5))
        shap_mean_abs_top = shap_mean_abs[top_idx]
        y_pos = np.arange(len(feature_names_kr))
        plt.barh(y_pos, shap_mean_abs_top, color=colors)
        plt.yticks(y_pos, feature_names_kr)
        plt.gca().invert_yaxis()
        plt.title(f"금형코드 {target_mold} - 변수 영향도 TOP10")
        plt.xlabel("변수 영향도 평균(|SHAP|)", fontsize=12)
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def pdp_plot():
        if not MODEL_LOADED_SUCCESSFULLY or df_pred.empty:
            return None

        target_mold = int(input.target_mold())
        target_feature = input.target_feature()

        df_seg = df_pred[df_pred["mold_code"] == target_mold].copy()
        if df_seg.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"mold_code {target_mold} 데이터 없음", ha="center", va="center")
            return fig

        X_seg = df_seg[feature_names]
        X_seg_sample = X_seg.sample(n=2000, random_state=42) if len(X_seg) > 2000 else X_seg

        display = PartialDependenceDisplay.from_estimator(
            estimator=pipeline,
            X=X_seg_sample,
            features=[target_feature],
            kind='average',
            grid_resolution=50,
            response_method="predict_proba"
        )
        ax = display.axes_[0, 0]

        pd_res = partial_dependence(
            estimator=pipeline,
            X=X_seg_sample,
            features=[target_feature],
            kind='average',
            grid_resolution=50,
            response_method="predict_proba"
        )
        xx = np.asarray(pd_res["grid_values"][0])
        yy = np.asarray(pd_res["average"][0])

        xvals = X_seg_sample[target_feature].values
        edges = np.r_[xx[0], (xx[1:] + xx[:-1]) / 2, xx[-1]]
        bin_ids = np.digitize(xvals, edges) - 1
        bin_ids = np.clip(bin_ids, 0, len(xx)-1)
        counts = np.bincount(bin_ids, minlength=len(xx))

        MIN_BIN = max(10, int(0.01 * len(X_seg_sample)))
        dense_mask = counts >= MIN_BIN

        EPS = 0.001
        ymin = float(yy.min())
        mask_good = (yy <= (ymin + EPS)) & dense_mask

        if mask_good.any():
            idx = np.where(mask_good)[0]
            splits = np.where(np.diff(idx) != 1)[0] + 1
            runs = np.split(idx, splits)
            best = max(runs, key=len)
            good_lo, good_hi = xx[best[0]], xx[best[-1]]

            ax.axvspan(good_lo, good_hi, color="green", alpha=0.20,
                       label=f"권장 구간 {good_lo:.2f} ≤ x ≤ {good_hi:.2f}")

        sample_info = f" (샘플: {len(X_seg_sample)}/{len(X_seg)}개)" if len(X_seg) > 2000 else ""
        ax.set_title(f"금형코드 {target_mold} - PDP: {get_kr_name(target_feature)}")
        ax.set_xlabel(get_kr_name(target_feature), fontsize=11)
        ax.set_ylabel("불량 확률 (predict_proba)",fontsize=11)
        ax.legend()
        plt.tight_layout()
        return display.figure_


    @reactive.Effect
    @reactive.event(input.predict_button)
    def run_prediction():
        r_correct_status.set(None)
        r_prediction_result.set(None)
        r_prediction_text.set("⏳ 모델 예측 중...")
        r_shap_data.set(None)
        r_pdp_rec_range.set(None)
        
        if not DATA_PRED_LOADED_SUCCESSFULLY or not MODEL_LOADED_SUCCESSFULLY:
            r_prediction_text.set(f"예측 불가: 파일/모델 오류")
            r_prediction_result.set("WARNING")
            return
            
        try:
            with reactive.isolate():
                working_numeric = {'가동': 1, '정지': 0}.get(input.working())
                mold_code_value = input.mold_code() 
                try:
                    mold_code_numeric = int(mold_code_value)
                except ValueError:
                    mold_code_numeric = mold_code_value

                all_slider_features = numerical_features + REQUIRED_HOUR_FEATURE 
                input_data_dict = {}
                
                for col in all_slider_features:
                    if col in input:
                        input_data_dict[col] = input[col]()
                
                input_data_dict['mold_code'] = mold_code_numeric
                input_data_dict['working'] = working_numeric                
                heating_furnace_map = {'A': 0, 'B': 1, 'C': 2}
                tryshot_signal_map = {'A': 0, 'D': 1}
                input_data_dict['heating_furnace'] = heating_furnace_map.get(input.heating_furnace())
                input_data_dict['tryshot_signal'] = tryshot_signal_map.get(input.tryshot_signal())


                input_df = pd.DataFrame([input_data_dict])
                prediction = pipeline.predict(input_df)
                prediction_proba = pipeline.predict_proba(input_df)
                
                preprocessor = pipeline.named_steps["preprocessor"]
                classifier = pipeline.named_steps["classifier"]
                X_custom = preprocessor.transform(input_df)
                feature_names_out = list(preprocessor.get_feature_names_out())
                
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_custom)
                
                if isinstance(shap_values, list):
                    shap_values_class1 = shap_values[1][0, :]
                else:
                    shap_values_class1 = shap_values[0, :]
                
                abs_shap = np.abs(shap_values_class1)
                top5_idx = np.argsort(abs_shap)[-5:][::-1]
                
                shap_data = {
                'shap_values': shap_values_class1[top5_idx],
                'feature_values': X_custom[0, top5_idx],
                'feature_names': [feature_names_out[i] for i in top5_idx]
                }
                r_shap_data.set(shap_data)
            
                top_feature_transformed = feature_names_out[top5_idx[0]]
                top_feature_original = top_feature_transformed.split('__')[-1] if '__' in top_feature_transformed else top_feature_transformed

                if top_feature_original in numerical_features:
                    r_top_shap_feature.set(top_feature_original)
        
            r_current_input.set(input_df)
        
            if prediction[0] == 0:
                prob = prediction_proba[0][0] * 100 
                result_text = f"✅ 정상 (양품)일 확률: {prob:.2f}%"
                result_class = "success"
            else:
                prob = prediction_proba[0][1] * 100
                result_text = f"🚨 불량일 확률: {prob:.2f}%"
                result_class = "danger"
                
            r_prediction_text.set(result_text)
            r_prediction_result.set(result_class)
            
        except Exception as e:
            r_prediction_text.set(f"예측 중 오류: {e}")
            r_prediction_result.set("WARNING")
            print(f"예측 오류 상세: {e}")

    @output
    @render.ui
    def prediction_output_ui():
        result_class, text = r_prediction_result(), r_prediction_text()
        if result_class is None: 
            return ui.div(ui.h5(text), class_="alert alert-info" if "예측 버튼" in text else "alert alert-secondary")
        final_class = {"success": "alert alert-success", "danger": "alert alert-danger", "WARNING": "alert alert-warning"}.get(result_class, "alert alert-warning")
        return ui.div(ui.h5(text), class_=final_class)

    @reactive.Effect
    @reactive.event(input.correct_btn)
    def set_correct(): 
        r_correct_status.set("✅ 불량 맞음")

    @reactive.Effect
    @reactive.event(input.incorrect_btn)
    def set_incorrect(): 
        r_correct_status.set("❌ 불량 아님")

    @output
    @render.text
    def selected_status():
        return f">> 현재 선택된 상태: {r_correct_status()}" if r_correct_status() else ">> 불량 여부를 선택해주세요."

    @reactive.Effect
    def update_pdp_selector():
        top_feature = r_top_shap_feature()
        if top_feature and top_feature in numerical_features:
            ui.update_select("pdp_variable_selector", selected=top_feature)

    @reactive.Effect
    @reactive.event(input.submit_btn)
    def save_feedback():
        prediction_text, correct_status = r_prediction_text(), r_correct_status()
        if correct_status is None or any(s in prediction_text for s in ["예측 버튼", "예측 불가", "모델 예측 중"]):
            ui.notification_show("🚨 예측 수행 후 실제 불량 여부를 선택해야 피드백 저장이 가능합니다.", duration=5, type="warning")
            return
        prediction_only = "불량" if "불량일 확률" in prediction_text else "정상"
        new_feedback = pd.DataFrame({"Time": [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')], "Prediction": [prediction_only], "Correct": [correct_status], "Feedback": [input.feedback()]})
        r_feedback_data.set(pd.concat([r_feedback_data(), new_feedback], ignore_index=True))
        r_correct_status.set(None)
        ui.update_text("feedback", value="")
        ui.notification_show("✅ 피드백이 성공적으로 저장되었습니다.", duration=3, type="success")

    @output
    @render.ui
    def feedback_table():
        df_feedback = r_feedback_data()
        if df_feedback.empty: 
            return ui.p("아직 저장된 피드백이 없습니다.")
        header = ui.tags.tr(*[ui.tags.th(col) for col in df_feedback.columns])
        rows = []
        for _, row in df_feedback.iterrows():
            correct_text = str(row['Correct'])
            correct_style = ""
            if "맞음" in correct_text: 
                correct_style = "background-color: #d4edda; color: #155724;"
            elif "아님" in correct_text: 
                correct_style = "background-color: #f8d7da; color: #721c24;"
            tds = [ui.tags.td(str(row['Time'])), ui.tags.td(str(row['Prediction'])), ui.tags.td(correct_text, style=correct_style), ui.tags.td(str(row['Feedback']))]
            rows.append(ui.tags.tr(*tds))
        return ui.tags.div(
            ui.tags.style("""
                table.custom-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
                .custom-table th, .custom-table td { border: 1px solid #ccc; padding: 8px; text-align: center; word-wrap: break-word; }
                .custom-table th { background-color: #f5f5f5; }
                .custom-table td:nth-child(1) { width: 15%; } 
                .custom-table td:nth-child(2) { width: 10%; }
                .custom-table td:nth-child(3) { width: 10%; font-weight: bold; } 
                .custom-table td:nth-child(4) { width: 65%; text-align: left; }
            """), 
            ui.tags.table({"class": "custom-table"}, ui.tags.thead(header), ui.tags.tbody(*rows))
        )


    @output
    @render.plot
    def shap_bar_plot():
        shap_data = r_shap_data()

        if shap_data is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "예측 버튼을 눌러 SHAP Bar Plot을 생성하세요", ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
            return fig

        try:
            shap_values = shap_data['shap_values']
            feature_values = shap_data['feature_values']
            feature_names_list = shap_data['feature_names']

            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(9, 4))
            colors = ['red' if val > 0 else 'blue' for val in shap_values]
            bars = ax.barh(range(len(shap_values)), shap_values, color=colors, alpha=0.7)

            ax.set_yticks(range(len(feature_names_list)))
            ax.set_yticklabels(feature_names_list, fontsize=10)
            ax.set_xlabel('SHAP 값 (불량 예측에 대한 기여도)', fontsize=11)
            ax.set_title('SHAP Bar Plot - 상위 5개 영향 변수', fontsize=13, pad=10)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)

            for i, (bar, val) in enumerate(zip(bars, shap_values)):
                ax.text(val/2, i, f'{val:.3f}', va='center', ha='center', fontsize=9, color='white', fontweight='bold')

            plt.tight_layout(pad=0.5)
            return fig

        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, f"Bar Plot 생성 오류: {str(e)}", ha='center', va='center', fontsize=10, color='red')
            ax.axis('off')
            print(f"Bar Plot 오류 상세: {e}")
            return fig

    @output
    @render.ui
    def shap_interpretation():
        shap_data = r_shap_data()
        prediction_result = r_prediction_result()
        
        if shap_data is None or prediction_result is None:
            return ui.div()
        
        try:
            shap_values = shap_data['shap_values']
            feature_names_list = shap_data['feature_names']
            
            abs_shap = np.abs(shap_values)
            top_idx = np.argmax(abs_shap)
            top_feature = feature_names_list[top_idx]
            top_shap_value = shap_values[top_idx]
            
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            baseline_prob = sigmoid(0) * 100
            adjusted_prob = sigmoid(top_shap_value) * 100
            prob_change = adjusted_prob - baseline_prob
            
            if prediction_result == "danger":
                if top_shap_value > 0:
                    interpretation = f"""
                    ** SHAP 값 해석:**
                    
                    예측 결과는 **불량**이며, 가장 큰 영향을 미친 변수는 **{top_feature}**입니다.
                    
                    - **SHAP 값**: {top_shap_value:.4f}
                    - **영향**: 이 변수가 불량 확률을 약 **{abs(prob_change):.2f}%p** 증가시켰습니다.
                    - **해석**: {top_feature}의 현재 값이 불량 예측에 가장 크게 기여했습니다.
                    
                     **개선 제안**: {top_feature}의 값을 조정하면 불량 확률을 낮출 수 있습니다.
                    """
                else:
                    interpretation = f"""
                    ** SHAP 값 해석:**
                    
                    예측 결과는 **불량**이지만, **{top_feature}**는 양품 방향으로 작용했습니다.
                    
                    - **SHAP 값**: {top_shap_value:.4f}
                    - **영향**: 이 변수가 양품 확률을 약 **{abs(prob_change):.2f}%p** 증가시키려 했으나, 다른 변수들의 영향으로 최종 예측은 불량입니다.
                    - **해석**: {top_feature}는 긍정적이나 다른 변수들을 개선해야 합니다.
                    """
            else:
                if top_shap_value < 0:
                    interpretation = f"""
                    ** SHAP 값 해석:**
                    
                    예측 결과는 **정상(양품)**이며, 가장 큰 영향을 미친 변수는 **{top_feature}**입니다.
                    
                    - **SHAP 값**: {top_shap_value:.4f}
                    - **영향**: 이 변수가 정상 확률을 약 **{abs(prob_change):.2f}%p** 증가시켰습니다.
                    - **해석**: {top_feature}의 현재 값이 양품 예측에 가장 크게 기여했습니다.
                    
                    ✅ **현재 상태 유지**: {top_feature}의 현재 값을 유지하는 것이 좋습니다.
                    """
                else:
                    interpretation = f"""
                    ** SHAP 값 해석:**
                    
                    예측 결과는 **정상(양품)**이지만, **{top_feature}**는 불량 방향으로 작용했습니다.
                    
                    - **SHAP 값**: {top_shap_value:.4f}
                    - **영향**: 이 변수가 불량 확률을 약 **{abs(prob_change):.2f}%p** 증가시키려 했으나, 다른 변수들의 긍정적 영향으로 최종예측은 양품입니다.
                    - **해석**: 다른 변수들이 우수하나 {top_feature}는 개선 여지가 있습니다.
                    """
            
            return ui.div(
                ui.markdown(interpretation),
                class_="alert alert-info",
                style="margin-top: 15px; margin-bottom: 20px;"
            )
            
        except Exception as e:
            return ui.div(
                ui.p(f"SHAP 해석 생성 중 오류 발생: {str(e)}"),
                class_="alert alert-warning"
            )

    @output
    @render.plot
    def prediction_pdp_plot():
        current_input = r_current_input()
    
        if current_input is None or not MODEL_LOADED_SUCCESSFULLY:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "예측 버튼을 먼저 눌러주세요", ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
            return fig
    
        try:
            selected_var = input.pdp_variable_selector()
            current_val = current_input[selected_var].values[0]

            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
    
            if selected_var in df_pred.columns:
                var_min = df_pred[selected_var].min()
                var_max = df_pred[selected_var].max()
            else:
                var_stats = feature_stats.get(selected_var, {})
                var_min = var_stats.get('min', current_val - 10)
                var_max = var_stats.get('max', current_val + 10)

            n_samples = 100
            X_pdp = pd.concat([current_input] * n_samples, ignore_index=True)
            X_pdp[selected_var] = np.linspace(var_min, var_max, n_samples)
    
            GRID = 50
            
            display = PartialDependenceDisplay.from_estimator(
                estimator=pipeline,
                X=X_pdp,
                features=[selected_var],
                kind='average',
                grid_resolution=GRID,
                response_method="predict_proba"
            )
            ax = display.axes_[0, 0]
            
            pd_res = partial_dependence(
                estimator=pipeline,
                X=X_pdp,
                features=[selected_var],
                kind='average',
                grid_resolution=GRID,
                response_method="predict_proba"
            )
            
            xx = np.asarray(pd_res["grid_values"][0])
            yy = np.asarray(pd_res["average"][0])
            
            # ✅ 두 번째 코드의 개선된 로직 적용
            # ✅ 변수별로 적절한 EPS 동적 계산
            y_range = float(yy.max() - yy.min())
            EPS = y_range * 0.01  # y축 범위의 3%를 임계값으로 사용
            ymin = float(yy.min())
            mask_good = yy <= (ymin + EPS)
            ymin = float(yy.min())
            mask_good = yy <= (ymin + EPS)
            
            good_lo, good_hi = None, None
            if mask_good.any():
                idx = np.where(mask_good)[0]
                
                # 현재 값에서 가장 가까운 최적 구간을 찾음
                closest_good_idx_in_idx = np.argmin(np.abs(xx[idx] - current_val))
                target_idx = idx[closest_good_idx_in_idx]
                
                splits = np.where(np.diff(idx) != 1)[0] + 1
                runs = np.split(idx, splits)
                
                best_run = None
                for run in runs:
                    if target_idx in run:
                        best_run = run
                        break
                
                if best_run is None:
                    best_run = max(runs, key=len)

                good_lo, good_hi = xx[best_run[0]], xx[best_run[-1]]
                
                r_pdp_rec_range.set({'lo': good_lo, 'hi': good_hi, 'var': selected_var})

                ax.axvspan(good_lo, good_hi, color="lightgreen", alpha=0.4,
                           label=f"권장 구간 {good_lo:.2f} ≤ x ≤ {good_hi:.2f}", zorder=1)
            else:
                r_pdp_rec_range.set(None)

            ax.axvline(x=current_val, color='red', linestyle='--', linewidth=2.5, 
                       label=f'현재 값: {current_val:.2f}', zorder=10)
            
            ax.legend(loc='best', fontsize=10)
            ax.set_title(f"PDP: {get_kr_name(selected_var)} 변화에 따른 불량 예측 확률", 
                         fontsize=14, pad=15, fontweight='bold')
            ax.set_xlabel(get_kr_name(selected_var), fontsize=12)
            ax.set_ylabel('불량 확률 (predict_proba)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            return display.figure_
    
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, f"PDP Plot 생성 오류: {str(e)}", ha='center', va='center',fontsize=10, color='red')
            ax.axis('off')
            print(f"PDP Plot 오류 상세: {e}")
            import traceback
            traceback.print_exc()
            return fig

    @output
    @render.ui
    def pdp_recommendation():
        current_input = r_current_input()
        prediction_result = r_prediction_result()
        pdp_range = r_pdp_rec_range()
        
        if current_input is None or prediction_result is None:
            return ui.div()
        
        if prediction_result == "success":
            return ui.div(
                ui.div(
                    ui.h5("✅ 현재 상태: 양품 예측", class_="text-success"),
                    ui.p("현재 입력된 공정 변수 값으로 양품이 예측됩니다. 현재 설정을 유지하시기 바랍니다.", class_="text-muted"),
                    class_="alert alert-success"
                )
            )
        
        try:
            selected_var = input.pdp_variable_selector()
            current_value = current_input[selected_var].values[0]
            
            if pdp_range and pdp_range['var'] == selected_var:
                rec_lo = pdp_range['lo']
                rec_hi = pdp_range['hi']
                recommendation_text = f"""
                ** {get_kr_name(selected_var)} 조정 권장사항:**
                
                - **현재 값:** {current_value:.2f}
                - **권장 범위:** **{rec_lo:.2f} ~ {rec_hi:.2f}**
                
                ** 개선 방안:**
                PDP 그래프 분석 결과, **{get_kr_name(selected_var)}** 값을 위 권장 범위 내로 조정하면 불량률을 낮출 수 있을 것으로 보입니다.
                
                 **주의:** 실제 공정 변경 시에는 반드시 현장 전문가와 상의하시기 바랍니다.
                """
            else:
                recommendation_text = f"""
                ** {get_kr_name(selected_var)} 조정 권장사항:**
                
                - **현재 값:** {current_value:.2f}

                ** 개선 방안:**
                PDP 그래프를 참고하여 불량 확률이 낮아지는 구간으로 **{get_kr_name(selected_var)}** 값을 조정하는 것을 고려해 보세요.
                
                 **주의:** 실제 공정 변경 시에는 반드시 현장 전문가와 상의하시기 바랍니다.
                """

            return ui.div(
                ui.markdown(recommendation_text),
                class_="alert alert-info",
                style="margin-top: 15px;"
            )
            
        except Exception as e:
            return ui.div(
                ui.p(f"권장사항 생성 중 오류 발생: {str(e)}"),
                class_="alert alert-warning"
            )


    @output
    @render.plot(alt="전처리 전 EDA 시각화")
    def eda_combined_plot_before():
        if not DATA_PRED_LOADED_SUCCESSFULLY: return

        selected_var = input.eda_variable_selector()
        if not selected_var or selected_var not in df_raw.columns: return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"'{get_kr_name(selected_var)}' 변수 분석 (전처리 전)", fontsize=16, weight='bold')
        sns.histplot(df_raw[selected_var], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("전체 데이터 분포", fontsize=12)
        axes[0].set_xlabel("값")
        axes[0].set_ylabel("빈도")
        sns.boxplot(x='passorfail', y=selected_var, data=df_raw, ax=axes[1])
        axes[1].set_title("불량 여부에 따른 분포 비교", fontsize=12)
        axes[1].set_xlabel("불량 여부 (0: 양품, 1: 불량)")
        axes[1].set_ylabel("값")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    @output
    @render.plot(alt="전처리 후 EDA 시각화")
    def eda_combined_plot_after():
        if not DATA_PRED_LOADED_SUCCESSFULLY: return

        selected_var = input.eda_variable_selector()
        if not selected_var or selected_var not in df_pred.columns: return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"'{get_kr_name(selected_var)}' 변수 분석 (전처리 후)", fontsize=16, weight='bold')
        sns.histplot(df_pred[selected_var], kde=True, ax=axes[0], color='lightgreen')
        axes[0].set_title("전체 데이터 분포", fontsize=12)
        axes[0].set_xlabel("값")
        axes[0].set_ylabel("빈도")
        sns.boxplot(x='passorfail', y=selected_var, data=df_pred, ax=axes[1])
        axes[1].set_title("불량 여부에 따른 분포 비교", fontsize=12)
        axes[1].set_xlabel("불량 여부 (0: 양품, 1: 불량)")
        axes[1].set_ylabel("값")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    @output
    @render.ui
    def eda_stats_ui():
        if not DATA_PRED_LOADED_SUCCESSFULLY or not DATA_PRED_LOADED_SUCCESSFULLY: return

        selected_var = input.eda_variable_selector()
        if not selected_var:
            return ui.div("변수를 선택해주세요.", class_="alert alert-warning")

        if selected_var in df_raw.columns:
            stats_before_df = df_raw.groupby('passorfail')[selected_var].describe().T
            stats_before_df.columns = ['양품 (0)', '불량 (1)']
            stats_before_html = stats_before_df.to_html(classes="table table-sm table-striped", float_format='{:,.2f}'.format)
            ui_before = ui.div(
                ui.h5("전처리 전 통계량"),
                ui.HTML(stats_before_html)
            )
        else:
            ui_before = ui.div(ui.h5("전처리 전 통계량"), ui.p("데이터 없음", class_="text-muted"))

        if selected_var in df_pred.columns:
            stats_after_df = df_pred.groupby('passorfail')[selected_var].describe().T
            stats_after_df.columns = ['양품 (0)', '불량 (1)']
            stats_after_html = stats_after_df.to_html(classes="table table-sm table-striped", float_format='{:,.2f}'.format)
            ui_after = ui.div(
                ui.h5("전처리 후 통계량", class_="mt-4"),
                ui.HTML(stats_after_html)
            )
        else:
            ui_after = ui.div(ui.h5("전처리 후 통계량", class_="mt-4"), ui.p("데이터 없음", class_="text-muted"))

        default_summary = "이 변수에는 일반적인 전처리가 적용되었습니다."
        summary_text = EDA_DESCRIPTIONS.get(selected_var, default_summary)
    
        return ui.card(
            ui.card_header(f"'{get_kr_name(selected_var)}' 기술 통계량"),
            ui_before,
            ui_after,
            ui.div(
                ui.h6("변수 처리 및 분석 요약", class_="mt-4"),
                ui.div(ui.HTML(summary_text), style="line-height: 1.6;")
            )
        )
        
    @output
    @render.ui
    def model_performance_table():
        report_data = {
            'precision': [1.00, 0.95, None, 0.97, 1.00],
            'recall': [1.00, 0.96, None, 0.98, 1.00],
            'f1-score': [1.00, 0.96, 1.00, 0.98, 1.00],
            'support': [14058, 662, 14720, 14720, 14720]
        }
        report_index = ['양품 (Class 0)', '불량 (Class 1)', 'accuracy', 'macro avg', 'weighted avg']
        df_report = pd.DataFrame(report_data, index=report_index)
        df_report['support'] = df_report['support'].astype(int)

        cm_data = [[14021, 37], [26, 636]]
        df_cm = pd.DataFrame(cm_data,
                             columns=pd.MultiIndex.from_product([['예측 (Predicted)'], ['양품 (0)', '불량 (1)']]),
                             index=pd.MultiIndex.from_product([['실제 (True)'], ['양품 (0)', '불량 (1)']]))

        cm_html = df_cm.to_html(classes="table table-bordered text-center", justify="center")
        report_html = df_report.to_html(classes="table table-striped table-hover", float_format='{:.2f}'.format, na_rep="")

        return ui.div(
            ui.h3("모델 성능 평가", style="text-align: center; margin-bottom: 20px;"),
            ui.row(
                ui.column(5,
                    ui.h5("혼동 행렬 (Confusion Matrix)", style="text-align: center;"),
                    ui.HTML(cm_html),
                    style="display: flex; flex-direction: column; align-items: center;"
                ),
                ui.column(7,
                    ui.h5("분류 리포트 (Classification Report)"),
                    ui.HTML(report_html),
                    ui.div(
                        ui.p(f"ROC-AUC: 0.9889", style="font-weight: bold; margin-top: 10px; display: inline-block; margin-right: 20px;"),
                        ui.p(f"(Threshold: 0.8346)", style="display: inline-block;")
                    )
                )
            )
        )

app = App(app_ui, server)