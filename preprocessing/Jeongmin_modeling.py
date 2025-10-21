from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.metrics import recall_score, classification_report, ConfusionMatrixDisplay,confusion_matrix,f1_score
import xgboost
import optuna
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.collections import LineCollection
from sklearn.model_selection import TimeSeriesSplit

train = pd.read_csv('./train.csv')
df = train.copy()
df = df.drop(index=19327) # 결측 많은 값 1개
df[df['physical_strength'] == 65535] # 3개
df = df.drop(index=[6000,11811,17598])
df[df['low_section_speed'] == 65535] # 1개
df = df.drop(index=[46546])
list(df[df['Coolant_temperature'] == 1449].index) # 9개
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index))
list(df[df['upper_mold_temp1'] == 1449].index) # 1개
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index))
list(df[df['upper_mold_temp2'] == 4232].index) # 1개
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index))

df['registration_time'] = pd.to_datetime(df['registration_time'])
df['hour'] = df['registration_time'].dt.hour
df['hour']=df['hour'].astype(object)

df.columns
df['tryshot_signal'] = df['tryshot_signal'].fillna('A')
df['molten_volume'] = df['molten_volume'].fillna(0)
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'

df["mold_code"] = df["mold_code"].astype(object)
df["EMS_operation_time"] = df["EMS_operation_time"].astype(object)
df.loc[df["molten_temp"] <= 80, "molten_temp"] = np.nan
df.loc[df["physical_strength"] <= 5, "physical_strength"] = np.nan

df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date','registration_time',
                      'upper_mold_temp3','lower_mold_temp3','working'])

X = df.drop(columns=["passorfail"])
y = df["passorfail"]

split_point = int(len(df) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

num_cols = X_train.select_dtypes(include=["int64","int32","float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# 전처리 파이프라인 정의
num_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),  # ✔️ 수치형 → KNN 기반 대치
    ("scaler", RobustScaler())               # 표준화 스케일링
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), # 범주형 → 최빈값 대치
    ("onehot", OneHotEncoder(handle_unknown="ignore"))    # Train에서 본 카테고리만 학습
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)
# 1. Objective 함수 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
def objective(trial):
    # 테스트할 하이퍼파라미터 값의 범위를 지정합니다.
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1, 10),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'scale_pos_weight': scale_pos_weight, # 기존에 정의된 값 사용
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # XGBoost 모델 생성
    xgb_opt = XGBClassifier(**params)
    
    # 전체 파이프라인 구성
    clf_opt = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_opt)
    ])
    
    # 교차 검증을 통해 안정적인 성능 측정
    # StratifiedKFold는 클래스 비율을 유지하며 데이터를 분할합니다.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    # X_train, y_train을 사용해 교차 검증 수행
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf_opt.fit(X_train_fold, y_train_fold)
        preds = clf_opt.predict(X_val_fold)
        f1 = f1_score(y_val_fold, preds)
        f1_scores.append(f1)
        
    # 교차 검증 F1-score의 평균을 반환
    return np.mean(f1_scores)

# 2. Study 생성 및 실행
# maximize 방향으로 f1-score를 최적화합니다.
study = optuna.create_study(direction='maximize')
# n_trials는 시도 횟수를 의미하며, 필요에 따라 조절합니다.
study.optimize(objective, n_trials=50, show_progress_bar=True)

# 최적화 결과 출력
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 3. 최적 파라미터로 최종 모델 학습 및 평가
best_params = trial.params
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'


final_xgb = XGBClassifier(**best_params)

final_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", final_xgb)
])

# 전체 학습 데이터로 최종 모델 학습
final_clf.fit(X_train, y_train)

# 테스트 데이터로 최종 평가
y_pred_final = final_clf.predict(X_test)

print("\n--- Final Model Evaluation with Best Params ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))



# 1. 테스트 데이터에 대한 예측 확률을 구합니다.
# predict_proba()는 [클래스 0의 확률, 클래스 1의 확률]을 반환하므로,
# 양성 클래스(1)의 확률만 선택합니다. ([:, 1])
y_pred_proba = final_clf.predict_proba(X_test)[:, 1]

# 재현율(Recall)을 높이고 싶다면 임계값을 낮추고,
# 정밀도(Precision)를 높이고 싶다면 임계값을 높입니다.
new_threshold = 0.77 # 예시로 0.4로 설정

# 3. 새로운 임계값을 기준으로 예측 클래스를 결정합니다.
# 확률이 임계값보다 크거나 같으면 1(양성), 작으면 0(음성)으로 변환합니다.
y_pred_adjusted = (y_pred_proba >= new_threshold).astype(int)

# 4. 새로운 예측 결과로 모델을 평가합니다.
print(f"--- Original Threshold (0.5) Evaluation ---")
y_pred_original = final_clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_original))
print("\nClassification Report:\n", classification_report(y_test, y_pred_original))

print(f"\n\n--- Adjusted Threshold ({new_threshold}) Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_adjusted))
print("\nClassification Report:\n", classification_report(y_test, y_pred_adjusted))





y_pred_proba = clf.predict_proba(X_test)[:, 1] 

thresholds = np.arange(0, 1.01, 0.01)
f1_scores = []

# 0부터 1까지 0.01씩 임계값(threshold)을 변경하며 F1-score 계산
for threshold in thresholds:
    # 현재 임계값으로 예측값 생성
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    
    # f1_score 계산하여 리스트에 추가
    f1 = f1_score(y_test, y_pred_temp)
    f1_scores.append(f1)

# F1-score가 최대가 되는 지점의 인덱스를 찾음
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"최적의 임계값 (Best Threshold): {best_threshold:.2f}")
print(f"최적의 F1-Score: {best_f1:.4f}")

# 최적 임계값으로 최종 평가
print("\n--- Evaluation with Best Threshold ---")
y_pred_final = (y_pred_proba >= best_threshold).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))







# ================= XGBOOST 성능평가 (기본모델) ==================
from imblearn.over_sampling import SMOTE
df = train.copy()
df = df.drop(index=19327) # 결측 많은 값 1개
df[df['physical_strength'] == 65535] # 3개
df = df.drop(index=[6000,11811,17598])
df[df['low_section_speed'] == 65535] # 1개
df = df.drop(index=[46546])
list(df[df['Coolant_temperature'] == 1449].index) # 9개
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index))
list(df[df['upper_mold_temp1'] == 1449].index) # 1개
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index))
list(df[df['upper_mold_temp2'] == 4232].index) # 1개
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index))

df['registration_time'] = pd.to_datetime(df['registration_time'])
df['hour'] = df['registration_time'].dt.hour
df['hour']=df['hour'].astype(object)

df.columns
df['tryshot_signal'] = df['tryshot_signal'].fillna('A')
df['molten_volume'] = df['molten_volume'].fillna(0)
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'

df["mold_code"] = df["mold_code"].astype(object)
df["EMS_operation_time"] = df["EMS_operation_time"].astype(object)
df.loc[df["molten_temp"] <= 80, "molten_temp"] = np.nan
df.loc[df["physical_strength"] <= 5, "physical_strength"] = np.nan

df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date','registration_time',
                      'upper_mold_temp3','lower_mold_temp3','working'])

X = df.drop(columns=["passorfail"])
y = df["passorfail"]

split_point = int(len(df) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

num_cols = X_train.select_dtypes(include=["int64","int32","float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns


# 5. 전처리 파이프라인 정의 (기존과 동일)
num_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ],
    remainder='passthrough' # 변환되지 않는 컬럼 유지
)

# 6. XGBoost 분류기 정의 (scale_pos_weight 파라미터 제거)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# 7. 데이터 전처리 실행
# 학습 데이터는 fit_transform, 테스트 데이터는 transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 8. SMOTE를 사용한 오버샘플링 (학습 데이터에만 적용!)
print("SMOTE 적용 전 학습 데이터 형태:", X_train_processed.shape)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print("SMOTE 적용 후 학습 데이터 형태:", X_train_resampled.shape)
print("SMOTE 적용 후 클래스 분포:\n", y_train_resampled.value_counts())


# 9. 모델 학습
# 오버샘플링된 데이터로 모델을 학습시킵니다.
xgb.fit(X_train_resampled, y_train_resampled)


# 10. 평가
# 원본 테스트 데이터로 성능을 검증합니다.
y_pred = xgb.predict(X_test_processed)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# lightGBM

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# -------------------
# 0) 데이터 불러오기 & 전처리
# -------------------
df = train.copy()

# low section speed 
try:
    # --- 1. 데이터 로딩 및 가설 검증 ---
    df = train.copy()

    # 조건: low_section_speed가 49 이하이면서, passorfail이 0(양품)인 경우
    anomalous_mask = (df['low_section_speed'] <= 49) & (df['passorfail'] == 0)
    anomalous_rows = df[anomalous_mask]

    unique_molds = anomalous_rows['mold_code'].unique()
    num_unique_molds = len(unique_molds)

    print(f"\n--- 1단계: 가설 검증 ---")
    print(f"low_section_speed가 49 이하인 양품 데이터의 개수: {len(anomalous_rows)}")
    print(f"해당 데이터들의 고유한 mold_code 개수: {num_unique_molds}")
    print(f"고유 mold_code 목록: {unique_molds}")

    # --- 2. 조건부 KNN 레이블 대치 ---
    if num_unique_molds > 1 and not anomalous_rows.empty:
        print("\n--- 2단계: KNN 레이블 재예측 시작 ---")
        
        # 분석에 사용할 컬럼만 선택 (기존 모델링과 유사하게)
        cols_to_use = [
            'count', 'molten_temp', 'low_section_speed', 'high_section_speed',
            'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1',
            'lower_mold_temp2', 'sleeve_temperature', 'physical_strength',
            'Coolant_temperature', 'EMS_operation_time', 'passorfail', 'mold_code'
        ]
        model_df = df[cols_to_use].copy()
        model_df.dropna(inplace=True) # 간단한 처리를 위해 결측치 행 제거

        # '의심스러운 데이터'와 '신뢰하는 데이터' 분리
        anomalous_mask_clean = (model_df['low_section_speed'] <= 49) & (model_df['passorfail'] == 0)
        trusted_df = model_df[~anomalous_mask_clean]
        anomalous_df = model_df[anomalous_mask_clean]

        X_trusted = trusted_df.drop("passorfail", axis=1)
        y_trusted = trusted_df["passorfail"]
        X_anomalous = anomalous_df.drop("passorfail", axis=1)

        # 전처리기 정의
        num_cols = X_trusted.select_dtypes(include=np.number).columns
        cat_cols = ['mold_code']
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        # KNN 모델 파이프라인
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])

        # '신뢰하는 데이터'로 모델 학습
        print("신뢰하는 데이터로 KNN 모델을 학습합니다...")
        knn_pipeline.fit(X_trusted, y_trusted)

        # '의심스러운 데이터'의 레이블 재예측
        print("의심스러운 데이터의 레이블을 재예측합니다...")
        new_labels = knn_pipeline.predict(X_anomalous)
        
        # 원본 데이터프레임(df)에 변경사항 적용
        original_indices = anomalous_df.index
        df.loc[original_indices, 'passorfail'] = new_labels

        # 변경 결과 요약
        num_changed = (anomalous_df['passorfail'] != new_labels).sum()
        print("\n--- 결과 요약 ---")
        print(f"총 {len(anomalous_df)}개의 의심스러운 데이터 중, {num_changed}개의 레이블이 0(pass)에서 1(fail)로 변경되었습니다.")
        
    elif anomalous_rows.empty:
        print("\n'low_section_speed <= 49' 조건에서 양품(0)인 데이터가 없어 추가 작업을 진행하지 않습니다.")
    else:
        print("\n모든 의심스러운 데이터가 단일 mold_code에서 발생하여 추가 작업을 진행하지 않습니다.")

except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다. 코드와 같은 폴더에 파일이 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

# 이상치/결측 행 제거
df = df.drop(index=19327) 
df = df.drop(index=[6000,11811,17598]) 
df = df.drop(index=[46546])
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index))
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index))
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index))

df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['molten_volume'] = df['molten_volume'].replace(635, np.nan)

# 시간 변수 가공
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['hour'] = df['registration_time'].dt.hour.astype(object)

# 결측치 보정
df['tryshot_signal'] = df['tryshot_signal'].fillna('A')
df['molten_volume'] = df['molten_volume'].fillna(0)
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'

# 타입 변경
df["mold_code"] = df["mold_code"].astype(object)
df["EMS_operation_time"] = df["EMS_operation_time"].astype(object)

# 값 조건 기반 결측 처리
df.loc[df["molten_temp"] <= 80, "molten_temp"] = np.nan
df.loc[df["physical_strength"] <= 5, "physical_strength"] = np.nan

# 불필요한 컬럼 제거
df = df.drop(columns=[
    'id','line','name','mold_name','emergency_stop','time','date','registration_time',
    'upper_mold_temp3','lower_mold_temp3','working'
])

# -------------------
# 1) X, y 분리
# -------------------
y = df['passorfail']
X = df.drop(columns=['passorfail'])

# 시간 순서 기준으로 split 
split_point = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# -------------------
# 2) 수치/범주형 컬럼 분리
# -------------------
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# -------------------
# 3) 전처리기 정의
# -------------------
# 수치형: KNN으로 결측치 채우기
num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

# 범주형: 최빈값 + OrdinalEncoder
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# ColumnTransformer로 합치기
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# -------------------
# 4) LightGBM 모델
# -------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))
])

best_trial={'num_leaves': 98, 'max_depth': 9, 'learning_rate': 0.008275183434186801, 'n_estimators': 677, 'subsample': 0.6035915042464275, 'colsample_bytree': 0.879104309394895, 'min_child_samples': 85}
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# 최적 파라미터로 최종 모델 재학습
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42,
                                 **best_trial))
])
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nFinal Model Report:")
print(classification_report(y_test, y_pred))
print("Final ROC-AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0","Pred 1"],
            yticklabels=["True 0","True 1"])
plt.title("Confusion Matrix (Final Model)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()