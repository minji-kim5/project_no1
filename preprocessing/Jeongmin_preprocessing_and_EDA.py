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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import recall_score, classification_report, ConfusionMatrixDisplay,confusion_matrix,f1_score
import xgboost
import optuna
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.collections import LineCollection
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


train = pd.read_csv('./train.csv')
train.info()
train.head()
train.columns
train.loc[train.low_section_speed.isna(),:]

test = pd.read_csv('./test.csv')

df = train[['low_section_speed', 'high_section_speed','molten_volume','mold_code',
            'cast_pressure','biscuit_thickness','sleeve_temperature','passorfail']]
df.loc[df.low_section_speed.isna(),:]
df.info()

# ================ pass/fail 그룹 비교 ===================
feature_columns = df.select_dtypes(include='number').columns.drop('passorfail')
for col in feature_columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"'{col}' 변수의 Pass/Fail 그룹 비교", fontsize=16)
    sns.kdeplot(data=df, x=col, hue='passorfail', 
                fill=True, common_norm=False, palette='Set1', ax=axes[0])
    axes[0].set_title('데이터 분포 비교 (KDE Plot)')
    axes[0].set_xlabel(f'{col} 값')
    axes[0].set_ylabel('밀도 (Density)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    sns.boxplot(data=df, x='passorfail', y=col, palette='Set1', ax=axes[1])
    axes[1].set_title('데이터 요약 비교 (Box Plot)')
    axes[1].set_xlabel('Pass/Fail')
    axes[1].set_ylabel(f'{col} 값')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



df.loc[df['low_section_speed']>40000,]
df1 = df.drop(index=46546)
df1.iloc[19327,:]
df1 = df1.drop(index=19327)
df1.info()
for col in feature_columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"'{col}' 변수의 Pass/Fail 그룹 비교", fontsize=16)
    sns.kdeplot(data=df1, x=col, hue='passorfail', 
                fill=True, common_norm=False, palette='Set1', ax=axes[0])
    axes[0].set_title('데이터 분포 비교 (KDE Plot)')
    axes[0].set_xlabel(f'{col} 값')
    axes[0].set_ylabel('밀도 (Density)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    sns.boxplot(data=df1, x='passorfail', y=col, palette='Set1', ax=axes[1])
    axes[1].set_title('데이터 요약 비교 (Box Plot)')
    axes[1].set_xlabel('Pass/Fail')
    axes[1].set_ylabel(f'{col} 값')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# ====================== EDA ===========================
df1.groupby('mold_code')['low_section_speed'].describe()
df1.loc[df1['low_section_speed']==0,'passorfail']
# 0일땐 다 불량
# 65000 하나 버리면 모든 code가 거의 비슷함. 나누는 게 의미없어보임

df1.loc[df1['low_section_speed']>190,:]
df1.loc[(df1['low_section_speed']>125)&(df1['passorfail']==1),:]['mold_code'].value_counts()
df1.loc[df1.mold_code == 8412]
df1.groupby('mold_code')['low_section_speed'].describe()
# 200인거 2개. 둘다 불량
# 125 넘는 것 중 양품은 전부다 8412(3290),불량은 8412(116),8573(3)
# ,8917(1). 근데 8412가 총 18000개라 ㅁㄴㅇㅇㅁㅇㄴㅁㅇ노ㅓ;ㅁㅎ
df1.loc[df1.low_section_speed<50,'passorfail'].sum()
# speed가 50 미만인 388개 데이터중 약 16개 빼고는 전부 불량
# >> 이 16개는 측정오류라고 보는게 맞지 않을까

df1.loc[df1['high_section_speed']<97,:]
# 84 이하는 전부 불량. 근데 코드는 제각각임
# 97,98,98 3개는 정상
df1.groupby('mold_code')['high_section_speed'].describe()

df1.loc[df1['molten_volume']>2500,['mold_code']].value_counts()
df1.loc[df1['molten_volume']>2500,'passorfail'].sum()
df1.loc[(df1['molten_volume']>500)&(df1['molten_volume']<1000),:]
df1.groupby('mold_code')['molten_volume'].describe()
# 600, 2700 주변에 값들이 모여있는데, 코드가 제각각임.
# 8573은 아예 전부다 결측치
# min max가 너무 차이가 많아서 뭐 중앙값을 써야하나..
# 600주변은 그나마 불량이 1개뿐인데 2700 주변은 1561개중 67개가 불량

df1.loc[(df1.cast_pressure >120)&(df1.cast_pressure<200)&(df1.passorfail==0),:]
# 149 아래는 다 불량. 151까지도 100개정도중에 5개만 정상이고 나머지는 다 불량
df1.loc[(df1.cast_pressure >151)&(df1.cast_pressure<=283)&(df1.passorfail==1),:]
# 1694개중에 20개 빼고 다 불량. (그중에도 152에 13개가 양품인데, 다 8412)
df1.groupby('mold_code')['cast_pressure'].describe()

df1.loc[df1['biscuit_thickness']>400,'passorfail'].sum()
# 400을 넘는 167개는 전부 불량. 나머지는 의미없어보임
df1.groupby('mold_code')['biscuit_thickness'].describe()
# 24보다 작은 66개중 하나만 양품

df1.loc[df1['sleeve_temperature']>1400,:]
# 1400을 넘는 값은 전부 8917(57), 그중 6개만 불량
df1.loc[(df1['sleeve_temperature']>610)&(df1['sleeve_temperature']<1400),:]
# 615이상 ~ 1400 이하는 전부 불량. code는 제각각임
df1.groupby('mold_code')['sleeve_temperature'].describe()
# 8412만 평균이 300대, 1분위수가 191. 그리고 분산이 매우높음.
# 다른 것들에 비해 8573, 8600은 미니멈이 매우높음
# 113까지는 2개 뺴고 다 불량. 


train.loc[(train.upper_mold_temp3 >200)&(train.mold_code == 8573),'upper_mold_temp3'].unique()



# ================= 이상치의 분포 =====================
target_column = 'sleeve_temperature'
# 1. passorfail = 0 그룹의 이상치 추출
Q1_0 = df.loc[df['passorfail'] == 0, target_column].quantile(0.25)
Q3_0 = df.loc[df['passorfail'] == 0, target_column].quantile(0.75)
IQR_0 = Q3_0 - Q1_0
lower_bound_0 = Q1_0 - 1.5 * IQR_0
upper_bound_0 = Q3_0 + 1.5 * IQR_0
outliers_pass0 = df.loc[
    (df['passorfail'] == 0) & 
    ((df[target_column] < lower_bound_0) | (df[target_column] > upper_bound_0)),
    target_column
].dropna()
# 2. passorfail = 1 그룹의 이상치 추출
Q1_1 = df.loc[df['passorfail'] == 1, target_column].quantile(0.25)
Q3_1 = df.loc[df['passorfail'] == 1, target_column].quantile(0.75)
IQR_1 = Q3_1 - Q1_1
lower_bound_1 = Q1_1 - 1.5 * IQR_1
upper_bound_1 = Q3_1 + 1.5 * IQR_1
outliers_pass1 = df.loc[
    (df['passorfail'] == 1) & 
    ((df[target_column] < lower_bound_1) | (df[target_column] > upper_bound_1)),
    target_column
].dropna()
# 3. 이상치 데이터 통합
outlier_df_combined = pd.DataFrame({
    target_column: pd.concat([outliers_pass0, outliers_pass1]),
    'passorfail': [0.0] * len(outliers_pass0) + [1.0] * len(outliers_pass1)
})
# 4. 시각화 (KDE Plot만 사용)
plt.figure(figsize=(10, 6))
if not outlier_df_combined.empty:
    sns.kdeplot(data=outlier_df_combined, x=target_column, hue='passorfail', 
                fill=True, common_norm=False, palette='Set1')
    plt.title(f"'{target_column}' 변수의 Pass/Fail 별 '이상치' 분포", fontsize=16)
else:
    plt.title(f"'{target_column}' 변수에는 이상치가 없습니다.", fontsize=16)
plt.xlabel(f'{target_column} 값', fontsize=12)
plt.ylabel('밀도 (Density)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Pass/Fail')
plt.show()



# =================== 상관관계 =======================
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계 히트맵', fontsize=16)
plt.show()




# ============== 랜덤포레스트 기반 변수 영향도 측정 ================
model_df = train.copy()
# 2. 불필요한 컬럼 제거
model_df = model_df.drop(['id','line','name','time','date','registration_time',
                          'mold_name','tryshot_signal'], axis=1)
# 3. 피처(X)와 타겟(y) 분리
X = model_df.drop('passorfail', axis=1)
y = model_df['passorfail']
# 4. 데이터 타입에 따라 컬럼명 분리
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
# 5. 전처리 파이프라인 구성
# 수치형 데이터: KNN Imputer로 결측치 처리
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))
])
# 범주형 데이터: 최빈값으로 결측치 처리 후 원핫인코딩
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# ColumnTransformer를 이용해 두 파이프라인 통합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
# 6. 랜덤 포레스트 모델과 전처리 파이프라인 연결
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
# 7. 모델 학습
model_pipeline.fit(X, y)
# 8. 변수 중요도 추출
# 원핫인코딩 후의 컬럼명 가져오기
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)
# 중요도 추출
importances = model_pipeline.named_steps['classifier'].feature_importances_
# 9. 시각화
feature_importances = pd.Series(importances, index=all_feature_names)
top_features = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(x=top_features, y=top_features.index)
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
print("\n가장 중요한 변수 Top 10:")
print(top_features.head(10))




# =========== 변수 제거 후 randomforestclassifier ==============
model_df = train.copy()

# 2. 불필요한 컬럼 제거
model_df = model_df.drop(['id','line','name','time','date','registration_time',
                          'mold_name','tryshot_signal','emergency_stop',
                          'working','heating_furnace','upper_mold_temp3',
                          'lower_mold_temp3'], axis=1)

# 3. 피처(X)와 타겟(y) 분리
X = model_df.drop('passorfail', axis=1)
y = model_df['passorfail']

# 4. 데이터 타입에 따라 컬럼명 분리
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# 'mold_code'를 수치형 리스트에서 제거하고 범주형 리스트에 추가
if 'mold_code' in numerical_features:
    numerical_features.remove('mold_code')
    categorical_features.append('mold_code')
    
# 5. 전처리 파이프라인 구성
# [수정] 수치형 데이터: KNN Imputer로 결측치 처리 후 StandardScaler로 스케일링
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()) # StandardScaler 단계 추가
])

# 범주형 데이터: 최빈값으로 결측치 처리 후 원핫인코딩
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer를 이용해 두 파이프라인 통합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. 랜덤 포레스트 모델과 전처리 파이프라인 연결
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])

# 7. Train/Test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. GridSearchCV 설정
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': ['balanced']
}
grid_search = GridSearchCV(model_pipeline, param_grid=param_grid,
                           scoring='recall', cv=3, n_jobs=-1, verbose=2)

# 9. 최적 파라미터 탐색 시작
print("GridSearchCV를 이용한 최적 하이퍼파라미터 탐색을 시작합니다...")
grid_search.fit(X_train, y_train)

# 10. 결과 출력
print("\n--- 탐색 결과 ---")
print("최적 하이퍼파라미터:", grid_search.best_params_)
print("교차 검증 최고 재현율(Recall) 점수:", f"{grid_search.best_score_:.4f}")

# 11. 최적 모델로 예측 및 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- 최적 모델의 Test 데이터 최종 성능 평가 ---")
print(classification_report(y_test, y_pred))

# 12. 혼동 행렬 시각화
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix (Optimized Model with Scaler)')
plt.show()


# best = {'classifier__class_weight': 'balanced', 'classifier__max_depth': 10, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}




# ================= XGBOOST 성능평가 (기본모델) ==================
df = train.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'heating_furnace','working','upper_mold_temp3','lower_mold_temp3'])

df["mold_code"] = df["mold_code"].astype(object)

df = df.drop(index=19327) # 결측 많은 값
df.describe()

df[df['physical_strength'] == 65535]
df = df.drop(index=[6000,11811,17598])

df[df['low_section_speed'] == 65535]
df = df.drop(index=[46546])

list(df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index))

list(df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index))

list(df[df['upper_mold_temp2'] == 4232].index)
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index))

df.info()

# 2. 입력변수(X), 타겟변수(y) 분리
X = df.drop(columns=["passorfail"])
y = df["passorfail"]

# 3. Train/Test Split (★ 누수 방지를 위해 전처리 전에 먼저 분리)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 수치형 / 범주형 컬럼 구분
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# 5. 전처리 파이프라인 정의
num_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),  # ✔️ 수치형 → KNN 기반 대치
    ("scaler", StandardScaler())             # 표준화 스케일링
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
from xgboost import XGBClassifier
# 6. XGBoost 분류기 정의

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 클래스 불균형 보정
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# 7. 전체 파이프라인 구성 (전처리 + 모델)
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])

# 8. 학습
clf.fit(X_train, y_train)

# 9. 평가
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(clf, "xgb_pipeline_model.pkl")





# ================= furnace yes/no 비교 ========================
data_A = train.loc[~train.molten_volume.isna(),:]
data_B = train.loc[~train.heating_furnace.isna(),:]

# 2. 데이터 합치기 및 그룹 구분
data_A['source'] = 'A'
data_B['source'] = 'B'
combined_df = pd.concat([data_A, data_B], ignore_index=True)

# 3. 시각화할 수치형 변수 선택
numerical_cols = combined_df.select_dtypes(include=np.number).columns.tolist()

# 4. 서브플롯(Subplot)을 이용한 시각화
# 변수 개수에 맞춰 자동으로 행과 열 개수 조절
n_cols = 3
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() # 2D 배열을 1D로 변환하여 다루기 쉽게 함

for i, col in enumerate(numerical_cols):
    sns.kdeplot(data=combined_df, x=col, hue='source', fill=True,
                palette={'A': 'coral', 'B': 'skyblue'}, ax=axes[i])
    axes[i].set_title(f'"{col}"의 분포 비교', fontsize=12)
    axes[i].set_xlabel('') # x축 라벨은 간결하게 생략
    axes[i].legend(title='데이터 그룹')

# 남는 빈 subplot이 있다면 숨기기
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(pad=2.0)
plt.suptitle('A/B 데이터 그룹별 변수 분포 비교', fontsize=18, y=1.02)
plt.legend()
plt.show()




# =========== 조기종료로 최적 파라미터탐색 ==================
# 'train' 데이터프레임이 이미 로드되어 있다고 가정합니다.
df = train.copy()

# 데이터 전처리 (기존과 동일)
# ... (이전 코드의 데이터 전처리 부분은 여기에 그대로 들어갑니다)
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=19327, errors='ignore')
df = df.drop(index=[6000,11811,17598], errors='ignore')
df = df.drop(index=[46546], errors='ignore')
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index), errors='ignore')

# 입력변수(X), 타겟변수(y) 분리
X = df.drop(columns=["passorfail"])
y = df["passorfail"]

# 1차 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 전처리기 학습 및 데이터 변환
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns
X_train['mold_code'] = X_train['mold_code'].astype(object)
X_test['mold_code'] = X_test['mold_code'].astype(object)
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

num_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ], remainder='passthrough'
)
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 조기 종료를 위한 검증 데이터셋 추가 분리
X_train_sub_t, X_val_t, y_train_sub, y_val = train_test_split(
    X_train_transformed, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# --- [수정] XGBoost 분류기 정의 시점에 파라미터 전달 ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    early_stopping_rounds=50 # 파라미터를 이 위치로 이동
)

# --- [수정] 모델 학습 시점에서는 eval_set만 전달 ---
print("\nXGBoost 모델 학습을 시작합니다 (수정된 조기 종료 적용)...")
xgb.fit(X_train_sub_t, y_train_sub,
        eval_set=[(X_val_t, y_val)], # early_stopping_rounds는 제거
        verbose=False)

print(f"\n최적의 트리 개수 (n_estimators): {xgb.best_iteration}")

# 최종 평가
y_pred = xgb.predict(X_test_transformed)
print("\n--- 조기 종료 적용 후 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix (with Early Stopping)")
plt.show()




# ================= 베이지안 최적화 =====================
# 'train' 데이터프레임이 이미 로드되어 있다고 가정합니다.
# ... (이전 코드의 데이터 전처리 부분은 여기에 그대로 들어갑니다)
df = train.copy()
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=19327, errors='ignore')
df = df.drop(index=[6000,11811,17598], errors='ignore')
df = df.drop(index=[46546], errors='ignore')
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index), errors='ignore')

# 입력변수(X), 타겟변수(y) 분리
X = df.drop(columns=["passorfail"])
y = df["passorfail"]

# 1차 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 데이터 전처리 파이프라인 구성 (기존과 동일)
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns
X_train['mold_code'] = X_train['mold_code'].astype(object)
X_test['mold_code'] = X_test['mold_code'].astype(object)
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

num_transformer = Pipeline(steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())])
cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)])

# --- Optuna를 위한 Objective 함수 정의 ---
def objective(trial, X, y, preprocessor):
    # 1. 데이터를 학습용과 검증용으로 분리
    X_train_obj, X_val_obj, y_train_obj, y_val_obj = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. 전처리기 학습 및 데이터 변환
    preprocessor.fit(X_train_obj)
    X_train_transformed = preprocessor.transform(X_train_obj)
    X_val_transformed = preprocessor.transform(X_val_obj)

    # 3. 탐색할 하이퍼파라미터 범위 설정
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }

    # 4. 모델 학습 및 평가
    scale_pos_weight = (y_train_obj == 0).sum() / (y_train_obj == 1).sum()
    model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_transformed, y_train_obj)
    preds = model.predict(X_val_transformed)
    
    # 5. 재현율(Recall) 반환 (Optuna는 이 값을 최대화하는 방향으로 탐색)
    recall = recall_score(y_val_obj, preds, pos_label=1)
    return recall

# --- Optuna 실행 ---
# 1. Study 생성 (재현율을 maximize하는 것이 목표)
study = optuna.create_study(direction='maximize')

# 2. 최적화 실행 (n_trials: 시도 횟수, 많을수록 좋지만 시간 소요)
print("Optuna를 이용한 베이지안 최적화를 시작합니다...")
study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=50)

# 3. 최적의 파라미터 확인
print("\n최적의 하이퍼파라미터:", study.best_params)
print("최적의 교차 검증 재현율:", study.best_value)

# --- 최종 모델 학습 및 평가 ---
# 1. 최적의 파라미터로 최종 모델 정의
best_params = study.best_params
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 2. 전체 학습 데이터로 전처리기 및 모델 재학습
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", final_xgb)])
final_pipeline.fit(X_train, y_train)

# 3. 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- Optuna 최적화 후 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix (Optuna Optimized)")
plt.show()






# =========== 베이지안 최적화 기반 최적파라미터 =============
df = train.copy()

# 피처 엔지니어링 및 정제
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['pressure_x_temp'] = df['cast_pressure'] * df['molten_temp']
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)

# --- 2. 입력/타겟 변수 분리 및 Train/Test Split ---
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. 그룹별 결측치 처리 ---
numerical_cols = X_train.select_dtypes(include=np.number).columns
grouped_medians = X_train.groupby('mold_code')[numerical_cols].median()
global_median = X_train[numerical_cols].median()
for mold_code, group_median in grouped_medians.iterrows():
    X_train.loc[X_train['mold_code'] == mold_code] = X_train.loc[X_train['mold_code'] == mold_code].fillna(group_median)
X_train.fillna(global_median, inplace=True)
for mold_code in X_test['mold_code'].unique():
    if mold_code in grouped_medians.index:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(grouped_medians.loc[mold_code])
    else:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(global_median)
X_test.fillna(global_median, inplace=True)

# --- 4. 최종 모델 학습 및 평가 ---
# [수정] Optuna 탐색 결과로 얻은 최적의 하이퍼파라미터를 직접 입력
best_params = {
    'n_estimators': 591, 'max_depth': 3,
    'learning_rate': 0.06213186050221397,
    'subsample': 0.6795232137595384,
    'colsample_bytree': 0.8259315529365058,
    'gamma': 3.5790023124651724, 'reg_alpha': 3.2638154400802755,
    'reg_lambda': 1.2306483183364956
}

# 최종 모델 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 최종 파이프라인 구성
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include="object").columns
final_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("classifier", final_xgb)])

# 전체 학습 데이터로 최종 모델 학습
print("최적 파라미터로 최종 모델 학습을 시작합니다...")
final_pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix (Final Optimized Model)")
plt.show()



# ============== low section speed 예측불량 치환 ================
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

# 피처 엔지니어링 및 정제
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['pressure_x_temp'] = df['cast_pressure'] * df['molten_temp']
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)

# --- 2. 입력/타겟 변수 분리 및 Train/Test Split ---
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. 그룹별 결측치 처리 ---
numerical_cols = X_train.select_dtypes(include=np.number).columns
grouped_medians = X_train.groupby('mold_code')[numerical_cols].median()
global_median = X_train[numerical_cols].median()
for mold_code, group_median in grouped_medians.iterrows():
    X_train.loc[X_train['mold_code'] == mold_code] = X_train.loc[X_train['mold_code'] == mold_code].fillna(group_median)
X_train.fillna(global_median, inplace=True)
for mold_code in X_test['mold_code'].unique():
    if mold_code in grouped_medians.index:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(grouped_medians.loc[mold_code])
    else:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(global_median)
X_test.fillna(global_median, inplace=True)

# --- 4. 최종 모델 학습 및 평가 ---
# [수정] Optuna 탐색 결과로 얻은 최적의 하이퍼파라미터를 직접 입력
best_params = {
    'n_estimators': 591, 'max_depth': 3,
    'learning_rate': 0.06213186050221397,
    'subsample': 0.6795232137595384,
    'colsample_bytree': 0.8259315529365058,
    'gamma': 3.5790023124651724, 'reg_alpha': 3.2638154400802755,
    'reg_lambda': 1.2306483183364956
}

# 최종 모델 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 최종 파이프라인 구성
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include="object").columns
final_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("classifier", final_xgb)])

# 전체 학습 데이터로 최종 모델 학습
print("최적 파라미터로 최종 모델 학습을 시작합니다...")
final_pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))






# ================ cast pressure 예측불량 치환 ===================
try:
    # --- 1. 데이터 로딩 및 가설 검증 ---
    df = train.copy()
    print("데이터 로딩 완료.")

    # [수정] 조건: cast_pressure가 283 미만이면서, passorfail이 0(양품)인 경우
    anomalous_mask = (df['cast_pressure'] < 283) & (df['passorfail'] == 0)
    anomalous_rows = df[anomalous_mask]

    unique_molds = anomalous_rows['mold_code'].unique()
    num_unique_molds = len(unique_molds)

    print(f"\n--- 1단계: 가설 검증 (cast_pressure < 283) ---")
    print(f"해당 조건의 양품 데이터 개수: {len(anomalous_rows)}")
    print(f"해당 데이터들의 고유한 mold_code 개수: {num_unique_molds}")
    print(f"고유 mold_code 목록: {unique_molds}")

    # --- 2. 조건부 KNN 레이블 대치 ---
    if num_unique_molds > 1 and not anomalous_rows.empty:
        print("\n--- 2단계: KNN 레이블 재예측 시작 ---")
        
        # 분석에 사용할 컬럼만 선택
        cols_to_use = [
            'count', 'molten_temp', 'low_section_speed', 'high_section_speed',
            'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'cast_pressure',
            'lower_mold_temp2', 'sleeve_temperature', 'physical_strength',
            'Coolant_temperature', 'EMS_operation_time', 'passorfail', 'mold_code'
        ]
        model_df = df[cols_to_use].copy()
        model_df.dropna(inplace=True)

        # '의심스러운 데이터'와 '신뢰하는 데이터' 분리
        anomalous_mask_clean = (model_df['cast_pressure'] < 283) & (model_df['passorfail'] == 0)
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
        print("\n해당 조건에서 양품(0)인 데이터가 없어 추가 작업을 진행하지 않습니다.")
    else:
        print("\n모든 의심스러운 데이터가 단일 mold_code에서 발생하여 추가 작업을 진행하지 않습니다.")

except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다. 코드와 같은 폴더에 파일이 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

# 피처 엔지니어링 및 정제
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['pressure_x_temp'] = df['cast_pressure'] * df['molten_temp']
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)

# --- 2. 입력/타겟 변수 분리 및 Train/Test Split ---
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. 그룹별 결측치 처리 ---
numerical_cols = X_train.select_dtypes(include=np.number).columns
grouped_medians = X_train.groupby('mold_code')[numerical_cols].median()
global_median = X_train[numerical_cols].median()
for mold_code, group_median in grouped_medians.iterrows():
    X_train.loc[X_train['mold_code'] == mold_code] = X_train.loc[X_train['mold_code'] == mold_code].fillna(group_median)
X_train.fillna(global_median, inplace=True)
for mold_code in X_test['mold_code'].unique():
    if mold_code in grouped_medians.index:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(grouped_medians.loc[mold_code])
    else:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(global_median)
X_test.fillna(global_median, inplace=True)

# --- 4. 최종 모델 학습 및 평가 ---
# [수정] Optuna 탐색 결과로 얻은 최적의 하이퍼파라미터를 직접 입력
best_params = {
    'n_estimators': 591, 'max_depth': 3,
    'learning_rate': 0.06213186050221397,
    'subsample': 0.6795232137595384,
    'colsample_bytree': 0.8259315529365058,
    'gamma': 3.5790023124651724, 'reg_alpha': 3.2638154400802755,
    'reg_lambda': 1.2306483183364956
}

# 최종 모델 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 최종 파이프라인 구성
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include="object").columns
final_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("classifier", final_xgb)])

# 전체 학습 데이터로 최종 모델 학습
print("최적 파라미터로 최종 모델 학습을 시작합니다...")
final_pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))




# ================== 예측불량 둘 다 치환 =====================
def relabel_anomalies(df, condition_mask, feature_cols):
    """
    주어진 조건(mask)에 해당하는 데이터의 레이블을 KNN으로 재예측하여 수정하는 함수
    """
    anomalous_rows = df[condition_mask]
    
    if anomalous_rows.empty:
        print("해당 조건의 양품 데이터가 없어 레이블 수정을 건너뜁니다.")
        return df

    print(f"\n{len(anomalous_rows)}개의 데이터에 대해 KNN 레이블 재예측을 시작합니다...")
    
    # 분석에 사용할 데이터프레임 복사 및 결측치 처리
    model_df = df[feature_cols + ['passorfail']].copy()
    model_df.dropna(inplace=True)

    anomalous_mask_clean = (model_df.index.isin(anomalous_rows.index))
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

    # '신뢰하는 데이터'로 모델 학습 및 '의심스러운 데이터' 재예측
    knn_pipeline.fit(X_trusted, y_trusted)
    new_labels = knn_pipeline.predict(X_anomalous)
    
    # 원본 데이터프레임(df)에 변경사항 적용
    original_indices = anomalous_df.index
    num_changed = (df.loc[original_indices, 'passorfail'] != new_labels).sum()
    df.loc[original_indices, 'passorfail'] = new_labels
    
    print(f"-> 완료: {num_changed}개의 레이블이 0(pass)에서 1(fail)로 변경되었습니다.")
    return df

try:
    # --- 1. 데이터 로딩 ---
    df = pd.read_csv('train.csv')
    print("데이터 로딩 완료.")

    # 분석에 사용할 주요 컬럼 정의
    feature_cols = [
        'count', 'molten_temp', 'low_section_speed', 'high_section_speed',
        'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'cast_pressure',
        'lower_mold_temp2', 'sleeve_temperature', 'physical_strength',
        'Coolant_temperature', 'EMS_operation_time', 'mold_code'
    ]

    # --- 2. 가설 기반 데이터 정제 ---
    # 가설 1: low_section_speed <= 49 이면서 양품인 경우
    print("\n--- 가설 1: low_section_speed 처리 ---")
    speed_mask = (df['low_section_speed'] <= 49) & (df['passorfail'] == 0)
    df = relabel_anomalies(df, speed_mask, feature_cols)

    # 가설 2: cast_pressure < 283 이면서 양품인 경우
    print("\n--- 가설 2: cast_pressure 처리 ---")
    pressure_mask = (df['cast_pressure'] < 283) & (df['passorfail'] == 0)
    df = relabel_anomalies(df, pressure_mask, feature_cols)

    # --- 3. 최종 모델링을 위한 전처리 ---
    # (이전 대화에서 논의된 다른 전처리들을 여기에 추가할 수 있습니다)
    X = df.drop(columns=["passorfail"])
    y = df["passorfail"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 최종 모델 학습 및 평가 (RandomForest 사용)
    # ... (이전 대화의 최종 RandomForest 모델링 코드를 여기에 붙여넣으시면 됩니다) ...
    # 예시:
    print("\n--- 최종 모델 학습 및 평가 ---")
    # (이 부분은 이전 대화에서 사용한 최종 모델링 파이프라인으로 채워주세요)
    print("데이터 정제가 완료되었습니다. 이 데이터를 바탕으로 최종 모델을 학습시켜 성능을 확인해보세요.")


except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다. 코드와 같은 폴더에 파일이 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

# 피처 엔지니어링 및 정제
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['pressure_x_temp'] = df['cast_pressure'] * df['molten_temp']
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','tryshot_signal','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)

# --- 2. 입력/타겟 변수 분리 및 Train/Test Split ---
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. 그룹별 결측치 처리 ---
numerical_cols = X_train.select_dtypes(include=np.number).columns
grouped_medians = X_train.groupby('mold_code')[numerical_cols].median()
global_median = X_train[numerical_cols].median()
for mold_code, group_median in grouped_medians.iterrows():
    X_train.loc[X_train['mold_code'] == mold_code] = X_train.loc[X_train['mold_code'] == mold_code].fillna(group_median)
X_train.fillna(global_median, inplace=True)
for mold_code in X_test['mold_code'].unique():
    if mold_code in grouped_medians.index:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(grouped_medians.loc[mold_code])
    else:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(global_median)
X_test.fillna(global_median, inplace=True)

# --- 4. 최종 모델 학습 및 평가 ---
# [수정] Optuna 탐색 결과로 얻은 최적의 하이퍼파라미터를 직접 입력
best_params = {
    'n_estimators': 591, 'max_depth': 3,
    'learning_rate': 0.06213186050221397,
    'subsample': 0.6795232137595384,
    'colsample_bytree': 0.8259315529365058,
    'gamma': 3.5790023124651724, 'reg_alpha': 3.2638154400802755,
    'reg_lambda': 1.2306483183364956
}

# 최종 모델 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 최종 파이프라인 구성
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include="object").columns
final_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("classifier", final_xgb)])

# 전체 학습 데이터로 최종 모델 학습
print("최적 파라미터로 최종 모델 학습을 시작합니다...")
final_pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))







# ============= 정지 이후 횟수와 불량의 관계 시각화 ================
# --- 1. 데이터 로딩 ---
df = pd.read_csv('train.csv')

# --- 2. 시각화 데이터 준비 ---
# x 좌표는 데이터 인덱스, y 좌표는 모두 1로 고정하여 수평선을 만듭니다.
x = df.index.to_numpy()
y = np.ones_like(x)

# 각 점을 연결하는 짧은 수평 선분들(segments)을 생성합니다.
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# passorfail 값에 따라 각 선분의 색상을 결정합니다.
colors = df['passorfail'].map({0: 'skyblue', 1: 'coral'}).iloc[:-1].tolist()

# '정지' 이벤트가 발생한 지점의 인덱스를 찾습니다.
stop_indices = df[df['working'] == '정지'].index

# --- 3. LineCollection을 이용한 시각화 ---
fig, ax = plt.subplots(figsize=(16, 4))

# 수평선분 모음(LineCollection)을 생성합니다. 선을 두껍게 하여 잘 보이게 합니다.
lc = LineCollection(segments, colors=colors, linewidths=10)
ax.add_collection(lc)

# '정지' 이벤트 지점에 검은색 수직선을 그립니다.
ax.vlines(x=stop_indices, ymin=0, ymax=2, colors='black', lw=2, 
          label='정지(Stop) 발생', alpha=0.8)

# 축 범위 자동 설정 및 조정
ax.autoscale_view()
ax.set_ylim(0.5, 1.5)

# 그래프 꾸미기
ax.set_title("시간에 따른 정상/불량 상태 및 '정지' 이벤트 발생 시점", fontsize=16)
ax.set_xlabel('데이터 인덱스 (시간 순서)')
ax.set_yticks([]) # Y축 눈금은 의미가 없으므로 제거

# 범례(Legend) 수동 생성
pass_patch = plt.Line2D([0], [0], color='skyblue', lw=4, label='정상 (Pass)')
fail_patch = plt.Line2D([0], [0], color='coral', lw=4, label='불량 (Fail)')
stop_patch = plt.Line2D([0], [0], color='black', lw=2, label='정지 (Stop)')
ax.legend(handles=[pass_patch, fail_patch, stop_patch], loc='upper right')

plt.tight_layout()
plt.show()





# ============ tryshot_signal 추가 (XGboost)===================
df = train.copy()
# ----------------------------------------------------
# 불필요한 컬럼 제거 (cycles_since_stop 생성 후 working 제거)
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date','registration_time',
'working','upper_mold_temp3','lower_mold_temp3'])
# 데이터 정제 (기존과 동일)
df.loc[df['tryshot_signal'].isna(),'tryshot_signal'] = 'A'
df["mold_code"] = df["mold_code"].astype(object)
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)
# 2. 입력변수(X), 타겟변수(y) 분리
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# 4. 수치형 / 범주형 컬럼 구분
# cycles_since_stop이 자동으로 수치형 변수에 포함됩니다.
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns
# 5. 전처리 파이프라인 정의
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
    ]
)
  
# 6. XGBoost 분류기 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
# 7. 전체 파이프라인 구성
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])
# 8. 학습
print("모델 학습을 시작합니다...")
clf.fit(X_train, y_train)
# 9. 평가
y_pred = clf.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ================ tryshot 추가(베이지안) =====================
df = train.copy()
# 피처 엔지니어링 및 정제
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df['molten_volume'] = df['molten_volume'].replace(2767, np.nan)
df['pressure_x_temp'] = df['cast_pressure'] * df['molten_temp']
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=df[df['physical_strength'] == 65535].index)
df = df.drop(index=df[df['low_section_speed'] == 65535].index)
df = df.drop(index=df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=df[df['upper_mold_temp2'] == 4232].index)

# --- 2. 입력/타겟 변수 분리 및 Train/Test Split ---
X = df.drop(columns=["passorfail"])
y = df["passorfail"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. 그룹별 결측치 처리 ---
numerical_cols = X_train.select_dtypes(include=np.number).columns
grouped_medians = X_train.groupby('mold_code')[numerical_cols].median()
global_median = X_train[numerical_cols].median()
for mold_code, group_median in grouped_medians.iterrows():
    X_train.loc[X_train['mold_code'] == mold_code] = X_train.loc[X_train['mold_code'] == mold_code].fillna(group_median)
X_train.fillna(global_median, inplace=True)
for mold_code in X_test['mold_code'].unique():
    if mold_code in grouped_medians.index:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(grouped_medians.loc[mold_code])
    else:
        X_test.loc[X_test['mold_code'] == mold_code] = X_test.loc[X_test['mold_code'] == mold_code].fillna(global_median)
X_test.fillna(global_median, inplace=True)

# --- 4. 최종 모델 학습 및 평가 ---
# [수정] Optuna 탐색 결과로 얻은 최적의 하이퍼파라미터를 직접 입력
best_params = {
    'n_estimators': 591, 'max_depth': 3,
    'learning_rate': 0.06213186050221397,
    'subsample': 0.6795232137595384,
    'colsample_bytree': 0.8259315529365058,
    'gamma': 3.5790023124651724, 'reg_alpha': 3.2638154400802755,
    'reg_lambda': 1.2306483183364956
}

# 최종 모델 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_xgb = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

# 최종 파이프라인 구성
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(include="object").columns
final_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("classifier", final_xgb)])

# 전체 학습 데이터로 최종 모델 학습
print("최적 파라미터로 최종 모델 학습을 시작합니다...")
final_pipeline.fit(X_train, y_train)
print("학습이 완료되었습니다.")

# 테스트 데이터로 최종 평가
y_pred = final_pipeline.predict(X_test)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))






# ============== 랜덤포레스트 기반 변수 영향도 측정 ================
model_df = train.copy()
# 2. 불필요한 컬럼 제거
model_df = model_df.drop(['id','line','name','time','date','registration_time',
                          'mold_name'], axis=1)
# 3. 피처(X)와 타겟(y) 분리
X = df.drop('passorfail', axis=1)
y = df['passorfail']
# 4. 데이터 타입에 따라 컬럼명 분리
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
# 5. 전처리 파이프라인 구성
# 수치형 데이터: KNN Imputer로 결측치 처리
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))
])
# 범주형 데이터: 최빈값으로 결측치 처리 후 원핫인코딩
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# ColumnTransformer를 이용해 두 파이프라인 통합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
# 6. 랜덤 포레스트 모델과 전처리 파이프라인 연결
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
# 7. 모델 학습
model_pipeline.fit(X, y)
# 8. 변수 중요도 추출
# 원핫인코딩 후의 컬럼명 가져오기
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)
# 중요도 추출
importances = model_pipeline.named_steps['classifier'].feature_importances_
# 9. 시각화
feature_importances = pd.Series(importances, index=all_feature_names)
top_features = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(x=top_features, y=top_features.index)
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
print("\n가장 중요한 변수 Top 10:")
print(top_features.head(10))







# ======================= 안정 범위 측정 ==========================
df = train[['low_section_speed','high_section_speed', 'molten_volume',
            'cast_pressure','biscuit_thickness', 'mold_code']]
import pandas as pd
import numpy as np

# 1. 분석할 수치형 컬럼 선택
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
del numerical_cols[5]

# 2. 결과를 저장할 빈 리스트 생성
normal_ranges = []

# 3. mold_code별로 그룹화하여 반복
for code, group_df in df.groupby('mold_code'):
    
    # 4. 각 수치형 컬럼에 대해 정상 범위 계산
    for col in numerical_cols:
        # 그룹별로 컬럼의 결측치를 제외하고 계산
        if group_df[col].isnull().all():
            continue

        Q1 = group_df[col].quantile(0.25)
        Q3 = group_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 5. 계산된 결과 저장
        normal_ranges.append({
            'mold_code': code,
            'feature': col,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

# 6. 리스트를 최종 데이터프레임으로 변환
normal_range_df = pd.DataFrame(normal_ranges)

# 결과 출력
print("--- mold_code별 정상 범위 계산 결과 ---")
normal_range_df






# ================ 크루스컬-윌리스 검정 ===================
df = model_df.copy()
df = df.loc[df['passorfail']==0,:]
df = df.drop(columns='passorfail', axis=1)
# 분석할 수치형 컬럼과 그룹핑 컬럼 정의
numerical_cols = df.select_dtypes(include=np.number).columns
grouping_col = 'mold_code'
mold_codes = df[grouping_col].unique()

print(f"--- Kruskal-Wallis 검정 및 효과 크기(Epsilon-squared) 분석 ---")
print("="*60)

for col in numerical_cols:
    if col == grouping_col:
        continue
    
    print(f"[{col}] 컬럼 분석")
    
    grouped_data = [df[col][df[grouping_col] == code].dropna() for code in mold_codes]

    # 크루스컬-월리스 검정
    h_stat, p_value = stats.kruskal(*grouped_data)

    # 효과 크기(Epsilon-squared) 계산
    n = sum(len(group) for group in grouped_data) # 전체 샘플 수
    epsilon_squared = h_stat / (n - 1)

    print(f"  - p-value: {p_value:.4f}")
    print(f"  - 효과 크기(ε²): {epsilon_squared:.4f}")
    
    # 효과 크기 해석
    if epsilon_squared >= 0.14:
        effect_size_interp = "큰 효과 (실질적으로 중요함)"
    elif epsilon_squared >= 0.06:
        effect_size_interp = "중간 효과 (어느정도 중요함)"
    elif epsilon_squared >= 0.01:
        effect_size_interp = "작은 효과 (차이가 미미함)"
    else:
        effect_size_interp = "매우 작은 효과 (거의 무시 가능)"

    print(f"  - 해석: {effect_size_interp}")

    if p_value >= 0.05:
         print("  - 최종 결론: 그룹 간 차이가 통계적으로 유의미하지 않음.")
    elif epsilon_squared < 0.01:
         print("  - 최종 결론: 통계적으로 유의미하지만, 효과 크기가 매우 작아 실질적 의미는 없을 가능성이 높음.")
    else:
         print("  - 최종 결론: 그룹 간에 실질적으로 의미있는 차이가 있을 가능성이 높음.")

    print("="*60)










# ================= XGBOOST 성능평가 (기본모델) ==================
df = train.copy()
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','registration_time',
                      'working','upper_mold_temp3','tryshot_signal','lower_mold_temp3'])
df["mold_code"] = df["mold_code"].astype(object)

df['date_dt'] = pd.to_datetime(df['date'], format='%H:%M:%S')
df['day'] = df['date_dt'].dt.day
df['hour'] = df['date_dt'].dt.hour.astype(object)
df['minute'] = df['date_dt'].dt.minute
df = df.drop(columns=['date','date_dt'])

df = df.drop(index=19327) # 결측 많은 값

df[df['physical_strength'] == 65535]
df = df.drop(index=[6000,11811,17598])

df[df['low_section_speed'] == 65535]
df = df.drop(index=[46546])

list(df[df['Coolant_temperature'] == 1449].index)
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index))

list(df[df['upper_mold_temp1'] == 1449].index)
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index))

list(df[df['upper_mold_temp2'] == 4232].index)
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index))

# 2. 입력변수(X), 타겟변수(y) 분리
X = df.drop(columns=["passorfail"])
y = df["passorfail"]

split_point = int(len(df) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

# 4. 수치형 / 범주형 컬럼 구분
num_cols = X_train.select_dtypes(include=["int64","int32","float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# 5. 전처리 파이프라인 정의
num_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),  # ✔️ 수치형 → KNN 기반 대치
    ("scaler", StandardScaler())             # 표준화 스케일링
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
from xgboost import XGBClassifier

# 6. XGBoost 분류기 정의
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 클래스 불균형 보정
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# 7. 전체 파이프라인 구성 (전처리 + 모델)
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])

# 8. 학습
clf.fit(X_train, y_train)

# 9. 평가
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# ===================== FP인 행 찾기 ====================
y_pred_series = pd.Series(y_pred, index=y_test.index)
fp_condition = (y_test == 0) & (y_pred_series == 1)
fp_indices = y_test[fp_condition].index

# 3. 추출된 인덱스와 개수 출력
print(f"False Positive (실제: 양품, 예측: 불량) 데이터 개수: {len(fp_indices)}")
print("\nFalse Positive 데이터의 인덱스 목록:")
print(fp_indices.tolist())

# 원본 데이터프레임(df)에서 인덱스를 사용해 실제 데이터를 조회합니다.
fp_data = train.loc[fp_indices]

fp_data.to_csv('./false positive.csv')


# ============== top5에 가장 많이 나온 변수 ====================
import shap
# 4. 변환된 데이터에 대한 SHAP 값 계산
shap_values = explainer.shap_values(fp_data_transformed_df)

# --- 5. Top 5 영향 변수 추출 및 데이터프레임 생성 ---

# 결과를 저장할 리스트 초기화
top_features_list = []

# 수정된 부분: shap_values[1] 대신 shap_values를 그대로 사용합니다.
# (XGBoost 분류기의 경우 shap_values()는 양성 클래스(1)에 대한 SHAP 값만 반환합니다)
shap_values_for_fail = shap_values

# 각 False Positive 데이터 행에 대해 반복
for i in range(len(fp_data)):
    # i번째 데이터의 원본 인덱스
    original_index = fp_data.index[i]
    
    # i번째 데이터의 SHAP 값들을 변수명과 함께 Series로 변환
    shap_series = pd.Series(shap_values_for_fail[i, :], index=all_feature_names)
    
    # SHAP 값의 '절대값'을 기준으로 내림차순 정렬하여 영향이 큰 변수를 찾음
    top_5 = shap_series.abs().sort_values(ascending=False).head(5)
    
    # Top 5 변수명과 원본 인덱스를 리스트에 추가
    top_features_list.append([original_index] + top_5.index.tolist())

# 리스트를 데이터프레임으로 변환
top_features_df = pd.DataFrame(top_features_list, 
                               columns=['Original_Index', 
                                        'Top1_Feature', 
                                        'Top2_Feature', 
                                        'Top3_Feature', 
                                        'Top4_Feature', 
                                        'Top5_Feature'])

# --- 결과 출력 ---
print("--- False Positive 예측별 Top 5 영향 변수 ---")
print(top_features_df.head())

# 1. Top1 ~ Top5 변수 컬럼들을 하나의 긴 Series로 변환
feature_cols = ['Top1_Feature', 'Top2_Feature', 'Top3_Feature', 'Top4_Feature', 'Top5_Feature']
# .melt()를 사용해 모든 변수 이름을 'value' 컬럼 아래로 모읍니다.
all_top_features = top_features_df[feature_cols].melt(value_name='feature')['feature']

# 2. 각 변수가 Top 5에 등장한 횟수를 계산
feature_counts = all_top_features.value_counts()

# 3. 결과 출력
print("--- False Positive Top 5 원인 변수 등장 빈도 ---")
print(feature_counts)

# 4. 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_counts, y=feature_counts.index)
plt.title('False Positive의 주요 원인 변수 빈도', fontsize=16)
plt.xlabel('Top 5에 포함된 횟수', fontsize=12)
plt.ylabel('변수명', fontsize=12)
plt.tight_layout()
plt.show()


# ================= TN와 FP 분포 비교 ==================
# --- 1. 데이터 준비 (이전 단계에서 생성된 변수들을 사용) ---
# train: 원본 데이터프레임
# y_test: 테스트 데이터의 실제 값
# y_pred: 모델의 예측 값

# 예측 결과를 y_test와 인덱스를 맞춘 Series로 변환
y_pred_series = pd.Series(y_pred, index=y_test.index)

# --- 2. 그룹 정의 ---
# True Negative (TN): 실제 양품(0) -> 예측 양품(0)
tn_condition = (y_test == 0) & (y_pred_series == 0)
tn_indices = y_test[tn_condition].index

# False Positive (FP): 실제 양품(0) -> 예측 불량(1)
fp_condition = (y_test == 0) & (y_pred_series == 1)
fp_indices = y_test[fp_condition].index

# --- 3. 비교용 데이터프레임 생성 ---
# 각 그룹에 해당하는 원본 데이터를 가져오기
tn_data = train.loc[tn_indices].copy()
fp_data = train.loc[fp_indices].copy()

# 어떤 그룹인지 표시하는 컬럼 추가
tn_data['result'] = 'Correct (True Negative)'
fp_data['result'] = 'Incorrect (False Positive)'

# 두 그룹의 데이터를 하나로 합치기
comparison_df = pd.concat([tn_data, fp_data])


# --- 4. 분포 시각화 ---
# 사용자께서 지정한 7개의 변수
features_to_compare = [
    'cast_pressure', 'sleeve_temperature', 'upper_mold_temp1', 
    'lower_mold_temp2', 'biscuit_thickness', 'count', 'low_section_speed'
]

print("--- TN 그룹과 FP 그룹의 변수 분포 비교 ---")

for feature in features_to_compare:
    plt.figure(figsize=(10, 6))
    
    # KDE 플롯으로 두 그룹의 분포를 겹쳐서 그리기
    sns.kdeplot(data=comparison_df, x=feature, hue='result', 
                fill=True, common_norm=False)
    
    plt.title(f"'{feature}'의 그룹별 분포 비교", fontsize=15)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()




# =============== 변수중요도 (순열중요도 기반) =====================
from sklearn.inspection import permutation_importance

# 불필요한 컬럼 및 결측치가 많은 컬럼 제거
df = train.copy()
condition = (df['molten_volume'].notna()) & (df['heating_furnace'].isna())
df.loc[condition, 'heating_furnace'] = 'C'
df.loc[df['tryshot_signal'].isna(),'tryshot_signal'] = 'A'
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date','registration_time',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df = df.drop(index=19327, errors='ignore')
df = df.drop(index=[6000,11811,17598], errors='ignore')
df = df.drop(index=[46546], errors='ignore')
df = df.drop(index=list(df[df['Coolant_temperature'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp1'] == 1449].index), errors='ignore')
df = df.drop(index=list(df[df['upper_mold_temp2'] == 4232].index), errors='ignore')

# 범주형(object) 변수와 수치형 변수 선택
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=np.number).columns.drop('passorfail')

# 결측치 처리

imputer = KNNImputer(n_neighbors=5)
df[numerical_features] = imputer.fit_transform(df[numerical_features])

for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

# 3. 특성(X)과 타겟(y) 분리
X = df.drop('passorfail', axis=1)
y = df['passorfail']

# 4. 훈련 데이터와 테스트 데이터 분리
split_point = int(len(df) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

# 5. 모델 학습
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 6. 순열 중요도 계산
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

# 7. 결과 시각화
sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

plt.figure(figsize=(12, 8))
sns.boxplot(data=importances, orient='h', whis=10)
plt.title('Permutation Importance (Test Set)', fontsize=16)
plt.axvline(x=0, color='k', linestyle='--')
plt.xlabel('Importance (Performance Drop)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 중요도 높은 변수 출력
importances_df = pd.DataFrame({
    'feature': X.columns[sorted_importances_idx],
    'mean_importance': result.importances_mean[sorted_importances_idx],
    'std_importance': result.importances_std[sorted_importances_idx]
}).sort_values(by='mean_importance', ascending=False)

print("\n--- 변수 중요도 (높은 순) ---")
print(importances_df)







df = train.copy()
df = df.loc[df['tryshot_signal']=='D',]
df = df.drop(columns=['id','line','name','mold_name','emergency_stop','time','date',
                      'working','upper_mold_temp3','lower_mold_temp3'])
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['hour'] = df['registration_time'].dt.hour
df = df.drop('registration_time', axis=1)
# 그래프 스타일 설정
sns.set(style="whitegrid")

# 데이터프레임의 모든 칼럼에 대해 반복
for column in df.columns:
    plt.figure(figsize=(10, 6))
    plt.title(f'Distribution of {column}', fontsize=15)

    # 칼럼의 데이터 타입이 수치형(int, float)인 경우 히스토그램 생성
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column], kde=True, bins=30)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
    # 칼럼의 데이터 타입이 범주형(object)이거나 고유값이 적은 경우 막대그래프 생성
    else:
        # 결측값(NaN)을 제외하고 상위 10개 카테고리만 시각화 (너무 많은 경우 대비)
        if df[column].nunique() > 10:
            top_10_categories = df[column].value_counts().nlargest(10).index
            sns.countplot(y=column, data=df[df[column].isin(top_10_categories)], order=top_10_categories)
            plt.title(f'Top 10 Categories in {column}', fontsize=15)
        else:
            sns.countplot(y=column, data=df, order = df[column].value_counts().index)

        plt.xlabel('Count', fontsize=12)
        plt.ylabel(column, fontsize=12)

    plt.tight_layout()
    plt.show()


