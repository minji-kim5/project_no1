import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('./train.csv')
train.head()
train.info()
train.head()

train = train.rename(columns={
    "time": "date",    # 날짜 데이터 → date
    "date": "time"     # 시간 데이터 → time
})

train.drop(train[train['upper_mold_temp1'] >= 1400].index, inplace=True)  #원본데이터에서도 삭제
train.drop(train[train['upper_mold_temp2'] >= 4000].index, inplace=True)


df=train[['upper_mold_temp1','upper_mold_temp2','lower_mold_temp1','lower_mold_temp2']]
df.info()
df.isnull().sum()
df.describe()


df['lower_mold_temp1'].describe()
df[df['upper_mold_temp1'].isnull()]
df=df.dropna()

# 결측 행 제거 후
df['upper_mold_temp2'].describe()

num_cols=df.select_dtypes(include='number').columns.to_list()
cat_cols=df.select_dtypes(include='object').columns.to_list()

# -----------------------------------------------------

# 상관관계 분석 -> 불량에 영향 주는 파라미터 추출
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr

# 불량 여부 컬럼명
features = train.drop(columns='passorfail').select_dtypes(include='number').columns

# 1. Point-Biserial Correlation (이진라벨과 연속형 변수 상관)
train_clean = train[features].copy()

train_clean = train_clean.dropna()
train_clean['passorfail'] = train['passorfail']
correlations = {}
for col in features:
    corr, _ = pointbiserialr(train_clean['passorfail'], train_clean[col])
    correlations[col] = corr

# -------------------------
# 상관계수 시각화
corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
print("불량과의 상관관계 (절대값 기준 정렬):")
print(corr_series)

sns.barplot(x=corr_series.values, y=corr_series.index)
plt.title('Point-Biserial Correlation with Defect')
plt.xlabel('Correlation')
plt.ylabel('Feature')
plt.show()

# ----------------------
# 2. 랜덤포레스트로 피처 중요도
X = train[features]
y = train['passorfail']
X = X.fillna(X.mean())
# 스케일링 (선택사항)
X_scaled = StandardScaler().fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("\n랜덤포레스트 기반 중요도:")
print(importances)

sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 랜덤포레스트 기반 중요도: 0.05 이상이면 관련성 높음!!
# -----------------------------------------
# 박스 플롯으로 각 변수의 분포 비교

plt.figure(figsize=(15, 10))
# 비교할 변수들
temp_cols = ['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
             'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3']

for i, col in enumerate(temp_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='passorfail', y=col, data=train)
    plt.title(f'{col} Distribution by Pass/Fail')
    plt.xlabel('Pass/Fail (0: Normal, 1: Defective)')
    plt.ylabel(f'{col} Temperature')

plt.tight_layout()
plt.suptitle('Temperature Distribution by Pass/Fail', fontsize=16, y=1.03)
plt.show()

df[df['upper_mold_temp1']>=1400]    #일단 삭제!!
df[df['upper_mold_temp2']>=4000]  # 이건 행 삭제함

# ===================================
# 이상치 -> mold_code와 관련 있는지 확인
train_pass=train[train['passorfail']==0].copy()
train_fail=train[train['passorfail']==1].copy()

train.loc[train['upper_mold_temp3']<=300,'mold_code'].value_counts()  #전부 8573
train.loc[train['lower_mold_temp1']>=300,'mold_code'].value_counts()   #전부 8722
train.loc[train['lower_mold_temp2']>=290,'mold_code'].value_counts()   #3가지 모델 존재
train_pass.loc[train['lower_mold_temp2']<=100,'mold_code'].value_counts()   # 4가지 모델 존재

# 극단치 제거
train[train['upper_mold_temp1']>=1400]   #1개/불량
train.drop(train[train['upper_mold_temp1'] >= 1400].index, inplace=True)  #원본데이터에서도 삭제
train[train['upper_mold_temp2']>=400]    #1개/불량
train.drop(train[train['upper_mold_temp2'] >= 400].index, inplace=True)
train[train['lower_mold_temp3']>=60000]  #1개/양품
train.drop(train[train['lower_mold_temp3'] >= 60000].index, inplace=True)


# ------------------------------------------------------
# [mold code별 box plot]
import seaborn as sns
import matplotlib.pyplot as plt

# mold_code별로 반복
unique_molds = train['mold_code'].unique()

# temp_cols는 기존 코드에서 정의한 것 사용
temp_cols = ['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
             'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3']

# mold_code별 boxplot 시각화
for mold in unique_molds:
    mold_df = train[train['mold_code'] == mold]
    
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(temp_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='passorfail', y=col, data=mold_df)
        plt.title(f'{col} by Pass/Fail (Mold Code: {mold})')
        plt.xlabel('Pass/Fail (0: Normal, 1: Defective)')
        plt.ylabel('Temperature')

    plt.suptitle(f'Temperature Distribution by Pass/Fail - Mold Code: {mold}', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

# [mold code 별 히스토그램]
# => 각 모델별 합격 스펙 구하기
import seaborn as sns
import matplotlib.pyplot as plt

train_pass.columns = train_pass.columns.str.strip()
# mold_code를 문자열로 변환
train_pass['mold_code'] = train_pass['mold_code'].astype(str)

# 시각화 대상 온도 변수
temp_cols = ['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
             'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3']
palette = sns.color_palette('Set1', n_colors=train_pass['mold_code'].nunique())
# 히스토그램 그리기
for temp_col in temp_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=train_pass,
        x=temp_col,
        hue='mold_code',
        element='step',
        stat='count',
        common_norm=False,
        palette=palette,
        linewidth=1.5
    )

    mold_codes = train_pass['mold_code'].unique()
    for mold_code in mold_codes:
        temp_values = train_pass[train_pass['mold_code'] == mold_code][temp_col]
        min_temp = temp_values.min()
        max_temp = temp_values.max()
        print(f"mold_code: {mold_code}, {temp_col} range: ({min_temp:.2f}, {max_temp:.2f})")

    plt.title(f'Distribution of {temp_col} by Mold Code')
    plt.xlabel('Temperature')
    plt.ylabel('count')
    plt.legend(title='Mold Code',
               bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.tight_layout()
    plt.show()

# lower_mold_temp3 8722 모델 이상치
train_pass[train_pass['lower_mold_temp3']==65503]
train_pass.drop(train_pass[train_pass['lower_mold_temp3']==65503].index, inplace=True)  #원본데이터에서도 삭제

temp_col = 'lower_mold_temp3'
mold_code = '8722'

temp_values = train_pass[train_pass['mold_code'] == mold_code][temp_col]
min_temp = temp_values.min()
max_temp = temp_values.max()
print(f"mold_code: {mold_code}, {temp_col} range: ({min_temp:.2f}, {max_temp:.2f})")


# 이상치 제거 후 8722 모델의 lower_temp3 범위
# lower_temp3
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # 예시 구간 설정
labels = ['0-50','50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500']

train['temp_bin'] = pd.cut(train['lower_mold_temp3'], bins=bins, labels=labels, right=False)

# 각 구간에 대해 Pass와 Fail의 개수 계산
pivot_table = train.groupby(['temp_bin', 'passorfail', 'mold_code']).size().unstack(fill_value=0)
# 결과 출력
pivot_table

train.loc[train['upper_mold_temp3']!=1449,'passorfail']
train['upper_mold_temp3'].isnull().sum()
train['lower_mold_temp3'].isnull().sum()

train['upper_mold_temp1'].isnull().sum()
train['upper_mold_temp2'].isnull().sum()
train['lower_mold_temp1'].isnull().sum()
train['lower_mold_temp2'].isnull().sum()
train.loc[train['lower_mold_temp3']>=1400,'passorfail'].value_counts()



# 각 mold 별로 온도 합격 범위 정하고 상한선, 하한선 그린 다음 시간 순서대로 보면서 합격한 데이터들이 범위 안에 실제로 들어오는지 확인
# ========================시계열 분석
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('./train.csv')

df.drop(df[df['upper_mold_temp1'] >= 1400].index, inplace=True)  #원본데이터에서도 삭제
df.drop(df[df['upper_mold_temp2'] >= 4000].index, inplace=True)


df['timestamp'] = pd.to_datetime(df['registration_time'])
df = df.sort_values('timestamp')

# 그릴 변수 리스트
temp_columns = ['upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2']

# moldcode 고유값과 colormap 설정
unique_moldcodes = df['mold_code'].unique()
color_list = [
    '#FF7F0E',  # 주황
    '#FFD700',  # 노랑
    '#2CA02C',  # 초록
    '#1F77B4',  # 파랑
    '#9467BD'] # 보라


for temp_col in temp_columns:
    plt.figure(figsize=(15, 5))

    # 몰드코드별 선 그리기
    for i, moldcode in enumerate(unique_moldcodes):
        subset = df[df['mold_code'] == moldcode]
        plt.plot(
            subset['timestamp'],
            subset[temp_col],
            label=f'Moldcode {moldcode}',
            color=color_list[i % len(color_list)]
        )

    if 'passorfail' in df.columns:
        ng_subset = df[df['passorfail'] == 1]
        plt.scatter(
            ng_subset['timestamp'],
            ng_subset[temp_col],
            color='red',
            label='Defect (NG)',
            marker='o',
            s=30,
            alpha=0.7
        )

    plt.title(f'{temp_col} over Time by Moldcode')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend(title='Moldcode', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 

# -----------------------------
# 합격한 애들 중 mold code 별 온도 변수 분포
mold_temp_vars = ['upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2']
# 합격한 데이터만 필터링
df_pass = df[df['passorfail'] == 0]

plt.figure(figsize=(16, 12))

for i, var in enumerate(mold_temp_vars):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='mold_code', y=var, data=df_pass)
    plt.title(f'Boxplot of {var} by mold_code (Passed only)')
    plt.xticks(rotation=45)
    plt.grid(True)

plt.tight_layout()
plt.show()


# 이를 기반으로 IQR 범위 설정 -상한선, 하한선
iqr_summary = []

# mold_code별 그룹
for mold, group in df.groupby('mold_code'):
    for var in mold_temp_vars:
        q1 = group[var].quantile(0.25)
        q3 = group[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_summary.append({
            'mold_code': mold,
            'variable': var,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'Lower Bound': round(lower_bound, 2),
            'Upper Bound': round(upper_bound, 2)
        })

# 보기 쉬운 데이터프레임으로 변환
iqr_df = pd.DataFrame(iqr_summary)

# 보기 좋게 정렬
iqr_df = iqr_df.sort_values(by=['variable', 'mold_code']).reset_index(drop=True)
iqr_df

