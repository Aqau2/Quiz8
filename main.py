# 필요한 라이브러리들을 임포트합니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# CSV 파일을 DataFrame으로 읽어옵니다.
df = pd.read_csv("smoke_detection_iot.csv")

# DataFrame의 첫 부분을 출력하여 데이터를 확인합니다.
df.head()

# DataFrame의 열(컬럼) 이름을 확인합니다.
df.columns

# DataFrame의 행과 열의 개수를 확인합니다.
df.shape

# 'Unnamed: 0' 열을 삭제합니다.
df.drop(columns=['Unnamed: 0'], inplace=True)

# 수치형 데이터에 대한 기술 통계 정보를 출력합니다.
df.describe()

# 각 열에 대해 결측치의 개수를 확인합니다.
df.isnull().sum()

# 중복된 행의 개수를 확인합니다.
df.duplicated().sum()

# DataFrame의 정보(열 데이터 타입, 비어있지 않은 값의 개수 등)를 출력합니다.
df.info()

# 각 열에 대해 고유한 값들의 개수와 실제 값들을 출력하는 함수를 정의합니다.
def uniquecounts(df):
    for x in df.columns:
        print(x, len(df[x].unique()), df[x].unique())

# 정의한 함수를 이용하여 DataFrame의 고유한 값들을 출력합니다.
uniquecounts(df)

# DataFrame의 마지막 부분을 출력하여 데이터를 확인합니다.
df.tail()

# 각 열들 간의 상관 관계를 계산합니다.
df.corr()

# 'TVOC[ppb]' 열의 각 값들의 개수를 확인합니다.
df['TVOC[ppb]'].value_counts()

# 각 특성(feature)에 대해 'Fire Alarm'에 따른 분포를 바이올린 플롯으로 시각화합니다.
def violinplot(df):
    for x in df.columns:
        sns.violinplot(x='Fire Alarm',y=x,data=df)
        plt.savefig(f'test1_{x}.png')  # 각 그래프를 다른 파일로 저장
        plt.show()

violinplot(df)

# 특성과 타겟 변수를 나누어줍니다.
X = df.drop('Fire Alarm', axis=1)  # 특성
y = df['Fire Alarm']  # 타겟 변수

# 데이터를 학습용과 평가용으로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier 모델을 초기화하고 학습합니다.
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 학습된 모델을 사용하여 평가용 데이터를 예측합니다.
y_pred = model.predict(X_test)

# 모델의 정확도를 계산합니다.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 모델의 정확도를 막대 그래프로 시각화합니다.
plt.figure(figsize=(6, 4))
sns.barplot(x=['Accuracy'], y=[accuracy])
plt.ylim(0, 1)  # y축 범위 설정 (0에서 1 사이)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.show()