import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 예측기")
st.write("과거 서울 기온 데이터를 바탕으로 선형 회귀 모델을 생성하여 미래 기온을 예측합니다.")

# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 데이터 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 필수 컬럼 결측치 제거
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측일수 및 평균기온 계산
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 2025년 이하 & 관측일수 300일 이상 조건 필터링
    filtered_yearly = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    
    return filtered_yearly

df_yearly = load_and_preprocess_data()

# 분석 기초 정보 산출
count_years = len(df_yearly)
min_year = int(df_yearly["연도"].min())
max_year = int(df_yearly["연도"].max())

# 2. 선형 회귀 계산 및 상관계수
X = df_yearly["연도"].values
Y = df_yearly["연평균기온"].values

# 1차 회귀 직선 기울기(slope) 및 절편(intercept)
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr = np.corrcoef(X, Y)[0, 1]

# 3. 사이드바 및 레이아웃 설정
st.sidebar.header("⚙️ 예측 연도 선택")
selected_year = st.sidebar.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2030)

# 예측 기온 계산
predicted_temp = slope * selected_year + intercept

# 상단 메트릭 및 정보 표시
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"📅 {selected_year}년 예상 평균기온", value=f"{predicted_temp:.2f} °C")
with col2:
    st.metric(label="📊 상관계수 (연도 vs 연평균기온)", value=f"{corr:.4f}")
with col3:
    st.metric(label="🔢 분석 대상 연도 수", value=f"{count_years}개 해")

st.info(f"💡 **직선 생성 기준 정보:** 총 **{count_years}개** 해의 데이터 사용 (시작 연도: **{min_year}년** / 끝 연도: **{max_year}년**)")

# 4. Plotly 산점도 및 회귀 직선 시각화
fig = go.Figure()

# 실제 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="연평균기온 (실제)",
    marker=dict(color="blue", size=6, opacity=0.7)
))

# 회귀 직선 (1900년 ~ 2100년 전체 구간)
x_range = np.linspace(1900, 2100, 201)
y_range = slope * x_range + intercept

fig.add_trace(go.Scatter(
    x=x_range,
    y=y_range,
    mode="lines",
    name="선형 회귀 직선",
    line=dict(color="red", width=2)
))

# 선택된 연도의 예측점 강조
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers+text",
    name=f"{selected_year}년 예측치",
    marker=dict(color="green", size=14, symbol="star"),
    text=[f"{predicted_temp:.2f}°C"],
    textposition="top center"
))

fig.update_layout(
    title="서울 연도별 연평균기온 추이 및 회귀 분석",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
