import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 예측기")
st.write("과거 서울 기온 데이터를 바탕으로 선형 회귀 모델을 생성하여 미래 기온을 예측하고, 상승 추세를 비교합니다.")

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

# 2. 전체 기간 선형 회귀 계산
X_all = df_yearly["연도"].values
Y_all = df_yearly["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
corr_all = np.corrcoef(X_all, Y_all)[0, 1]

# 3. 최근 20년 데이터 필터링 및 선형 회귀 계산
recent_start_year = max_year - 19  # 최근 20년 (예: 2006~2025년)
df_recent = df_yearly[df_yearly["연도"] >= recent_start_year]
X_recent = df_recent["연도"].values
Y_recent = df_recent["연평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)

# 100년당 기온 상승 폭 계산 (°C / 100년)
rate_100_all = slope_all * 100
rate_100_recent = slope_recent * 100

# 4. 사이드바 및 레이아웃 설정
st.sidebar.header("⚙️ 예측 연도 선택")
selected_year = st.sidebar.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2030)

# 전체 기간 회귀 직선 기준 예측 기온 계산
predicted_temp = slope_all * selected_year + intercept_all

# --- 화면 표시: 메트릭 강조 ---
st.subheader("🔥 기온 변화율 및 예측 정보")

# 100년당 상승폭 비교 (메인 강조)
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label=f"📈 전체 기간 100년당 상승 폭 ({min_year}~{max_year}년)",
        value=f"{rate_100_all:+.2f} °C / 100년",
        help="전체 수집된 데이터를 기준으로 100년 동안 변하는 기온의 양입니다."
    )
with col2:
    st.metric(
        label=f"🚀 최근 20년 100년당 상승 폭 ({recent_start_year}~{max_year}년)",
        value=f"{rate_100_recent:+.2f} °C / 100년",
        delta=f"{rate_100_recent - rate_100_all:+.2f} °C (전체 대비 가속도)",
        help="최근 20년 데이터를 기준으로 추산한 100년 단위 기온 변화 추세입니다."
    )

st.markdown("---")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(label=f"📅 {selected_year}년 예상 평균기온 (전체 추세 기준)", value=f"{predicted_temp:.2f} °C")
with col_b:
    st.metric(label="📊 전체 기간 상관계수", value=f"{corr_all:.4f}")
with col_c:
    st.metric(label="🔢 분석 대상 연도 수", value=f"{count_years}개 해")

st.info(f"💡 **직선 생성 기준 정보:** 총 **{count_years}개** 해의 데이터 사용 (시작 연도: **{min_year}년** / 끝 연도: **{max_year}년**)")

# 5. Plotly 시각화 (두 개의 회귀선 포함)
fig = go.Figure()

# 실제 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="연평균기온 (실제)",
    marker=dict(color="blue", size=6, opacity=0.6)
))

x_range = np.linspace(1900, 2100, 201)

# 1) 전체 기간 회귀 직선
y_range_all = slope_all * x_range + intercept_all
fig.add_trace(go.Scatter(
    x=x_range,
    y=y_range_all,
    mode="lines",
    name=f"전체 기간 회귀선 (+{rate_100_all:.2f}°C/100년)",
    line=dict(color="red", width=2)
))

# 2) 최근 20년 회귀 직선
y_range_recent = slope_recent * x_range + intercept_recent
fig.add_trace(go.Scatter(
    x=x_range,
    y=y_range_recent,
    mode="lines",
    name=f"최근 20년 회귀선 (+{rate_100_recent:.2f}°C/100년)",
    line=dict(color="orange", width=2, dash="dash")
))

# 선택된 연도의 예측점 강조 (전체 기준)
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
    title="서울 연도별 연평균기온 추이 및 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
