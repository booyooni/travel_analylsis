from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="하나여행 리뷰 개선 대시보드", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hanatour_reviews.csv"
CHART_DIR = BASE_DIR / "charts"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [re.sub(r"\s+", "", str(c)) for c in normalized.columns]
    return normalized


def _extract_schedule(product_name: str) -> str:
    if not isinstance(product_name, str):
        return "미식별"
    m = re.search(r"(\d+)\s*박\s*(\d+)\s*일", product_name)
    if m:
        return f"{m.group(1)}박{m.group(2)}일"
    m = re.search(r"(\d+)\s*일", product_name)
    if m:
        return f"{m.group(1)}일"
    return "미식별"


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace(".", "-", regex=False), errors="coerce")


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_reviews() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    df = _normalize_columns(df)

    required = ["상품코드", "대상도시", "상품명", "리뷰ID", "평점", "내용", "작성일", "리뷰요약"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        return pd.DataFrame()

    df["평점"] = pd.to_numeric(df["평점"], errors="coerce")
    df["작성일_dt"] = _parse_date(df["작성일"])
    df["여행일정"] = df["상품명"].map(_extract_schedule)
    df["작성월"] = df["작성일_dt"].dt.to_period("M").dt.to_timestamp()
    df["분석텍스트"] = (df["리뷰요약"].fillna("").astype(str) + " " + df["내용"].fillna("").astype(str)).map(_clean_text)
    return df


def top_summary_keywords(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in series.dropna().astype(str):
        parts = [p.strip() for p in text.split(",") if p.strip()]
        counter.update(parts)
    items = counter.most_common(top_n)
    return pd.DataFrame(items, columns=["키워드", "언급수"])


def tfidf_pos_neg_terms(df: pd.DataFrame, top_n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df[df["평점"].notna()].copy()
    labeled = labeled[(labeled["평점"] >= 4.5) | (labeled["평점"] <= 3.0)].copy()

    if labeled.empty:
        return pd.DataFrame(columns=["키워드", "점수"]), pd.DataFrame(columns=["키워드", "점수"])

    labeled["감성"] = labeled["평점"].map(lambda x: "긍정" if x >= 4.5 else "부정")
    labeled = labeled[labeled["분석텍스트"].str.len() > 0].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["키워드", "점수"]), pd.DataFrame(columns=["키워드", "점수"])

    vec = TfidfVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2), max_features=7000)
    X = vec.fit_transform(labeled["분석텍스트"])
    terms = vec.get_feature_names_out()

    pos_idx = labeled["감성"].values == "긍정"
    neg_idx = labeled["감성"].values == "부정"
    if not pos_idx.any() or not neg_idx.any():
        return pd.DataFrame(columns=["키워드", "점수"]), pd.DataFrame(columns=["키워드", "점수"])

    pos_mean = X[pos_idx].mean(axis=0).A1
    neg_mean = X[neg_idx].mean(axis=0).A1
    delta = pos_mean - neg_mean

    pos_order = delta.argsort()[::-1][:top_n]
    neg_order = delta.argsort()[:top_n]

    pos_df = pd.DataFrame(
        [(terms[i], float(delta[i])) for i in pos_order if delta[i] > 0],
        columns=["키워드", "점수"],
    )
    neg_df = pd.DataFrame(
        [(terms[i], float(-delta[i])) for i in neg_order if delta[i] < 0],
        columns=["키워드", "점수"],
    )
    return pos_df, neg_df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("필터")

    cities = sorted(df["대상도시"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect("도시", options=cities, default=cities)

    min_rating = float(df["평점"].min()) if df["평점"].notna().any() else 1.0
    max_rating = float(df["평점"].max()) if df["평점"].notna().any() else 5.0
    rating_range = st.sidebar.slider("평점 범위", min_value=1.0, max_value=5.0, value=(min_rating, max_rating), step=0.1)

    min_date = df["작성일_dt"].min()
    max_date = df["작성일_dt"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input("작성일 범위", value=(min_date.date(), max_date.date()))
    else:
        date_range = None

    filtered = df[df["대상도시"].isin(selected_cities)].copy()
    filtered = filtered[(filtered["평점"] >= rating_range[0]) & (filtered["평점"] <= rating_range[1])]

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0])
        end_ts = pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["작성일_dt"] >= start_ts) & (filtered["작성일_dt"] <= end_ts)]

    return filtered


def compute_code_risk_scores(
    df: pd.DataFrame,
    w_scale: float = 0.4,
    w_low_rating: float = 0.4,
    w_recency: float = 0.2,
) -> pd.DataFrame:
    scored = df.copy()
    scored = scored[scored["평점"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()

    latest_date = scored["작성일_dt"].max()
    if pd.isna(latest_date):
        scored["recent_weight"] = 1.0
    else:
        days_diff = (latest_date - scored["작성일_dt"]).dt.days.fillna(365)
        scored["recent_weight"] = 1.0 / (1.0 + (days_diff / 90.0))

    grouped = (
        scored.groupby(["대상도시", "상품코드"])
        .agg(
            리뷰수=("리뷰ID", "count"),
            평균평점=("평점", "mean"),
            저평점비중=("평점", lambda s: float((s <= 3.0).mean())),
            최근성가중치=("recent_weight", "mean"),
            대표상품명=("상품명", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .reset_index()
    )

    if grouped.empty:
        return grouped

    grouped["규모점수"] = grouped["리뷰수"] / max(1, grouped["리뷰수"].max())
    grouped["저평점점수"] = grouped["저평점비중"]
    grouped["최근성점수"] = grouped["최근성가중치"]

    weight_sum = max(1e-9, (w_scale + w_low_rating + w_recency))
    ws = w_scale / weight_sum
    wl = w_low_rating / weight_sum
    wr = w_recency / weight_sum

    grouped["리스크점수"] = (
        ws * grouped["규모점수"]
        + wl * grouped["저평점점수"]
        + wr * grouped["최근성점수"]
    ) * 100
    grouped = grouped.sort_values("리스크점수", ascending=False)
    return grouped


def render_kpis(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    total_reviews = len(df)
    city_count = int(df["대상도시"].nunique()) if not df.empty else 0
    code_count = int(df["상품코드"].nunique()) if not df.empty else 0
    avg_rating = float(df["평점"].mean()) if df["평점"].notna().any() else 0.0
    low_rating_ratio = float((df["평점"] <= 3.0).mean()) if df["평점"].notna().any() else 0.0

    c1.metric("리뷰수", f"{total_reviews:,}")
    c2.metric("도시수", f"{city_count}")
    c3.metric("여행코드수", f"{code_count}")
    c4.metric("평균평점", f"{avg_rating:.2f}")
    c5.metric("저평점비중", f"{low_rating_ratio:.1%}")


def render_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("도시별 현황")
    city_stats = (
        df.groupby("대상도시")
        .agg(
            리뷰수=("리뷰ID", "count"),
            평균평점=("평점", "mean"),
            저평점비중=("평점", lambda s: float((s <= 3.0).mean())),
        )
        .sort_values("리뷰수", ascending=False)
        .reset_index()
    )
    if city_stats.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    st.bar_chart(city_stats.set_index("대상도시")[["리뷰수"]])
    st.dataframe(city_stats, use_container_width=True)

    st.subheader("월별 리뷰량")
    monthly = (
        df.dropna(subset=["작성월"])
        .groupby(["작성월", "대상도시"])
        .size()
        .reset_index(name="리뷰수")
    )
    if not monthly.empty:
        pivot = monthly.pivot(index="작성월", columns="대상도시", values="리뷰수").fillna(0)
        st.line_chart(pivot)


def render_schedule_code_tab(df: pd.DataFrame) -> None:
    st.subheader("여행일정 분포")
    sched = df["여행일정"].fillna("미식별").value_counts().head(10).rename_axis("여행일정").reset_index(name="리뷰수")
    if not sched.empty:
        st.bar_chart(sched.set_index("여행일정")[["리뷰수"]])
        st.dataframe(sched, use_container_width=True)

    st.subheader("상위 여행코드")
    code_stats = (
        df.groupby("상품코드")
        .agg(
            리뷰수=("리뷰ID", "count"),
            평균평점=("평점", "mean"),
            대표상품명=("상품명", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .sort_values(["리뷰수", "평균평점"], ascending=[False, False])
        .head(20)
        .reset_index()
    )
    if not code_stats.empty:
        st.bar_chart(code_stats.set_index("상품코드")[["리뷰수"]])
        st.dataframe(code_stats, use_container_width=True)


def render_keyword_sentiment_tab(df: pd.DataFrame) -> None:
    st.subheader("리뷰요약 키워드")
    top_kw = top_summary_keywords(df["리뷰요약"], top_n=15)
    if not top_kw.empty:
        st.bar_chart(top_kw.set_index("키워드")[["언급수"]])
        st.dataframe(top_kw, use_container_width=True)

    st.subheader("TF-IDF 긍정/부정 키워드")
    pos_df, neg_df = tfidf_pos_neg_terms(df, top_n=12)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 긍정 대표")
        if not pos_df.empty:
            st.bar_chart(pos_df.set_index("키워드")[["점수"]])
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("긍정 키워드를 계산하기 위한 데이터가 부족합니다.")

    with col2:
        st.markdown("#### 부정 대표")
        if not neg_df.empty:
            st.bar_chart(neg_df.set_index("키워드")[["점수"]])
            st.dataframe(neg_df, use_container_width=True)
        else:
            st.info("부정 키워드를 계산하기 위한 데이터가 부족합니다.")


def render_chart_gallery_tab() -> None:
    st.subheader("개선포인트 시각화 갤러리")

    chart_files = [
        ("도시별 개선 우선순위 버블", CHART_DIR / "city_priority_bubble.png"),
        ("도시별 부정 키워드 히트맵", CHART_DIR / "city_negative_keyword_heatmap.png"),
        ("월별 리뷰량/저평점 비중 추이", CHART_DIR / "monthly_low_rating_trend.png"),
    ]

    for title, path in chart_files:
        st.markdown(f"#### {title}")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"차트를 찾을 수 없습니다: {path.name}")


def render_risk_priority_tab(df: pd.DataFrame) -> None:
    st.subheader("여행코드 리스크 우선순위")
    st.markdown("#### 리스크 가중치 시뮬레이터")
    preset_options = ["균형형", "운영중심형", "CX중심형", "매출영향형", "커스텀"]
    preset = st.selectbox("프리셋", options=preset_options, index=0)

    preset_map = {
        "균형형": (0.4, 0.4, 0.2),
        "운영중심형": (0.2, 0.6, 0.2),
        "CX중심형": (0.2, 0.7, 0.1),
        "매출영향형": (0.6, 0.25, 0.15),
    }

    if preset == "커스텀":
        wc1, wc2, wc3 = st.columns(3)
        w_scale = wc1.slider("규모(리뷰수) 가중치", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        w_low = wc2.slider("저평점비중 가중치", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        w_recent = wc3.slider("최근성 가중치", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    else:
        w_scale, w_low, w_recent = preset_map[preset]
        st.caption(
            f"선택된 프리셋 가중치 - 규모: {w_scale:.2f}, 저평점: {w_low:.2f}, 최근성: {w_recent:.2f}"
        )

    if (w_scale + w_low + w_recent) == 0:
        st.warning("가중치 합이 0입니다. 하나 이상 값을 올려주세요.")
        return

    risk_df = compute_code_risk_scores(df, w_scale=w_scale, w_low_rating=w_low, w_recency=w_recent)

    if risk_df.empty:
        st.info("리스크 점수 계산을 위한 데이터가 부족합니다.")
        return

    min_reviews = st.slider("최소 리뷰수", min_value=10, max_value=300, value=30, step=10)
    city_options = ["전체"] + sorted(risk_df["대상도시"].dropna().unique().tolist())
    selected_city = st.selectbox("도시 선택", options=city_options)

    view_df = risk_df[risk_df["리뷰수"] >= min_reviews].copy()
    if selected_city != "전체":
        view_df = view_df[view_df["대상도시"] == selected_city].copy()

    if view_df.empty:
        st.warning("현재 조건에 맞는 여행코드가 없습니다.")
        return

    topn = st.slider("표시 개수", min_value=10, max_value=50, value=20, step=5)
    top_df = view_df.head(topn).copy()

    st.markdown("#### 리스크 우선순위 Top")
    show_cols = ["대상도시", "상품코드", "리뷰수", "평균평점", "저평점비중", "최근성가중치", "리스크점수", "대표상품명"]
    st.dataframe(top_df[show_cols], use_container_width=True)

    st.markdown("#### 리스크 매트릭스 (여행코드 산점도)")
    scatter_df = view_df.copy()
    scatter_df["버블크기"] = (scatter_df["리뷰수"] / max(1, scatter_df["리뷰수"].max()) * 250) + 30
    st.scatter_chart(
        data=scatter_df,
        x="평균평점",
        y="저평점비중",
        size="버블크기",
        color="대상도시",
    )

    st.caption(
        "리스크점수 = (w1×규모점수 + w2×저평점점수 + w3×최근성점수) × 100 (가중치는 자동 정규화)"
    )


def main() -> None:
    st.title("하나여행 리뷰 기반 상품개선 대시보드")
    st.caption("목표: 리뷰 데이터로 개선 우선순위를 도출하고 개선된 여행상품 설계에 활용")

    raw = load_reviews()
    if raw.empty:
        st.error("리뷰 데이터를 불러오지 못했습니다. 파일 경로와 컬럼을 확인해주세요.")
        return

    filtered = filter_data(raw)
    if filtered.empty:
        st.warning("현재 필터 조건에 맞는 리뷰가 없습니다.")
        return

    render_kpis(filtered)

    tabs = st.tabs(["요약", "일정/코드", "키워드/감성", "리스크 우선순위", "차트 갤러리"])
    with tabs[0]:
        render_overview_tab(filtered)
    with tabs[1]:
        render_schedule_code_tab(filtered)
    with tabs[2]:
        render_keyword_sentiment_tab(filtered)
    with tabs[3]:
        render_risk_priority_tab(filtered)
    with tabs[4]:
        render_chart_gallery_tab()

    st.markdown("---")
    st.markdown("### 운영 팁")
    st.markdown("- 저평점비중이 높은 도시/코드부터 개선 액션을 우선 적용하세요.")
    st.markdown("- TF-IDF 부정 키워드는 운영 품질 이슈(가이드/일정/식사/숙소) 점검 체크리스트로 연결하세요.")


if __name__ == "__main__":
    main()
