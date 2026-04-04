from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="동남아 3도시 리뷰 분석 대시보드", layout="wide")


BASE_DIR = Path(__file__).resolve().parent / "20260328_분석패키지" / "분석자료"
BUSINESS_DIR = BASE_DIR / "business_analysis_outputs"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "volume_summary": _safe_read_csv(BASE_DIR / "reviews_volume_summary_3cities.csv"),
        "volume_monthly": _safe_read_csv(BASE_DIR / "reviews_volume_monthly_3cities.csv"),
        "biz_kpi": _safe_read_csv(BUSINESS_DIR / "biz_city_kpis.csv"),
        "biz_monthly": _safe_read_csv(BUSINESS_DIR / "biz_monthly_city_reviews.csv"),
        "biz_theme": _safe_read_csv(BUSINESS_DIR / "biz_theme_mentions_by_city.csv"),
        "biz_product": _safe_read_csv(BUSINESS_DIR / "biz_top_products_by_reviews.csv"),
        "biz_duration": _safe_read_csv(BUSINESS_DIR / "biz_trip_duration_by_city.csv"),
        "age_cov": _safe_read_csv(BASE_DIR / "reviews_age_mentions_coverage_3cities.csv"),
        "age_summary": _safe_read_csv(BASE_DIR / "reviews_age_mentions_summary_3cities.csv"),
        "age_samples": _safe_read_csv(BASE_DIR / "reviews_age_mentions_samples_3cities.csv"),
    }


def render_overview(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("전체 요약")

    volume_summary = data["volume_summary"]
    biz_kpi = data["biz_kpi"]

    total_reviews = int(volume_summary["review_count"].sum()) if not volume_summary.empty else 0
    city_count = int(volume_summary["city"].nunique()) if not volume_summary.empty else 0
    avg_positive = float(biz_kpi["positive_rate"].mean()) if not biz_kpi.empty else 0.0
    avg_trip_days = float(biz_kpi["avg_trip_days"].mean()) if not biz_kpi.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 리뷰 수", f"{total_reviews:,}")
    c2.metric("분석 도시 수", f"{city_count}")
    c3.metric("평균 긍정률", f"{avg_positive:.1%}")
    c4.metric("평균 일정(일)", f"{avg_trip_days:.2f}")

    st.markdown("#### 도시별 리뷰 점유율")
    if not volume_summary.empty:
        share_df = volume_summary[["city", "share_pct"]].copy()
        share_df = share_df.set_index("city")
        st.bar_chart(share_df)
        st.dataframe(volume_summary, use_container_width=True)
    else:
        st.info("요약 데이터가 없습니다.")


def render_volume_trend(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("리뷰량 추이")

    volume_monthly = data["volume_monthly"]
    biz_monthly = data["biz_monthly"]

    if not volume_monthly.empty:
        city_cols = [c for c in volume_monthly.columns if c not in ["year_month", "합계"]]
        selected = st.multiselect("표시할 도시", options=city_cols, default=city_cols)

        if selected:
            chart_df = volume_monthly[["year_month", *selected]].copy()
            chart_df = chart_df.set_index("year_month")
            st.line_chart(chart_df)

        st.dataframe(volume_monthly, use_container_width=True)
    else:
        st.info("월별 리뷰량 데이터가 없습니다.")

    st.markdown("#### 비즈니스 월별 데이터(영문 도시코드)")
    if not biz_monthly.empty:
        pivot = biz_monthly.pivot_table(
            index="year_month", columns="city", values="review_count", aggfunc="sum"
        ).fillna(0)
        st.line_chart(pivot)
    else:
        st.info("비즈니스 월별 데이터가 없습니다.")


def render_business_metrics(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("비즈니스 지표")

    biz_kpi = data["biz_kpi"]
    biz_theme = data["biz_theme"]
    biz_product = data["biz_product"]
    biz_duration = data["biz_duration"]

    if biz_kpi.empty:
        st.info("비즈니스 KPI 데이터가 없습니다.")
        return

    cities = biz_kpi["city"].dropna().unique().tolist()
    selected_city = st.selectbox("도시 선택", options=cities)

    city_row = biz_kpi[biz_kpi["city"] == selected_city].head(1)
    if not city_row.empty:
        r = city_row.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("리뷰 수", f"{int(r['reviews']):,}")
        c2.metric("상품 코드 수", f"{int(r['unique_products']):,}")
        c3.metric("긍정률", f"{float(r['positive_rate']):.1%}")
        c4.metric("부정률", f"{float(r['negative_rate']):.1%}")
        c5.metric("평균 일정(일)", f"{float(r['avg_trip_days']):.2f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 테마 언급률")
        theme_city = biz_theme[biz_theme["city"] == selected_city].copy()
        if not theme_city.empty:
            theme_city = theme_city.sort_values("mention_rate_pct", ascending=False)
            st.bar_chart(theme_city.set_index("theme")[["mention_rate_pct"]])
            st.dataframe(theme_city, use_container_width=True)
        else:
            st.info("해당 도시 테마 데이터가 없습니다.")

    with col_right:
        st.markdown("#### 일정(일수) 분포")
        duration_city = biz_duration[biz_duration["city"] == selected_city].copy()
        if not duration_city.empty:
            duration_city = duration_city.sort_values("trip_days")
            duration_city = duration_city.set_index("trip_days")
            st.bar_chart(duration_city[["review_count"]])
            st.dataframe(duration_city.reset_index(), use_container_width=True)
        else:
            st.info("해당 도시 일정 분포 데이터가 없습니다.")

    st.markdown("#### 상위 상품 코드(리뷰 수)")
    product_city = biz_product[biz_product["city"] == selected_city].copy()
    if not product_city.empty:
        product_city = product_city.sort_values("review_count", ascending=False).head(15)
        st.bar_chart(product_city.set_index("product_code")[["review_count"]])
        st.dataframe(product_city, use_container_width=True)
    else:
        st.info("해당 도시 상품 코드 데이터가 없습니다.")


def render_age_mentions(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("연령 언급 분석")

    age_cov = data["age_cov"]
    age_summary = data["age_summary"]
    age_samples = data["age_samples"]

    if not age_cov.empty:
        st.markdown("#### 연령 언급 커버리지")
        cov_chart = age_cov.set_index("city")[["age_mention_ratio_pct"]]
        st.bar_chart(cov_chart)
        st.dataframe(age_cov, use_container_width=True)
    else:
        st.info("연령 언급 커버리지 데이터가 없습니다.")

    if not age_summary.empty:
        st.markdown("#### 연령대 분포")
        city_options = sorted(age_summary["city"].dropna().unique().tolist())
        selected_city = st.selectbox("연령 분포 도시 선택", city_options)
        city_age = age_summary[age_summary["city"] == selected_city].copy()
        city_age = city_age.sort_values("mention_reviews", ascending=False)
        if not city_age.empty:
            st.bar_chart(city_age.set_index("age_band")[["mention_reviews"]])
            st.dataframe(city_age, use_container_width=True)
    else:
        st.info("연령 요약 데이터가 없습니다.")

    if not age_samples.empty:
        st.markdown("#### 연령 언급 샘플")
        city_opt = ["전체"] + sorted(age_samples["city"].dropna().unique().tolist())
        age_opt = ["전체"] + sorted(age_samples["age_band"].dropna().unique().tolist())
        col1, col2 = st.columns(2)
        selected_city = col1.selectbox("샘플 도시", city_opt)
        selected_age = col2.selectbox("연령대", age_opt)

        samples = age_samples.copy()
        if selected_city != "전체":
            samples = samples[samples["city"] == selected_city]
        if selected_age != "전체":
            samples = samples[samples["age_band"] == selected_age]

        st.dataframe(samples.head(200), use_container_width=True)


def main() -> None:
    st.title("동남아 3개 도시 여행 리뷰 대시보드")
    st.caption("데이터 기준: 20260328 분석 패키지")

    data = load_data()

    tabs = st.tabs(["요약", "리뷰 추이", "비즈니스 지표", "연령 언급"])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_volume_trend(data)
    with tabs[2]:
        render_business_metrics(data)
    with tabs[3]:
        render_age_mentions(data)


if __name__ == "__main__":
    main()