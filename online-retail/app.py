from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import filter_transactions, get_default_data_path, load_online_retail
from utils.formatters import build_data_basis, build_interpretation, fmt_currency, fmt_int, fmt_pct, to_csv_bytes
from utils.metrics import (
    aggregate_countries,
    aggregate_customers,
    aggregate_orders,
    aggregate_products,
    compute_kpis,
    customer_monthly_sales,
    daily_activity,
    monthly_business_metrics,
    monthly_summary,
    product_monthly_sales,
)
from utils.retention import build_retention_table
from utils.recommender import (
    build_cf_model,
    build_content_catalog,
    build_content_model,
    purchased_products_for_customer,
    recommend_products_cf,
    recommend_similar_products,
)
from utils.rfm import build_rfm_table, summarize_rfm


st.set_page_config(page_title="Online Retail Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_app_data() -> dict[str, pd.DataFrame | dict[str, object]]:
    return load_online_retail(get_default_data_path(APP_DIR))


def render_figure(fig: go.Figure, data_basis: str, interpretation: str, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.markdown(f"**데이터 기준** {data_basis}")
    st.markdown(f"**해석 방법** {interpretation}")


def render_footer(filtered_df: pd.DataFrame, source_label: str, include_cancellations: bool) -> None:
    st.markdown("---")
    st.markdown("### 데이터 기준 및 가공 방식")
    st.markdown(
        f"이 대시보드는 {source_label} 데이터셋 기준 현재 필터가 적용된 {len(filtered_df):,}행을 사용하며, "
        f"`online-retail/data/online_retail.parquet`에서 데이터를 읽은 뒤 `Sales=Quantity*UnitPrice`, `OrderDate`, `YearMonth` 파생 컬럼을 생성하여 집계한다. "
        f"취소 주문 포함 여부는 현재 `{include_cancellations}` 설정을 따르며, 고객 분석은 `CustomerID`가 존재하는 거래만 대상으로 계산한다."
    )
    st.markdown("### 대시보드 해석 가이드")
    st.markdown(
        "지표를 해석할 때는 매출, 주문 수, 구매 고객 수를 반드시 함께 비교해야 하며, 특정 구간의 급등이나 급락은 시즌성, 대량 주문, 국가 편중, 취소 주문 포함 여부에 따라 달라질 수 있다. "
        "따라서 단일 차트만 보기보다 상단 필터 조건과 다른 보조 지표를 함께 확인하면서 실제 성장과 일시적 변동을 구분해야 한다."
    )


def get_filtered_data(data: dict[str, pd.DataFrame | dict[str, object]]) -> tuple[pd.DataFrame, str, bool]:
    use_clean = st.sidebar.toggle("정제 데이터 사용", value=True)
    include_cancellations = st.sidebar.toggle("취소 주문 포함", value=False)

    source_df = data["clean"] if use_clean else data["raw"]
    source_label = "정제 데이터" if use_clean else "원본 데이터"
    assert isinstance(source_df, pd.DataFrame)

    min_date = source_df["OrderDate"].min()
    max_date = source_df["OrderDate"].max()
    date_range = st.sidebar.date_input("날짜 범위", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    start_date, end_date = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_date, max_date)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    countries = sorted(source_df["CountryStr"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("국가 선택", countries, default=[])
    product_query = st.sidebar.text_input("상품 검색", placeholder="상품명 또는 상품코드")
    customer_query = st.sidebar.text_input("고객 검색", placeholder="고객 ID")

    filtered = filter_transactions(
        source_df,
        start_date=start_date,
        end_date=end_date,
        countries=selected_countries,
        product_query=product_query,
        customer_query=customer_query,
        include_cancellations=include_cancellations,
    )
    return filtered, source_label, include_cancellations


def render_overview_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("종합 대시보드")
    kpis = compute_kpis(filtered_df)
    monthly = monthly_summary(filtered_df)
    countries = aggregate_countries(filtered_df).head(15)
    products = aggregate_products(filtered_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 데이터 수", fmt_int(kpis["rows"]))
    c2.metric("전체 주문 수", fmt_int(kpis["orders"]))
    c3.metric("전체 상품 수", fmt_int(kpis["products"]))
    c4.metric("전체 고객 수", fmt_int(kpis["customers"]))
    c5.metric("총 매출", fmt_currency(kpis["sales"]))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("평균 주문금액 AOV", fmt_currency(kpis["aov"]))
    c7.metric("평균 고객당 매출", fmt_currency(kpis["arppu"]))
    c8.metric("취소 주문 수", fmt_int(kpis["cancelled_orders"]))
    c9.metric("취소율", fmt_pct(kpis["cancel_rate"]))
    st.metric("선택 기간 신규 고객 수", fmt_int(kpis["new_customers"]))

    if not monthly.empty:
        sales_fig = px.line(monthly, x="YearMonthStart", y="sales", markers=True, title="월별 매출 추이")
        render_figure(
            sales_fig,
            build_data_basis(len(filtered_df), "InvoiceDate, Quantity, UnitPrice", "월 단위 매출 합계", "`Sales=Quantity*UnitPrice`를 생성하고 월 단위로 합산"),
            build_interpretation("월별 매출", "급등 구간은 주문 수와 구매 고객 수를 같이 보아 대량 주문 영향인지 확인해야 한다"),
            "overview-monthly-sales",
        )

        activity_fig = px.line(
            monthly,
            x="YearMonthStart",
            y=["orders", "buyers"],
            markers=True,
            title="월별 주문 수와 구매 고객 수",
        )
        render_figure(
            activity_fig,
            build_data_basis(len(filtered_df), "InvoiceDate, InvoiceNo, CustomerID", "월 단위 주문 수와 구매 고객 수 집계", "고유 주문번호와 고유 고객 수를 월별로 계산"),
            build_interpretation("주문 수와 구매 고객 수의 동행 여부", "주문만 늘고 고객 수가 정체되면 기존 고객의 구매 빈도 상승일 가능성을 함께 봐야 한다"),
            "overview-monthly-activity",
        )

    if not countries.empty:
        country_bar = px.bar(countries, x="CountryStr", y="sales", title="국가별 매출")
        render_figure(
            country_bar,
            build_data_basis(len(filtered_df), "Country, Quantity, UnitPrice", "국가별 매출 합계", "국가별로 `Sales`를 합산하고 매출 상위 국가만 표시"),
            build_interpretation("국가별 매출 편중도", "상위 국가 비중이 과도하게 높으면 특정 국가 의존도가 큰 구조인지 점검해야 한다"),
            "overview-country-bar",
        )

        country_tree = px.treemap(countries, path=[px.Constant("전체"), "CountryStr"], values="sales", color="orders", title="국가별 매출 비중 트리맵")
        render_figure(
            country_tree,
            build_data_basis(len(filtered_df), "Country, Sales, InvoiceNo", "국가별 매출 비중 시각화", "국가별 매출을 면적, 주문 수를 색상 기준으로 표현"),
            build_interpretation("국가별 기여도 구조", "면적은 매출 비중을 뜻하므로 큰 블록이 시장 핵심이며 색상은 주문 수 수준과 함께 해석해야 한다"),
            "overview-country-tree",
        )

    if not products.empty:
        top_sales = products.head(10)
        product_sales_fig = px.bar(top_sales, x="DescriptionStr", y="sales", title="상위 상품 매출", hover_data=["StockCodeStr"])
        render_figure(
            product_sales_fig,
            build_data_basis(len(filtered_df), "StockCode, Description, Quantity, UnitPrice", "상품별 매출 합계", "상품 단위로 `Sales`를 합산한 뒤 상위 10개를 표시"),
            build_interpretation("상위 상품 매출 랭킹", "매출이 높은 상품이라도 주문 수와 수량이 함께 높은지 확인해 단가 주도 상품인지 구분해야 한다"),
            "overview-product-sales",
        )

        product_qty_fig = px.bar(top_sales, x="DescriptionStr", y="quantity", title="상위 상품 판매수량", hover_data=["StockCodeStr"])
        render_figure(
            product_qty_fig,
            build_data_basis(len(filtered_df), "StockCode, Description, Quantity", "상품별 판매수량 합계", "상품 단위로 수량을 합산하고 판매수량 상위 상품을 표시"),
            build_interpretation("판매수량 중심 베스트셀러", "수량 상위 상품과 매출 상위 상품이 다르면 저가 다판매 구조인지 함께 해석해야 한다"),
            "overview-product-qty",
        )


def render_products_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("상품 분석")
    products = aggregate_products(filtered_df)
    if products.empty:
        st.info("표시할 상품 데이터가 없습니다.")
        return

    search = st.text_input("상품 상세 검색", placeholder="상품명 또는 상품코드", key="product-tab-search")
    shown = products.copy()
    if search:
        query = search.lower()
        shown = shown[
            shown["StockCodeStr"].str.lower().str.contains(query, na=False, regex=False)
            | shown["DescriptionStr"].str.lower().str.contains(query, na=False, regex=False)
        ]

    if shown.empty:
        st.info("검색 조건에 맞는 상품이 없습니다.")
        return

    st.dataframe(shown.head(100), use_container_width=True)
    options = shown.head(100)[["StockCodeStr", "DescriptionStr"]]
    if options.empty:
        st.info("표시 가능한 상품이 없습니다.")
        return
    selected_label = st.selectbox(
        "상세 상품 선택",
        options=[f"{row.StockCodeStr} | {row.DescriptionStr}" for row in options.itertuples(index=False)],
    )
    selected_code = selected_label.split(" | ", 1)[0]

    product_row = products[products["StockCodeStr"] == selected_code].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("누적 매출", fmt_currency(product_row["sales"]))
    c2.metric("판매수량", fmt_int(product_row["quantity"]))
    c3.metric("주문 수", fmt_int(product_row["orders"]))
    c4.metric("구매 고객 수", fmt_int(product_row["customers"]))

    monthly = product_monthly_sales(filtered_df, selected_code)
    if not monthly.empty:
        monthly_fig = px.line(monthly, x="YearMonthStart", y="sales", markers=True, title="선택 상품 월별 매출 추이")
        render_figure(
            monthly_fig,
            build_data_basis(len(filtered_df), "StockCode, InvoiceDate, Quantity, UnitPrice", "선택 상품의 월별 매출 합계", "선택 상품만 필터링한 뒤 월 단위로 `Sales`를 합산"),
            build_interpretation("선택 상품의 시계열 성과", "특정 월의 급증은 일회성 프로모션이나 대량 주문일 수 있으므로 주문 수도 함께 봐야 한다"),
            "product-monthly-sales",
        )

    product_countries = filtered_df[filtered_df["StockCodeStr"] == selected_code]
    country_summary = aggregate_countries(product_countries)
    if not country_summary.empty:
        country_fig = px.bar(country_summary.head(10), x="CountryStr", y="sales", title="선택 상품 국가별 매출")
        render_figure(
            country_fig,
            build_data_basis(len(product_countries), "Country, Quantity, UnitPrice", "선택 상품의 국가별 매출 합계", "선택 상품 데이터만 추출해 국가별로 `Sales`를 집계"),
            build_interpretation("상품의 국가별 판매 편중", "특정 국가 비중이 높다면 해당 상품이 지역 특화 수요를 갖는지 판단하는 근거가 된다"),
            "product-country-sales",
        )

        sunburst_fig = px.sunburst(
            product_countries.groupby(["CountryStr", "DescriptionStr"], as_index=False).agg(sales=("Sales", "sum")),
            path=["CountryStr", "DescriptionStr"],
            values="sales",
            title="국가와 상품 기준 매출 선버스트",
        )
        render_figure(
            sunburst_fig,
            build_data_basis(len(product_countries), "Country, Description, Sales", "국가-상품 계층형 매출 비중", "국가와 상품을 계층 구조로 묶어 `Sales` 비중을 시각화"),
            build_interpretation("계층형 매출 비중", "상위 고리의 국가 비중과 하위 상품 비중을 함께 보면 어디서 어떤 상품이 기여하는지 빠르게 판단할 수 있다"),
            "product-sunburst",
        )


def render_customers_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("고객 분석")
    customers = aggregate_customers(filtered_df)
    if customers.empty:
        st.info("표시할 고객 데이터가 없습니다.")
        return

    search = st.text_input("고객 검색", placeholder="고객 ID", key="customer-tab-search")
    shown = customers.copy()
    if search:
        shown = shown[shown["CustomerIDStr"].str.lower().str.contains(search.strip().lower(), na=False, regex=False)]

    if shown.empty:
        st.info("검색 조건에 맞는 고객이 없습니다.")
        return

    st.dataframe(shown.head(100), use_container_width=True)
    customer_options = shown.head(100)["CustomerIDStr"].tolist()
    if not customer_options:
        st.info("표시 가능한 고객이 없습니다.")
        return
    selected_customer = st.selectbox("상세 고객 선택", options=customer_options)
    customer_row = customers[customers["CustomerIDStr"] == selected_customer].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 구매금액", fmt_currency(customer_row["sales"]))
    c2.metric("총 주문 수", fmt_int(customer_row["orders"]))
    c3.metric("평균 주문금액", fmt_currency(customer_row["avg_order_value"]))
    c4.metric("최근 구매일", str(customer_row["last_purchase"]))

    top_customers = customers.head(15)
    rank_fig = px.bar(top_customers, x="CustomerIDStr", y="sales", title="상위 고객 매출 랭킹")
    render_figure(
        rank_fig,
        build_data_basis(len(filtered_df), "CustomerID, Quantity, UnitPrice", "고객별 매출 합계", "고객 단위로 `Sales`를 합산해 상위 고객을 정렬"),
        build_interpretation("고객 매출 집중도", "상위 고객의 비중이 높으면 핵심 고객 유지 전략이 중요하고 편중 위험도도 함께 고려해야 한다"),
        "customer-rank",
    )

    monthly = customer_monthly_sales(filtered_df, selected_customer)
    if not monthly.empty:
        monthly_fig = px.line(monthly, x="YearMonthStart", y="sales", markers=True, title="선택 고객 월별 구매금액")
        render_figure(
            monthly_fig,
            build_data_basis(len(filtered_df[filtered_df["CustomerIDStr"] == selected_customer]), "CustomerID, InvoiceDate, Sales", "선택 고객의 월별 구매금액", "선택 고객 거래를 월 단위로 집계하여 구매금액을 합산"),
            build_interpretation("고객별 구매 패턴", "구매가 주기적으로 발생하는지, 특정 월에만 집중되는지 보면 재구매 패턴과 이탈 위험을 가늠할 수 있다"),
            "customer-monthly-sales",
        )

    country_counts = customers.groupby("country", as_index=False).agg(customers=("CustomerIDStr", "nunique")).sort_values("customers", ascending=False)
    country_fig = px.bar(country_counts.head(15), x="country", y="customers", title="국가별 고객 분포")
    render_figure(
        country_fig,
        build_data_basis(len(customers), "CustomerID, Country", "국가별 고유 고객 수 집계", "고객별 대표 국가를 기준으로 고유 고객 수를 계산"),
        build_interpretation("국가별 고객 기반 규모", "고객 수가 많은 국가와 매출이 높은 국가가 다를 수 있으므로 국가별 AOV와 함께 비교해야 한다"),
        "customer-country-distribution",
    )


def render_orders_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("판매내역 조회")
    orders = aggregate_orders(filtered_df)
    if orders.empty:
        st.info("표시할 주문 데이터가 없습니다.")
        return

    order_query = st.text_input("주문번호 검색", placeholder="예: 536365")
    shown = orders.copy()
    if order_query:
        shown = shown[shown["InvoiceNoStr"].str.contains(order_query.strip(), na=False, regex=False)]

    if shown.empty:
        st.info("검색 조건에 맞는 주문이 없습니다.")
        return

    st.dataframe(shown.head(200), use_container_width=True)
    st.download_button("주문 데이터 CSV 다운로드", to_csv_bytes(shown), file_name="online_retail_orders.csv", mime="text/csv")

    order_options = shown.head(200)["InvoiceNoStr"].tolist()
    if not order_options:
        st.info("표시 가능한 주문이 없습니다.")
        return
    selected_order = st.selectbox("상세 주문 선택", options=order_options)
    order_lines = filtered_df[filtered_df["InvoiceNoStr"] == selected_order].copy()
    st.dataframe(
        order_lines[["InvoiceNoStr", "InvoiceDate", "StockCodeStr", "DescriptionStr", "Quantity", "UnitPrice", "Sales", "CustomerIDStr", "CountryStr"]],
        use_container_width=True,
    )

    daily = orders.copy()
    daily["OrderDate"] = pd.to_datetime(daily["InvoiceDate"]).dt.date
    daily_orders = daily.groupby("OrderDate", as_index=False).agg(orders=("InvoiceNoStr", "nunique"), sales=("sales", "sum"))
    fig = px.line(daily_orders, x="OrderDate", y="orders", markers=True, title="일별 주문 수")
    render_figure(
        fig,
        build_data_basis(len(filtered_df), "InvoiceNo, InvoiceDate", "일 단위 고유 주문 수 집계", "주문번호를 일자 기준으로 고유 집계하여 주문 수를 계산"),
        build_interpretation("일별 주문량 변동", "급증일은 프로모션이나 특정 고객 대량 주문일 수 있으므로 매출과 함께 확인해야 한다"),
        "orders-daily-count",
    )


def render_countries_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("국가별 거래 현황")
    countries = aggregate_countries(filtered_df)
    if countries.empty:
        st.info("표시할 국가 데이터가 없습니다.")
        return

    st.dataframe(countries, use_container_width=True)
    selected_country = st.selectbox("국가 선택", options=countries["CountryStr"].tolist())
    country_df = filtered_df[filtered_df["CountryStr"] == selected_country]
    monthly = monthly_summary(country_df)

    comp_fig = px.bar(countries.head(15), x="CountryStr", y=["sales", "orders"], barmode="group", title="상위 국가 비교")
    render_figure(
        comp_fig,
        build_data_basis(len(filtered_df), "Country, Sales, InvoiceNo", "국가별 매출과 주문 수 비교", "국가 단위로 매출 합계와 주문 수를 함께 집계"),
        build_interpretation("국가별 거래 효율", "매출은 높지만 주문 수가 낮다면 고가 주문 구조일 수 있고 반대면 저단가 다빈도 구조일 수 있다"),
        "countries-compare",
    )

    tree_fig = px.treemap(countries.head(15), path=[px.Constant("전체"), "CountryStr"], values="sales", color="customers", title="국가별 거래 비중 트리맵")
    render_figure(
        tree_fig,
        build_data_basis(len(filtered_df), "Country, Sales, CustomerID", "국가별 매출 비중과 고객 수 표현", "트리맵 면적은 매출, 색상은 고객 수 기준으로 표시"),
        build_interpretation("국가별 기여도와 고객 규모", "면적과 색상을 함께 보면 매출 중심 국가와 고객 기반이 넓은 국가의 차이를 읽을 수 있다"),
        "countries-tree",
    )

    if not monthly.empty:
        monthly_fig = px.line(monthly, x="YearMonthStart", y="sales", markers=True, title=f"{selected_country} 월별 매출 추이")
        render_figure(
            monthly_fig,
            build_data_basis(len(country_df), "Country, InvoiceDate, Sales", "선택 국가의 월별 매출 합계", "선택 국가 데이터만 추출한 뒤 월 단위로 매출을 합산"),
            build_interpretation("선택 국가의 시계열 추이", "특정 국가의 성장세가 전체 매출 상승을 이끄는지 다른 탭과 비교해 해석할 수 있다"),
            "countries-monthly",
        )


def render_kpis_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("주요 지표")
    daily = daily_activity(filtered_df)
    monthly = monthly_business_metrics(filtered_df)
    if daily.empty or monthly.empty:
        st.info("주요 지표를 계산할 충분한 고객 데이터가 없습니다.")
        return

    dau_fig = px.line(daily, x="OrderDate", y="buyers", markers=True, title="DAU")
    render_figure(
        dau_fig,
        build_data_basis(len(filtered_df), "OrderDate, CustomerID", "일자별 고유 구매 고객 수 집계", "고객 ID가 있는 거래만 사용해 일별 고유 구매 고객 수를 계산"),
        build_interpretation("일별 활성 고객 추이", "단기 변동 폭이 클 수 있으므로 주간 패턴과 프로모션 일정을 함께 고려해 해석해야 한다"),
        "kpi-dau",
    )

    mau_fig = px.line(monthly, x="YearMonthStart", y="buyers", markers=True, title="MAU")
    render_figure(
        mau_fig,
        build_data_basis(len(filtered_df), "InvoiceDate, CustomerID", "월별 고유 구매 고객 수 집계", "고객 ID 기준으로 월별 고유 구매자 수를 계산"),
        build_interpretation("월별 활성 고객 기반", "MAU가 증가하면 고객 기반이 확장되는 것이며 매출과 같이 보면 고객 규모와 구매력 변화를 함께 볼 수 있다"),
        "kpi-mau",
    )

    arppu_fig = px.line(monthly, x="YearMonthStart", y="arppu", markers=True, title="월별 ARPPU")
    render_figure(
        arppu_fig,
        build_data_basis(len(filtered_df), "Sales, CustomerID, InvoiceDate", "월별 ARPPU 계산", "월 총매출을 월별 구매 고객 수로 나누어 ARPPU를 계산"),
        build_interpretation("고객당 매출 수준", "ARPPU 상승은 구매 단가나 빈도 증가를 뜻할 수 있으므로 주문 수와 AOV를 같이 봐야 원인을 분리할 수 있다"),
        "kpi-arppu",
    )

    acquire_fig = go.Figure()
    acquire_fig.add_bar(x=monthly["YearMonthStart"], y=monthly["acquired_customers"], name="신규 고객 수")
    acquire_fig.add_bar(x=monthly["YearMonthStart"], y=monthly["repeat_customers"], name="재구매 고객 수")
    acquire_fig.add_trace(go.Scatter(x=monthly["YearMonthStart"], y=monthly["repeat_rate"], name="재구매율", yaxis="y2"))
    acquire_fig.update_layout(title="월별 고객 획득과 재구매", yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
    render_figure(
        acquire_fig,
        build_data_basis(len(filtered_df), "CustomerID, InvoiceDate, Sales", "월별 신규 고객과 재구매 고객 집계", "고객의 첫 구매 월을 계산한 뒤 월별 신규 고객, 재구매 고객, 재구매율을 계산"),
        build_interpretation("획득과 재구매의 균형", "신규 유입만 높고 재구매율이 낮으면 유지 전략 보강이 필요하며 반대면 충성고객 중심 구조일 수 있다"),
        "kpi-acquire-repeat",
    )

    st.dataframe(monthly, use_container_width=True)


def render_retention_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("리텐션 분석")
    counts, retention = build_retention_table(filtered_df)
    if retention.empty:
        st.info("리텐션을 계산할 충분한 고객 데이터가 없습니다.")
        return

    heatmap = px.imshow(
        retention,
        labels=dict(x="개월차", y="첫 구매 월 코호트", color="리텐션 비율"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="코호트 리텐션 히트맵",
        text_auto=".0%",
    )
    render_figure(
        heatmap,
        build_data_basis(len(filtered_df), "CustomerID, InvoiceDate", "첫 구매 월 기준 코호트 리텐션 계산", "고객의 첫 구매 월을 코호트로 정의하고 이후 개월차별 재구매 고객 비율을 계산"),
        build_interpretation("코호트 잔존율", "같은 행에서 오른쪽으로 갈수록 시간 경과에 따른 유지율이며 색이 진할수록 리텐션이 높은 코호트로 해석한다"),
        "retention-heatmap",
    )

    st.dataframe((retention * 100).round(1), use_container_width=True)
    st.dataframe(counts, use_container_width=True)


def render_rfm_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("RFM 분석")
    rfm = build_rfm_table(filtered_df)
    summary = summarize_rfm(rfm)
    if rfm.empty or summary.empty:
        st.info("RFM을 계산할 충분한 고객 데이터가 없습니다.")
        return

    tree_fig = px.treemap(summary, path=[px.Constant("전체"), "Segment"], values="sales", color="customers", title="RFM 세그먼트 매출 비중")
    render_figure(
        tree_fig,
        build_data_basis(len(filtered_df), "CustomerID, InvoiceDate, Sales", "고객별 RFM 계산 후 세그먼트별 매출 비중 표현", "고객 단위 RFM을 계산하고 세그먼트별 매출과 고객 수를 집계"),
        build_interpretation("세그먼트 기여도", "면적이 큰 세그먼트일수록 매출 기여도가 높고 색상은 고객 수 수준을 나타내므로 두 지표를 함께 읽어야 한다"),
        "rfm-tree",
    )

    bar_fig = px.bar(summary, x="Segment", y="sales", color="share", title="RFM 세그먼트별 매출 기여도")
    render_figure(
        bar_fig,
        build_data_basis(len(filtered_df), "CustomerID, Sales", "RFM 세그먼트별 매출 합계", "세그먼트별 고객 수와 매출을 집계하고 매출 기여도를 계산"),
        build_interpretation("세그먼트 우선순위", "Champions와 Loyal Customers 비중이 높으면 안정적 구조이며 At Risk 비중이 높으면 이탈 방지 전략이 중요하다"),
        "rfm-bar",
    )

    selected_segment = st.selectbox("세그먼트 선택", options=summary["Segment"].tolist())
    st.dataframe(rfm[rfm["Segment"] == selected_segment].head(200), use_container_width=True)


def render_content_recommendation_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("콘텐츠 기반 추천")
    catalog = build_content_catalog(filtered_df)
    if catalog.empty:
        st.info("콘텐츠 기반 추천에 사용할 상품 데이터가 없습니다.")
        return

    query = st.text_input("상품명/상품코드 검색", placeholder="예: HEART, 85123A", key="content-reco-search")
    shown = catalog.copy()
    if query:
        q = query.strip().lower()
        shown = shown[
            shown["StockCodeStr"].str.lower().str.contains(q, na=False, regex=False)
            | shown["DescriptionStr"].str.lower().str.contains(q, na=False, regex=False)
        ]

    if shown.empty:
        st.info("검색 조건에 맞는 상품이 없습니다.")
        return

    st.markdown("상품 목록에서 상품을 선택하면 유사 상품을 보여줍니다.")
    st.dataframe(shown[["StockCodeStr", "DescriptionStr", "sales", "orders", "quantity"]].head(200), use_container_width=True)

    options = shown.head(200)[["StockCodeStr", "DescriptionStr"]]
    selected_label = st.selectbox(
        "추천 기준 상품 선택",
        options=[f"{row.StockCodeStr} | {row.DescriptionStr}" for row in options.itertuples(index=False)],
        key="content-reco-product-select",
    )
    selected_code = selected_label.split(" | ", 1)[0]

    content_model = build_content_model(catalog)
    recs = recommend_similar_products(content_model, selected_code, top_n=15)
    if recs.empty:
        st.info("유사 상품을 계산할 수 없습니다.")
        return

    st.markdown("#### 유사 상품 추천 결과")
    st.dataframe(recs, use_container_width=True)


def render_collaborative_filtering_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("협업필터링 추천")
    rfm = build_rfm_table(filtered_df)
    if rfm.empty:
        st.info("협업필터링을 계산할 충분한 고객 데이터가 없습니다.")
        return

    customer_query = st.text_input("고객 ID 검색", placeholder="예: 17850", key="cf-customer-search")
    customer_view = rfm[["CustomerIDStr", "R", "F", "M", "recency_days", "frequency", "monetary", "Segment"]].copy()
    if customer_query:
        q = customer_query.strip().lower()
        customer_view = customer_view[customer_view["CustomerIDStr"].str.lower().str.contains(q, na=False, regex=False)]

    if customer_view.empty:
        st.info("검색 조건에 맞는 고객이 없습니다.")
        return

    st.markdown("R, F, M 기반 고객 목록")
    st.dataframe(customer_view.head(300), use_container_width=True)

    customer_options = customer_view["CustomerIDStr"].head(300).tolist()
    selected_customer = st.selectbox("고객 선택", options=customer_options, key="cf-customer-select")

    purchased = purchased_products_for_customer(filtered_df, selected_customer)
    st.markdown("#### 선택 고객의 구매 상품")
    if purchased.empty:
        st.info("해당 고객의 구매 이력이 없습니다.")
    else:
        st.dataframe(purchased.head(200), use_container_width=True)

    cf_model = build_cf_model(filtered_df)
    recommendations = recommend_products_cf(cf_model, selected_customer, top_n=15, item_neighbors=20)
    st.markdown("#### 협업필터링 추천 상품")
    if recommendations.empty:
        st.info("협업필터링 추천 결과가 없습니다. 필터 조건을 넓혀보세요.")
    else:
        st.dataframe(recommendations, use_container_width=True)


def main() -> None:
    st.title("Online Retail 종합 대시보드")
    st.caption("Plotly 기반 시각화와 거래, 고객, 국가, 리텐션, RFM 분석을 포함한 업무용 대시보드")

    try:
        data = load_app_data()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {exc}")
        st.info("데이터 파일 경로와 컬럼 구성을 확인한 뒤 다시 실행해주세요.")
        return

    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        st.caption(
            f"데이터 경로: {metadata.get('data_path', 'unknown')} | "
            f"원본 {fmt_int(int(metadata.get('raw_rows', 0)))}행 / "
            f"정제 {fmt_int(int(metadata.get('clean_rows', 0)))}행"
        )

    filtered_df, source_label, include_cancellations = get_filtered_data(data)

    if filtered_df.empty:
        st.warning("현재 필터 조건에 해당하는 데이터가 없습니다.")
        return

    tabs = st.tabs(["종합", "상품", "고객", "판매내역", "국가", "주요 지표", "리텐션", "RFM", "콘텐츠 추천", "협업필터링"])
    with tabs[0]:
        render_overview_tab(filtered_df)
    with tabs[1]:
        render_products_tab(filtered_df)
    with tabs[2]:
        render_customers_tab(filtered_df)
    with tabs[3]:
        render_orders_tab(filtered_df)
    with tabs[4]:
        render_countries_tab(filtered_df)
    with tabs[5]:
        render_kpis_tab(filtered_df)
    with tabs[6]:
        render_retention_tab(filtered_df)
    with tabs[7]:
        render_rfm_tab(filtered_df)
    with tabs[8]:
        render_content_recommendation_tab(filtered_df)
    with tabs[9]:
        render_collaborative_filtering_tab(filtered_df)

    render_footer(filtered_df, source_label, include_cancellations)


if __name__ == "__main__":
    main()