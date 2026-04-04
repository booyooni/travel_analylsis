from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def format_number(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "online_retail.parquet"
    report_path = base_dir / "online_retail_basic_eda_report.md"

    df = pd.read_parquet(data_path)

    # Raw overview
    row_count, col_count = df.shape
    start_date = df["InvoiceDate"].min()
    end_date = df["InvoiceDate"].max()

    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)

    duplicate_rows = int(df.duplicated().sum())
    cancel_mask = df["InvoiceNo"].astype(str).str.startswith("C")
    cancellation_rows = int(cancel_mask.sum())

    negative_qty_rows = int((df["Quantity"] <= 0).sum())
    non_positive_price_rows = int((df["UnitPrice"] <= 0).sum())

    # Cleaned dataset for business EDA
    clean = df.copy()
    clean = clean[
        (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
        & clean["CustomerID"].notna()
        & clean["Description"].notna()
    ].copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int)
    clean["Sales"] = clean["Quantity"] * clean["UnitPrice"]
    clean["YearMonth"] = clean["InvoiceDate"].dt.to_period("M").astype(str)

    # Aggregations
    summary = {
        "orders": clean["InvoiceNo"].nunique(),
        "customers": clean["CustomerID"].nunique(),
        "products": clean["StockCode"].nunique(),
        "countries": clean["Country"].nunique(),
        "total_sales": clean["Sales"].sum(),
        "avg_order_value": clean.groupby("InvoiceNo")["Sales"].sum().mean(),
    }

    country_sales = (
        clean.groupby("Country", observed=True)
        .agg(
            sales=("Sales", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .sort_values("sales", ascending=False)
        .head(10)
        .reset_index()
    )

    product_sales = (
        clean.groupby(["StockCode", "Description"], observed=True)
        .agg(
            sales=("Sales", "sum"),
            quantity=("Quantity", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
        .sort_values("sales", ascending=False)
        .head(10)
        .reset_index()
    )

    monthly_sales = (
        clean.groupby("YearMonth", observed=True)
        .agg(
            sales=("Sales", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .sort_values("YearMonth")
        .reset_index()
    )

    sales_stats = clean["Sales"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])

    lines: list[str] = []
    lines.append("# Online Retail 기본 EDA 리포트")
    lines.append("")
    lines.append(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 데이터: online-retail/online_retail.parquet")
    lines.append("")

    lines.append("## 1) 데이터 개요")
    lines.append("")
    lines.append(f"- 행 수: {format_number(row_count)}")
    lines.append(f"- 열 수: {format_number(col_count)}")
    lines.append(f"- 기간: {start_date} ~ {end_date}")
    lines.append(f"- 고유 주문번호 수: {format_number(df['InvoiceNo'].nunique())}")
    lines.append(f"- 고유 고객 수(원본): {format_number(df['CustomerID'].nunique(dropna=True))}")
    lines.append(f"- 고유 상품 수(원본): {format_number(df['StockCode'].nunique())}")
    lines.append(f"- 고유 국가 수(원본): {format_number(df['Country'].nunique())}")
    lines.append("")

    lines.append("## 2) 결측치/이상치 점검")
    lines.append("")
    miss_rows = []
    for col in missing.index:
        miss_rows.append([
            col,
            format_number(int(missing[col])),
            f"{missing_pct[col]:.2f}%",
        ])
    lines.append(md_table(["Column", "MissingCount", "MissingPct"], miss_rows))
    lines.append("")
    lines.append(f"- 완전 중복 행 수: {format_number(duplicate_rows)}")
    lines.append(f"- 취소 주문 행 수(InvoiceNo가 C로 시작): {format_number(cancellation_rows)}")
    lines.append(f"- 수량 0 이하 행 수: {format_number(negative_qty_rows)}")
    lines.append(f"- 단가 0 이하 행 수: {format_number(non_positive_price_rows)}")
    lines.append("")

    lines.append("## 3) 정제 후 분석 기준")
    lines.append("")
    lines.append("- Quantity > 0")
    lines.append("- UnitPrice > 0")
    lines.append("- CustomerID 결측 제거")
    lines.append("- Description 결측 제거")
    lines.append("")
    lines.append(f"- 정제 후 행 수: {format_number(len(clean))}")
    lines.append(f"- 정제 후 주문 수: {format_number(summary['orders'])}")
    lines.append(f"- 정제 후 고객 수: {format_number(summary['customers'])}")
    lines.append(f"- 정제 후 상품 수: {format_number(summary['products'])}")
    lines.append(f"- 정제 후 국가 수: {format_number(summary['countries'])}")
    lines.append(f"- 총 매출(정제 후): {format_number(summary['total_sales'])}")
    lines.append(f"- 평균 주문금액(AOV): {format_number(summary['avg_order_value'])}")
    lines.append("")

    lines.append("## 4) 국가별 매출 상위 10")
    lines.append("")
    country_rows = []
    for _, row in country_sales.iterrows():
        country_rows.append([
            str(row["Country"]),
            format_number(float(row["sales"])),
            format_number(int(row["orders"])),
            format_number(int(row["customers"])),
        ])
    lines.append(md_table(["Country", "Sales", "Orders", "Customers"], country_rows))
    lines.append("")

    lines.append("## 5) 상품별 매출 상위 10")
    lines.append("")
    product_rows = []
    for _, row in product_sales.iterrows():
        product_rows.append([
            str(row["StockCode"]),
            str(row["Description"]),
            format_number(float(row["sales"])),
            format_number(int(row["quantity"])),
            format_number(int(row["orders"])),
        ])
    lines.append(md_table(["StockCode", "Description", "Sales", "Quantity", "Orders"], product_rows))
    lines.append("")

    lines.append("## 6) 월별 매출 추이")
    lines.append("")
    monthly_rows = []
    for _, row in monthly_sales.iterrows():
        monthly_rows.append([
            str(row["YearMonth"]),
            format_number(float(row["sales"])),
            format_number(int(row["orders"])),
            format_number(int(row["customers"])),
        ])
    lines.append(md_table(["YearMonth", "Sales", "Orders", "Customers"], monthly_rows))
    lines.append("")

    lines.append("## 7) 라인아이템 매출(Sales=Quantity*UnitPrice) 분포")
    lines.append("")
    sales_stat_rows = []
    for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "90%", "99%", "max"]:
        sales_stat_rows.append([stat_name, format_number(float(sales_stats[stat_name]))])
    lines.append(md_table(["Statistic", "Value"], sales_stat_rows))
    lines.append("")

    lines.append("## 8) 핵심 인사이트")
    lines.append("")
    top_country = country_sales.iloc[0]
    top_product = product_sales.iloc[0]
    peak_month = monthly_sales.sort_values("sales", ascending=False).iloc[0]
    lines.append(
        f"- 매출 1위 국가는 {top_country['Country']}이며, 정제 기준 누적 매출은 {format_number(float(top_country['sales']))}입니다."
    )
    lines.append(
        f"- 매출 1위 상품은 {top_product['Description']}({top_product['StockCode']})이며, 누적 매출은 {format_number(float(top_product['sales']))}입니다."
    )
    lines.append(
        f"- 월 매출 피크는 {peak_month['YearMonth']}이며, 해당 월 매출은 {format_number(float(peak_month['sales']))}입니다."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
