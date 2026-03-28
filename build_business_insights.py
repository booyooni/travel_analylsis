from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

PRODUCT_PATH = BASE_DIR / "tb_product.csv"
SCHEDULE_PATH = BASE_DIR / "tb_price_schedule.csv"


def classify_status(value: str) -> str:
    text = str(value)
    if "취소" in text:
        return "취소"
    if "마감" in text:
        return "마감"
    if "판매" in text or "가능" in text:
        return "판매중"
    return "기타"


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    preferred_fonts = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Nanum Gothic",
        "NanumGothic",
        "DejaVu Sans",
    ]

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((name for name in preferred_fonts if name in available_names), "DejaVu Sans")

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = preferred_fonts
    plt.rcParams["axes.unicode_minus"] = False


def save_kpi_tables(merged: pd.DataFrame) -> dict:
    group_cols = ["travel_style"]

    kpi = (
        merged.groupby(group_cols)
        .agg(
            products=("god_id", "nunique"),
            events=("event_cd", "nunique"),
            avg_price=("price_adt", "mean"),
            median_price=("price_adt", "median"),
            min_price=("price_adt", "min"),
            max_price=("price_adt", "max"),
            avg_days=("evt_dt_cnt", "mean"),
            avg_nights=("evt_night_cnt", "mean"),
        )
        .reset_index()
    )

    status_mix = (
        merged.groupby(["travel_style", "status_bucket"]).size().reset_index(name="count")
    )
    status_total = status_mix.groupby("travel_style")["count"].transform("sum")
    status_mix["ratio"] = status_mix["count"] / status_total

    carrier_top = (
        merged.groupby(["travel_style", "carrer_nm"]).size().reset_index(name="count")
        .sort_values(["travel_style", "count"], ascending=[True, False])
        .groupby("travel_style")
        .head(5)
    )

    kpi.to_csv(OUT_DIR / "kpi_by_style.csv", index=False, encoding="utf-8-sig")
    status_mix.to_csv(OUT_DIR / "status_mix_by_style.csv", index=False, encoding="utf-8-sig")
    carrier_top.to_csv(OUT_DIR / "top5_carrier_by_style.csv", index=False, encoding="utf-8-sig")

    return {
        "kpi": kpi,
        "status_mix": status_mix,
        "carrier_top": carrier_top,
    }


def build_charts(merged: pd.DataFrame, tables: dict) -> None:
    setup_plot_style()

    kpi = tables["kpi"]
    status_mix = tables["status_mix"]
    carrier_top = tables["carrier_top"]
    style_label_map = {"휴양형": "Resort", "도시형": "City"}
    status_label_map = {"판매중": "On Sale", "마감": "Sold Out", "취소": "Canceled", "기타": "Other"}

    kpi_plot = kpi.copy()
    kpi_plot["style_en"] = kpi_plot["travel_style"].map(style_label_map).fillna(kpi_plot["travel_style"])

    merged_plot = merged.copy()
    merged_plot["style_en"] = merged_plot["travel_style"].map(style_label_map).fillna(merged_plot["travel_style"])

    status_mix_plot = status_mix.copy()
    status_mix_plot["style_en"] = status_mix_plot["travel_style"].map(style_label_map).fillna(status_mix_plot["travel_style"])
    status_mix_plot["status_en"] = status_mix_plot["status_bucket"].map(status_label_map).fillna(status_mix_plot["status_bucket"])

    carrier_top_plot = carrier_top.copy()
    carrier_top_plot["style_en"] = carrier_top_plot["travel_style"].map(style_label_map).fillna(carrier_top_plot["travel_style"])

    # 1) 평균 가격 비교
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=kpi_plot, x="style_en", y="avg_price", ax=ax)
    ax.set_title("Average Adult Price: Resort vs City")
    ax.set_xlabel("Travel Style")
    ax.set_ylabel("Average Adult Price (KRW)")
    for i, v in enumerate(kpi_plot["avg_price"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_avg_price_by_style.png", dpi=150)
    plt.close(fig)

    # 2) 가격 분포 박스플롯
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=merged_plot, x="style_en", y="price_adt", ax=ax)
    ax.set_title("Adult Price Distribution by Travel Style")
    ax.set_xlabel("Travel Style")
    ax.set_ylabel("Adult Price (KRW)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_price_distribution_boxplot.png", dpi=150)
    plt.close(fig)

    # 3) 상태 구성 비율 (100% stacked)
    pivot = (
        status_mix_plot.pivot(index="style_en", columns="status_en", values="ratio")
        .fillna(0)
        .reindex(columns=[c for c in ["On Sale", "Sold Out", "Canceled", "Other"] if c in status_mix_plot["status_en"].unique()])
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = pd.Series([0] * len(pivot), index=pivot.index)
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col)
        bottom += pivot[col]
    ax.set_title("Status Mix by Travel Style")
    ax.set_xlabel("Travel Style")
    ax.set_ylabel("Ratio")
    ax.legend(title="Status")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_status_mix_stacked.png", dpi=150)
    plt.close(fig)

    # 4) 일정 길이 비교
    melted = kpi.melt(
        id_vars=["travel_style"],
        value_vars=["avg_days", "avg_nights"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {"avg_days": "Avg Trip Days", "avg_nights": "Avg Nights"}
    melted["metric"] = melted["metric"].map(metric_map)
    melted["style_en"] = melted["travel_style"].map(style_label_map).fillna(melted["travel_style"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=melted, x="style_en", y="value", hue="metric", ax=ax)
    ax.set_title("Trip Duration Comparison by Travel Style")
    ax.set_xlabel("Travel Style")
    ax.set_ylabel("Days")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_duration_compare.png", dpi=150)
    plt.close(fig)

    # 5) 상위 항공사 점유
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=carrier_top_plot, x="count", y="carrer_nm", hue="style_en", ax=ax)
    ax.set_title("Top Carriers by Travel Style")
    ax.set_xlabel("Count")
    ax.set_ylabel("Carrier")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_top_carrier_by_style.png", dpi=150)
    plt.close(fig)



def write_purpose_and_insights(merged: pd.DataFrame, tables: dict) -> None:
    kpi = tables["kpi"].set_index("travel_style")

    resort_avg = float(kpi.loc["휴양형", "avg_price"]) if "휴양형" in kpi.index else 0
    city_avg = float(kpi.loc["도시형", "avg_price"]) if "도시형" in kpi.index else 0
    price_ratio = (city_avg / resort_avg) if resort_avg else 0

    status_mix = tables["status_mix"]
    cancelled = (
        status_mix[status_mix["status_bucket"] == "취소"]
        .set_index("travel_style")["ratio"]
        .to_dict()
    )

    purpose_text = [
        "[분석 목적]",
        "1) 휴양형(다낭·나트랑) vs 도시형(싱가포르)의 가격, 일정, 판매상태 구조를 비교해 상품 포트폴리오 전략을 수립한다.",
        "2) 어떤 유형이 고가/저가 포지션인지 정량화해 캠페인 메시지와 채널 예산 배분 근거를 만든다.",
        "3) 항공사 편중 및 상태(판매중/마감/취소) 비율을 통해 운영 리스크와 공급전략을 점검한다.",
        "",
        "[핵심 비즈니스 인사이트]",
        f"- 도시형 평균가가 휴양형 대비 {price_ratio:.2f}배로 나타나, 동일 가격정책보다 유형별 가격커뮤니케이션 분리가 필요하다.",
        f"- 휴양형 상품/행사 볼륨이 크므로(규모 우위), 업셀링(호텔·특식·옵션 강화) 전략이 효율적이다.",
        f"- 취소 비율은 휴양형 {cancelled.get('휴양형', 0):.1%}, 도시형 {cancelled.get('도시형', 0):.1%}로 모니터링이 필요하다.",
        "- 항공사 노출 상위 집중도가 높아 단일 파트너 의존 리스크를 점검하고 대체편 공급력을 병행 확보해야 한다.",
        "",
        "[추천 액션]",
        "- 휴양형: 가성비 세그먼트 중심 + 선택옵션 업셀(마사지/특식/리조트룸).",
        "- 도시형: 프리미엄 가치(도심 접근성, 자유일정, 브랜드호텔) 중심 메시지 강화.",
        "- 공통: 상태별(판매중/마감/취소) 전환성과를 주간 대시보드로 관리.",
    ]

    (OUT_DIR / "business_purpose_insights.txt").write_text("\n".join(purpose_text), encoding="utf-8")


def build_html_report() -> None:
    charts = [
        "01_avg_price_by_style.png",
        "02_price_distribution_boxplot.png",
        "03_status_mix_stacked.png",
        "04_duration_compare.png",
        "05_top_carrier_by_style.png",
    ]
    html = [
        "<html><head><meta charset='utf-8'><title>Travel Style Analysis</title></head><body>",
        "<h1>휴양형 vs 도시형 여행 비교 리포트</h1>",
        "<p>원천 데이터: tb_product.csv + tb_price_schedule.csv</p>",
        "<h2>분석 목적 및 인사이트</h2>",
        "<pre>" + (OUT_DIR / "business_purpose_insights.txt").read_text(encoding="utf-8") + "</pre>",
        "<h2>시각화</h2>",
    ]
    for c in charts:
        html.append(f"<h3>{c}</h3><img src='{c}' style='max-width:1000px; width:100%;'><hr>")
    html.append("</body></html>")
    (OUT_DIR / "travel_style_analysis_report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    product = pd.read_csv(PRODUCT_PATH)
    schedule = pd.read_csv(SCHEDULE_PATH)

    merged = schedule.merge(
        product[["god_id", "travel_style", "search_keyword", "agency_id"]],
        on="god_id",
        how="left",
    )

    merged["price_adt"] = pd.to_numeric(merged["price_adt"], errors="coerce").fillna(0)
    merged["evt_dt_cnt"] = pd.to_numeric(merged["evt_dt_cnt"], errors="coerce").fillna(0)
    merged["evt_night_cnt"] = pd.to_numeric(merged["evt_night_cnt"], errors="coerce").fillna(0)
    merged["status_bucket"] = merged["evt_status_nm"].apply(classify_status)

    # 분석대상: 휴양형/도시형 + price > 0
    merged = merged[merged["travel_style"].isin(["휴양형", "도시형"])].copy()
    merged = merged[merged["price_adt"] > 0].copy()

    tables = save_kpi_tables(merged)
    build_charts(merged, tables)
    write_purpose_and_insights(merged, tables)
    build_html_report()

    summary = {
        "rows_for_analysis": int(len(merged)),
        "styles": merged["travel_style"].value_counts().to_dict(),
        "outputs": [
            "kpi_by_style.csv",
            "status_mix_by_style.csv",
            "top5_carrier_by_style.csv",
            "business_purpose_insights.txt",
            "travel_style_analysis_report.html",
            "01_avg_price_by_style.png",
            "02_price_distribution_boxplot.png",
            "03_status_mix_stacked.png",
            "04_duration_compare.png",
            "05_top_carrier_by_style.png",
        ],
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("분석 완료")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
