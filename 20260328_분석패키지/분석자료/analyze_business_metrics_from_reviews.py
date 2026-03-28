from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "verygoodtour_reviews_3cities_1year.csv"
OUT_DIR = BASE_DIR / "business_analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def normalize_city(keyword: str) -> str:
    keyword = (keyword or "").strip()
    if keyword == "싱가포르":
        return "싱가폴"
    return keyword


def city_en(city: str) -> str:
    return {
        "다낭": "Danang",
        "나트랑": "Nha Trang",
        "싱가폴": "Singapore",
    }.get(city, city)


def extract_days_and_nights(product_title: str) -> tuple[int | None, int | None]:
    title = product_title or ""

    night_day = re.search(r"(\d+)\s*박\s*(\d+)\s*일", title)
    if night_day:
        return int(night_day.group(2)), int(night_day.group(1))

    candidates = [int(x) for x in re.findall(r"(\d+)\s*일", title)]
    candidates = [x for x in candidates if 2 <= x <= 10]
    if not candidates:
        return None, None

    day = min(candidates)
    return day, day - 2 if day >= 3 else None


def keyword_flag_series(text_series: pd.Series, words: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(w) for w in words)
    return text_series.str.contains(pattern, case=False, na=False)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["city"] = df["keyword"].map(normalize_city)
    df["city_en"] = df["city"].map(city_en)
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna(""))
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    days_nights = df["product_title"].fillna("").apply(extract_days_and_nights)
    df["trip_days"] = days_nights.apply(lambda x: x[0])
    df["trip_nights"] = days_nights.apply(lambda x: x[1])

    positive_words = [
        "좋", "만족", "친절", "추천", "행복", "최고", "감사", "완벽", "즐거", "인생샷",
    ]
    negative_words = [
        "아쉽", "불편", "별로", "실망", "불친절", "지연", "문제", "최악", "비추", "힘들",
    ]

    df["positive_flag"] = keyword_flag_series(df["text"], positive_words)
    df["negative_flag"] = keyword_flag_series(df["text"], negative_words)

    theme_words = {
        "Guide": ["가이드", "인솔", "차장님", "소장님"],
        "Hotel": ["호텔", "숙소", "객실", "룸", "리조트"],
        "Meal": ["식사", "음식", "맛집", "조식", "석식", "중식"],
        "Schedule": ["일정", "시간", "자유시간", "이동", "코스"],
        "Price": ["가격", "가성비", "비용", "비싸", "저렴"],
    }

    for theme, words in theme_words.items():
        df[f"theme_{theme}"] = keyword_flag_series(df["text"], words)

    city_kpi = (
        df.groupby("city", as_index=False)
        .agg(
            reviews=("board_seq", "count"),
            unique_products=("product_code", lambda s: s.fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
            positive_rate=("positive_flag", "mean"),
            negative_rate=("negative_flag", "mean"),
            avg_trip_days=("trip_days", "mean"),
        )
        .sort_values("reviews", ascending=False)
    )
    city_kpi["positive_rate"] = (city_kpi["positive_rate"] * 100).round(2)
    city_kpi["negative_rate"] = (city_kpi["negative_rate"] * 100).round(2)
    city_kpi["avg_trip_days"] = city_kpi["avg_trip_days"].round(2)

    monthly_city = (
        df.groupby(["year_month", "city", "city_en"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    top_products = (
        df[df["product_code"].notna() & (df["product_code"].astype(str).str.strip() != "")]
        .groupby(["product_code", "city", "city_en"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
        .sort_values("review_count", ascending=False)
        .head(20)
    )

    duration_city = (
        df[df["trip_days"].notna()]
        .groupby(["city", "city_en", "trip_days"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
        .sort_values(["city_en", "trip_days"])
    )

    theme_city_rows = []
    for theme in theme_words:
        col = f"theme_{theme}"
        sub = df.groupby(["city", "city_en"], as_index=False)[col].mean()
        sub["theme"] = theme
        sub["mention_rate_pct"] = (sub[col] * 100).round(2)
        theme_city_rows.append(sub[["city", "city_en", "theme", "mention_rate_pct"]])
    theme_city = pd.concat(theme_city_rows, ignore_index=True)

    city_kpi.to_csv(OUT_DIR / "biz_city_kpis.csv", index=False, encoding="utf-8-sig")
    monthly_city.to_csv(OUT_DIR / "biz_monthly_city_reviews.csv", index=False, encoding="utf-8-sig")
    top_products.to_csv(OUT_DIR / "biz_top_products_by_reviews.csv", index=False, encoding="utf-8-sig")
    duration_city.to_csv(OUT_DIR / "biz_trip_duration_by_city.csv", index=False, encoding="utf-8-sig")
    theme_city.to_csv(OUT_DIR / "biz_theme_mentions_by_city.csv", index=False, encoding="utf-8-sig")

    # 1) 월별 리뷰 추이
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly_city, x="year_month", y="review_count", hue="city_en", marker="o")
    plt.title("Monthly Review Volume by City")
    plt.xlabel("Year-Month")
    plt.ylabel("Review Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_monthly_review_trend.png", dpi=150)
    plt.close()

    # 2) 상위 상품코드
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_products, y="product_code", x="review_count", hue="city_en")
    plt.title("Top 20 Product Codes by Review Count")
    plt.xlabel("Review Count")
    plt.ylabel("Product Code")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_top_product_codes.png", dpi=150)
    plt.close()

    # 3) 테마 언급률 heatmap
    heat = theme_city.pivot(index="theme", columns="city_en", values="mention_rate_pct").fillna(0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Theme Mention Rate by City (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_theme_mentions_heatmap.png", dpi=150)
    plt.close()

    # 4) 일정 길이 분포
    plt.figure(figsize=(10, 5))
    sns.barplot(data=duration_city, x="trip_days", y="review_count", hue="city_en")
    plt.title("Trip Duration Distribution by City")
    plt.xlabel("Days")
    plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_trip_duration_distribution.png", dpi=150)
    plt.close()

    total_reviews = len(df)
    lines = []
    lines.append("# 리뷰 기반 비즈니스 추가 분석")
    lines.append("")
    lines.append(f"- 대상 리뷰 수: {total_reviews:,}건")
    lines.append(f"- 분석 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    lines.append("")
    lines.append("## 1) 도시별 KPI")
    lines.append("")
    lines.append("| 도시 | 리뷰수 | 상품코드수 | 긍정언급률(%) | 부정언급률(%) | 평균 일정일수 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in city_kpi.iterrows():
        lines.append(
            f"| {row['city']} | {int(row['reviews']):,} | {int(row['unique_products'])} | {row['positive_rate']:.2f} | {row['negative_rate']:.2f} | {row['avg_trip_days']:.2f} |"
        )

    lines.append("")
    lines.append("## 2) 핵심 관찰")
    lines.append("")
    top_city = city_kpi.sort_values("reviews", ascending=False).iloc[0]
    lines.append(f"- 리뷰 볼륨 1위 도시는 {top_city['city']} (리뷰 {int(top_city['reviews']):,}건)")
    best_positive = city_kpi.sort_values("positive_rate", ascending=False).iloc[0]
    lines.append(f"- 긍정언급률 최고 도시는 {best_positive['city']} ({best_positive['positive_rate']:.2f}%)")
    high_negative = city_kpi.sort_values("negative_rate", ascending=False).iloc[0]
    lines.append(f"- 부정언급률 점검 필요 도시는 {high_negative['city']} ({high_negative['negative_rate']:.2f}%)")
    lines.append("- 테마 언급률은 가이드/호텔/식사/일정/가격 축으로 비교")

    lines.append("")
    lines.append("## 3) 생성 산출물")
    lines.append("")
    lines.append("- CSV: biz_city_kpis.csv")
    lines.append("- CSV: biz_monthly_city_reviews.csv")
    lines.append("- CSV: biz_top_products_by_reviews.csv")
    lines.append("- CSV: biz_trip_duration_by_city.csv")
    lines.append("- CSV: biz_theme_mentions_by_city.csv")
    lines.append("- PNG: viz_monthly_review_trend.png")
    lines.append("- PNG: viz_top_product_codes.png")
    lines.append("- PNG: viz_theme_mentions_heatmap.png")
    lines.append("- PNG: viz_trip_duration_distribution.png")

    (OUT_DIR / "비즈니스_추가지표_분석_0328.md").write_text("\n".join(lines), encoding="utf-8")

    print("done")
    print(f"output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
