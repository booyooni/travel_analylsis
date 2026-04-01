from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hanatour_reviews.csv"
OUTPUT_MD = BASE_DIR / "hanatour_reviews_analysis.md"
CHART_DIR = BASE_DIR / "charts"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    preferred_fonts = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Nanum Gothic",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred_fonts if name in available), "DejaVu Sans")

    plt.rcParams["font.family"] = selected
    plt.rcParams["font.sans-serif"] = preferred_fonts
    plt.rcParams["axes.unicode_minus"] = False


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Some files include hidden spaces in Korean column names.
    normalized = df.copy()
    normalized.columns = [re.sub(r"\s+", "", str(c)) for c in normalized.columns]
    return normalized


def extract_schedule(product_name: str) -> str | None:
    if not isinstance(product_name, str):
        return None

    nights_days = re.search(r"(\d+)\s*박\s*(\d+)\s*일", product_name)
    if nights_days:
        return f"{nights_days.group(1)}박{nights_days.group(2)}일"

    days = re.search(r"(\d+)\s*일", product_name)
    if days:
        return f"{days.group(1)}일"

    return None


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace(".", "-", regex=False), errors="coerce")


def top_summary_keywords(series: pd.Series, top_n: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in series.dropna().astype(str):
        parts = [p.strip() for p in text.split(",") if p.strip()]
        counter.update(parts)
    return counter.most_common(top_n)


def top_content_tokens(series: pd.Series, top_n: int = 10) -> list[tuple[str, int]]:
    stopwords = {
        "가이드", "가이드님", "여행", "일정", "호텔", "음식", "관광", "관광지", "패키지", "하나투어",
        "정말", "너무", "진짜", "아주", "그리고", "또한", "이번", "덕분", "같아요", "있어요", "했어요",
        "좋았어요", "좋아요", "추천", "만족", "최고", "다낭", "나트랑", "싱가포르",
    }
    counter: Counter[str] = Counter()
    for text in series.dropna().astype(str):
        tokens = re.findall(r"[가-힣]{2,}", text)
        filtered = [tok for tok in tokens if tok not in stopwords and len(tok) >= 2]
        counter.update(filtered)
    return counter.most_common(top_n)


def make_city_volume_table(df: pd.DataFrame) -> str:
    city_stats = (
        df.groupby("대상도시")
        .agg(
            리뷰수=("리뷰ID", "count"),
            고유상품코드수=("상품코드", "nunique"),
            평균평점=("평점", "mean"),
        )
        .sort_values("리뷰수", ascending=False)
        .reset_index()
    )

    total = city_stats["리뷰수"].sum()
    city_stats["리뷰비중"] = city_stats["리뷰수"] / total

    rows = []
    for _, r in city_stats.iterrows():
        rows.append(
            [
                str(r["대상도시"]),
                f"{int(r['리뷰수']):,}",
                f"{r['리뷰비중']:.1%}",
                f"{int(r['고유상품코드수']):,}",
                f"{r['평균평점']:.2f}",
            ]
        )

    return md_table(["도시", "리뷰수", "리뷰비중", "고유여행코드수", "평균평점"], rows)


def make_city_schedule_tables(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    blocks: list[str] = []

    for city in df["대상도시"].dropna().value_counts().index:
        city_df = df[df["대상도시"] == city].copy()
        sched = (
            city_df["여행일정"]
            .fillna("미식별")
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        sched.columns = ["여행일정", "리뷰수"]
        sched["비중"] = sched["리뷰수"] / len(city_df)

        rows: list[list[str]] = []
        for _, r in sched.iterrows():
            rows.append([str(r["여행일정"]), f"{int(r['리뷰수']):,}", f"{r['비중']:.1%}"])

        blocks.append(f"### {city} 일정 분포 Top {top_n}")
        blocks.append("")
        blocks.append(md_table(["여행일정", "리뷰수", "비중"], rows))
        blocks.append("")

    return blocks


def make_city_code_tables(df: pd.DataFrame, top_n: int = 7) -> list[str]:
    blocks: list[str] = []

    for city in df["대상도시"].dropna().value_counts().index:
        city_df = df[df["대상도시"] == city].copy()
        code_stats = (
            city_df.groupby("상품코드")
            .agg(
                리뷰수=("리뷰ID", "count"),
                평균평점=("평점", "mean"),
                대표상품명=("상품명", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            )
            .sort_values(["리뷰수", "평균평점"], ascending=[False, False])
            .head(top_n)
            .reset_index()
        )

        rows: list[list[str]] = []
        for _, r in code_stats.iterrows():
            rows.append(
                [
                    str(r["상품코드"]),
                    f"{int(r['리뷰수']):,}",
                    f"{r['평균평점']:.2f}",
                    str(r["대표상품명"])[:46] + ("..." if len(str(r["대표상품명"])) > 46 else ""),
                ]
            )

        blocks.append(f"### {city} 여행코드 Top {top_n}")
        blocks.append("")
        blocks.append(md_table(["여행코드", "리뷰수", "평균평점", "대표상품명(요약)"], rows))
        blocks.append("")

    return blocks


def make_keyword_tables(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    blocks: list[str] = []

    for city in df["대상도시"].dropna().value_counts().index:
        city_df = df[df["대상도시"] == city].copy()

        summary_kw = top_summary_keywords(city_df["리뷰요약"], top_n=top_n)
        content_kw = top_content_tokens(city_df["내용"], top_n=top_n)

        blocks.append(f"### {city} 키워드")
        blocks.append("")

        sum_rows = [[w, f"{c:,}"] for w, c in summary_kw]
        blocks.append("- 리뷰요약 기반 상위 키워드")
        blocks.append("")
        blocks.append(md_table(["키워드", "언급수"], sum_rows))
        blocks.append("")

        cont_rows = [[w, f"{c:,}"] for w, c in content_kw]
        blocks.append("- 리뷰본문 기반 상위 토큰")
        blocks.append("")
        blocks.append(md_table(["토큰", "빈도"], cont_rows))
        blocks.append("")

    return blocks


def make_monthly_volume_table(df: pd.DataFrame, months: int = 12) -> str:
    monthly = (
        df.dropna(subset=["작성월"])
        .groupby(["작성월", "대상도시"])
        .size()
        .reset_index(name="리뷰수")
    )

    latest_month = monthly["작성월"].max()
    if pd.isna(latest_month):
        return "월별 리뷰 데이터가 없습니다."

    start_month = latest_month - pd.offsets.MonthBegin(months - 1)
    monthly = monthly[monthly["작성월"] >= start_month].copy()

    pivot = (
        monthly.pivot(index="작성월", columns="대상도시", values="리뷰수")
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    rows: list[list[str]] = []
    for idx, row in pivot.iterrows():
        row_values = [idx.strftime("%Y-%m")]
        row_values.extend([f"{int(v):,}" for v in row.tolist()])
        row_values.append(f"{int(row.sum()):,}")
        rows.append(row_values)

    headers = ["월"] + [str(c) for c in pivot.columns.tolist()] + ["합계"]
    return md_table(headers, rows)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tfidf_sentiment_keywords(df: pd.DataFrame, top_n: int = 12) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    labeled = df[df["평점"].notna()].copy()
    labeled = labeled[(labeled["평점"] >= 4.5) | (labeled["평점"] <= 3.0)].copy()
    if labeled.empty:
        return [], []

    labeled["감성"] = labeled["평점"].map(lambda x: "긍정" if x >= 4.5 else "부정")
    labeled["분석텍스트"] = (
        labeled["리뷰요약"].fillna("").astype(str) + " " + labeled["내용"].fillna("").astype(str)
    ).map(clean_text)
    labeled = labeled[labeled["분석텍스트"].str.len() > 0].copy()
    if labeled.empty:
        return [], []

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2), max_features=7000)
    X = vectorizer.fit_transform(labeled["분석텍스트"])
    terms = vectorizer.get_feature_names_out()

    pos_idx = labeled["감성"].values == "긍정"
    neg_idx = labeled["감성"].values == "부정"
    if not pos_idx.any() or not neg_idx.any():
        return [], []

    pos_mean = X[pos_idx].mean(axis=0).A1
    neg_mean = X[neg_idx].mean(axis=0).A1
    delta = pos_mean - neg_mean

    pos_order = delta.argsort()[::-1][:top_n]
    neg_order = delta.argsort()[:top_n]

    pos_terms = [(str(terms[i]), float(delta[i])) for i in pos_order if delta[i] > 0]
    neg_terms = [(str(terms[i]), float(-delta[i])) for i in neg_order if delta[i] < 0]
    return pos_terms, neg_terms


def extract_city_negative_keywords(df: pd.DataFrame, top_n: int = 8) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for city in df["대상도시"].dropna().value_counts().index:
        city_df = df[(df["대상도시"] == city) & (df["평점"].notna()) & (df["평점"] <= 3.0)].copy()
        if len(city_df) < 15:
            result[city] = []
            continue

        city_df["분석텍스트"] = (
            city_df["리뷰요약"].fillna("").astype(str) + " " + city_df["내용"].fillna("").astype(str)
        ).map(clean_text)
        city_df = city_df[city_df["분석텍스트"].str.len() > 0].copy()
        if city_df.empty:
            result[city] = []
            continue

        vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=3000)
        X = vec.fit_transform(city_df["분석텍스트"])
        mean_w = X.mean(axis=0).A1
        terms = vec.get_feature_names_out()
        order = mean_w.argsort()[::-1][:top_n]
        result[city] = [(str(terms[i]), float(mean_w[i])) for i in order]

    return result


def extract_city_similar_review_pairs(df: pd.DataFrame, top_n: int = 5, max_docs_per_city: int = 1800) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}

    for city in df["대상도시"].dropna().value_counts().index:
        city_df = df[df["대상도시"] == city].copy()
        city_df = city_df.sort_values("작성일_dt", ascending=False)
        if len(city_df) > max_docs_per_city:
            city_df = city_df.head(max_docs_per_city).copy()

        city_df["리뷰요약"] = city_df["리뷰요약"].fillna("").astype(str)
        city_df["분석텍스트"] = (
            city_df["리뷰요약"].fillna("").astype(str) + " " + city_df["내용"].fillna("").astype(str)
        ).map(clean_text)
        city_df = city_df[city_df["분석텍스트"].str.len() > 0].copy()
        city_df = city_df[city_df["리뷰요약"].str.strip().str.len() > 0].copy()
        city_df = city_df.drop_duplicates(subset=["리뷰ID", "분석텍스트"]).copy()
        city_df = city_df.reset_index(drop=True)

        if len(city_df) < 3:
            result[city] = []
            continue

        vec = TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1, 2), max_features=8000)
        X = vec.fit_transform(city_df["분석텍스트"])

        nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
        nn.fit(X)
        distances, indices = nn.kneighbors(X, return_distance=True)

        pairs: list[dict[str, str]] = []
        used_pairs: set[tuple[int, int]] = set()
        for i in range(X.shape[0]):
            j = int(indices[i, 1])
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            sim = 1.0 - float(distances[i, 1])
            review1 = str(city_df.loc[pair[0], "리뷰ID"])
            review2 = str(city_df.loc[pair[1], "리뷰ID"])
            if review1 == review2:
                continue
            pairs.append(
                {
                    "similarity": f"{sim:.4f}",
                    "review1": review1,
                    "review2": review2,
                    "summary1": str(city_df.loc[pair[0], "리뷰요약"])[:40],
                    "summary2": str(city_df.loc[pair[1], "리뷰요약"])[:40],
                }
            )

        pairs = sorted(pairs, key=lambda x: float(x["similarity"]), reverse=True)[:top_n]
        result[city] = pairs

    return result


def build_tfidf_sections(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    pos_terms, neg_terms = extract_tfidf_sentiment_keywords(df, top_n=12)
    city_neg = extract_city_negative_keywords(df, top_n=8)
    city_pairs = extract_city_similar_review_pairs(df, top_n=5, max_docs_per_city=1800)

    lines.append("## 6) TF-IDF 기반 긍정/부정 키워드 분석")
    lines.append("")
    lines.append("- 기준: 평점 4.5 이상=긍정, 3.0 이하=부정")
    lines.append("- 방법: 리뷰요약+리뷰본문 텍스트 TF-IDF 벡터화 후 클래스 평균 가중치 차이(긍정-부정) 계산")
    lines.append("")

    pos_rows = [[w, f"{s:.4f}"] for w, s in pos_terms] if pos_terms else [["데이터 부족", "-"]]
    neg_rows = [[w, f"{s:.4f}"] for w, s in neg_terms] if neg_terms else [["데이터 부족", "-"]]

    lines.append("### 전체 긍정 대표 키워드")
    lines.append("")
    lines.append(md_table(["키워드", "긍정우세점수"], pos_rows))
    lines.append("")

    lines.append("### 전체 부정 대표 키워드")
    lines.append("")
    lines.append(md_table(["키워드", "부정우세점수"], neg_rows))
    lines.append("")

    lines.append("### 도시별 부정 키워드 (평점 3.0 이하 리뷰)")
    lines.append("")
    for city in df["대상도시"].dropna().value_counts().index:
        lines.append(f"#### {city}")
        lines.append("")
        kw_rows = [[w, f"{s:.4f}"] for w, s in city_neg.get(city, [])]
        if not kw_rows:
            kw_rows = [["데이터 부족", "-"]]
        lines.append(md_table(["키워드", "평균TF-IDF"], kw_rows))
        lines.append("")

    lines.append("## 7) 코사인 유사도 기반 유사 리뷰 페어")
    lines.append("")
    lines.append("- 방법: 도시별 최신 리뷰(최대 1,800건) TF-IDF 벡터화 후 최근접 이웃(코사인) 탐색")
    lines.append("- 활용: 중복 VOC(동일 불만/동일 칭찬) 묶음 탐지 및 개선과제 우선순위화")
    lines.append("")

    for city in df["대상도시"].dropna().value_counts().index:
        lines.append(f"### {city} 유사 리뷰 Top 5")
        lines.append("")
        pairs = city_pairs.get(city, [])
        if not pairs:
            lines.append(md_table(["Similarity", "ReviewID-1", "ReviewID-2", "요약1", "요약2"], [["-", "-", "-", "데이터 부족", "-"]]))
            lines.append("")
            continue

        pair_rows = [[p["similarity"], p["review1"], p["review2"], p["summary1"], p["summary2"]] for p in pairs]
        lines.append(md_table(["Similarity", "ReviewID-1", "ReviewID-2", "요약1", "요약2"], pair_rows))
        lines.append("")

    lines.append("## TF-IDF/유사도 기반 추가 인사이트")
    lines.append("")
    lines.append("- 긍정 키워드와 부정 키워드의 분리도가 높은 항목은 상품 상세페이지 메시지와 운영 KPI의 핵심 후보입니다.")
    lines.append("- 도시별 부정 키워드는 지역 특화 개선과제(예: 이동 동선, 식사, 가이드 운영)를 설계하는 근거로 사용할 수 있습니다.")
    lines.append("- 코사인 유사 리뷰 페어는 VOC를 테마 단위로 묶어, 건별 대응이 아닌 묶음 대응(템플릿 개선)에 활용 가능합니다.")
    lines.append("")

    return lines


def save_city_priority_bubble(df: pd.DataFrame) -> Path:
    city_stats = (
        df.groupby("대상도시")
        .agg(
            리뷰수=("리뷰ID", "count"),
            평균평점=("평점", "mean"),
            저평점비중=("평점", lambda s: float((s <= 3.0).mean())),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = (city_stats["리뷰수"] / max(1, city_stats["리뷰수"].max()) * 2400) + 300
    palette = sns.color_palette("Set2", n_colors=len(city_stats))

    ax.scatter(
        city_stats["평균평점"],
        city_stats["저평점비중"] * 100,
        s=sizes,
        c=palette,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.8,
    )

    for _, row in city_stats.iterrows():
        ax.text(
            row["평균평점"] + 0.01,
            row["저평점비중"] * 100 + 0.1,
            f"{row['대상도시']} ({int(row['리뷰수']):,})",
            fontsize=9,
        )

    ax.set_title("도시별 개선 우선순위 버블")
    ax.set_xlabel("평균평점")
    ax.set_ylabel("저평점 비중 (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = CHART_DIR / "city_priority_bubble.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_negative_keyword_heatmap(df: pd.DataFrame) -> Path:
    city_neg = extract_city_negative_keywords(df, top_n=12)
    all_terms = []
    for city, items in city_neg.items():
        for term, _ in items:
            if term not in all_terms:
                all_terms.append(term)

    top_terms = all_terms[:12]
    cities = [c for c in df["대상도시"].dropna().value_counts().index.tolist() if c in city_neg]

    matrix = pd.DataFrame(0.0, index=cities, columns=top_terms)
    for city in cities:
        for term, score in city_neg.get(city, []):
            if term in matrix.columns:
                matrix.loc[city, term] = score

    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("도시별 부정 키워드 강도 히트맵 (평점 3.0 이하)")
    ax.set_xlabel("키워드")
    ax.set_ylabel("도시")
    fig.tight_layout()

    out_path = CHART_DIR / "city_negative_keyword_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_monthly_low_rating_trend(df: pd.DataFrame) -> Path:
    monthly = (
        df.dropna(subset=["작성월", "평점"])
        .groupby(["작성월", "대상도시"])
        .agg(
            리뷰수=("리뷰ID", "count"),
            저평점비중=("평점", lambda s: float((s <= 3.0).mean())),
        )
        .reset_index()
    )

    if monthly.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.set_title("월별 저평점 비중 추이")
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
        fig.tight_layout()
        out_path = CHART_DIR / "monthly_low_rating_trend.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    latest_month = monthly["작성월"].max()
    start_month = latest_month - pd.offsets.MonthBegin(11)
    monthly = monthly[monthly["작성월"] >= start_month].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    sns.lineplot(
        data=monthly,
        x="작성월",
        y="리뷰수",
        hue="대상도시",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("최근 12개월 도시별 리뷰량")
    axes[0].set_xlabel("월")
    axes[0].set_ylabel("리뷰수")
    axes[0].tick_params(axis="x", rotation=30)

    sns.lineplot(
        data=monthly,
        x="작성월",
        y=monthly["저평점비중"] * 100,
        hue="대상도시",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("최근 12개월 도시별 저평점 비중")
    axes[1].set_xlabel("월")
    axes[1].set_ylabel("저평점 비중 (%)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_path = CHART_DIR / "monthly_low_rating_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_visualization_section(df: pd.DataFrame) -> list[str]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    bubble_path = save_city_priority_bubble(df)
    heatmap_path = save_negative_keyword_heatmap(df)
    trend_path = save_monthly_low_rating_trend(df)

    lines: list[str] = []
    lines.append("## 8) 개선포인트 시각화")
    lines.append("")
    lines.append("### 도시별 개선 우선순위 버블")
    lines.append("")
    lines.append(f"![도시별 개선 우선순위 버블](charts/{bubble_path.name})")
    lines.append("")
    lines.append("- 우상단(저평점 비중 높고 평점 낮음)으로 갈수록 우선 개선 필요")
    lines.append("")

    lines.append("### 도시별 부정 키워드 히트맵")
    lines.append("")
    lines.append(f"![도시별 부정 키워드 히트맵](charts/{heatmap_path.name})")
    lines.append("")
    lines.append("- 도시마다 강하게 나타나는 부정 신호 키워드를 빠르게 비교")
    lines.append("")

    lines.append("### 월별 리뷰량/저평점 비중 추이")
    lines.append("")
    lines.append(f"![월별 리뷰량 저평점 비중 추이](charts/{trend_path.name})")
    lines.append("")
    lines.append("- 특정 월 급증한 저평점 구간을 찾아 운영 이슈 시점과 연결 분석")
    lines.append("")

    return lines


def build_insights(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    city_counts = df["대상도시"].value_counts()
    city_scores = df.groupby("대상도시")["평점"].mean().sort_values(ascending=False)

    top_city = city_counts.index[0] if not city_counts.empty else "-"
    top_city_ratio = city_counts.iloc[0] / city_counts.sum() if not city_counts.empty else 0

    best_score_city = city_scores.index[0] if not city_scores.empty else "-"
    best_score = city_scores.iloc[0] if not city_scores.empty else 0

    lines.append("## 비즈니스 인사이트")
    lines.append("")
    lines.append(f"- 리뷰 볼륨은 {top_city}에 집중되어 있으며 비중은 {top_city_ratio:.1%}입니다. 해당 도시는 상품 개선 실험(A/B) 우선 적용 후보입니다.")
    lines.append(f"- 평균 평점은 {best_score_city}이(가) 가장 높고({best_score:.2f}), 고평점 구조를 다른 도시 상품 기획에 이식할 여지가 있습니다.")
    lines.append("- 리뷰요약에서는 '일정 알참/가이드 품질/객실 청결/현지 음식' 요소가 반복적으로 확인되어, 판매 페이지 핵심 메시지로 활용 가치가 높습니다.")
    lines.append("- 여행코드별 리뷰 쏠림이 커서, 상위 코드의 강점을 템플릿화해 중하위 코드로 확산하면 상품 포트폴리오 전체 전환 개선이 가능합니다.")
    lines.append("")

    lines.append("## 개선된 여행상품 개발 제안")
    lines.append("")
    lines.append("- 일정 설계: 도시별 주력 일정(예: 5일) 중심으로 핵심 동선을 표준화하고, 선택옵션은 피로도 낮은 모듈형으로 재구성")
    lines.append("- 가이드 운영: 리뷰 상위 여행코드의 가이드 운영 방식(설명 밀도, 케어 방식, 사진 지원)을 운영 매뉴얼로 문서화")
    lines.append("- 숙소/식사: '객실 청결', '현지 음식' 키워드가 높은 상품은 프리미엄 라인으로, 낮은 상품은 협력사 SLA 재협상")
    lines.append("- 리뷰 KPI: 도시-여행코드 단위로 월간 리뷰량, 평점, 핵심 키워드 언급률(일정/가이드/객실/음식)을 대시보드 지표로 관리")

    return lines


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)

    required_columns = ["상품코드", "대상도시", "상품명", "리뷰ID", "평점", "내용", "작성일", "리뷰요약"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["평점"] = pd.to_numeric(df["평점"], errors="coerce")
    df["작성일_dt"] = parse_date(df["작성일"])
    df["작성월"] = df["작성일_dt"].dt.to_period("M").dt.to_timestamp()
    df["여행일정"] = df["상품명"].map(extract_schedule)

    lines: list[str] = []
    lines.append("# 하나여행 리뷰 분석 보고서")
    lines.append("")
    lines.append(f"- 생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 분석데이터: 하나여행/data/hanatour_reviews.csv")
    lines.append(f"- 전체 리뷰수: {len(df):,}건")
    lines.append(f"- 도시 수: {df['대상도시'].nunique():,}개")
    lines.append(f"- 여행코드 수: {df['상품코드'].nunique():,}개")
    min_date = df["작성일_dt"].min()
    max_date = df["작성일_dt"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        lines.append(f"- 리뷰 기간: {min_date.date()} ~ {max_date.date()}")
    lines.append("")

    lines.append("## 분석 목적")
    lines.append("")
    lines.append("- 도시별 리뷰 데이터를 기반으로 고객이 반응하는 일정/상품코드/경험요소를 도출")
    lines.append("- 리뷰량과 키워드 신호를 통해 상품 개선 우선순위를 정의")
    lines.append("- 결과를 신규/개선 여행상품 기획(일정 설계, 운영 품질, 커뮤니케이션)에 반영")
    lines.append("")

    lines.append("## 1) 도시별 리뷰량 현황")
    lines.append("")
    lines.append(make_city_volume_table(df))
    lines.append("")

    lines.append("## 2) 도시별 여행일정 분포")
    lines.append("")
    lines.extend(make_city_schedule_tables(df, top_n=5))

    lines.append("## 3) 도시별 여행코드 반응(리뷰수 기준)")
    lines.append("")
    lines.extend(make_city_code_tables(df, top_n=7))

    lines.append("## 4) 리뷰 키워드 분석")
    lines.append("")
    lines.append("- 리뷰요약: 고객이 선택한 구조화 문구 기반")
    lines.append("- 리뷰본문: 자유서술 내용에서 상위 토큰 추출")
    lines.append("")
    lines.extend(make_keyword_tables(df, top_n=10))

    lines.append("## 5) 최근 12개월 월별 리뷰량")
    lines.append("")
    lines.append(make_monthly_volume_table(df, months=12))
    lines.append("")

    lines.extend(build_tfidf_sections(df))
    lines.extend(build_visualization_section(df))

    lines.extend(build_insights(df))
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
