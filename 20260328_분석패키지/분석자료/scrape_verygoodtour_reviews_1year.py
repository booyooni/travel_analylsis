from __future__ import annotations

import argparse
import ssl
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://m.verygoodtour.com/TravelInfo/GetReviewList"
REFERER_URL = "https://m.verygoodtour.com/TravelInfo/Review"
SSL_CONTEXT = ssl._create_unverified_context()


@dataclass
class ReviewRecord:
    board_seq: str
    title: str
    date: str
    writer: str
    product_code: str
    category: str
    product_title: str
    content: str
    keyword: str
    page: int


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\xa0", " ").split())


def parse_date_and_writer(raw_text: str) -> tuple[str, str]:
    text = clean_text(raw_text).replace(" 님", "")
    if not text:
        return "", ""
    parts = text.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], clean_text(parts[1])


def parse_date(raw_date: str) -> date | None:
    raw_date = clean_text(raw_date)
    if not raw_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw_date, fmt).date()
        except ValueError:
            continue
    return None


def fetch_review_page(
    page: int,
    keyword: str,
    category: str = "",
    region: str = "",
    page_size: int = 10,
) -> str:
    payload = urlencode(
        {
            "page": page,
            "pageSize": page_size,
            "regionName": region,
            "category": category,
            "serachText": keyword,
            "masterSeq": 1,
            "myFlag": "false",
        }
    ).encode()
    request = Request(
        BASE_URL,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": REFERER_URL,
        },
        method="POST",
    )
    with urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        return response.read().decode("utf-8")


def parse_review_cards(html: str, keyword: str, page: int) -> list[ReviewRecord]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(".review_wrap.toggle_wrap")
    reviews: list[ReviewRecord] = []

    for card in cards:
        review_prod = card.select_one(".review_prod")
        if review_prod is None:
            continue

        title_node = card.select_one(".review_tit > .tit")
        date_node = card.select_one(".date_person")
        content_node = card.select_one(".review_txt .cont")
        product_code_node = review_prod.select_one(".prod_code")
        category_node = review_prod.select_one(".prod_location")
        product_title_node = review_prod.select_one(".prod_tit")
        button_node = card.select_one("button.btn_toggle")

        title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
        raw_date_text = date_node.get_text(" ", strip=True) if date_node else ""
        date_text, writer = parse_date_and_writer(raw_date_text)
        content = clean_text(content_node.get_text(" ", strip=True) if content_node else "")
        product_code = clean_text(review_prod.get("data-product", "")) or clean_text(
            product_code_node.get_text(" ", strip=True) if product_code_node else ""
        )
        category = clean_text(category_node.get_text(" ", strip=True) if category_node else "")
        product_title = clean_text(product_title_node.get_text(" ", strip=True) if product_title_node else "")
        board_seq = clean_text(button_node.get("data-boardseq", "") if button_node else "")

        if not title or not date_text or not content:
            continue

        reviews.append(
            ReviewRecord(
                board_seq=board_seq,
                title=title,
                date=date_text,
                writer=writer,
                product_code=product_code,
                category=category,
                product_title=product_title,
                content=content,
                keyword=keyword,
                page=page,
            )
        )

    return reviews


def make_review_key(record: ReviewRecord) -> tuple[str, str, str, str, str]:
    return (
        record.board_seq,
        record.date,
        record.writer,
        record.title,
        record.product_title or record.content[:80],
    )


def text_matches_keywords(record: ReviewRecord, keywords: Iterable[str]) -> bool:
    haystack = " ".join([record.title, record.product_title, record.content, record.product_code])
    return any(keyword in haystack for keyword in keywords)


def collect_reviews(
    keywords: list[str],
    since_date: date,
    max_pages: int,
    category: str = "",
    region: str = "",
    page_size: int = 10,
) -> pd.DataFrame:
    collected: list[ReviewRecord] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for keyword in keywords:
        print(f"키워드 '{keyword}' 수집 시작")
        for page in range(1, max_pages + 1):
            html = fetch_review_page(page=page, keyword=keyword, category=category, region=region, page_size=page_size)
            page_reviews = parse_review_cards(html, keyword=keyword, page=page)
            if not page_reviews:
                print(f"- {page}페이지에서 리뷰가 없어 중단")
                break

            kept_count = 0
            oldest_on_page: date | None = None
            for record in page_reviews:
                record_date = parse_date(record.date)
                if record_date is None:
                    continue
                if oldest_on_page is None or record_date < oldest_on_page:
                    oldest_on_page = record_date
                if record_date < since_date:
                    continue
                if not text_matches_keywords(record, keywords):
                    continue
                key = make_review_key(record)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(record)
                kept_count += 1

            print(f"- {page}페이지 파싱 {len(page_reviews)}건 / 저장 {kept_count}건")
            if oldest_on_page and oldest_on_page < since_date:
                print(f"- {page}페이지 최소 날짜 {oldest_on_page}로 기준일 도달, 중단")
                break

    df = pd.DataFrame([asdict(record) for record in collected])
    if df.empty:
        return df
    return df.sort_values(by=["date", "board_seq", "title"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", required=True)
    parser.add_argument("--since-date", required=True)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--category", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keywords = [clean_text(keyword) for keyword in args.keywords.split(",") if clean_text(keyword)]
    since_date = datetime.strptime(args.since_date, "%Y-%m-%d").date()
    output_path = Path(args.output)

    df = collect_reviews(
        keywords=keywords,
        since_date=since_date,
        max_pages=max(1, args.max_pages),
        category=args.category,
        region=args.region,
        page_size=max(1, args.page_size),
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if df.empty:
        print(f"수집 결과가 없습니다: {output_path}")
        return

    print(df.head())
    print(f"총 {len(df)}개 리뷰 저장 완료: {output_path}")
    print(f"수집 기간: {df['date'].min()} ~ {df['date'].max()}")


if __name__ == "__main__":
    main()
