from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import Locator, Page, sync_playwright

URL = "https://m.verygoodtour.com/TravelInfo/Review"
OUTPUT_PATH = Path("verygoodtour_reviews.csv")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_inner_text(locator: Locator) -> str:
    try:
        return clean_text(locator.inner_text())
    except Exception:
        return ""


def safe_attribute(locator: Locator, name: str) -> str:
    try:
        if locator.count() == 0:
            return ""
        return clean_text(locator.first.get_attribute(name) or "")
    except Exception:
        return ""


def parse_date_and_writer(raw_text: str) -> tuple[str, str]:
    text = clean_text(raw_text).replace(" 님", "")
    match = re.match(r"(\d{4}[.-]\d{2}[.-]\d{2})\s*(.*)", text)
    if not match:
        return "", text
    return match.group(1), clean_text(match.group(2))


def apply_search_filters(
    page: Page,
    keyword: str = "",
    category: str = "",
    region: str = "",
) -> None:
    if category:
        try:
            page.locator("select#category").select_option(label=category)
        except Exception:
            pass

    if region:
        try:
            page.locator("select#region").select_option(label=region)
        except Exception:
            pass

    if keyword:
        try:
            page.locator("input#serachText").fill(keyword)
        except Exception:
            pass

    if keyword or category or region:
        page.locator("button:has-text('검색')").click()
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)


def expand_review_cards(page: Page) -> None:
    buttons = page.locator(".review_wrap.toggle_wrap button.btn_toggle:visible")
    for i in range(buttons.count()):
        button = buttons.nth(i)
        try:
            button.click(timeout=2000)
            page.wait_for_timeout(200)
        except Exception:
            continue


def extract_reviews_from_page(page: Page) -> list[dict[str, str]]:
    reviews: list[dict[str, str]] = []
    cards = page.locator(".review_wrap.toggle_wrap:has(.review_prod)")

    for i in range(cards.count()):
        card = cards.nth(i)
        title = safe_inner_text(card.locator(".review_tit > .tit").first)
        date, writer = parse_date_and_writer(
            safe_inner_text(card.locator(".review_tit .date_person").first)
        )
        product_code = safe_attribute(card.locator(".review_prod"), "data-product")
        category = safe_inner_text(card.locator(".review_prod .prod_location").first)
        product_title = safe_inner_text(card.locator(".review_prod .prod_tit").first)
        content = safe_inner_text(card.locator(".review_txt .cont").first)
        board_seq = safe_attribute(card.locator("button.btn_toggle"), "data-boardseq")

        if not title or not content:
            continue

        reviews.append(
            {
                "board_seq": board_seq,
                "title": title,
                "date": date,
                "writer": writer,
                "product_code": product_code,
                "category": category,
                "product_title": product_title,
                "content": content,
            }
        )

    return reviews


def go_to_next_page(page: Page, current_first_title: str) -> bool:
    next_button = page.locator("a.btnNext:visible").last
    if next_button.count() == 0:
        return False

    try:
        next_button.click(timeout=5000)
    except PlaywrightTimeoutError:
        try:
            next_button.evaluate("(element) => element.click()")
        except Exception:
            return False
    except Exception:
        return False

    for _ in range(20):
        page.wait_for_timeout(500)
        next_first_title = safe_inner_text(
            page.locator(".review_wrap.toggle_wrap .review_tit > .tit").first
        )
        if next_first_title and next_first_title != current_first_title:
            return True

    return False


def scrape_reviews(
    max_pages: int = 1,
    headless: bool = True,
    keyword: str = "",
    category: str = "",
    region: str = "",
) -> pd.DataFrame:
    all_reviews: list[dict[str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle", timeout=60000)
        apply_search_filters(page, keyword=keyword, category=category, region=region)

        for page_number in range(1, max_pages + 1):
            page.wait_for_timeout(1000)
            expand_review_cards(page)
            page_reviews = extract_reviews_from_page(page)
            all_reviews.extend(page_reviews)
            print(f"{page_number}페이지 리뷰 {len(page_reviews)}건 수집")

            if page_number == max_pages or not page_reviews:
                break

            current_first_title = page_reviews[0]["title"]
            moved = go_to_next_page(page, current_first_title=current_first_title)
            if not moved:
                break

        browser.close()

    return pd.DataFrame(all_reviews).drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--keyword", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)
    df = scrape_reviews(
        max_pages=max(1, args.max_pages),
        headless=not args.headed,
        keyword=args.keyword,
        category=args.category,
        region=args.region,
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(df.head())
    print(f"총 {len(df)}개 리뷰 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
