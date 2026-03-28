import csv
import re
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.lottetour.com"
PROMOTION_PAGE_URL = "https://www.lottetour.com/promotion/3682"
PROMOTION_LIST_URL = "https://www.lottetour.com/promotion/disp/list"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

PRICE_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})+\s*원\s*~?)")


@dataclass
class TourItem:
    title: str
    price: str
    url: str
    source: str


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_promotion_params(promotion_html: str) -> dict[str, str]:
    soup = BeautifulSoup(promotion_html, "html.parser")
    form = soup.select_one("#promotionForm")
    if not form:
        raise ValueError("promotionForm not found in promotion page")

    params: dict[str, str] = {}
    for inp in form.select("input[name]"):
        params[inp["name"]] = inp.get("value", "")

    if "exhiId" not in params or "godTemplateType" not in params:
        raise ValueError("Required params not found in promotion form")

    return params


def extract_items_from_html(html: str, source_url: str) -> list[TourItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[TourItem] = []

    for card in soup.select("#goods_list li .item_wrap"):
        title_el = card.select_one(".txt a strong")
        link_el = card.select_one(".txt a[href]")
        price_el = card.select_one(".txt .price")

        if not title_el or not link_el:
            continue

        title = normalize_text(title_el.get_text(" ", strip=True))
        if not title or "다낭" not in title:
            continue

        href = link_el.get("href", "").strip()
        absolute_url = urljoin(BASE_URL, href)
        context_text = normalize_text(card.get_text(" ", strip=True))

        price_text = normalize_text(price_el.get_text(" ", strip=True)) if price_el else ""
        price_match = PRICE_RE.search(price_text or context_text)
        price = price_match.group(1) if price_match else ""

        items.append(
            TourItem(
                title=title,
                price=price,
                url=absolute_url,
                source=source_url,
            )
        )

    dedup: dict[tuple[str, str], TourItem] = {}
    for item in items:
        key = (item.title, item.url)
        if key not in dedup:
            dedup[key] = item

    return list(dedup.values())


def save_csv(items: list[TourItem], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "price", "url", "source"])
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def main() -> None:
    promotion_html = fetch_html(PROMOTION_PAGE_URL)
    params = build_promotion_params(promotion_html)

    list_response = requests.get(
        PROMOTION_LIST_URL,
        params=params,
        headers={**HEADERS, "Referer": PROMOTION_PAGE_URL},
        timeout=15,
    )
    list_response.raise_for_status()

    results = extract_items_from_html(list_response.text, list_response.url)
    output_path = "danang_lottetour.csv"
    save_csv(results, output_path)

    print(f"총 수집 건수: {len(results)}")
    print(f"저장 파일: {output_path}")

    for index, item in enumerate(results[:10], start=1):
        print(f"{index}. {item.title} | {item.price} | {item.url}")


if __name__ == "__main__":
    main()
