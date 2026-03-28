import csv
import math
import statistics
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.lottetour.com"
PAGE_SIZE = 10

DESTINATIONS = {
    "다낭": "휴양형",
    "나트랑": "휴양형",
    "싱가포르": "도시형",
}

COMMON_PARAMS = {
    "selectKey": "KEY",
    "subkwd": "",
    "subdatekwd": "",
    "minprice": "",
    "maxprice": "",
    "checkdtcnt": "",
    "checkday": "",
    "checkcomm": "",
    "checkairline": "",
    "checkclass": "",
    "checkstatus": "",
    "calenderKwd": "",
    "startPrice": "",
    "endPrice": "",
    "subcategory": "",
    "selectcategory": "globalpackage",
    "category": "globalpackage",
}


@dataclass
class TourRecord:
    keyword: str
    travel_style: str
    dep_dt: str
    arr_dt: str
    evt_status_nm: str
    evt_nm1: str
    evt_nm2: str
    god_nm1: str
    god_nm2: str
    menu_4lvl_nm: str
    price_adt: int
    evt_dt_cnt: int
    evt_night_cnt: int
    carrier: str
    dep_place: str
    event_code: str
    god_id: str
    detail_url: str


def parse_yyyymmdd(value: str) -> date | None:
    if not value or len(value) != 8 or not value.isdigit():
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def strip_html(text: str) -> str:
    if not text:
        return ""
    value = unescape(text)
    while "<" in value and ">" in value:
        start = value.find("<")
        end = value.find(">", start)
        if end == -1:
            break
        value = value[:start] + value[end + 1 :]
    return " ".join(value.split())


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_detail_url(item: dict[str, Any]) -> str:
    if item.get("detail_url"):
        return item["detail_url"]

    evt_cd = item.get("evevt_cd") or item.get("evt_cd")
    menu_1 = item.get("menu_1lvl_cd")
    menu_2 = item.get("menu_2lvl_cd")
    menu_3 = item.get("menu_3lvl_cd")
    menu_4 = item.get("menu_4lvl_cd")

    if evt_cd and menu_1 and menu_2 and menu_3 and menu_4:
        return f"{BASE}/evtDetail/{menu_1}/{menu_2}/{menu_3}/{menu_4}?evtCd={evt_cd}"

    return ""


def total_count_for_keyword(session: requests.Session, keyword: str) -> int:
    params = {"kwd": keyword, "category": "globalpackage"}
    response = session.get(
        f"{BASE}/search/totaljson",
        params=params,
        headers={"Referer": f"{BASE}/search?kwd={quote(keyword)}"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    for menu in data.get("menucnt", []):
        if menu.get("menuid") == "globalpackage":
            return safe_int(menu.get("totalcnt"))
    return 0


def fetch_page_items(session: requests.Session, keyword: str, page_num: int) -> list[dict[str, Any]]:
    params = {
        **COMMON_PARAMS,
        "kwd": keyword,
        "pageNum": str(page_num),
        "pageSize": str(PAGE_SIZE),
    }
    response = session.get(
        f"{BASE}/search/json",
        params=params,
        headers={"Referer": f"{BASE}/search?kwd={quote(keyword)}"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("ITEMS", []) if isinstance(data.get("ITEMS"), list) else []


def collect_keyword_records(session: requests.Session, keyword: str) -> list[TourRecord]:
    total = total_count_for_keyword(session, keyword)
    page_count = max(1, math.ceil(total / PAGE_SIZE))

    records: list[TourRecord] = []
    for page in range(1, page_count + 1):
        try:
            items = fetch_page_items(session, keyword, page)
        except requests.RequestException:
            continue
        if not items:
            continue

        for item in items:
            records.append(
                TourRecord(
                    keyword=keyword,
                    travel_style=DESTINATIONS[keyword],
                    dep_dt=str(item.get("dep_dt") or ""),
                    arr_dt=str(item.get("arr_dt") or ""),
                    evt_status_nm=str(item.get("evt_status_nm") or ""),
                    evt_nm1=strip_html(str(item.get("evt_nm1") or "")),
                    evt_nm2=strip_html(str(item.get("evt_nm2") or "")),
                    god_nm1=strip_html(str(item.get("god_nm1") or "")),
                    god_nm2=strip_html(str(item.get("god_nm2") or "")),
                    menu_4lvl_nm=str(item.get("menu_4lvl_nm") or ""),
                    price_adt=safe_int(item.get("price_adt")),
                    evt_dt_cnt=safe_int(item.get("evt_dt_cnt")),
                    evt_night_cnt=safe_int(item.get("evt_night_cnt")),
                    carrier=str(item.get("carrer_nm") or ""),
                    dep_place=str(item.get("dep_place_nm") or ""),
                    event_code=str(item.get("evevt_cd") or item.get("evt_cd") or ""),
                    god_id=str(item.get("god_id") or ""),
                    detail_url=build_detail_url(item),
                )
            )

    dedup: dict[str, TourRecord] = {}
    for rec in records:
        key = rec.event_code or f"{rec.god_id}_{rec.dep_dt}_{rec.price_adt}"
        if key not in dedup:
            dedup[key] = rec

    return list(dedup.values())


def write_csv(path: str, records: list[TourRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else [
            "keyword", "travel_style", "dep_dt", "arr_dt", "evt_status_nm", "evt_nm1", "evt_nm2",
            "god_nm1", "god_nm2", "menu_4lvl_nm", "price_adt", "evt_dt_cnt", "evt_night_cnt",
            "carrier", "dep_place", "event_code", "god_id", "detail_url"
        ])
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def summarize(records: list[TourRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[TourRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.travel_style, rec.keyword), []).append(rec)

    rows: list[dict[str, Any]] = []
    for (style, keyword), items in grouped.items():
        prices = [x.price_adt for x in items if x.price_adt > 0]
        days = [x.evt_dt_cnt for x in items if x.evt_dt_cnt > 0]
        nights = [x.evt_night_cnt for x in items if x.evt_night_cnt > 0]
        cancelled = sum(1 for x in items if "취소" in x.evt_status_nm)
        rows.append(
            {
                "travel_style": style,
                "keyword": keyword,
                "count": len(items),
                "avg_price": round(statistics.mean(prices), 0) if prices else 0,
                "median_price": round(statistics.median(prices), 0) if prices else 0,
                "min_price": min(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "avg_days": round(statistics.mean(days), 2) if days else 0,
                "avg_nights": round(statistics.mean(nights), 2) if nights else 0,
                "cancelled_ratio": round(cancelled / len(items), 3) if items else 0,
            }
        )

    rows.sort(key=lambda x: (x["travel_style"], x["keyword"]))
    return rows


def write_summary_csv(path: str, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def filter_last_two_years(records: list[TourRecord], today: date) -> list[TourRecord]:
    start = today - timedelta(days=365 * 2)
    filtered: list[TourRecord] = []
    for rec in records:
        dep = parse_yyyymmdd(rec.dep_dt)
        if dep is None:
            continue
        if start <= dep <= today:
            filtered.append(rec)
    return filtered


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def main() -> None:
    today = date.today()
    session = build_session()

    all_records: list[TourRecord] = []
    for keyword in DESTINATIONS:
        all_records.extend(collect_keyword_records(session, keyword))

    two_year_records = filter_last_two_years(all_records, today)

    raw_path = "resort_city_raw_all.csv"
    two_year_path = "resort_city_last2years.csv"
    summary_path = "resort_city_last2years_summary.csv"

    write_csv(raw_path, all_records)
    write_csv(two_year_path, two_year_records)

    summary_rows = summarize(two_year_records)
    write_summary_csv(summary_path, summary_rows)

    print(f"전체 수집 건수(중복제거): {len(all_records)}")
    print(f"최근 2년 출발일 필터 건수: {len(two_year_records)}")
    print(f"저장: {raw_path}, {two_year_path}, {summary_path}")

    if summary_rows:
        print("\n[최근 2년 비교 요약]")
        for row in summary_rows:
            print(
                f"- {row['travel_style']} | {row['keyword']} | 건수={row['count']} | "
                f"평균가={int(row['avg_price']):,}원 | 평균일수={row['avg_days']}일"
            )


if __name__ == "__main__":
    main()
