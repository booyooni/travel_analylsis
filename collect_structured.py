"""
롯데관광 여행 상품 구조화 수집기
──────────────────────────────────────────
TB_AGENCY        여행사정보   (1건, 롯데관광 고정)
TB_PRODUCT       상품정보     (god_id 기준 중복제거)
TB_PRICE_SCHEDULE 가격/일정정보 (event_cd 기준, 상품 1건당 N개)

수집 대상 키워드: 다낭(휴양형) | 나트랑(휴양형) | 싱가포르(도시형)
출력 파일: tb_agency.csv, tb_product.csv, tb_price_schedule.csv
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
BASE_URL = "https://www.lottetour.com"
PAGE_SIZE = 10
OUT_DIR = Path(__file__).parent

DESTINATIONS: dict[str, str] = {
    "다낭": "휴양형",
    "나트랑": "휴양형",
    "싱가포르": "도시형",
}

SEARCH_PARAMS_COMMON: dict[str, str] = {
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


# ──────────────────────────────────────────
# 테이블 데이터 클래스
# ──────────────────────────────────────────
@dataclass
class Agency:
    """TB_AGENCY 여행사정보"""
    agency_id: int           # PK
    agency_nm: str           # 여행사명
    agency_url: str          # 홈페이지


@dataclass
class Product:
    """TB_PRODUCT 상품정보"""
    god_id: str              # PK (상품번호)
    agency_id: int           # FK → TB_AGENCY
    god_cd: str              # 내부 상품코드 (B27A-076xxxx)
    god_nm1: str             # 상품명 (메인)
    god_nm2: str             # 상품명 (서브)
    god_title: str           # 상품 부제목
    god_point: str           # 상품 설명·특징
    grd_nm: str              # 상품등급 (정통/실속/품격/HIGH&)
    god_gb1_nm: str          # 상품 대분류 (패키지)
    god_gb2_nm: str          # 상품 소분류 (일반패키지)
    area_mc_nm: str          # 지역 대분류 (동남아)
    area_sc_nm: str          # 지역 소분류 (다낭)
    city_list: str           # 방문 도시 목록
    thema_nm: str            # 테마
    menu_4lvl_nm: str        # 최하위 메뉴명 (다낭/호이안)
    search_keyword: str      # 검색 키워드
    travel_style: str        # 여행유형 (휴양형/도시형)
    image_url: str           # 썸네일 이미지 URL
    reg_dt: str              # 상품 등록일
    upd_dt: str              # 상품 최종수정일


@dataclass
class PriceSchedule:
    """TB_PRICE_SCHEDULE 가격/일정정보"""
    event_cd: str            # PK (행사코드)
    god_id: str              # FK → TB_PRODUCT
    dep_dt: str              # 출발일 (YYYY-MM-DD)
    arr_dt: str              # 도착일 (YYYY-MM-DD)
    dep_place_nm: str        # 출발 도시/공항
    depart_deptm: str        # 출발편 출발시간 (HH:MM)
    retn_deptm: str          # 귀국편 출발시간 (HH:MM)
    retn_arrtm: str          # 귀국편 도착시간 (HH:MM)
    evt_dt_cnt: int          # 총 여행 일수
    evt_night_cnt: int       # 박수
    carrer_nm: str           # 이용 항공사명
    carrer_abbr: str         # 항공사 코드 (KE, TW …)
    carry_apo_nm: str        # 항공편명
    price_adt: int           # 성인 가격 (원)
    price_chd: int           # 아동 가격 (원)
    price_inf: int           # 유아 가격 (원)
    evt_person_cnt: int      # 잔여 인원
    evt_status_nm: str       # 행사 상태 (예약가능/마감/행사취소 …)
    evt_nm1: str             # 행사명 1행
    evt_nm2: str             # 행사명 2행
    upd_dt_evt: str          # 행사 최종수정일


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────
_HTML_TAG = re.compile(r"<[^>]+>")


def clean(text: Any) -> str:
    """HTML 태그 제거 + 공백 정규화"""
    if not text:
        return ""
    return " ".join(_HTML_TAG.sub(" ", unescape(str(text))).split())


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fmt_time(hhmm: Any) -> str:
    """'1855' → '18:55', '0100' → '01:00'"""
    s = str(hhmm or "").zfill(4)
    if len(s) >= 4 and s.isdigit():
        return f"{s[:2]}:{s[2:4]}"
    return str(hhmm or "")


def fmt_date(yyyymmdd: Any) -> str:
    """'20260328' → '2026-03-28'"""
    s = str(yyyymmdd or "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def build_detail_url(item: dict[str, Any]) -> str:
    """상품 상세 페이지 URL 구성"""
    ec = item.get("evevt_cd") or ""
    m2 = item.get("menu_2lvl_id") or ""
    m3 = item.get("menu_3lvl_id") or ""
    m4 = item.get("menu_4lvl_id") or ""
    if ec and m2 and m3 and m4:
        return (
            f"{BASE_URL}/evtDetail/826/{m2}/{m3}/{m4}?evtCd={ec}"
        )
    god = item.get("god_id") or ""
    area = item.get("area_sc_cd") or ""
    if god:
        return f"{BASE_URL}/evtList/826/{area}?godId={god}"
    return ""


# ──────────────────────────────────────────
# HTTP 세션
# ──────────────────────────────────────────
def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


# ──────────────────────────────────────────
# API 호출
# ──────────────────────────────────────────
def get_total_count(session: requests.Session, keyword: str) -> int:
    r = session.get(
        f"{BASE_URL}/search/totaljson",
        params={"kwd": keyword, "category": "globalpackage"},
        headers={"Referer": f"{BASE_URL}/search?kwd={quote(keyword)}"},
        timeout=20,
    )
    r.raise_for_status()
    for m in r.json().get("menucnt", []):
        if m.get("menuid") == "globalpackage":
            return to_int(m.get("totalcnt"))
    return 0


def get_page(session: requests.Session, keyword: str, page: int) -> list[dict]:
    params = {
        **SEARCH_PARAMS_COMMON,
        "kwd": keyword,
        "pageNum": str(page),
        "pageSize": str(PAGE_SIZE),
    }
    r = session.get(
        f"{BASE_URL}/search/json",
        params=params,
        headers={"Referer": f"{BASE_URL}/search?kwd={quote(keyword)}"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("ITEMS") or []


# ──────────────────────────────────────────
# 변환: API 응답 → 데이터 클래스
# ──────────────────────────────────────────
AGENCY_INSTANCE = Agency(
    agency_id=1,
    agency_nm="롯데관광",
    agency_url=BASE_URL,
)


def item_to_product(item: dict, keyword: str, style: str) -> Product:
    return Product(
        god_id=str(item.get("god_id") or ""),
        agency_id=AGENCY_INSTANCE.agency_id,
        god_cd=str(item.get("god_cd") or ""),
        god_nm1=clean(item.get("god_nm1")),
        god_nm2=clean(item.get("god_nm2")),
        god_title=clean(item.get("god_title")),
        god_point=clean(item.get("god_point")),
        grd_nm=str(item.get("grd_nm") or ""),
        god_gb1_nm=str(item.get("god_gb1_nm") or ""),
        god_gb2_nm=str(item.get("god_gb2_nm") or ""),
        area_mc_nm=str(item.get("area_mc_nm") or ""),
        area_sc_nm=str(item.get("area_sc_nm") or ""),
        city_list=str(item.get("city_list") or ""),
        thema_nm=str(item.get("thema_nm") or ""),
        menu_4lvl_nm=str(item.get("menu_4lvl_nm") or ""),
        search_keyword=keyword,
        travel_style=style,
        image_url=str(item.get("durl1") or ""),
        reg_dt=str(item.get("reg_dt_god") or ""),
        upd_dt=str(item.get("upd_dt_god") or ""),
    )


def item_to_price_schedule(item: dict) -> PriceSchedule:
    return PriceSchedule(
        event_cd=str(item.get("evevt_cd") or ""),
        god_id=str(item.get("god_id") or ""),
        dep_dt=fmt_date(item.get("dep_dt")),
        arr_dt=fmt_date(item.get("arr_dt")),
        dep_place_nm=str(item.get("dep_place_nm") or ""),
        depart_deptm=fmt_time(item.get("depart_deptm")),
        retn_deptm=fmt_time(item.get("retn_deptm")),
        retn_arrtm=fmt_time(item.get("retn_arrtm")),
        evt_dt_cnt=to_int(item.get("evt_dt_cnt")),
        evt_night_cnt=to_int(item.get("evt_night_cnt")),
        carrer_nm=str(item.get("carrer_nm") or ""),
        carrer_abbr=str(item.get("carrer_abbr") or ""),
        carry_apo_nm=str(item.get("carry_apo_nm") or ""),
        price_adt=to_int(item.get("price_adt")),
        price_chd=to_int(item.get("price_chd")),
        price_inf=to_int(item.get("price_inf")),
        evt_person_cnt=to_int(item.get("evt_person_cnt")),
        evt_status_nm=str(item.get("evt_status_nm") or ""),
        evt_nm1=clean(item.get("evt_nm1")),
        evt_nm2=clean(item.get("evt_nm2")),
        upd_dt_evt=str(item.get("upd_dt_evt") or ""),
    )


# ──────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────
def write_csv(path: Path, rows: list, header: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if hasattr(row, "__dataclass_fields__") else row)
    print(f"  ✓ {path.name}: {len(rows)}행")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main() -> None:
    session = make_session()

    products: dict[str, Product] = {}          # god_id → Product (중복제거)
    schedules: dict[str, PriceSchedule] = {}   # event_cd → PriceSchedule (중복제거)

    for keyword, style in DESTINATIONS.items():
        total = get_total_count(session, keyword)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        print(f"[{keyword}] 총 {total}건 / {pages}페이지 수집 시작...")

        for page in range(1, pages + 1):
            try:
                items = get_page(session, keyword, page)
            except requests.RequestException as e:
                print(f"  페이지 {page} 실패: {e}")
                continue

            for item in items:
                # 상품정보: god_id 기준 (첫 번째만 저장)
                gid = str(item.get("god_id") or "")
                if gid and gid not in products:
                    products[gid] = item_to_product(item, keyword, style)

                # 가격/일정정보: event_cd 기준
                ec = str(item.get("evevt_cd") or "")
                if ec and ec not in schedules:
                    schedules[ec] = item_to_price_schedule(item)

        print(f"  → 누적 상품={len(products)}, 행사={len(schedules)}")

    # ─── CSV 출력 ────────────────────────────
    print("\n파일 저장 중...")
    prod_list = list(products.values())
    sched_list = list(schedules.values())

    write_csv(
        OUT_DIR / "tb_agency.csv",
        [AGENCY_INSTANCE],
        list(asdict(AGENCY_INSTANCE).keys()),
    )
    write_csv(
        OUT_DIR / "tb_product.csv",
        prod_list,
        list(asdict(prod_list[0]).keys()) if prod_list else [],
    )
    write_csv(
        OUT_DIR / "tb_price_schedule.csv",
        sched_list,
        list(asdict(sched_list[0]).keys()) if sched_list else [],
    )

    # ─── 간단 통계 ────────────────────────────
    print("\n[수집 결과 요약]")
    from collections import Counter
    kw_prod = Counter(p.search_keyword for p in prod_list)
    kw_sched = Counter(schedules[ec].god_id for ec in schedules
                       if schedules[ec].god_id in products)
    style_count = Counter(p.travel_style for p in prod_list)

    print(f"  여행사: 1 (롯데관광)")
    print(f"  상품 수 (상품정보): {len(prod_list)}")
    print(f"    키워드별: {dict(kw_prod)}")
    print(f"    여행유형별: {dict(style_count)}")
    print(f"  행사 수 (가격/일정정보): {len(sched_list)}")

    if sched_list:
        a_prices = [s.price_adt for s in sched_list if s.price_adt > 0]
        style_avg: dict[str, list[int]] = {}
        for s in sched_list:
            p = products.get(s.god_id)
            if p and s.price_adt > 0:
                style_avg.setdefault(p.travel_style, []).append(s.price_adt)
        print("\n  [여행유형별 성인 평균 가격]")
        for style, prices in style_avg.items():
            avg = int(sum(prices) / len(prices))
            print(f"    {style}: {avg:,}원 (n={len(prices)})")

        carrier_cnt = Counter(s.carrer_nm for s in sched_list if s.carrer_nm)
        print(f"\n  [항공사별 행사 건수 Top5]")
        for name, cnt in carrier_cnt.most_common(5):
            print(f"    {name}: {cnt}건")


if __name__ == "__main__":
    main()
