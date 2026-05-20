# =============================================================================
# [파일 설명]
# - 수행 기능: 네이버 뉴스 API를 호출하고, HTML/날짜/언론사/URL을 정리한 뒤 섹션별 후보 뉴스를 수집합니다.
# - 프로세스: 키워드별 API 요청 -> 응답 항목 정규화 -> 최근성/샘플링 적용 -> 통계 갱신 -> 후보 목록 반환
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 네이버 API 환경변수, 검색 키워드, 정렬/페이지/시간 필터 설정
# - 리턴값/출력: 뉴스 dict 목록과 LAST_SCRAPE_STATS 기반 수집 통계를 제공합니다.
# =============================================================================

"""
네이버 뉴스 검색 API 스크래퍼

기능:
- 네이버 뉴스 검색 API로 뉴스 수집
- 최근 N시간 이내 뉴스만 필터링
- 언론사명 추정
- 설정된 정렬 방식(date/sim)에 따라 키워드당 최대 display_per_keyword × pages_per_keyword개 조회
- URL 중복 제거
- 의미 중복/최근 반복 이슈 제거는 issue_history.py에서 별도로 수행
- 시간대별 비례 샘플링으로 최신 기사 쏠림 완화
- AI 선별 단계로 넘길 최종 후보는 기본 100개로 제한
- 매핑되지 않은 언론사 도메인은 개별 로그 대신 파일로 누적 저장
- 마지막 수집 통계를 main.py에서 가져갈 수 있도록 제공
"""

import os
import requests
import json
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
import logging
import pytz

from press_map import PRESS_MAP

load_dotenv()

# 로깅 설정
# 콘솔 출력만 사용하며, 파일 로그는 생성하지 않음
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================
# 네이버 API 설정
# ====================================
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"

# 한국 시간대
KST = pytz.timezone("Asia/Seoul")

# 매핑되지 않은 언론사 도메인 수집용
# 개별 로그를 찍지 않고, 실행 종료 시 파일로 누적 저장한다.
UNMAPPED_PRESS_DOMAINS = set()
DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH = "data/unmapped_press_domains.json"

# 마지막 수집 통계
# main.py에서 섹션별 대시보드 데이터로 가져간다.
LAST_SCRAPE_STATS = {}


# [코드 이해 주석]
# - 역할: 현재 한국 시간 반환.
# - 호출하는 곳: naver_news_scraper.is_within_last_hours, naver_news_scraper.sample_news_by_time_bucket,
# naver_news_scraper.save_unmapped_press_domains, naver_news_scraper.search_multiple_keywords
# - 파라미터: 없음
# - 리턴값: datetime 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_now_kst() -> datetime:
    """
    현재 한국 시간 반환
    """
    return datetime.now(KST)


# [코드 이해 주석]
# - 역할: 마지막 search_multiple_keywords 실행 통계를 반환한다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: 없음
# - 리턴값: dict 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_last_scrape_stats() -> dict:
    """
    마지막 search_multiple_keywords 실행 통계를 반환한다.

    main.py에서 섹션별 대시보드 데이터로 사용한다.
    """
    return dict(LAST_SCRAPE_STATS)


# [코드 이해 주석]
# - 역할: 네이버 뉴스 pubDate 문자열을 datetime으로 변환.
# - 호출하는 곳: naver_news_scraper.get_news_date_range, naver_news_scraper.is_within_last_hours,
# naver_news_scraper.search_multiple_keywords
# - 파라미터: pub_date: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 문자열/설정을 읽습니다 -> 가능한 형식으로 변환을 시도합니다 -> 실패 시 안전한 기본값을 반환합니다.
def parse_naver_pubdate(pub_date: str):
    """
    네이버 뉴스 pubDate 문자열을 datetime으로 변환

    예:
    Tue, 07 May 2026 14:20:00 +0900
    """
    try:
        dt = parsedate_to_datetime(pub_date)

        # timezone 정보가 없는 경우 KST로 처리
        if dt.tzinfo is None:
            dt = KST.localize(dt)

        return dt.astimezone(KST)

    except Exception as e:
        logger.warning(f"⚠️ pubDate 파싱 실패: {pub_date} / {e}")
        return None

# [코드 이해 주석]
# - 역할: 뉴스 목록에서 가장 최근 발행일과 가장 오래된 발행일을 반환한다.
# - 호출하는 곳: naver_news_scraper.search_multiple_keywords, naver_news_scraper.search_naver_news
# - 파라미터: news_list: list
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_date_range(news_list: list):
    """
    뉴스 목록에서 가장 최근 발행일과 가장 오래된 발행일을 반환한다.

    Returns:
        {
            "latest": datetime | None,
            "oldest": datetime | None
        }
    """
    published_dates = []

    for news in news_list or []:
        published_dt = news.get("_published_dt")

        if published_dt is None:
            published_at_kst = news.get("published_at_kst")

            if published_at_kst:
                try:
                    published_dt = datetime.fromisoformat(published_at_kst)
                except Exception:
                    published_dt = None

        if published_dt is None:
            pub_date = news.get("pubDate") or news.get("published_at")

            if pub_date:
                published_dt = parse_naver_pubdate(pub_date)

        if published_dt is not None:
            published_dates.append(published_dt)

    if not published_dates:
        return {
            "latest": None,
            "oldest": None
        }

    return {
        "latest": max(published_dates),
        "oldest": min(published_dates)
    }


# [코드 이해 주석]
# - 역할: 날짜 범위 로그 문자열 생성.
# - 호출하는 곳: naver_news_scraper.search_multiple_keywords, naver_news_scraper.search_naver_news
# - 파라미터: date_range: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def format_date_range_for_log(date_range: dict) -> str:
    """
    날짜 범위 로그 문자열 생성
    """
    latest = date_range.get("latest")
    oldest = date_range.get("oldest")

    if latest is None or oldest is None:
        return "발행일 범위 확인 불가"

    return (
        f"최근 {latest.strftime('%Y-%m-%d %H:%M')} / "
        f"가장 오래됨 {oldest.strftime('%Y-%m-%d %H:%M')}"
    )

# [코드 이해 주석]
# - 역할: 뉴스 발행일이 현재 시각 기준 최근 N시간 이내인지 확인.
# - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
# - 파라미터: pub_date: str, hours: int = 24
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_within_last_hours(pub_date: str, hours: int = 24) -> bool:
    """
    뉴스 발행일이 현재 시각 기준 최근 N시간 이내인지 확인
    """
    published_dt = parse_naver_pubdate(pub_date)

    if published_dt is None:
        return False

    now_kst = get_now_kst()
    cutoff_dt = now_kst - timedelta(hours=hours)

    return cutoff_dt <= published_dt <= now_kst


# [코드 이해 주석]
# - 역할: HTML 태그 제거.
# - 호출하는 곳: naver_news_scraper.search_naver_news
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def remove_html_tags(text: str) -> str:
    """
    HTML 태그 제거
    """
    if text is None:
        return ""

    clean = re.sub("<.*?>", "", text)
    clean = clean.replace("&quot;", '"').replace("&amp;", "&")
    clean = clean.replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&#39;", "'")

    return clean.strip()


# [코드 이해 주석]
# - 역할: 중복 제거용 URL 정규화.
# - 호출하는 곳: naver_news_scraper.search_multiple_keywords
# - 파라미터: url: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_news_url(url: str) -> str:
    """
    중복 제거용 URL 정규화

    - scheme 제거
    - www. 제거
    - query string 제거
    - fragment 제거
    - 마지막 slash 제거
    """
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().strip()
        path = parsed.path.strip()

        if domain.startswith("www."):
            domain = domain[4:]

        return f"{domain}{path}".rstrip("/")

    except Exception:
        return str(url).lower().strip().rstrip("/")


# [코드 이해 주석]
# - 역할: 네이버 뉴스 API 응답에는 언론사명 전용 필드가 없으므로,.
# - 호출하는 곳: naver_news_scraper.search_multiple_keywords, naver_news_scraper.search_naver_news
# - 파라미터: item: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_press_name(item: dict) -> str:
    """
    네이버 뉴스 API 응답에는 언론사명 전용 필드가 없으므로,
    originallink 또는 link의 도메인을 기준으로 언론사명을 추정한다.

    1. PRESS_MAP에 매핑된 도메인이 있으면 한글 언론사명 반환
    2. 매핑이 없으면 UNMAPPED_PRESS_DOMAINS에 저장
    3. 매핑이 없을 때도 콘솔에 개별 출력하지 않음
    4. 매핑이 없으면 도메인 앞부분을 임시 언론사명으로 반환
    """
    url = item.get("originallink") or item.get("link") or ""

    try:
        domain = urlparse(url).netloc.lower().strip()

        if not domain:
            return "언론사 미상"

        # 포트 제거
        domain = domain.split(":")[0]

        # www. 제거
        if domain.startswith("www."):
            domain = domain[4:]

        # 정확히 일치하거나 하위 도메인인 경우 매핑
        for key, name in PRESS_MAP.items():
            if domain == key or domain.endswith("." + key):
                return name

        # 매핑 안 된 도메인은 개별 로그를 찍지 않고 모아둔다.
        UNMAPPED_PRESS_DOMAINS.add(domain)

        # 매핑 안 된 경우: 도메인에서 대표 영문명 추출
        suffixes = [
            ".co.kr",
            ".or.kr",
            ".go.kr",
            ".ac.kr",
            ".com",
            ".net",
            ".org",
            ".kr",
        ]

        press_code = domain

        for suffix in suffixes:
            if press_code.endswith(suffix):
                press_code = press_code[: -len(suffix)]
                break

        return press_code or "언론사 미상"

    except Exception as e:
        logger.warning(f"⚠️ 언론사명 추출 실패: {url} / {e}")
        return "언론사 미상"


# [코드 이해 주석]
# - 역할: 매핑되지 않은 언론사 도메인을 누적 저장한다.
# - 호출하는 곳: main.main, naver_news_scraper.search_multiple_keywords
# - 파라미터: filename: str = DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 저장할 구조를 준비합니다 -> 대상 파일에 기록합니다 -> 실패 시 로그/예외 흐름에 맡깁니다.
def save_unmapped_press_domains(filename: str = DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH):
    """
    매핑되지 않은 언론사 도메인을 누적 저장한다.

    - 콘솔에는 개별 도메인을 찍지 않고 전체 건수만 출력
    - 기존 파일이 있으면 기존 domains와 이번 실행 domains를 합친다
    - 중복 도메인은 set으로 제거한다
    """
    if not UNMAPPED_PRESS_DOMAINS:
        logger.debug("🧩 이번 실행 언론사 미매핑 도메인: 0개")
        return

    existing_domains = set()

    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)

            for domain in existing_payload.get("domains", []):
                if domain:
                    existing_domains.add(str(domain).strip())

        except Exception as e:
            logger.warning(f"⚠️ 기존 미매핑 도메인 파일 읽기 실패, 새로 생성합니다: {e}")

    current_domains = set(UNMAPPED_PRESS_DOMAINS)
    merged_domains = sorted(existing_domains | current_domains)

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    payload = {
        "updated_at": get_now_kst().isoformat(),
        "total_count": len(merged_domains),
        "current_run_count": len(current_domains),
        "new_count": len(current_domains - existing_domains),
        "domains": merged_domains
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        "🧩 미매핑 언론사 도메인: 이번 %s개 / 신규 %s개 / 누적 %s개",
        len(current_domains),
        len(current_domains - existing_domains),
        len(merged_domains),
    )


# [코드 이해 주석]
# - 역할: 네이버 뉴스 검색.
# - 호출하는 곳: naver_news_scraper.search_multiple_keywords
# - 파라미터: query: str, display: int = 100, sort: str = 'date', start: int = 1
# - 리턴값: dict 타입 값을 반환합니다.
# - 프로세스 흐름: 요청 파라미터를 준비합니다 -> 외부 API를 호출합니다 -> 응답을 정규화하고 통계를 갱신합니다.
def search_naver_news(
    query: str,
    display: int = 100,
    sort: str = "date",
    start: int = 1
) -> dict:
    """
    네이버 뉴스 검색

    Args:
        query: 검색어 예: "EMR 전자의무기록"
        display: 결과 개수. 최대 100
        sort: 정렬. date=최신순
        start: 검색 시작 위치. 기본 1

    Returns:
        {
            'success': bool,
            'items': [...],
            'total': int
        }
    """

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.error("❌ 네이버 API 키가 없습니다!")
        return {"success": False, "error": "API 키 없음"}

    try:
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }

        display = min(max(int(display), 1), 100)
        start = max(int(start), 1)

        params = {
            "query": query,
            "display": display,
            "sort": sort,
            "start": start,
        }

        logger.debug(
            f"🔍 검색 중: '{query}' "
            f"(sort={sort}, start={start}, display={display})"
        )

        response = requests.get(
            NAVER_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])

        for item in items:
            item["title"] = remove_html_tags(item.get("title", ""))
            item["description"] = remove_html_tags(item.get("description", ""))
            item["source"] = extract_press_name(item)

        api_date_range = get_news_date_range(items)

        logger.debug(
            f"🕒 조회 결과 발행일 범위: "
            f"query='{query}', sort={sort}, start={start}, display={display} | "
            f"{format_date_range_for_log(api_date_range)}"
        )

        return {
            "success": True,
            "items": items,
            "total": data.get("total", 0),
        }

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.error("❌ API 호출 한도 초과!")
        else:
            logger.error(f"❌ HTTP 오류: {e}")

        return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"❌ 검색 실패: {e}")
        return {"success": False, "error": str(e)}


# [코드 이해 주석]
# - 역할: 시간대별 비례 샘플링.
# - 호출하는 곳: main.collect_select_and_summarize, naver_news_scraper.search_multiple_keywords
# - 파라미터: news_list: list, bucket_hours: int = 4, max_total_news: int = 100, min_per_bucket: int = 0, recent_hours:
# int = 24
# - 리턴값: list 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def sample_news_by_time_bucket(
    news_list: list,
    bucket_hours: int = 4,
    max_total_news: int = 100,
    min_per_bucket: int = 0,
    recent_hours: int = 24
) -> list:
    """
    시간대별 비례 샘플링

    목적:
    - 최근 기사에만 몰리는 현상 방지
    - 뉴스가 많이 나온 시간대는 많이 남김
    - 뉴스가 적은 시간대는 최소 개수만 보장
    - 전체 후보 수는 max_total_news 이하로 제한

    방식:
    1. 최근 recent_hours 범위를 bucket_hours 단위로 나눔
    2. 각 시간대에 min_per_bucket 최소 보장
    3. 나머지 슬롯은 시간대별 기사량 비율대로 배분
    4. 전체 뉴스 수가 max_total_news 이하이면 그대로 통과
    """
    if not news_list:
        return []

    if bucket_hours <= 0:
        logger.warning("bucket_hours가 0 이하이므로 시간대별 샘플링을 건너뜁니다.")
        return news_list

    if max_total_news is None or max_total_news <= 0:
        logger.warning("max_total_news가 없거나 0 이하이므로 시간대별 샘플링을 건너뜁니다.")
        return news_list

    if min_per_bucket < 0:
        min_per_bucket = 0

    now_kst = get_now_kst()
    cutoff_dt = now_kst - timedelta(hours=recent_hours)

    bucket_map = {}

    for news in news_list:
        published_dt = news.get("_published_dt")

        if published_dt is None:
            published_at_kst = news.get("published_at_kst")

            if published_at_kst:
                try:
                    published_dt = datetime.fromisoformat(published_at_kst)
                except Exception:
                    published_dt = None

        if published_dt is None:
            continue

        if not (cutoff_dt <= published_dt <= now_kst):
            continue

        diff_seconds = (now_kst - published_dt).total_seconds()
        diff_hours = int(diff_seconds // 3600)

        bucket_index = diff_hours // bucket_hours
        bucket_start = now_kst - timedelta(hours=(bucket_index + 1) * bucket_hours)
        bucket_end = now_kst - timedelta(hours=bucket_index * bucket_hours)

        if bucket_index not in bucket_map:
            bucket_map[bucket_index] = {
                "start": bucket_start,
                "end": bucket_end,
                "items": []
            }

        bucket_map[bucket_index]["items"].append(news)

    if not bucket_map:
        return []

    # 최신 구간부터 정렬
    sorted_bucket_keys = sorted(bucket_map.keys())

    total_items = sum(len(bucket_map[key]["items"]) for key in sorted_bucket_keys)

    if total_items <= max_total_news:
        logger.debug(
            f"🧺 시간대별 샘플링 불필요: "
            f"전체 {total_items}개가 max_total_news={max_total_news} 이하"
        )

        sampled_news = []

        for key in sorted_bucket_keys:
            sampled_news.extend(bucket_map[key]["items"])

        return sampled_news

    # 1차: 각 시간대 최소 보장
    allocation = {}
    guaranteed_total = 0

    for key in sorted_bucket_keys:
        bucket_size = len(bucket_map[key]["items"])
        guaranteed = min(bucket_size, min_per_bucket)
        allocation[key] = guaranteed
        guaranteed_total += guaranteed

    # 최소 보장만으로 max_total_news를 넘는 경우 방어
    if guaranteed_total >= max_total_news:
        logger.warning(
            f"⚠️ 최소 보장 개수만으로 최대 후보 수 초과: "
            f"guaranteed_total={guaranteed_total}, max_total_news={max_total_news}"
        )

        sampled_news = []

        for key in sorted_bucket_keys:
            if len(sampled_news) >= max_total_news:
                break

            remain = max_total_news - len(sampled_news)
            sampled_news.extend(bucket_map[key]["items"][:remain])

        return sampled_news

    remaining_slots = max_total_news - guaranteed_total

    # 2차: 남은 슬롯을 기사량 비율대로 배분
    remaining_capacity_total = 0

    for key in sorted_bucket_keys:
        bucket_size = len(bucket_map[key]["items"])
        remaining_capacity_total += max(bucket_size - allocation[key], 0)

    if remaining_capacity_total <= 0:
        sampled_news = []

        for key in sorted_bucket_keys:
            sampled_news.extend(bucket_map[key]["items"][:allocation[key]])

        return sampled_news[:max_total_news]

    raw_extra_allocation = {}
    floor_extra_total = 0

    for key in sorted_bucket_keys:
        bucket_size = len(bucket_map[key]["items"])
        remaining_capacity = max(bucket_size - allocation[key], 0)

        raw_extra = remaining_slots * (remaining_capacity / remaining_capacity_total)
        floor_extra = int(raw_extra)

        raw_extra_allocation[key] = {
            "raw": raw_extra,
            "floor": floor_extra,
            "fraction": raw_extra - floor_extra,
            "capacity": remaining_capacity
        }

        floor_extra_total += floor_extra

    for key in sorted_bucket_keys:
        allocation[key] += raw_extra_allocation[key]["floor"]

    # 3차: 소수점 때문에 남은 슬롯을 fraction 큰 순서대로 배분
    leftover_slots = remaining_slots - floor_extra_total

    fraction_sorted_keys = sorted(
        sorted_bucket_keys,
        key=lambda key: raw_extra_allocation[key]["fraction"],
        reverse=True
    )

    for key in fraction_sorted_keys:
        if leftover_slots <= 0:
            break

        bucket_size = len(bucket_map[key]["items"])

        if allocation[key] < bucket_size:
            allocation[key] += 1
            leftover_slots -= 1

    # 4차: 혹시 아직 남은 슬롯이 있으면 최신 구간부터 추가 배분
    while leftover_slots > 0:
        added = False

        for key in sorted_bucket_keys:
            if leftover_slots <= 0:
                break

            bucket_size = len(bucket_map[key]["items"])

            if allocation[key] < bucket_size:
                allocation[key] += 1
                leftover_slots -= 1
                added = True

        if not added:
            break

    sampled_news = []

    for key in sorted_bucket_keys:
        bucket = bucket_map[key]
        bucket_items = bucket["items"]
        selected_count = min(allocation[key], len(bucket_items))
        selected_items = bucket_items[:selected_count]

        sampled_news.extend(selected_items)

        ratio = len(bucket_items) / total_items if total_items else 0

        logger.debug(
            f"🧺 시간대 비례 샘플링: "
            f"{bucket['start'].strftime('%Y-%m-%d %H:%M')} ~ "
            f"{bucket['end'].strftime('%Y-%m-%d %H:%M')} | "
            f"전체 {len(bucket_items)}개 "
            f"({ratio:.1%}) 중 {selected_count}개 선택"
        )

    sampled_news = sampled_news[:max_total_news]

    logger.debug(
        f"🧺 시간대별 비례 샘플링 완료: "
        f"{len(news_list)}개 → {len(sampled_news)}개 "
        f"(bucket_hours={bucket_hours}, max_total_news={max_total_news}, min_per_bucket={min_per_bucket})"
    )

    return sampled_news


# [코드 이해 주석]
# - 역할: 여러 키워드로 뉴스 검색.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: keywords: list, display_per_keyword: int = 100, recent_hours: int = 24, sorts: list = None,
# pages_per_keyword: int = 3, enable_time_bucket_sampling: bool = True, bucket_hours: int = 4, max_total_news: int =
# 100, min_per_bucket: int = 0, unmapped_press_domains_file_path: str = None, save_unmapped_domains_at_end: bool =
# True
# - 리턴값: list 타입 값을 반환합니다.
# - 프로세스 흐름: 요청 파라미터를 준비합니다 -> 외부 API를 호출합니다 -> 응답을 정규화하고 통계를 갱신합니다.
def search_multiple_keywords(
    keywords: list,
    display_per_keyword: int = 100,
    recent_hours: int = 24,
    sorts: list = None,
    pages_per_keyword: int = 3,
    enable_time_bucket_sampling: bool = True,
    bucket_hours: int = 4,
    max_total_news: int = 100,
    min_per_bucket: int = 0,
    unmapped_press_domains_file_path: str = None,
    save_unmapped_domains_at_end: bool = True,
) -> list:
    """
    여러 키워드로 뉴스 검색

    공통 방식:
    - date 최신순만 사용
    - 키워드당 최대 300개 조회
      display_per_keyword=100, pages_per_keyword=3이면 start=1, 101, 201 호출
    - 최근 recent_hours 이내 뉴스만 유지
    - URL 중복 제거
- 의미 중복/최근 반복 이슈 제거는 issue_history.py에서 별도로 수행
    - max_total_news 초과 시 시간대별 비례 샘플링
    - AI 선별 단계로 넘길 최종 후보는 기본 100개 이하

    Args:
        keywords:
            ['EMR', '전자의무기록', ...]

        display_per_keyword:
            키워드당, 페이지당 뉴스 개수.
            네이버 API 최대값은 100.

        recent_hours:
            최근 몇 시간 이내 뉴스만 포함할지. 기본 24시간.

        sorts:
            사용할 정렬 방식 리스트.
            기본값은 ["date"].
            매일 자동 브리핑에서는 sim 검색을 쓰지 않는다.

        pages_per_keyword:
            키워드별 몇 페이지까지 가져올지.
            기본 3.
            display_per_keyword=100, pages_per_keyword=3이면
            start=1, 101, 201로 키워드당 최대 300개 조회.

        enable_time_bucket_sampling:
            True이면 max_total_news 초과 시 시간대별 비례 샘플링.

        bucket_hours:
            시간대 묶음 크기.
            4이면 4시간 단위.

        max_total_news:
            AI 선별 단계로 넘길 최종 후보 최대 개수.
            전체 수집 결과가 이 값 이하이면 샘플링 없이 그대로 통과.

        min_per_bucket:
            각 시간대 최소 보장 개수.

        unmapped_press_domains_file_path:
            매핑되지 않은 언론사 도메인을 저장할 파일 경로.
            main.py에서 설정 파일명 기준으로 동적 생성한 경로를 넘긴다.

        save_unmapped_domains_at_end:
            True이면 search_multiple_keywords 실행 종료 시 미매핑 도메인을 저장한다.

    Returns:
        [
            {
                'title': str,
                'description': str,
                'url': str,
                'originallink': str,
                'source': str,
                'published_at': str,
                'published_at_kst': str,
                'keyword': str,
                'sort': str,
                'scraped_at': str
            },
            ...
        ]
    """
    global LAST_SCRAPE_STATS

    # 매일 자동 브리핑 기준: 정확도순 sim
    if sorts is None:
        sorts = ["sim"]

    # 실행마다 미매핑 도메인 목록 초기화
    UNMAPPED_PRESS_DOMAINS.clear()

    # 실행 시작 시 통계 초기화
    LAST_SCRAPE_STATS = {}

    display_per_keyword = min(max(int(display_per_keyword), 1), 100)
    pages_per_keyword = max(int(pages_per_keyword), 1)

    all_news = []
    seen_links = set()

    total_seen_count = 0
    old_news_count = 0
    duplicate_count = 0
    parse_fail_or_invalid_count = 0
    failed_search_count = 0

    now_kst = get_now_kst()
    cutoff_dt = now_kst - timedelta(hours=recent_hours)

    logger.info(
        "🕒 뉴스 수집 조건: 키워드 %s개 / 최근 %s시간 / sorts=%s / "
        "키워드당 최대 %s개",
        len(keywords or []),
        recent_hours,
        sorts,
        display_per_keyword * pages_per_keyword,
    )

    for keyword in keywords:
        logger.debug(f"🔍 키워드: '{keyword}'")

        for sort in sorts:
            for page_index in range(pages_per_keyword):
                start = page_index * display_per_keyword + 1

                # 네이버 검색 API start는 일반적으로 1000 이내에서 사용
                if start > 1000:
                    logger.warning(f"⚠️ start={start}는 너무 커서 검색을 건너뜁니다.")
                    continue

                result = search_naver_news(
                    keyword,
                    display=display_per_keyword,
                    sort=sort,
                    start=start,
                )

                if not result["success"]:
                    failed_search_count += 1
                    logger.warning(f"⚠️ '{keyword}' 검색 실패 sort={sort}, start={start}")
                    continue

                for item in result["items"]:
                    total_seen_count += 1

                    link = item.get("link", "")
                    original_link = item.get("originallink", "")
                    pub_date = item.get("pubDate", "")

                    published_dt = parse_naver_pubdate(pub_date)

                    if published_dt is None:
                        parse_fail_or_invalid_count += 1
                        continue

                    if not (cutoff_dt <= published_dt <= now_kst):
                        old_news_count += 1
                        continue

                    # URL 완전 중복 제거
                    # 의미상 같은 사건인지 여부는 issue_history.py와 news_selector.py에서 별도로 판단한다.
                    normalized_urls = [
                        normalize_news_url(link),
                        normalize_news_url(original_link),
                    ]
                    normalized_urls = [url for url in normalized_urls if url]

                    if any(url in seen_links for url in normalized_urls):
                        duplicate_count += 1
                        continue

                    for url in normalized_urls:
                        seen_links.add(url)

                    all_news.append(
                        {
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "url": link,
                            "originallink": original_link,
                            "source": item.get("source") or extract_press_name(item),
                            "published_at": pub_date,
                            "published_at_kst": published_dt.isoformat(),
                            "keyword": keyword,
                            "sort": sort,
                            "scraped_at": now_kst.isoformat(),

                            # 내부 샘플링용. 반환 전 제거한다.
                            "_published_dt": published_dt,
                        }
                    )

    pre_sampling_count = len(all_news)

    logger.info(
        f"✅ 뉴스 1차 수집: 검색 {total_seen_count}개 → 후보 {pre_sampling_count}개 "
        f"(시간초과 {old_news_count}, URL중복 {duplicate_count}, 날짜실패 {parse_fail_or_invalid_count}, 검색실패 {failed_search_count})"
    )

    # 시간대별 비례 샘플링 적용
    # 주의: enable_time_bucket_sampling=False이면 여기서는 max_total_news 제한도 걸지 않는다.
    # main.py가 규칙 기반 반복/내부중복 제거 후 sample_news_by_time_bucket()을 호출해
    # 최종 후보 100개 제한을 적용한다.
    scraper_sampling_applied = False

    if enable_time_bucket_sampling and max_total_news is not None and max_total_news > 0:
        before_sampling_count = len(all_news)
        all_news = sample_news_by_time_bucket(
            all_news,
            bucket_hours=bucket_hours,
            max_total_news=max_total_news,
            min_per_bucket=min_per_bucket,
            recent_hours=recent_hours
        )
        scraper_sampling_applied = before_sampling_count > len(all_news)
    else:
        logger.debug(
            "🧺 스크래퍼 내부 시간대 샘플링 비활성화: "
            "반복 이슈 필터 후 main.py에서 최종 후보 수를 제한합니다."
        )

    final_candidate_count = len(all_news)

    final_candidate_date_range = get_news_date_range(all_news)

    logger.debug(
        f"🕒 AI 선별 후보 발행일 범위: "
        f"후보 {final_candidate_count}개 | "
        f"{format_date_range_for_log(final_candidate_date_range)}"
    )

    # 대시보드용 마지막 수집 통계 저장
    LAST_SCRAPE_STATS = {
        "total_seen_count": total_seen_count,
        "pre_sampling_count": pre_sampling_count,
        "final_candidate_count": final_candidate_count,
        "old_news_count": old_news_count,
        "duplicate_count": duplicate_count,
        "url_duplicate_count": duplicate_count,
        "parse_fail_or_invalid_count": parse_fail_or_invalid_count,
        "failed_search_count": failed_search_count,
        "recent_hours": recent_hours,
        "display_per_keyword": display_per_keyword,
        "pages_per_keyword": pages_per_keyword,
        "sorts": list(sorts),
        "max_total_news": max_total_news,
        "enable_time_bucket_sampling": enable_time_bucket_sampling,
        "scraper_sampling_enabled": enable_time_bucket_sampling,
        "scraper_sampling_applied": scraper_sampling_applied,
        "scraper_return_count": final_candidate_count,
        "bucket_hours": bucket_hours,
        "min_per_bucket": min_per_bucket,
        "keyword_count": len(keywords),
        "keywords": list(keywords),
        "searched_at": now_kst.isoformat(),
    }

    # 내부용 datetime 제거
    for news in all_news:
        news.pop("_published_dt", None)

    logger.info(f"✅ 뉴스 수집 최종 후보: {final_candidate_count}개")

    if save_unmapped_domains_at_end:
        save_unmapped_press_domains(
            filename=unmapped_press_domains_file_path or DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH
        )

    return all_news


# [코드 이해 주석]
# - 역할: main.py에서 반복 이슈 필터 후 시간대 샘플링을 적용한 결과를.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: before_issue_filter_count: int, after_issue_filter_count: int, before_sampling_count: int,
# after_sampling_count: int
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def update_post_issue_filter_sampling_stats(
    before_issue_filter_count: int,
    after_issue_filter_count: int,
    before_sampling_count: int,
    after_sampling_count: int,
):
    """
    main.py에서 반복 이슈 필터 후 시간대 샘플링을 적용한 결과를
    마지막 수집 통계에 반영한다.

    search_multiple_keywords(enable_time_bucket_sampling=False)로 넓게 수집한 뒤,
    규칙 기반 중복 제거를 먼저 수행하고, 마지막에 max_total_news 제한을 적용하는
    구조를 로그와 메일 대시보드에 정확히 보여주기 위한 함수다.
    """
    global LAST_SCRAPE_STATS

    LAST_SCRAPE_STATS["issue_filter_before_count"] = int(before_issue_filter_count or 0)
    LAST_SCRAPE_STATS["issue_filter_after_count"] = int(after_issue_filter_count or 0)
    LAST_SCRAPE_STATS["after_issue_filter_before_sampling_count"] = int(before_sampling_count or 0)
    LAST_SCRAPE_STATS["after_issue_filter_after_sampling_count"] = int(after_sampling_count or 0)
    LAST_SCRAPE_STATS["post_issue_filter_sampling_applied"] = int(before_sampling_count or 0) > int(after_sampling_count or 0)
    LAST_SCRAPE_STATS["final_candidate_count"] = int(after_sampling_count or 0)
    LAST_SCRAPE_STATS["scraper_return_count"] = int(before_issue_filter_count or 0)



# ====================================
# 뉴스 검색만 테스트
# 파일 저장 없이 콘솔 출력만 수행
# ====================================
if __name__ == "__main__":
    test_keywords = [
        "EMR", "의료정보시스템", "의료IT", "전자의무기록", "헬스케어"
    ]

    news_list = search_multiple_keywords(
        keywords=test_keywords,
        display_per_keyword=100,
        recent_hours=24,
        sorts=["sim"],
        pages_per_keyword=3,
        enable_time_bucket_sampling=True,
        bucket_hours=4,
        max_total_news=100,
        min_per_bucket=0,
    )

    print(f"\n✅ 수집된 뉴스 수: {len(news_list)}개")
    print(f"📊 마지막 수집 통계: {get_last_scrape_stats()}")

    for i, news in enumerate(news_list[:30], 1):
        print(f"\n[{i}] {news.get('title')}")
        print(f"    언론사: {news.get('source')}")
        print(f"    키워드: {news.get('keyword')}")
        print(f"    정렬: {news.get('sort')}")
        print(f"    발행일: {news.get('published_at')}")
