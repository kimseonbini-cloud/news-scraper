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
logger = logging.getLogger(__name__)  # 모듈로거

# ====================================
# 네이버 API 설정
# ====================================
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")                    # 네이버API클라이언트ID
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")            # 네이버API클라이언트시크릿

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"  # 네이버뉴스검색API주소

# 한국 시간대
KST = pytz.timezone("Asia/Seoul")                                 # 한국시간대객체

# 매핑되지 않은 언론사 도메인 수집용
# 개별 로그를 찍지 않고, 실행 종료 시 파일로 누적 저장한다.
UNMAPPED_PRESS_DOMAINS = set()                                    # 이번실행미매핑언론사도메인
DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH = "data/unmapped_press_domains.json"  # 기본미매핑언론사저장경로

# 마지막 수집 통계
# main.py에서 섹션별 대시보드 데이터로 가져간다.
LAST_SCRAPE_STATS = {}                                            # 마지막수집통계


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
        dt = parsedate_to_datetime(pub_date)  # 일시

        # timezone 정보가 없는 경우 KST로 처리
        if dt.tzinfo is None:
            dt = KST.localize(dt)  # 일시

        return dt.astimezone(KST)

    except Exception as e:  # 예외객체
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
    published_dates = []                                          # 비교가능한발행일목록

    for news in news_list or []:                                  # 후보뉴스
        published_dt = news.get("_published_dt")                  # 내부샘플링용발행시각

        if published_dt is None:
            published_at_kst = news.get("published_at_kst")       # ISO형식발행시각

            if published_at_kst:
                try:
                    published_dt = datetime.fromisoformat(published_at_kst)  # 발행일시
                except Exception:
                    published_dt = None  # 발행일시

        if published_dt is None:
            pub_date = news.get("pubDate") or news.get("published_at")  # 네이버원본발행일문자열

            if pub_date:
                published_dt = parse_naver_pubdate(pub_date)  # 발행일시

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
    latest = date_range.get("latest")  # 최신
    oldest = date_range.get("oldest")  # 가장오래된

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
    published_dt = parse_naver_pubdate(pub_date)  # 발행일시

    if published_dt is None:
        return False

    # 1) 최근 recent_hours 범위를 bucket_hours 단위로 나눠 기사들을 시간대별 통에 담는다.
    #    입력 news_list는 이미 URL/날짜 필터를 통과한 후보이며,
    #    _published_dt가 있으면 파싱 비용을 줄이기 위해 그대로 쓴다.
    now_kst = get_now_kst()  # 현재한국시간
    cutoff_dt = now_kst - timedelta(hours=hours)  # 필터기준시각

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

    clean = re.sub("<.*?>", "", text)  # 정제
    clean = clean.replace("&quot;", '"').replace("&amp;", "&")  # 정제
    clean = clean.replace("&lt;", "<").replace("&gt;", ">")  # 정제
    clean = clean.replace("&#39;", "'")  # 정제

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
        parsed = urlparse(url)  # parsed
        domain = parsed.netloc.lower().strip()  # 도메인
        path = parsed.path.strip()  # 경로

        if domain.startswith("www."):
            domain = domain[4:]  # 도메인

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
    url = item.get("originallink") or item.get("link") or ""  # URL

    try:
        domain = urlparse(url).netloc.lower().strip()  # 도메인

        if not domain:
            return "언론사 미상"

        # 포트 제거
        domain = domain.split(":")[0]  # 도메인

        # www. 제거
        if domain.startswith("www."):
            domain = domain[4:]  # 도메인

        # 정확히 일치하거나 하위 도메인인 경우 매핑
        for key, name in PRESS_MAP.items():  # 키,이름
            if domain == key or domain.endswith("." + key):
                return name

        # 매핑 안 된 도메인은 개별 로그를 찍지 않고 모아둔다.
        UNMAPPED_PRESS_DOMAINS.add(domain)

        # 매핑 안 된 경우: 도메인에서 대표 영문명 추출
        suffixes = [  # suffixes
            ".co.kr",
            ".or.kr",
            ".go.kr",
            ".ac.kr",
            ".com",
            ".net",
            ".org",
            ".kr",
        ]

        press_code = domain  # 언론사code

        for suffix in suffixes:  # suffix
            if press_code.endswith(suffix):
                press_code = press_code[: -len(suffix)]  # 언론사code
                break

        return press_code or "언론사 미상"

    except Exception as e:  # 예외객체
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

    existing_domains = set()  # existingdomains

    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:  # 파일객체
                existing_payload = json.load(f)  # existing데이터

            for domain in existing_payload.get("domains", []):  # 도메인
                if domain:
                    existing_domains.add(str(domain).strip())

        except Exception as e:  # 예외객체
            logger.warning(f"⚠️ 기존 미매핑 도메인 파일 읽기 실패, 새로 생성합니다: {e}")

    current_domains = set(UNMAPPED_PRESS_DOMAINS)  # 현재domains
    merged_domains = sorted(existing_domains | current_domains)  # 병합domains

    dirname = os.path.dirname(filename)  # dirname
    if dirname:
        os.makedirs(dirname, exist_ok=True)  # existok

    payload = {  # 데이터
        "updated_at": get_now_kst().isoformat(),
        "total_count": len(merged_domains),
        "current_run_count": len(current_domains),
        "new_count": len(current_domains - existing_domains),
        "domains": merged_domains
    }

    with open(filename, "w", encoding="utf-8") as f:  # 파일객체
        json.dump(payload, f, ensure_ascii=False, indent=2)  # 파일객체,ensureascii

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
    display: int = 100,  # 표시건수
    sort: str = "date",
    start: int = 1  # 시작값
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
        # 1) 네이버 API가 요구하는 인증 헤더와 검색 파라미터를 만든다.
        #    display/start는 API 허용 범위 안으로 보정해 잘못된 설정값 때문에 검색 전체가 실패하지 않게 한다.
        headers = {  # 요청헤더
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }

        display = min(max(int(display), 1), 100)  # 표시건수
        start = max(int(start), 1)  # 시작값

        params = {  # 파라미터
            "query": query,
            "display": display,
            "sort": sort,
            "start": start,
        }

        logger.debug(
            f"🔍 검색 중: '{query}' "
            f"(sort={sort}, start={start}, display={display})"
        )

        # 2) 이 함수는 "한 키워드, 한 정렬 방식, 한 페이지"만 조회한다.
        #    여러 키워드/페이지를 도는 책임은 search_multiple_keywords()가 맡는다.
        response = requests.get(  # 응답
            NAVER_SEARCH_URL,
            headers=headers,  # 요청헤더
            params=params,  # 파라미터
            timeout=10,  # timeout
        )

        response.raise_for_status()
        data = response.json()  # 데이터

        items = data.get("items", [])  # 항목목록

        # 3) 네이버 응답은 제목/요약에 HTML 태그가 섞이고 언론사 필드가 따로 없다.
        #    여기서 메일/중복판정 단계가 바로 쓸 수 있도록 title, description, source를 표준화한다.
        for item in items:  # 항목
            item["title"] = remove_html_tags(item.get("title", ""))  # 처리값
            item["description"] = remove_html_tags(item.get("description", ""))  # 처리값
            item["source"] = extract_press_name(item)  # 처리값

        api_date_range = get_news_date_range(items)  # API조회발행일범위

        logger.debug(
            f"🕒 조회 결과 발행일 범위: "
            f"query='{query}', sort={sort}, start={start}, display={display} | "
            f"{format_date_range_for_log(api_date_range)}"
        )

        # 4) 반환값은 API 원본 전체가 아니라, 후속 수집 루프가 필요한 items/total 중심의 얇은 dict다.
        return {
            "success": True,
            "items": items,
            "total": data.get("total", 0),
        }

    except requests.exceptions.HTTPError as e:  # 예외객체
        if e.response is not None and e.response.status_code == 429:
            logger.error("❌ API 호출 한도 초과!")
        else:
            logger.error(f"❌ HTTP 오류: {e}")

        return {"success": False, "error": str(e)}

    except Exception as e:  # 예외객체
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
    bucket_hours: int = 4,  # bucket시간수
    max_total_news: int = 100,  # 최대전체뉴스
    min_per_bucket: int = 0,  # 최소기사별bucket
    recent_hours: int = 24  # recent시간수
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
        min_per_bucket = 0  # 최소기사별bucket

    now_kst = get_now_kst()                                       # 샘플링기준현재시각
    cutoff_dt = now_kst - timedelta(hours=recent_hours)           # 샘플링대상최소발행시각

    # 시간대 bucket은 "최근 몇 시간 안의 후보를 균등하게 조금씩 남기기" 위한 임시 구조다.
    # key는 현재 시각으로부터 몇 번째 bucket인지이고, items에는 그 시간대에 발행된 뉴스가 쌓인다.
    bucket_map = {}                                               # 시간대별후보뉴스묶음

    for news in news_list:                                        # 샘플링대상뉴스
        published_dt = news.get("_published_dt")                  # 내부발행시각

        if published_dt is None:
            published_at_kst = news.get("published_at_kst")       # ISO형식발행시각

            if published_at_kst:
                try:
                    published_dt = datetime.fromisoformat(published_at_kst)  # 발행일시
                except Exception:
                    published_dt = None  # 발행일시

        if published_dt is None:
            continue

        if not (cutoff_dt <= published_dt <= now_kst):
            continue

        diff_seconds = (now_kst - published_dt).total_seconds()   # 현재로부터경과초
        diff_hours = int(diff_seconds // 3600)                    # 현재로부터경과시간

        bucket_index = diff_hours // bucket_hours                 # 시간대버킷번호
        bucket_start = now_kst - timedelta(hours=(bucket_index + 1) * bucket_hours)  # 버킷시작시각
        bucket_end = now_kst - timedelta(hours=bucket_index * bucket_hours)          # 버킷종료시각

        if bucket_index not in bucket_map:
            bucket_map[bucket_index] = {  # 처리값
                "start": bucket_start,
                "end": bucket_end,
                "items": []
            }

        bucket_map[bucket_index]["items"].append(news)

    if not bucket_map:
        return []

    # 최신 구간부터 정렬
    sorted_bucket_keys = sorted(bucket_map.keys())                # 최신순버킷키목록

    total_items = sum(len(bucket_map[key]["items"]) for key in sorted_bucket_keys)  # 샘플링전전체후보수

    # 2) 후보 수가 이미 상한 이하라면 샘플링하지 않는다.
    #    이 경우에도 시간대 순서로 다시 펼쳐, 이후 단계가 최신 구간부터 후보를 보게 한다.
    if total_items <= max_total_news:
        logger.debug(
            f"🧺 시간대별 샘플링 불필요: "
            f"전체 {total_items}개가 max_total_news={max_total_news} 이하"
        )

        sampled_news = []                                         # 샘플링없이정렬된후보뉴스

        for key in sorted_bucket_keys:  # 키
            sampled_news.extend(bucket_map[key]["items"])

        return sampled_news

    # 3) 1차 배분: 각 시간대 최소 보장.
    #    최근 특정 시간에 기사가 몰려도 오래된 구간의 대표 후보가 완전히 사라지지 않게 하는 장치다.
    allocation = {}                                               # 버킷별선택개수
    guaranteed_total = 0                                          # 최소보장으로이미배정된개수

    for key in sorted_bucket_keys:                                # 시간대버킷번호
        bucket_size = len(bucket_map[key]["items"])               # 해당버킷후보수
        guaranteed = min(bucket_size, min_per_bucket)             # 해당버킷최소보장개수
        allocation[key] = guaranteed  # 처리값
        guaranteed_total += guaranteed  # 처리값

    # 최소 보장만으로 max_total_news를 넘는 경우 방어
    if guaranteed_total >= max_total_news:
        logger.warning(
            f"⚠️ 최소 보장 개수만으로 최대 후보 수 초과: "
            f"guaranteed_total={guaranteed_total}, max_total_news={max_total_news}"
        )

        sampled_news = []                                         # 최소보장초과시최신순후보뉴스

        for key in sorted_bucket_keys:  # 키
            if len(sampled_news) >= max_total_news:
                break

            remain = max_total_news - len(sampled_news)           # 아직채울수있는후보수
            sampled_news.extend(bucket_map[key]["items"][:remain])

        return sampled_news

    remaining_slots = max_total_news - guaranteed_total           # 비례배분가능슬롯수

    # 4) 2차 배분: 남은 슬롯을 기사량 비율대로 나눈다.
    #    기사량이 많은 시간대는 더 많은 후보를 받고, 적은 시간대는 최소 보장분 위주로 남는다.
    remaining_capacity_total = 0                                  # 최소보장이후남은전체수용량

    for key in sorted_bucket_keys:  # 키
        bucket_size = len(bucket_map[key]["items"])               # 해당버킷후보수
        remaining_capacity_total += max(bucket_size - allocation[key], 0)  # 처리값

    if remaining_capacity_total <= 0:
        sampled_news = []                                         # 수용량없는경우최소보장후보뉴스

        for key in sorted_bucket_keys:  # 키
            sampled_news.extend(bucket_map[key]["items"][:allocation[key]])

        return sampled_news[:max_total_news]

    raw_extra_allocation = {}                                     # 버킷별비례배분계산값
    floor_extra_total = 0                                         # 정수내림으로확정된추가배정수

    for key in sorted_bucket_keys:                                # 시간대버킷번호
        bucket_size = len(bucket_map[key]["items"])               # 해당버킷후보수
        remaining_capacity = max(bucket_size - allocation[key], 0)  # 해당버킷추가수용량

        raw_extra = remaining_slots * (remaining_capacity / remaining_capacity_total)  # 비례배분원값
        floor_extra = int(raw_extra)                              # 정수로확정된추가배정수

        raw_extra_allocation[key] = {  # 처리값
            "raw": raw_extra,
            "floor": floor_extra,
            "fraction": raw_extra - floor_extra,
            "capacity": remaining_capacity
        }

        floor_extra_total += floor_extra  # 처리값

    for key in sorted_bucket_keys:  # 키
        allocation[key] += raw_extra_allocation[key]["floor"]  # 처리값

    # 5) 3차 배분: 소수점 때문에 남은 슬롯을 fraction 큰 순서대로 배분한다.
    #    floor 처리로 버려진 슬롯을 다시 채워 최종 후보 수가 max_total_news보다 불필요하게 작아지지 않게 한다.
    leftover_slots = remaining_slots - floor_extra_total          # 소수점처리후남은슬롯수

    fraction_sorted_keys = sorted(                                # 소수점큰순서버킷목록
        sorted_bucket_keys,
        key=lambda key: raw_extra_allocation[key]["fraction"],
        reverse=True  # reverse
    )

    for key in fraction_sorted_keys:  # 키
        if leftover_slots <= 0:
            break

        bucket_size = len(bucket_map[key]["items"])               # 해당버킷후보수

        if allocation[key] < bucket_size:
            allocation[key] += 1  # 처리값
            leftover_slots -= 1  # 처리값

    # 6) 4차 배분: 혹시 아직 남은 슬롯이 있으면 최신 구간부터 추가 배분한다.
    #    모든 시간대가 이미 꽉 찼다면 added=False가 되어 루프를 종료한다.
    while leftover_slots > 0:
        added = False                                             # 이번루프에서추가배정여부

        for key in sorted_bucket_keys:  # 키
            if leftover_slots <= 0:
                break

            bucket_size = len(bucket_map[key]["items"])           # 해당버킷후보수

            if allocation[key] < bucket_size:
                allocation[key] += 1  # 처리값
                leftover_slots -= 1  # 처리값
                added = True  # added

        if not added:
            break

    sampled_news = []                                             # 최종샘플링후보뉴스

    # 7) 계산된 allocation만큼 각 시간대의 기사 목록을 앞에서부터 가져온다.
    #    bucket 안의 기사 순서는 수집 단계의 최신순을 유지하므로, 같은 시간대 안에서는 더 최신 기사가 우선된다.
    for key in sorted_bucket_keys:  # 키
        bucket = bucket_map[key]                                  # 현재시간대버킷데이터
        bucket_items = bucket["items"]                            # 현재버킷후보뉴스목록
        selected_count = min(allocation[key], len(bucket_items))  # 현재버킷실제선택개수
        selected_items = bucket_items[:selected_count]            # 현재버킷선택뉴스

        sampled_news.extend(selected_items)

        ratio = len(bucket_items) / total_items if total_items else 0  # 전체대비현재버킷비율

        logger.debug(
            f"🧺 시간대 비례 샘플링: "
            f"{bucket['start'].strftime('%Y-%m-%d %H:%M')} ~ "
            f"{bucket['end'].strftime('%Y-%m-%d %H:%M')} | "
            f"전체 {len(bucket_items)}개 "
            f"({ratio:.1%}) 중 {selected_count}개 선택"
        )

    sampled_news = sampled_news[:max_total_news]  # 샘플링뉴스뉴스

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
    display_per_keyword: int = 100,  # 표시건수기사별키워드
    recent_hours: int = 24,  # recent시간수
    sorts: list = None,  # 정렬목록
    pages_per_keyword: int = 3,  # 페이지목록기사별키워드
    enable_time_bucket_sampling: bool = True,  # enable시간bucketsampling
    bucket_hours: int = 4,  # bucket시간수
    max_total_news: int = 100,  # 최대전체뉴스
    min_per_bucket: int = 0,  # 최소기사별bucket
    unmapped_press_domains_file_path: str = None,  # unmapped언론사domains파일경로
    save_unmapped_domains_at_end: bool = True,  # saveunmappeddomainsat종료값
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

    # 1) 이번 실행의 수집 상태를 초기화한다.
    #    LAST_SCRAPE_STATS는 main.py의 메일 대시보드/로그에 그대로 쓰이므로 실행마다 새로 쌓아야 한다.
    #    UNMAPPED_PRESS_DOMAINS도 이번 실행에서 발견한 미매핑 언론사만 따로 모은다.
    # 매일 자동 브리핑 기준: 정확도순 sim
    if sorts is None:
        sorts = ["sim"]                                           # 기본검색정렬목록

    # 실행마다 미매핑 도메인 목록 초기화
    UNMAPPED_PRESS_DOMAINS.clear()

    # 실행 시작 시 통계 초기화
    LAST_SCRAPE_STATS = {}  # 마지막수집통계

    display_per_keyword = min(max(int(display_per_keyword), 1), 100)  # 키워드별페이지당수집수
    pages_per_keyword = max(int(pages_per_keyword), 1)                # 키워드별조회페이지수

    all_news = []                                                 # 최근성URL필터통과뉴스
    seen_links = set()                                            # URL중복검사용정규화링크

    total_seen_count = 0                                          # 네이버응답확인건수
    old_news_count = 0                                            # 최근시간범위초과건수
    duplicate_count = 0                                           # URL중복제외건수
    parse_fail_or_invalid_count = 0                               # 발행일파싱실패건수
    failed_search_count = 0                                       # API검색실패횟수

    now_kst = get_now_kst()                                       # 수집기준현재시각
    cutoff_dt = now_kst - timedelta(hours=recent_hours)           # 최근뉴스판정최소시각

    logger.info(
        "🕒 뉴스 수집 조건: 키워드 %s개 / 최근 %s시간 / sorts=%s / "
        "키워드당 최대 %s개",
        len(keywords or []),
        recent_hours,
        sorts,
        display_per_keyword * pages_per_keyword,
    )

    # 2) 키워드 × 정렬방식 × 페이지 조합을 전부 순회한다.
    #    이 루프에서 얻는 item은 아직 "API 원본에 가까운 기사"라서 날짜/중복/최근성 필터를 바로 적용한다.
    for keyword in keywords:  # 키워드
        logger.debug(f"🔍 키워드: '{keyword}'")

        for sort in sorts:  # 정렬
            for page_index in range(pages_per_keyword):  # 페이지순번
                start = page_index * display_per_keyword + 1      # 네이버API검색시작위치

                # 네이버 검색 API start는 일반적으로 1000 이내에서 사용
                if start > 1000:
                    logger.warning(f"⚠️ start={start}는 너무 커서 검색을 건너뜁니다.")
                    continue

                result = search_naver_news(                       # 단일키워드페이지검색결과
                    keyword,
                    display=display_per_keyword,  # 표시건수
                    sort=sort,  # 정렬
                    start=start,  # 시작값
                )

                if not result["success"]:
                    failed_search_count += 1  # 처리값
                    logger.warning(f"⚠️ '{keyword}' 검색 실패 sort={sort}, start={start}")
                    continue

                # 3) 검색 결과 item을 실제 후보 뉴스 dict로 변환한다.
                #    여기서 최근 시간 밖 기사, 날짜 파싱 실패 기사, URL 중복 기사는 all_news에 넣지 않는다.
                for item in result["items"]:  # 항목
                    total_seen_count += 1  # 처리값

                    link = item.get("link", "")                   # 네이버뉴스링크
                    original_link = item.get("originallink", "")  # 원문기사링크
                    pub_date = item.get("pubDate", "")            # 네이버발행일문자열

                    published_dt = parse_naver_pubdate(pub_date)  # KST발행시각

                    if published_dt is None:
                        parse_fail_or_invalid_count += 1  # 처리값
                        continue

                    if not (cutoff_dt <= published_dt <= now_kst):
                        old_news_count += 1  # 처리값
                        continue

                    # URL 완전 중복 제거
                    # 의미상 같은 사건인지 여부는 issue_history.py와 news_selector.py에서 별도로 판단한다.
                    normalized_urls = [                           # 중복판정용정규화URL후보
                        normalize_news_url(link),
                        normalize_news_url(original_link),
                    ]
                    normalized_urls = [url for url in normalized_urls if url]  # 정규화URL목록

                    if any(url in seen_links for url in normalized_urls):
                        duplicate_count += 1  # 처리값
                        continue

                    for url in normalized_urls:  # URL
                        seen_links.add(url)

                    # 4) 이 dict가 이후 파이프라인의 기본 뉴스 단위다.
                    #    _published_dt는 시간대 샘플링 전용 내부 필드라 반환 직전에 제거한다.
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

    pre_sampling_count = len(all_news)                            # 샘플링전수집후보수

    logger.info(
        f"✅ 뉴스 1차 수집: 검색 {total_seen_count}개 → 후보 {pre_sampling_count}개 "
        f"(시간초과 {old_news_count}, URL중복 {duplicate_count}, 날짜실패 {parse_fail_or_invalid_count}, 검색실패 {failed_search_count})"
    )

    # 5) 시간대별 비례 샘플링 적용.
    #    enable_time_bucket_sampling=False이면 여기서는 max_total_news 제한도 걸지 않는다.
    #    main.py가 반복 이슈/그룹 중복을 먼저 제거한 뒤 sample_news_by_time_bucket()을 다시 호출해
    #    최종 후보 상한을 적용한다. 순서를 이렇게 둬야 중복 후보가 상한 슬롯을 낭비하지 않는다.
    scraper_sampling_applied = False                              # 스크래퍼내부샘플링적용여부

    if enable_time_bucket_sampling and max_total_news is not None and max_total_news > 0:
        before_sampling_count = len(all_news)                     # 스크래퍼샘플링전후보수
        all_news = sample_news_by_time_bucket(  # 전체뉴스
            all_news,
            bucket_hours=bucket_hours,  # bucket시간수
            max_total_news=max_total_news,  # 최대전체뉴스
            min_per_bucket=min_per_bucket,  # 최소기사별bucket
            recent_hours=recent_hours  # recent시간수
        )
        scraper_sampling_applied = before_sampling_count > len(all_news)  # scrapersamplingapplied
    else:
        logger.debug(
            "🧺 스크래퍼 내부 시간대 샘플링 비활성화: "
            "반복 이슈 필터 후 main.py에서 최종 후보 수를 제한합니다."
        )

    final_candidate_count = len(all_news)                         # 스크래퍼반환후보수

    final_candidate_date_range = get_news_date_range(all_news)    # 최종후보발행일범위

    logger.debug(
        f"🕒 AI 선별 후보 발행일 범위: "
        f"후보 {final_candidate_count}개 | "
        f"{format_date_range_for_log(final_candidate_date_range)}"
    )

    # 대시보드용 마지막 수집 통계 저장
    # 1) main.py는 이 dict를 가져와 반복이슈/그룹화/요약 통계와 병합한다.
    # 2) email_sender.py는 병합된 scrape_stats를 읽어 메일 상단 운영 대시보드의 숫자를 표시한다.
    # 3) 따라서 여기에는 "네이버 API를 본 순간부터 최근성/URL 중복/샘플링까지"의 수집 단계 숫자만 저장한다.
    LAST_SCRAPE_STATS = {  # 마지막수집통계
        "total_seen_count": total_seen_count,                     # 네이버응답확인건수
        "pre_sampling_count": pre_sampling_count,                 # 샘플링전후보수
        "final_candidate_count": final_candidate_count,           # 최종AI후보수
        "old_news_count": old_news_count,                         # 최근시간범위초과건수
        "duplicate_count": duplicate_count,                       # URL중복제외건수
        "url_duplicate_count": duplicate_count,                   # URL중복제외건수호환키
        "parse_fail_or_invalid_count": parse_fail_or_invalid_count,  # 발행일파싱실패건수
        "failed_search_count": failed_search_count,               # API검색실패횟수
        "recent_hours": recent_hours,                             # 최근뉴스시간범위
        "display_per_keyword": display_per_keyword,               # 키워드별페이지당수집수
        "pages_per_keyword": pages_per_keyword,                   # 키워드별조회페이지수
        "sorts": list(sorts),                                     # 검색정렬방식목록
        "max_total_news": max_total_news,                         # 최종후보상한
        "enable_time_bucket_sampling": enable_time_bucket_sampling,  # 스크래퍼샘플링설정값
        "scraper_sampling_enabled": enable_time_bucket_sampling,  # 스크래퍼샘플링설정값호환키
        "scraper_sampling_applied": scraper_sampling_applied,     # 스크래퍼샘플링적용여부
        "scraper_return_count": final_candidate_count,            # 스크래퍼반환뉴스수
        "bucket_hours": bucket_hours,                             # 샘플링버킷시간크기
        "min_per_bucket": min_per_bucket,                         # 버킷별최소보장수
        "keyword_count": len(keywords),                           # 검색키워드수
        "keywords": list(keywords),                               # 검색키워드목록
        "searched_at": now_kst.isoformat(),                       # 수집실행시각
    }

    # 내부용 datetime 제거
    for news in all_news:  # 뉴스
        news.pop("_published_dt", None)

    logger.info(f"✅ 뉴스 수집 최종 후보: {final_candidate_count}개")

    if save_unmapped_domains_at_end:
        save_unmapped_press_domains(
            filename=unmapped_press_domains_file_path or DEFAULT_UNMAPPED_PRESS_DOMAINS_FILE_PATH  # filename
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

    # main.py가 반복이슈/제외어/그룹화 이후 다시 샘플링한 결과를 수집 통계에 덧씌운다.
    # 1) 스크래퍼 내부 샘플링을 끈 실행에서는 최초 수집 수와 최종 AI 후보 수가 다르다.
    # 2) 이 보정이 있어야 메일 대시보드의 final_candidate_count가 실제 OpenAI에 넘어간 후보 수와 맞는다.
    LAST_SCRAPE_STATS["issue_filter_before_count"] = int(before_issue_filter_count or 0)                     # 반복필터전후보수
    LAST_SCRAPE_STATS["issue_filter_after_count"] = int(after_issue_filter_count or 0)                       # 반복필터후후보수
    LAST_SCRAPE_STATS["after_issue_filter_before_sampling_count"] = int(before_sampling_count or 0)          # 후처리샘플링전후보수
    LAST_SCRAPE_STATS["after_issue_filter_after_sampling_count"] = int(after_sampling_count or 0)            # 후처리샘플링후후보수
    LAST_SCRAPE_STATS["post_issue_filter_sampling_applied"] = int(before_sampling_count or 0) > int(after_sampling_count or 0)  # 후처리샘플링적용여부
    LAST_SCRAPE_STATS["final_candidate_count"] = int(after_sampling_count or 0)                              # 최종AI후보수
    LAST_SCRAPE_STATS["scraper_return_count"] = int(before_issue_filter_count or 0)                          # 최초스크래퍼반환수



# ====================================
# 뉴스 검색만 테스트
# 파일 저장 없이 콘솔 출력만 수행
# ====================================
if __name__ == "__main__":
    test_keywords = [  # 테스트키워드목록
        "EMR", "의료정보시스템", "의료IT", "전자의무기록", "헬스케어"
    ]

    news_list = search_multiple_keywords(  # 뉴스list
        keywords=test_keywords,  # 키워드목록
        display_per_keyword=100,  # 표시건수기사별키워드
        recent_hours=24,  # recent시간수
        sorts=["sim"],
        pages_per_keyword=3,  # 페이지목록기사별키워드
        enable_time_bucket_sampling=True,  # enable시간bucketsampling
        bucket_hours=4,  # bucket시간수
        max_total_news=100,  # 최대전체뉴스
        min_per_bucket=0,  # 최소기사별bucket
    )

    print(f"\n✅ 수집된 뉴스 수: {len(news_list)}개")
    print(f"📊 마지막 수집 통계: {get_last_scrape_stats()}")

    for i, news in enumerate(news_list[:30], 1):  # i,뉴스
        print(f"\n[{i}] {news.get('title')}")
        print(f"    언론사: {news.get('source')}")
        print(f"    키워드: {news.get('keyword')}")
        print(f"    정렬: {news.get('sort')}")
        print(f"    발행일: {news.get('published_at')}")
