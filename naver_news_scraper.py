"""
네이버 뉴스 검색 API 스크래퍼
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


def get_now_kst() -> datetime:
    """
    현재 한국 시간 반환
    """
    return datetime.now(KST)


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


def is_within_last_hours(pub_date: str, hours: int = 24) -> bool:
    """
    뉴스 발행일이 현재 시각 기준 최근 N시간 이내인지 확인

    Args:
        pub_date: 네이버 뉴스 pubDate
        hours: 기준 시간. 기본 24시간

    Returns:
        bool
    """
    published_dt = parse_naver_pubdate(pub_date)

    if published_dt is None:
        return False

    now_kst = get_now_kst()
    cutoff_dt = now_kst - timedelta(hours=hours)

    return cutoff_dt <= published_dt <= now_kst


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


def extract_press_name(item: dict) -> str:
    """
    네이버 뉴스 API 응답에는 언론사명 전용 필드가 없으므로,
    originallink 또는 link의 도메인을 기준으로 언론사명을 추정한다.

    1. PRESS_MAP에 매핑된 도메인이 있으면 한글 언론사명 반환
    2. 매핑이 없으면 도메인 앞부분 반환

    예:
    www.example.co.kr      -> example
    example.co.kr          -> example
    news.example.co.kr     -> news.example
    www.medicaltimes.com   -> 메디컬타임즈
    health.chosun.com      -> 헬스조선
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
        # 예: health.chosun.com -> health.chosun.com 우선 매칭
        # 예: news.health.chosun.com -> health.chosun.com 하위 도메인 매칭
        for key, name in PRESS_MAP.items():
            if domain == key or domain.endswith("." + key):
                return name

        logger.info(f"🧩 언론사 매핑 없음: {domain}")

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


def search_naver_news(query: str, display: int = 10, sort: str = "date") -> dict:
    """
    네이버 뉴스 검색

    Args:
        query: 검색어 예: "EMR 전자의무기록"
        display: 결과 개수. 최대 100
        sort: 정렬. date=최신순, sim=관련도순

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

        params = {
            "query": query,
            "display": min(display, 100),
            "sort": sort,
        }

        logger.info(f"🔍 검색 중: '{query}' (최대 {display}개)")

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

        logger.info(f"✅ {len(items)}개 뉴스 수집 (전체 {data.get('total', 0)}개)")

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


def search_multiple_keywords(
    keywords: list,
    display_per_keyword: int = 10,
    recent_hours: int = 24,
) -> list:
    """
    여러 키워드로 뉴스 검색

    현재 조회 시각 기준 recent_hours 시간 이내 뉴스만 저장한다.

    Args:
        keywords: ['EMR', '전자의무기록', ...]
        display_per_keyword: 키워드당 뉴스 개수
        recent_hours: 최근 몇 시간 이내 뉴스만 포함할지. 기본 24시간

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
                'scraped_at': str
            },
            ...
        ]
    """
    all_news = []
    seen_links = set()

    total_seen_count = 0
    old_news_count = 0
    duplicate_count = 0
    parse_fail_or_invalid_count = 0

    now_kst = get_now_kst()
    cutoff_dt = now_kst - timedelta(hours=recent_hours)

    logger.info("\n" + "=" * 60)
    logger.info("🕒 최근 뉴스 필터 적용")
    logger.info(f"기준 현재 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"저장 기준: {cutoff_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} 이후 발행 뉴스")
    logger.info("=" * 60)

    for keyword in keywords:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🔍 키워드: '{keyword}'")
        logger.info(f"{'=' * 60}")

        result = search_naver_news(
            keyword,
            display=display_per_keyword,
            sort="date",
        )

        if result["success"]:
            for item in result["items"]:
                total_seen_count += 1

                link = item.get("link", "")
                pub_date = item.get("pubDate", "")

                published_dt = parse_naver_pubdate(pub_date)

                if published_dt is None:
                    parse_fail_or_invalid_count += 1
                    logger.info(f"   ⏭️ 날짜 파싱 실패로 제외: {item.get('title', '')[:50]}")
                    continue

                if not (cutoff_dt <= published_dt <= now_kst):
                    old_news_count += 1
                    logger.info(
                        f"   ⏭️ {recent_hours}시간 초과 뉴스 제외: "
                        f"{published_dt.strftime('%Y-%m-%d %H:%M')} / "
                        f"{item.get('title', '')[:50]}"
                    )
                    continue

                # 중복 제거
                if link in seen_links:
                    duplicate_count += 1
                    logger.info(f"   ⏭️ 중복 링크 제외: {item.get('title', '')[:50]}")
                    continue

                seen_links.add(link)

                all_news.append(
                    {
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "url": link,
                        "originallink": item.get("originallink", ""),
                        "source": item.get("source") or extract_press_name(item),
                        "published_at": pub_date,
                        "published_at_kst": published_dt.isoformat(),
                        "keyword": keyword,
                        "scraped_at": now_kst.isoformat(),
                    }
                )

        else:
            logger.warning(f"⚠️ '{keyword}' 검색 실패")

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ 뉴스 수집 완료")
    logger.info(f"전체 조회 기사 수: {total_seen_count}개")
    logger.info(f"{recent_hours}시간 이내 저장 기사 수: {len(all_news)}개")
    logger.info(f"{recent_hours}시간 초과 제외: {old_news_count}개")
    logger.info(f"중복 제외: {duplicate_count}개")
    logger.info(f"날짜 파싱 실패 제외: {parse_fail_or_invalid_count}개")
    logger.info(f"{'=' * 60}")

    return all_news


def save_to_json(news_list: list, filename: str = "data/naver_news.json"):
    """
    JSON 파일로 저장
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 저장 완료: {filename}")

    except Exception as e:
        logger.error(f"❌ 저장 실패: {e}")