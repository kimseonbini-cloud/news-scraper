"""
네이버 뉴스 검색 API 스크래퍼
"""
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

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


def search_naver_news(query: str, display: int = 10, sort: str = "date") -> dict:
    """
    네이버 뉴스 검색
    
    Args:
        query: 검색어 (예: "EMR 전자의무기록")
        display: 결과 개수 (최대 100)
        sort: 정렬 (date=최신순, sim=관련도순)
    
    Returns:
        {
            'success': bool,
            'items': [...],
            'total': int
        }
    """
    
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.error("❌ 네이버 API 키가 없습니다!")
        return {'success': False, 'error': 'API 키 없음'}
    
    try:
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        
        params = {
            "query": query,
            "display": min(display, 100),  # 최대 100개
            "sort": sort
        }
        
        logger.info(f"🔍 검색 중: '{query}' (최대 {display}개)")
        
        response = requests.get(
            NAVER_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        # HTML 태그 제거
        items = data.get('items', [])
        for item in items:
            item['title'] = remove_html_tags(item['title'])
            item['description'] = remove_html_tags(item['description'])
        
        logger.info(f"✅ {len(items)}개 뉴스 수집 (전체 {data.get('total', 0)}개)")
        
        return {
            'success': True,
            'items': items,
            'total': data.get('total', 0)
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error("❌ API 호출 한도 초과!")
        else:
            logger.error(f"❌ HTTP 오류: {e}")
        return {'success': False, 'error': str(e)}
        
    except Exception as e:
        logger.error(f"❌ 검색 실패: {e}")
        return {'success': False, 'error': str(e)}


def remove_html_tags(text: str) -> str:
    """
    HTML 태그 제거
    """
    import re
    clean = re.sub('<.*?>', '', text)
    clean = clean.replace('&quot;', '"').replace('&amp;', '&')
    clean = clean.replace('&lt;', '<').replace('&gt;', '>')
    return clean.strip()


def search_multiple_keywords(keywords: list, display_per_keyword: int = 10) -> list:
    """
    여러 키워드로 뉴스 검색
    
    Args:
        keywords: ['EMR', '전자의무기록', ...]
        display_per_keyword: 키워드당 뉴스 개수
    
    Returns:
        [{'title': str, 'description': str, 'link': str, 'pubDate': str}, ...]
    """
    all_news = []
    seen_links = set()  # 중복 제거용
    
    for keyword in keywords:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 키워드: '{keyword}'")
        logger.info(f"{'='*60}")
        
        result = search_naver_news(keyword, display=display_per_keyword)
        
        if result['success']:
            for item in result['items']:
                # 중복 제거
                if item['link'] not in seen_links:
                    seen_links.add(item['link'])
                    
                    all_news.append({
                        'title': item['title'],
                        'description': item['description'],
                        'url': item['link'],
                        'published_at': item['pubDate'],
                        'keyword': keyword,
                        'scraped_at': datetime.now().isoformat()
                    })
        else:
            logger.warning(f"⚠️ '{keyword}' 검색 실패")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 총 {len(all_news)}개 뉴스 수집 완료 (중복 제거됨)")
    logger.info(f"{'='*60}")
    
    return all_news


def save_to_json(news_list: list, filename: str = "data/naver_news.json"):
    """
    JSON 파일로 저장
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 저장 완료: {filename}")
        
    except Exception as e:
        logger.error(f"❌ 저장 실패: {e}")


# ====================================
# 테스트 코드
# ====================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 네이버 뉴스 스크래퍼 테스트")
    print("="*60)
    
    # 테스트 1: 단일 검색
    print("\n[테스트 1] 'EMR' 검색")
    result = search_naver_news("EMR", display=5)
    
    if result['success']:
        print(f"\n✅ 총 {result['total']}개 중 {len(result['items'])}개 가져옴\n")
        
        for i, item in enumerate(result['items'][:3], 1):
            print(f"[{i}] {item['title']}")
            print(f"    📝 {item['description'][:80]}...")
            print(f"    🔗 {item['link']}")
            print(f"    📅 {item['pubDate']}\n")
    else:
        print(f"❌ 오류: {result.get('error')}")
    
    # 테스트 2: 다중 키워드
    print("\n" + "="*60)
    print("[테스트 2] 다중 키워드 검색")
    print("="*60)
    
    keywords = ["EMR", "전자의무기록", "디지털헬스케어"]
    news_list = search_multiple_keywords(keywords, display_per_keyword=5)
    
    if news_list:
        save_to_json(news_list)
        
        print("\n🔥 상위 5개 뉴스:")
        for i, news in enumerate(news_list[:5], 1):
            print(f"\n[{i}] {news['title']}")
            print(f"    키워드: {news['keyword']}")
            print(f"    {news['description'][:80]}...")
    
    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)