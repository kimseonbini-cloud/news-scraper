"""
뉴스 스크래퍼
네이버 뉴스 수집 → OpenAI 요약 → 이메일 발송
"""
import logging
from datetime import datetime
from difflib import SequenceMatcher
import json
import os

# 모듈 임포트
import naver_news_scraper
import summarizer
import email_sender

# 로깅 설정
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def select_balanced_news(news_list, keywords, total_limit=10):
    """
    키워드별로 균등하게 배분하되, 각 키워드 내에서는 최신순
    """
    from datetime import datetime
    
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y.%m.%d')
        except:
            return datetime.now()
    
    # 키워드당 할당량
    per_keyword = max(1, total_limit // len(keywords))
    
    selected_news = []
    
    for keyword in keywords:
        # 키워드별 필터링
        keyword_news = [
            news for news in news_list 
            if news.get('keyword') == keyword
        ]
        
        # 날짜순 정렬
        keyword_news_sorted = sorted(
            keyword_news,
            key=lambda x: parse_date(x.get('date', '')),
            reverse=True
        )
        
        # 할당량만큼 선택
        selected = keyword_news_sorted[:per_keyword]
        selected_news.extend(selected)
        
        logger.info(f"   📌 '{keyword}': {len(keyword_news)}개 중 최신 {len(selected)}개 선택")
    
    # 전체 제한
    if len(selected_news) > total_limit:
        selected_news = selected_news[:total_limit]
    
    return selected_news

def calculate_similarity(str1, str2):
    """
    두 문자열의 유사도 계산 (0.0 ~ 1.0)
    
    Args:
        str1: 첫 번째 문자열
        str2: 두 번째 문자열
    
    Returns:
        float: 유사도 (0.0 ~ 1.0)
    """
    return SequenceMatcher(None, str1, str2).ratio()


def remove_duplicates(news_list, similarity_threshold=0.8):
    """
    중복 뉴스 제거 (제목 유사도 기반)
    
    Args:
        news_list: 뉴스 리스트
        similarity_threshold: 유사도 임계값 (0.8 = 80% 이상 유사하면 중복)
    
    Returns:
        list: 중복 제거된 뉴스 리스트
    """
    if not news_list:
        return []
    
    unique_news = []
    removed_count = 0
    
    for news in news_list:
        is_duplicate = False
        current_title = news.get('title', '').strip()
        
        # 기존 뉴스들과 비교
        for unique in unique_news:
            existing_title = unique.get('title', '').strip()
            
            # 제목 유사도 계산
            similarity = calculate_similarity(current_title, existing_title)
            
            if similarity >= similarity_threshold:
                is_duplicate = True
                removed_count += 1
                logger.info(f"   🗑️  중복 제거: '{current_title[:30]}...' (유사도: {similarity:.2%})")
                break
        
        # 중복이 아니면 추가
        if not is_duplicate:
            unique_news.append(news)
    
    logger.info(f"   ✅ 중복 제거 완료: {removed_count}개 제거, {len(unique_news)}개 남음")
    
    return unique_news


def main():
    """
    메인 실행 함수
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 뉴스 스크래퍼 시작")
    logger.info(f"⏰ {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    logger.info("="*60)
    
    # ====================================
    # Step 1: 네이버 뉴스 수집
    # ====================================
    logger.info("\n📰 [Step 1] 네이버 뉴스 수집 중...")
    
    keywords = [
        "EMR",
        "의료정보시스템",
        "병원정보시스템",
        "전자의무기록",
        "헬스케어"
    ]
    
    news_list = naver_news_scraper.search_multiple_keywords(
        keywords,
        display_per_keyword=15  # 키워드당 15개
    )
    
    if not news_list:
        logger.error("❌ 수집된 뉴스가 없습니다!")
        return
    
    logger.info(f"✅ 총 {len(news_list)}개 뉴스 수집 완료")
    
    # [Step 1.5] 중복 제거
    logger.info("\n🔍 [Step 1.5] 중복 뉴스 제거 중...")
    
    news_list = remove_duplicates(
        news_list=news_list,
        similarity_threshold=0.5  # 50% 이상 유사하면 중복으로 판단
    )
    # 원본 저장
    naver_news_scraper.save_to_json(news_list, 'data/raw_news.json')
    
    logger.info("\n📊 [Step 2] 요약할 뉴스 선택 중...")
    
    news_to_summarize = select_balanced_news(
        news_list=news_list,
        keywords=keywords,
        total_limit=10  # 총 10개 (키워드당 3~4개씩)
    )
    
    logger.info(f"✅ {len(news_to_summarize)}개 뉴스 선택 완료")
    
    # [Step 3] 요약
    logger.info("\n🤖 [Step 3] AI 요약 중...")
    
    for news in news_to_summarize:
        news['content'] = news['description']
    
    # [Step 3] AI 요약
    logger.info("\n🤖 [Step 3] AI 요약 중...")
    
    summaries = summarizer.summarize_batch(news_to_summarize)
    
    if not summaries:
        logger.error("❌ 요약 생성 실패!")
        return
    
    logger.info(f"✅ {len(summaries)}개 뉴스 요약 완료")
    
    # 요약 저장
    # 요약 저장
    with open('data/summaries.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logger.info("💾 요약 저장 완료: data/summarized_news.json")
    
    # ====================================
    # Step 3: 이메일 발송
    # ====================================
    logger.info("\n📧 [Step 3] 이메일 발송 중...")
    
    result = email_sender.send_email(summaries)
    
    if result['success']:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")
    
    # ====================================
    # 최종 결과
    # ====================================
    logger.info("\n" + "="*60)
    logger.info("📊 작업 완료 요약")
    logger.info("="*60)
    
    total_tokens = sum(s.get('tokens_used', 0) for s in summaries)
    cost = total_tokens * 0.00015  # gpt-4o-mini 가격
    
    logger.info(f"📰 수집: {len(news_list)}개")
    logger.info(f"✨ 요약: {len(summaries)}개")
    logger.info(f"💰 비용: ${cost:.4f} USD")
    logger.info(f"📧 발송: {result['success']}")
    
    # 주요 뉴스 출력
    logger.info("\n" + "="*60)
    logger.info("🔥 주요 뉴스 TOP 3")
    logger.info("="*60)
    
    for i, summary in enumerate(summaries[:3], 1):
        logger.info(f"\n[{i}] {summary['title']}")
        logger.info(f"📝 {summary['summary']}")
        logger.info(f"🔗 {summary['url']}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 모든 작업 완료!")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자가 작업을 중단했습니다.")
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()