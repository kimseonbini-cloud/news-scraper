"""
EMR 뉴스 스크래퍼 - 최종 버전
네이버 뉴스 수집 → OpenAI 요약 → 이메일 발송
"""
import logging
from datetime import datetime
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


def main():
    """
    메인 실행 함수
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 EMR 뉴스 스크래퍼 시작")
    logger.info(f"⏰ {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    logger.info("="*60)
    
    # ====================================
    # Step 1: 네이버 뉴스 수집
    # ====================================
    logger.info("\n📰 [Step 1] 네이버 뉴스 수집 중...")
    
    keywords = [
        "EMR",
        "전자의무기록",
        "디지털헬스케어",
        "원격의료",
        "의료정보시스템"
    ]
    
    news_list = naver_news_scraper.search_multiple_keywords(
        keywords,
        display_per_keyword=15  # 키워드당 15개
    )
    
    if not news_list:
        logger.error("❌ 수집된 뉴스가 없습니다!")
        return
    
    logger.info(f"✅ 총 {len(news_list)}개 뉴스 수집 완료")
    
    # 원본 저장
    naver_news_scraper.save_to_json(news_list, 'data/raw_news.json')
    
    # ====================================
    # Step 2: OpenAI로 요약
    # ====================================
    logger.info("\n🤖 [Step 2] AI 요약 시작...")
    
    # 상위 10개만 요약 (비용 절감)
    news_to_summarize = news_list[:10]
    
    # description을 content로 변환
    for news in news_to_summarize:
        news['content'] = news['description']
    
    summaries = summarizer.summarize_batch(news_to_summarize)
    
    if not summaries:
        logger.error("❌ 요약 실패!")
        return
    
    logger.info(f"✅ {len(summaries)}개 뉴스 요약 완료")
    
    # 요약 저장
    with open('data/summaries.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
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