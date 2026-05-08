"""
뉴스 스크래퍼
네이버 뉴스 수집 → OpenAI 뉴스 선별 → OpenAI 요약 → 이메일 발송
"""
import logging
from datetime import datetime
import json
import os

# 모듈 임포트
import naver_news_scraper
import news_selector
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


def collect_select_and_summarize(
    section_name,
    keywords,
    topic_description,
    raw_json_path,
    selected_json_path,
    summary_json_path,
    display_per_keyword=30,
    select_limit=10
):
    """
    뉴스 수집 → OpenAI 선별 → 요약까지 공통 처리

    Args:
        section_name: 브리핑 섹션명
        keywords: 검색 키워드 리스트
        topic_description: OpenAI 선별 기준 설명
        raw_json_path: 원본 뉴스 저장 경로
        selected_json_path: 선별 뉴스 저장 경로
        summary_json_path: 요약 뉴스 저장 경로
        display_per_keyword: 키워드당 네이버 API 검색 개수
        select_limit: 최종 선별 개수

    Returns:
        summaries: 요약 결과 리스트
        raw_count: 수집 뉴스 수
        selected_count: 선별 뉴스 수
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"📰 [{section_name}] 뉴스 수집 시작")
    logger.info("=" * 60)

    news_list = naver_news_scraper.search_multiple_keywords(
        keywords,
        display_per_keyword=display_per_keyword
    )

    if not news_list:
        logger.error(f"❌ [{section_name}] 수집된 뉴스가 없습니다.")
        return [], 0, 0

    logger.info(f"✅ [{section_name}] 후보 뉴스 {len(news_list)}개 수집 완료")

    naver_news_scraper.save_to_json(
        news_list,
        raw_json_path
    )

    logger.info(f"💾 [{section_name}] 원본 뉴스 저장 완료: {raw_json_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"🧠 [{section_name}] OpenAI 뉴스 선별 시작")
    logger.info("=" * 60)

    selected_news = news_selector.select_important_news(
        news_list=news_list,
        topic_name=section_name,
        topic_description=topic_description,
        limit=select_limit
    )

    if not selected_news:
        logger.error(f"❌ [{section_name}] 선별된 뉴스가 없습니다.")
        return [], len(news_list), 0

    with open(selected_json_path, 'w', encoding='utf-8') as f:
        json.dump(selected_news, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 [{section_name}] 선별 뉴스 저장 완료: {selected_json_path}")
    logger.info(f"✅ [{section_name}] {len(selected_news)}개 뉴스 선별 완료")

    logger.info("\n" + "=" * 60)
    logger.info(f"🤖 [{section_name}] AI 요약 시작")
    logger.info("=" * 60)

    for news in selected_news:
        news['content'] = news.get('description', '')

    summaries = summarizer.summarize_batch(selected_news)

    if not summaries:
        logger.error(f"❌ [{section_name}] 요약 생성 실패")
        return [], len(news_list), len(selected_news)

    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 [{section_name}] 요약 저장 완료: {summary_json_path}")
    logger.info(f"✅ [{section_name}] {len(summaries)}개 뉴스 요약 완료")

    return summaries, len(news_list), len(selected_news)


def main():
    """
    메인 실행 함수
    """
    logger.info("\n" + "=" * 60)
    logger.info("🚀 뉴스 스크래퍼 시작")
    logger.info(f"⏰ {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    logger.info("=" * 60)

    # ====================================
    # 의료 뉴스
    # ====================================
    medical_keywords = [
        "EMR",
        "의료정보시스템",
        "의료IT",
        "전자의무기록",
        "헬스케어"
    ]

    medical_topic_description = """
의료 IT, 병원 정보시스템, EMR, 전자의무기록, 디지털 헬스케어, 의료 데이터,
병원 DX, 의료기관 시스템, 보건의료 기술 변화와 직접 관련 있는 뉴스를 선택한다.
단순 건강정보, 일반 제약/바이오 기사, 병원 홍보성 기사는 우선순위를 낮춘다.
"""

    medical_summaries, medical_raw_count, medical_selected_count = collect_select_and_summarize(
        section_name="롯데그룹 의료뉴스브리핑",
        keywords=medical_keywords,
        topic_description=medical_topic_description,
        raw_json_path="data/raw_medical_news.json",
        selected_json_path="data/selected_medical_news.json",
        summary_json_path="data/medical_summaries.json",
        display_per_keyword=30,
        select_limit=10
    )

    # ====================================
    # 롯데 관련 뉴스
    # ====================================
    lotte_keywords = [
        "롯데이노베이트",
        "롯데그룹"
    ]

    lotte_topic_description = """
롯데그룹, 롯데이노베이트, 롯데 계열사, 그룹 전략, IT/DX 사업, 신사업,
실적, 투자, 제휴, 인사, 경영 변화와 직접 관련 있는 뉴스를 선택한다.
단순 이벤트, 광고성 기사, 유통 할인 행사, 키워드만 포함된 관련성 낮은 기사는 제외한다.
"""

    lotte_summaries, lotte_raw_count, lotte_selected_count = collect_select_and_summarize(
        section_name="롯데 관련 뉴스브리핑",
        keywords=lotte_keywords,
        topic_description=lotte_topic_description,
        raw_json_path="data/raw_lotte_news.json",
        selected_json_path="data/selected_lotte_news.json",
        summary_json_path="data/lotte_summaries.json",
        display_per_keyword=30,
        select_limit=10
    )

    # ====================================
    # 이메일 발송
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info("📧 이메일 발송 시작")
    logger.info("=" * 60)

    result = email_sender.send_email(
        medical_summaries=medical_summaries,
        lotte_summaries=lotte_summaries
    )

    if result['success']:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")

    # ====================================
    # 최종 결과
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 작업 완료 요약")
    logger.info("=" * 60)

    all_summaries = medical_summaries + lotte_summaries
    total_tokens = sum(s.get('tokens_used', 0) for s in all_summaries)
    summary_cost = total_tokens * 0.00015

    logger.info(f"📰 의료 뉴스 후보 수집: {medical_raw_count}개")
    logger.info(f"🧠 의료 뉴스 선별: {medical_selected_count}개")
    logger.info(f"✨ 의료 뉴스 요약: {len(medical_summaries)}개")

    logger.info(f"📰 롯데 뉴스 후보 수집: {lotte_raw_count}개")
    logger.info(f"🧠 롯데 뉴스 선별: {lotte_selected_count}개")
    logger.info(f"✨ 롯데 뉴스 요약: {len(lotte_summaries)}개")

    logger.info(f"💰 요약 기준 예상 비용: ${summary_cost:.4f} USD")
    logger.info(f"📧 발송: {result['success']}")


    logger.info("\n" + "=" * 60)
    logger.info("✅ 모든 작업 완료")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자가 작업을 중단했습니다.")
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()