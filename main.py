"""
뉴스 스크래퍼
설정 파일 기반 뉴스 수집 → OpenAI 뉴스 선별 → OpenAI 요약 → 이메일 발송

사용 예:
python main.py --config configs/company_briefing.json
python main.py --config configs/economy_briefing.json
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime

# 모듈 임포트
import naver_news_scraper
import news_selector
import summarizer
import email_sender


# ====================================
# 기본 설정
# ====================================
DEFAULT_CONFIG_PATH = "configs/company_briefing.json"
DEFAULT_DISPLAY_PER_KEYWORD = 30
DEFAULT_SELECT_LIMIT = 10
DEFAULT_RECENT_HOURS = 24


# ====================================
# 디렉터리 생성
# ====================================
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ====================================
# 로깅 설정
# ====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """
    실행 인자 파싱
    """
    parser = argparse.ArgumentParser(
        description="뉴스 브리핑 스크래퍼"
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"브리핑 설정 JSON 파일 경로. 기본값: {DEFAULT_CONFIG_PATH}"
    )

    return parser.parse_args()


def load_config(config_path):
    """
    브리핑 설정 파일 로드
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config, config_path)

    return config


def validate_config(config, config_path):
    """
    설정 파일 필수값 검증
    """
    required_top_keys = ["briefing_name", "subject_prefix", "receiver_env", "sections"]

    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"{config_path} 설정 파일에 '{key}' 값이 없습니다.")

    if not isinstance(config["sections"], list) or not config["sections"]:
        raise ValueError(f"{config_path} 설정 파일의 'sections'는 비어 있지 않은 리스트여야 합니다.")

    for idx, section in enumerate(config["sections"], 1):
        for key in ["section_name", "keywords", "topic_description"]:
            if key not in section:
                raise ValueError(f"{config_path} sections[{idx}]에 '{key}' 값이 없습니다.")

        if not isinstance(section["keywords"], list) or not section["keywords"]:
            raise ValueError(f"{config_path} sections[{idx}].keywords는 비어 있지 않은 리스트여야 합니다.")


def make_safe_filename(value):
    """
    파일명에 안전한 문자열 생성
    한글은 유지하고, 특수문자/공백은 _로 치환
    """
    value = str(value).strip()
    value = re.sub(r"[^\w가-힣]+", "_", value)
    value = value.strip("_")

    if not value:
        return "section"

    return value


def get_run_paths(config_path, briefing_name):
    """
    실행 결과 저장 경로 생성

    예:
    data/company_briefing/2026-05-08/raw_의료_뉴스_브리핑.json
    """
    config_base_name = os.path.splitext(os.path.basename(config_path))[0]
    today = datetime.now().strftime("%Y-%m-%d")

    base_dir = os.path.join(
        "data",
        make_safe_filename(config_base_name),
        today
    )

    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def save_json(data, file_path):
    """
    JSON 저장
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 저장 완료: {file_path}")


def collect_select_and_summarize(
    section_name,
    keywords,
    topic_description,
    output_dir,
    display_per_keyword=DEFAULT_DISPLAY_PER_KEYWORD,
    select_limit=DEFAULT_SELECT_LIMIT,
    recent_hours=DEFAULT_RECENT_HOURS
):
    """
    섹션별 뉴스 수집 → OpenAI 선별 → 요약 처리

    Args:
        section_name: 섹션명
        keywords: 검색 키워드 리스트
        topic_description: OpenAI 선별 기준 설명
        output_dir: 결과 저장 디렉터리
        display_per_keyword: 키워드당 네이버 API 검색 개수
        select_limit: 최종 선별 개수
        recent_hours: 최근 몇 시간 뉴스만 수집할지

    Returns:
        {
            "section_name": str,
            "summaries": list,
            "raw_count": int,
            "selected_count": int
        }
    """
    safe_section_name = make_safe_filename(section_name)

    raw_json_path = os.path.join(output_dir, f"raw_{safe_section_name}.json")
    selected_json_path = os.path.join(output_dir, f"selected_{safe_section_name}.json")
    summary_json_path = os.path.join(output_dir, f"summaries_{safe_section_name}.json")

    logger.info("\n" + "=" * 60)
    logger.info(f"📰 [{section_name}] 뉴스 수집 시작")
    logger.info("=" * 60)
    logger.info(f"키워드: {', '.join(keywords)}")
    logger.info(f"최근 뉴스 기준: {recent_hours}시간 이내")
    logger.info(f"키워드당 조회 개수: {display_per_keyword}")
    logger.info(f"최종 선별 개수: {select_limit}")

    try:
        news_list = naver_news_scraper.search_multiple_keywords(
            keywords,
            display_per_keyword=display_per_keyword,
            recent_hours=recent_hours
        )
    except TypeError:
        # naver_news_scraper.py에 recent_hours 인자를 아직 반영하지 않은 경우를 위한 임시 호환 처리
        logger.warning("⚠️ naver_news_scraper.search_multiple_keywords()가 recent_hours 인자를 지원하지 않습니다.")
        logger.warning("⚠️ 최근 24시간 필터를 적용하려면 naver_news_scraper.py 수정본이 필요합니다.")

        news_list = naver_news_scraper.search_multiple_keywords(
            keywords,
            display_per_keyword=display_per_keyword
        )

    if not news_list:
        logger.error(f"❌ [{section_name}] 수집된 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": 0,
            "selected_count": 0
        }

    logger.info(f"✅ [{section_name}] 후보 뉴스 {len(news_list)}개 수집 완료")

    save_json(news_list, raw_json_path)

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
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": len(news_list),
            "selected_count": 0
        }

    save_json(selected_news, selected_json_path)

    logger.info(f"✅ [{section_name}] {len(selected_news)}개 뉴스 선별 완료")

    logger.info("\n" + "=" * 60)
    logger.info(f"🤖 [{section_name}] AI 요약 시작")
    logger.info("=" * 60)

    for news in selected_news:
        news["content"] = news.get("description", "")

    summaries = summarizer.summarize_batch(selected_news)

    if not summaries:
        logger.error(f"❌ [{section_name}] 요약 생성 실패")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": len(news_list),
            "selected_count": len(selected_news)
        }

    save_json(summaries, summary_json_path)

    logger.info(f"✅ [{section_name}] {len(summaries)}개 뉴스 요약 완료")

    return {
        "section_name": section_name,
        "summaries": summaries,
        "raw_count": len(news_list),
        "selected_count": len(selected_news)
    }


def main():
    """
    메인 실행 함수
    """
    args = parse_args()

    config_path = args.config
    config = load_config(config_path)

    briefing_name = config["briefing_name"]
    subject_prefix = config["subject_prefix"]
    sections = config["sections"]
    receiver_env = config.get("receiver_env", "EMAIL_RECEIVER")

    display_per_keyword = int(config.get("display_per_keyword", DEFAULT_DISPLAY_PER_KEYWORD))
    select_limit = int(config.get("select_limit", DEFAULT_SELECT_LIMIT))
    recent_hours = int(config.get("recent_hours", DEFAULT_RECENT_HOURS))

    output_dir = get_run_paths(
        config_path=config_path,
        briefing_name=briefing_name
    )

    logger.info("\n" + "=" * 60)
    logger.info("🚀 뉴스 스크래퍼 시작")
    logger.info(f"⏰ {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    logger.info(f"설정 파일: {config_path}")
    logger.info(f"브리핑 이름: {briefing_name}")
    logger.info(f"수신자 환경변수: {receiver_env}")
    logger.info(f"메일 제목 prefix: {subject_prefix}")
    logger.info(f"섹션 수: {len(sections)}개")
    logger.info(f"결과 저장 경로: {output_dir}")
    logger.info("=" * 60)

    section_results = []

    for section in sections:
        section_name = section["section_name"]
        keywords = section["keywords"]
        topic_description = section["topic_description"]

        section_result = collect_select_and_summarize(
            section_name=section_name,
            keywords=keywords,
            topic_description=topic_description,
            output_dir=output_dir,
            display_per_keyword=int(section.get("display_per_keyword", display_per_keyword)),
            select_limit=int(section.get("select_limit", select_limit)),
            recent_hours=int(section.get("recent_hours", recent_hours))
        )

        section_results.append(section_result)

    # 전체 결과 저장
    all_results_path = os.path.join(output_dir, "briefing_results.json")
    save_json(
        {
            "briefing_name": briefing_name,
            "subject_prefix": subject_prefix,
            "config_path": config_path,
            "created_at": datetime.now().isoformat(),
            "sections": section_results
        },
        all_results_path
    )

    # ====================================
    # 이메일 발송
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info("📧 이메일 발송 시작")
    logger.info("=" * 60)

    # 다음 단계에서 email_sender.py를 이 인자 구조에 맞게 수정해야 합니다.
    result = email_sender.send_email(
        briefing_name=briefing_name,
        subject_prefix=subject_prefix,
        section_results=section_results,
        receiver_env_name=receiver_env
    )

    if result["success"]:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")

    # ====================================
    # 최종 결과
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 작업 완료 요약")
    logger.info("=" * 60)

    total_raw_count = 0
    total_selected_count = 0
    total_summary_count = 0
    total_tokens = 0

    for section_result in section_results:
        section_name = section_result["section_name"]
        summaries = section_result["summaries"]

        raw_count = section_result["raw_count"]
        selected_count = section_result["selected_count"]
        summary_count = len(summaries)

        total_raw_count += raw_count
        total_selected_count += selected_count
        total_summary_count += summary_count
        total_tokens += sum(summary.get("tokens_used", 0) for summary in summaries)

        logger.info(f"📰 [{section_name}] 후보 수집: {raw_count}개")
        logger.info(f"🧠 [{section_name}] 뉴스 선별: {selected_count}개")
        logger.info(f"✨ [{section_name}] 뉴스 요약: {summary_count}개")

    # 기존 코드의 비용 계산 방식 유지
    # 정확한 비용은 입력/출력 토큰 단가가 달라 별도 계산이 필요합니다.
    rough_cost = total_tokens * 0.00015

    logger.info("-" * 60)
    logger.info(f"📰 전체 후보 수집: {total_raw_count}개")
    logger.info(f"🧠 전체 뉴스 선별: {total_selected_count}개")
    logger.info(f"✨ 전체 뉴스 요약: {total_summary_count}개")
    logger.info(f"🧾 요약 토큰 합계: {total_tokens:,}")
    logger.info(f"💰 기존 방식 기준 예상 비용: ${rough_cost:.4f} USD")
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