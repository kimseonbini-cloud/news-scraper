"""
뉴스 스크래퍼
설정 파일 기반 뉴스 수집 → 최근 이슈 중복 제거 → OpenAI 뉴스 선별 → OpenAI 요약 → 이메일 발송 → 이슈 히스토리 저장

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
import issue_history


# ====================================
# 기본 설정
# ====================================
DEFAULT_CONFIG_PATH = "configs/company_briefing.json"

# 네이버 뉴스 API는 1회 호출 display 최대 100개.
# 기본값: 키워드당 관련도순(sim) 100개 × 3페이지 = 최대 300개 조회.
# 최신성 비중을 높이고 싶으면 설정 파일에서 sorts를 ["date", "sim"] 또는 ["date"]로 지정한다.
DEFAULT_DISPLAY_PER_KEYWORD = 100
DEFAULT_PAGES_PER_KEYWORD = 3
DEFAULT_SORTS = ["sim"]

# 최근 몇 시간 이내 뉴스만 사용할지
DEFAULT_RECENT_HOURS = 24

# 시간대별 샘플링 후 AI 선별 단계로 넘길 최종 후보 최대 개수
DEFAULT_MAX_TOTAL_NEWS = 100

# OpenAI 최종 선별 개수
DEFAULT_SELECT_LIMIT = 10

# 최근 며칠간 이미 다룬 이슈를 비교할지
DEFAULT_ISSUE_HISTORY_DAYS = 3


# ====================================
# 로깅 설정
# - GitHub Actions 콘솔 로그만 남김
# - logs/scraper.log 파일 저장 안 함
# ====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
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


def normalize_sorts(value):
    """
    설정 파일의 sorts 값을 안전하게 리스트로 변환한다.

    지원:
    - ["date"]
    - ["sim"]
    - ["sim", "date"]
    - "date"
    - "sim"
    - 값 없음 → DEFAULT_SORTS
    """
    if value is None:
        return DEFAULT_SORTS

    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or DEFAULT_SORTS

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return DEFAULT_SORTS
        return [value]

    return DEFAULT_SORTS


def collect_select_and_summarize(
    briefing_name,
    receiver_env,
    section_name,
    keywords,
    topic_description,
    display_per_keyword=DEFAULT_DISPLAY_PER_KEYWORD,
    pages_per_keyword=DEFAULT_PAGES_PER_KEYWORD,
    sorts=None,
    max_total_news=DEFAULT_MAX_TOTAL_NEWS,
    select_limit=DEFAULT_SELECT_LIMIT,
    recent_hours=DEFAULT_RECENT_HOURS,
    issue_history_days=DEFAULT_ISSUE_HISTORY_DAYS
):
    """
    섹션별 뉴스 수집 → 최근 반복 이슈 제거 → OpenAI 선별 → 요약 처리

    Args:
        briefing_name: 브리핑명
        receiver_env: 수신자 환경변수명. 사내용/경제/이노하스 등 히스토리 구분용
        section_name: 섹션명
        keywords: 검색 키워드 리스트
        topic_description: OpenAI 선별 기준 설명
        display_per_keyword: 키워드당 페이지별 네이버 API 검색 개수
        pages_per_keyword: 키워드당 조회 페이지 수
        sorts: 정렬 방식 리스트
        max_total_news: AI 선별 단계로 넘길 최종 후보 최대 개수
        select_limit: 최종 선별 개수
        recent_hours: 최근 몇 시간 뉴스만 수집할지
        issue_history_days: 최근 며칠간 이미 다룬 이슈와 비교할지

    Returns:
        {
            "section_name": str,
            "summaries": list,
            "raw_count": int,
            "selected_count": int,
            "scrape_stats": dict
        }
    """
    sorts = normalize_sorts(sorts)

    logger.info("\n" + "=" * 60)
    logger.info(f"📰 [{section_name}] 뉴스 수집 시작")
    logger.info("=" * 60)
    logger.info(f"브리핑 이름: {briefing_name}")
    logger.info(f"수신자 환경변수: {receiver_env}")
    logger.info(f"키워드: {', '.join(keywords)}")
    logger.info(f"최근 뉴스 기준: {recent_hours}시간 이내")
    logger.info(f"정렬 방식: {sorts}")
    logger.info(f"키워드당 페이지별 조회 개수: {display_per_keyword}")
    logger.info(f"키워드당 페이지 수: {pages_per_keyword}")
    logger.info(f"키워드당 최대 조회 개수: {display_per_keyword * pages_per_keyword}")
    logger.info(f"AI 선별 전달 최대 후보 수: {max_total_news}")
    logger.info(f"최종 선별 개수: {select_limit}")
    logger.info(f"반복 이슈 비교 기간: 최근 {issue_history_days}일")

    # 1) 먼저 시간대 샘플링 없이 넓게 수집한다.
    # 반복/내부중복 제거를 먼저 하고, 그 다음 max_total_news 제한을 적용해야
    # 중복 기사들이 100개 후보 슬롯을 차지했다가 나중에 버려지는 손실을 막을 수 있다.
    news_list = naver_news_scraper.search_multiple_keywords(
        keywords=keywords,
        display_per_keyword=display_per_keyword,
        recent_hours=recent_hours,
        sorts=sorts,
        pages_per_keyword=pages_per_keyword,
        enable_time_bucket_sampling=False,
        max_total_news=max_total_news
    )

    # naver_news_scraper.py에서 마지막 수집 통계를 가져온다.
    # 이 시점의 news_list는 URL/시간 필터까지만 거친 전체 후보다.
    scrape_stats = naver_news_scraper.get_last_scrape_stats()
    scrape_stats["pre_issue_filter_candidate_count"] = len(news_list)

    if not news_list:
        logger.error(f"❌ [{section_name}] 수집된 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": 0,
            "selected_count": 0,
            "scrape_stats": scrape_stats
        }

    logger.info(f"✅ [{section_name}] 후보 뉴스 {len(news_list)}개 수집 완료")

    # ====================================
    # 최근 N일 반복 이슈 제거
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info(f"🧹 [{section_name}] 최근 {issue_history_days}일 반복 이슈 제거 시작")
    logger.info("=" * 60)

    issue_filter_result = issue_history.filter_seen_issues_with_llm(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        candidate_news=news_list,
        days=issue_history_days
    )

    before_issue_filter_count = len(news_list)
    news_list = issue_filter_result.get("filtered_news", news_list)
    after_issue_filter_count = len(news_list)

    logger.info(
        f"🧹 [{section_name}] 반복 이슈 제거: "
        f"후보 {before_issue_filter_count}개 중 "
        f"{issue_filter_result.get('excluded_count', 0)}개 제외, "
        f"{after_issue_filter_count}개 유지 "
        f"(최근 {issue_history_days}일 이슈 {issue_filter_result.get('past_issue_count', 0)}개 비교)"
    )

    if issue_filter_result.get("message"):
        logger.info(f"🧹 [{section_name}] 반복 이슈 필터 상태: {issue_filter_result.get('message')}")

    scrape_stats["issue_filter_before_count"] = before_issue_filter_count
    scrape_stats["issue_filter_after_count"] = after_issue_filter_count
    scrape_stats["issue_filter_excluded_count"] = issue_filter_result.get("excluded_count", 0)
    scrape_stats["issue_filter_past_issue_count"] = issue_filter_result.get("past_issue_count", 0)
    scrape_stats["issue_filter_days"] = issue_history_days
    scrape_stats["issue_filter_success"] = issue_filter_result.get("success", False)
    scrape_stats["issue_filter_message"] = issue_filter_result.get("message", "")
    scrape_stats["issue_filter_url_excluded_count"] = issue_filter_result.get("url_excluded_count", 0)
    scrape_stats["issue_filter_title_excluded_count"] = issue_filter_result.get("title_excluded_count", 0)
    # issue_history.py는 이제 LLM/issue_key를 쓰지 않고 규칙 기반 fingerprint 비교만 수행한다.
    scrape_stats["issue_filter_core_key_excluded_count"] = issue_filter_result.get("core_key_excluded_count", 0)
    scrape_stats["issue_filter_llm_excluded_count"] = issue_filter_result.get("llm_excluded_count", 0)
    scrape_stats["issue_filter_internal_duplicate_count"] = issue_filter_result.get("internal_duplicate_count", 0)
    scrape_stats["issue_filter_text_excluded_count"] = issue_filter_result.get("text_excluded_count", 0)
    scrape_stats["issue_filter_token_overlap_excluded_count"] = issue_filter_result.get("token_overlap_excluded_count", 0)
    scrape_stats["issue_filter_simhash_excluded_count"] = issue_filter_result.get("simhash_excluded_count", 0)

    issue_token_stats = issue_filter_result.get("token_stats", {}) or {}
    scrape_stats["issue_key_tokens"] = issue_token_stats.get("issue_key_tokens", 0)
    scrape_stats["issue_duplicate_tokens"] = issue_token_stats.get("llm_duplicate_tokens", 0)

    # 2) 반복/내부중복 제거가 끝난 뒤에 시간대별 샘플링으로 최종 후보 수를 제한한다.
    # 이렇게 해야 AI 선별 후보가 중복 제거 후에도 최대한 max_total_news에 가깝게 유지된다.
    before_sampling_after_filter_count = len(news_list)
    news_list = naver_news_scraper.sample_news_by_time_bucket(
        news_list=news_list,
        bucket_hours=4,
        max_total_news=max_total_news,
        min_per_bucket=0,
        recent_hours=recent_hours
    )
    after_sampling_count = len(news_list)

    naver_news_scraper.update_post_issue_filter_sampling_stats(
        before_issue_filter_count=before_issue_filter_count,
        after_issue_filter_count=after_issue_filter_count,
        before_sampling_count=before_sampling_after_filter_count,
        after_sampling_count=after_sampling_count,
    )

    scrape_stats = naver_news_scraper.get_last_scrape_stats()

    # issue_history 결과 통계는 update_post_issue_filter_sampling_stats 이후에도 보존되도록 다시 반영한다.
    scrape_stats["issue_filter_excluded_count"] = issue_filter_result.get("excluded_count", 0)
    scrape_stats["issue_filter_past_issue_count"] = issue_filter_result.get("past_issue_count", 0)
    scrape_stats["issue_filter_days"] = issue_history_days
    scrape_stats["issue_filter_success"] = issue_filter_result.get("success", False)
    scrape_stats["issue_filter_message"] = issue_filter_result.get("message", "")
    scrape_stats["issue_filter_url_excluded_count"] = issue_filter_result.get("url_excluded_count", 0)
    scrape_stats["issue_filter_title_excluded_count"] = issue_filter_result.get("title_excluded_count", 0)
    scrape_stats["issue_filter_core_key_excluded_count"] = issue_filter_result.get("core_key_excluded_count", 0)
    scrape_stats["issue_filter_llm_excluded_count"] = issue_filter_result.get("llm_excluded_count", 0)
    scrape_stats["issue_filter_internal_duplicate_count"] = issue_filter_result.get("internal_duplicate_count", 0)
    scrape_stats["issue_filter_text_excluded_count"] = issue_filter_result.get("text_excluded_count", 0)
    scrape_stats["issue_filter_token_overlap_excluded_count"] = issue_filter_result.get("token_overlap_excluded_count", 0)
    scrape_stats["issue_filter_simhash_excluded_count"] = issue_filter_result.get("simhash_excluded_count", 0)
    scrape_stats["issue_key_tokens"] = issue_token_stats.get("issue_key_tokens", 0)
    scrape_stats["issue_duplicate_tokens"] = issue_token_stats.get("llm_duplicate_tokens", 0)

    logger.info(
        f"🧺 [{section_name}] 필터 후 시간대 샘플링: "
        f"{before_sampling_after_filter_count}개 → {after_sampling_count}개 "
        f"(max_total_news={max_total_news})"
    )

    if not news_list:
        logger.error(f"❌ [{section_name}] 최근 {issue_history_days}일 반복 이슈 제거 후 남은 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": 0,
            "selected_count": 0,
            "scrape_stats": scrape_stats
        }

    # ====================================
    # OpenAI 뉴스 선별
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info(f"🧠 [{section_name}] OpenAI 뉴스 선별 시작")
    logger.info("=" * 60)

    selected_news = news_selector.select_important_news(
        news_list=news_list,
        topic_name=section_name,
        topic_description=topic_description,
        limit=select_limit
    )

    selection_stats = news_selector.get_last_selection_stats()
    scrape_stats["selection_tokens"] = selection_stats.get("selection_tokens", 0)
    scrape_stats["event_group_tokens"] = selection_stats.get("event_group_tokens", 0)

    if not selected_news:
        logger.error(f"❌ [{section_name}] 선별된 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": len(news_list),
            "selected_count": 0,
            "scrape_stats": scrape_stats
        }

    logger.info(f"✅ [{section_name}] {len(selected_news)}개 뉴스 선별 완료")

    # ====================================
    # OpenAI 뉴스 요약
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info(f"🤖 [{section_name}] AI 요약 시작")
    logger.info("=" * 60)

    for news in selected_news:
        news["content"] = news.get("description", "")

    summaries = summarizer.summarize_batch(selected_news)
    scrape_stats["summary_tokens"] = sum(summary.get("tokens_used", 0) for summary in summaries or [])

    if not summaries:
        logger.error(f"❌ [{section_name}] 요약 생성 실패")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": len(news_list),
            "selected_count": len(selected_news),
            "scrape_stats": scrape_stats
        }

    logger.info(f"✅ [{section_name}] {len(summaries)}개 뉴스 요약 완료")

    return {
        "section_name": section_name,
        "summaries": summaries,
        "raw_count": len(news_list),
        "selected_count": len(selected_news),
        "scrape_stats": scrape_stats
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
    pages_per_keyword = int(config.get("pages_per_keyword", DEFAULT_PAGES_PER_KEYWORD))
    sorts = normalize_sorts(config.get("sorts", DEFAULT_SORTS))
    max_total_news = int(config.get("max_total_news", DEFAULT_MAX_TOTAL_NEWS))
    select_limit = int(config.get("select_limit", DEFAULT_SELECT_LIMIT))
    recent_hours = int(config.get("recent_hours", DEFAULT_RECENT_HOURS))
    issue_history_days = int(config.get("issue_history_days", DEFAULT_ISSUE_HISTORY_DAYS))

    logger.info("\n" + "=" * 60)
    logger.info("🚀 뉴스 스크래퍼 시작")
    logger.info(f"⏰ {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    logger.info(f"설정 파일: {config_path}")
    logger.info(f"브리핑 이름: {briefing_name}")
    logger.info(f"수신자 환경변수: {receiver_env}")
    logger.info(f"메일 제목 prefix: {subject_prefix}")
    logger.info(f"섹션 수: {len(sections)}개")
    logger.info(f"기본 정렬 방식: {sorts}")
    logger.info(f"기본 키워드당 페이지별 조회 개수: {display_per_keyword}")
    logger.info(f"기본 키워드당 페이지 수: {pages_per_keyword}")
    logger.info(f"기본 키워드당 최대 조회 개수: {display_per_keyword * pages_per_keyword}")
    logger.info(f"기본 AI 선별 전달 최대 후보 수: {max_total_news}")
    logger.info(f"기본 반복 이슈 비교 기간: 최근 {issue_history_days}일")
    logger.info("결과 JSON 파일 저장: 비활성화")
    logger.info("=" * 60)

    section_results = []

    for section in sections:
        section_name = section["section_name"]
        keywords = section["keywords"]
        topic_description = section["topic_description"]

        section_display_per_keyword = int(
            section.get("display_per_keyword", display_per_keyword)
        )
        section_pages_per_keyword = int(
            section.get("pages_per_keyword", pages_per_keyword)
        )
        section_sorts = normalize_sorts(
            section.get("sorts", sorts)
        )
        section_max_total_news = int(
            section.get("max_total_news", max_total_news)
        )
        section_select_limit = int(
            section.get("select_limit", select_limit)
        )
        section_recent_hours = int(
            section.get("recent_hours", recent_hours)
        )
        section_issue_history_days = int(
            section.get("issue_history_days", issue_history_days)
        )

        section_result = collect_select_and_summarize(
            briefing_name=briefing_name,
            receiver_env=receiver_env,
            section_name=section_name,
            keywords=keywords,
            topic_description=topic_description,
            display_per_keyword=section_display_per_keyword,
            pages_per_keyword=section_pages_per_keyword,
            sorts=section_sorts,
            max_total_news=section_max_total_news,
            select_limit=section_select_limit,
            recent_hours=section_recent_hours,
            issue_history_days=section_issue_history_days
        )

        section_results.append(section_result)

    # ====================================
    # 이메일 발송
    # ====================================
    logger.info("\n" + "=" * 60)
    logger.info("📧 이메일 발송 시작")
    logger.info("=" * 60)

    result = email_sender.send_email(
        briefing_name=briefing_name,
        subject_prefix=subject_prefix,
        section_results=section_results,
        receiver_env_name=receiver_env
    )

    if result["success"]:
        logger.info(f"✅ {result['message']}")

        history_result = issue_history.append_sent_issues(
            briefing_name=briefing_name,
            subject_prefix=subject_prefix,
            receiver_env=receiver_env,
            section_results=section_results
        )

        logger.info(
            f"🗂️ 이슈 히스토리 저장 완료: "
            f"신규 {history_result['saved_count']}개 / "
            f"오래된 이슈 삭제 {history_result.get('pruned_count', 0)}개 / "
            f"누적 {history_result['total_count']}개 / "
            f"{history_result['file_path']}"
        )

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
    total_issue_key_tokens = 0
    total_issue_duplicate_tokens = 0
    total_selection_tokens = 0
    total_event_group_tokens = 0
    total_summary_tokens = 0
    total_insight_tokens = 0

    for section_result in section_results:
        section_name = section_result["section_name"]
        summaries = section_result["summaries"]
        scrape_stats = section_result.get("scrape_stats", {})

        raw_count = section_result["raw_count"]
        selected_count = section_result["selected_count"]
        summary_count = len(summaries)

        duplicate_count = scrape_stats.get("duplicate_count", 0)
        old_news_count = scrape_stats.get("old_news_count", 0)
        pre_sampling_count = scrape_stats.get("pre_sampling_count", raw_count)
        final_candidate_count = scrape_stats.get("final_candidate_count", raw_count)

        issue_filter_before_count = scrape_stats.get("issue_filter_before_count", final_candidate_count)
        issue_filter_after_count = scrape_stats.get("issue_filter_after_count", raw_count)
        issue_filter_excluded_count = scrape_stats.get("issue_filter_excluded_count", 0)
        issue_filter_past_issue_count = scrape_stats.get("issue_filter_past_issue_count", 0)

        total_raw_count += raw_count
        total_selected_count += selected_count
        total_summary_count += summary_count
        issue_key_tokens = scrape_stats.get("issue_key_tokens", 0)
        issue_duplicate_tokens = scrape_stats.get("issue_duplicate_tokens", 0)
        selection_tokens = scrape_stats.get("selection_tokens", 0)
        event_group_tokens = scrape_stats.get("event_group_tokens", 0)
        summary_tokens = scrape_stats.get("summary_tokens", sum(summary.get("tokens_used", 0) for summary in summaries))
        insight_tokens = scrape_stats.get("insight_tokens", 0)

        section_total_tokens = (
            issue_key_tokens
            + issue_duplicate_tokens
            + selection_tokens
            + event_group_tokens
            + summary_tokens
            + insight_tokens
        )

        total_issue_key_tokens += issue_key_tokens
        total_issue_duplicate_tokens += issue_duplicate_tokens
        total_selection_tokens += selection_tokens
        total_event_group_tokens += event_group_tokens
        total_summary_tokens += summary_tokens
        total_insight_tokens += insight_tokens
        total_tokens += section_total_tokens

        logger.info(f"📰 [{section_name}] 후보 수집: {raw_count}개")
        logger.info(f"🧠 [{section_name}] 뉴스 선별: {selected_count}개")
        logger.info(f"✨ [{section_name}] 뉴스 요약: {summary_count}개")
        after_issue_filter_before_sampling_count = scrape_stats.get(
            "after_issue_filter_before_sampling_count", issue_filter_after_count
        )
        after_issue_filter_after_sampling_count = scrape_stats.get(
            "after_issue_filter_after_sampling_count", final_candidate_count
        )
        logger.info(
            f"📊 [{section_name}] 수집 통계: "
            f"URL/시간 필터 후 {pre_sampling_count}개 / "
            f"스크래퍼 내부 샘플링 "
            f"{'적용' if scrape_stats.get('scraper_sampling_applied') else '없음'} / "
            f"URL 중복 제외 {duplicate_count}개 / "
            f"{scrape_stats.get('recent_hours', DEFAULT_RECENT_HOURS)}시간 초과 제외 {old_news_count}개"
        )
        logger.info(
            f"🧹 [{section_name}] 반복 이슈 필터: "
            f"{issue_filter_before_count}개 → {issue_filter_after_count}개 / "
            f"제외 {issue_filter_excluded_count}개 / "
            f"비교 이슈 {issue_filter_past_issue_count}개"
        )
        logger.info(
            f"🧺 [{section_name}] 필터 후 시간대 샘플링: "
            f"{after_issue_filter_before_sampling_count}개 → "
            f"{after_issue_filter_after_sampling_count}개 / "
            f"AI 선별 후보 {final_candidate_count}개"
        )
        logger.info(
            f"🧹 [{section_name}] 반복 이슈 제외 상세: "
            f"URL {scrape_stats.get('issue_filter_url_excluded_count', 0)}개 / "
            f"제목 {scrape_stats.get('issue_filter_title_excluded_count', 0)}개 / "
            f"본문유사 {scrape_stats.get('issue_filter_text_excluded_count', 0)}개 / "
            f"토큰겹침 {scrape_stats.get('issue_filter_token_overlap_excluded_count', 0)}개 / "
            f"SimHash {scrape_stats.get('issue_filter_simhash_excluded_count', 0)}개 / "
            f"LLM {scrape_stats.get('issue_filter_llm_excluded_count', 0)}개 / "
            f"오늘 후보 내부 {scrape_stats.get('issue_filter_internal_duplicate_count', 0)}개"
        )
        logger.info(
            f"🧾 [{section_name}] 토큰 사용량: "
            f"issue_key {issue_key_tokens:,}(사용 안 함) / "
            f"반복판단LLM {issue_duplicate_tokens:,}(사용 안 함) / "
            f"선별 {selection_tokens:,} / "
            f"사건그룹 {event_group_tokens:,} / "
            f"요약 {summary_tokens:,} / "
            f"메일3줄 {insight_tokens:,} / "
            f"합계 {section_total_tokens:,}"
        )

    # 단순 추정 비용. 실제 비용은 모델별 입력/출력 토큰 단가에 따라 달라질 수 있다.
    rough_cost = total_tokens * 0.00015

    logger.info("-" * 60)
    logger.info(f"📰 전체 후보 수집: {total_raw_count}개")
    logger.info(f"🧠 전체 뉴스 선별: {total_selected_count}개")
    logger.info(f"✨ 전체 뉴스 요약: {total_summary_count}개")
    logger.info(f"🧾 issue_key 생성 토큰 합계: {total_issue_key_tokens:,} (사용 안 함)")
    logger.info(f"🧾 반복 이슈 LLM 판단 토큰 합계: {total_issue_duplicate_tokens:,} (사용 안 함)")
    logger.info(f"🧾 뉴스 선별 토큰 합계: {total_selection_tokens:,}")
    logger.info(f"🧾 사건 그룹화 토큰 합계: {total_event_group_tokens:,}")
    logger.info(f"🧾 뉴스 요약 토큰 합계: {total_summary_tokens:,}")
    logger.info(f"🧾 메일 핵심 3줄 토큰 합계: {total_insight_tokens:,}")
    logger.info(f"🧾 전체 추적 토큰 합계: {total_tokens:,}")
    logger.info(f"💰 단순 추정 예상 비용: ${rough_cost:.4f} USD")
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