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
import news_grouper
import summarizer
import email_sender
import issue_history
from openai_usage import reset_openai_usage_totals, log_openai_usage_summary


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

# OpenAI 그룹 선별에 실제로 전달할 후보 그룹 수
# 로컬 그룹화는 넓게 하되, AI 입력은 설정별로 압축해 토큰을 줄인다.
DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT = 45

# 요약 최대 길이. 설정 파일/섹션별로 조절 가능하다.
DEFAULT_SUMMARY_MAX_LENGTH = 180

# 메일 상단 3줄 흐름 요약을 별도 AI 호출로 만들지 여부.
# 기본은 꺼서 섹션당 1회씩 추가되는 AI 호출을 줄인다.
DEFAULT_EMAIL_INSIGHT_AI = False

# 이메일 발송 방식: individual=수신자별 개별 발송, bulk=수신자 전체에게 1회 발송
DEFAULT_EMAIL_SEND_MODE = "individual"

# 최근 며칠간 이미 다룬 이슈를 비교할지
DEFAULT_ISSUE_HISTORY_DAYS = 3

# 발송 성공 후 seen_issues.json에 저장할지 여부
DEFAULT_SAVE_ISSUE_HISTORY = True


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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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

    parser.add_argument(
        "--no-save-history",
        action="store_true",
        help="테스트 실행용. 이메일 발송 성공 후 seen_issues.json 저장을 건너뜁니다."
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


def get_config_slug(config_path):
    """
    설정 파일 경로에서 상태 파일 구분용 slug를 생성한다.

    예:
    - configs/company_briefing.json -> company_briefing
    - configs/economy_briefing.json -> economy_briefing
    - configs/inohas_briefing.json -> inohas_briefing

    config 파일이 추가되어도 파일명 기준으로 자동 분리된다.
    """
    config_base_name = os.path.splitext(os.path.basename(config_path))[0]
    return make_safe_filename(config_base_name)


def get_state_file_paths(config_path):
    """
    설정 파일 기준으로 상태 파일 경로를 동적으로 생성한다.

    같은 저장소에서 여러 브리핑 workflow가 실행되더라도
    브리핑별 상태 파일을 따로 사용해 Git 충돌 가능성을 낮춘다.
    """
    config_slug = get_config_slug(config_path)
    state_dir = os.path.join("data", "state", config_slug)
    os.makedirs(state_dir, exist_ok=True)

    return {
        "config_slug": config_slug,
        "state_dir": state_dir,
        "issue_history_file_path": os.path.join(state_dir, "seen_issues.json"),
        "unmapped_press_domains_file_path": os.path.join(state_dir, "unmapped_press_domains.json"),
    }


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




def normalize_exclude_keywords(value):
    """
    설정 파일의 exclude_keywords 값을 안전하게 리스트로 변환한다.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def normalize_bool(value, default=False):
    """
    설정 파일의 bool 값을 안전하게 변환한다.
    JSON bool뿐 아니라 "true"/"false", "1"/"0" 문자열도 지원한다.
    """
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    return bool(default)


def normalize_email_send_mode(value, default=DEFAULT_EMAIL_SEND_MODE):
    """
    설정 파일의 이메일 발송 방식을 안전하게 변환한다.

    지원:
    - individual: 수신자별 개별 발송
    - bulk: 전체 수신자를 To에 넣어 한 번에 발송
    """
    default = str(default or DEFAULT_EMAIL_SEND_MODE).strip().lower()
    if default not in {"individual", "bulk"}:
        default = DEFAULT_EMAIL_SEND_MODE

    text = str(value or default).strip().lower()
    bulk_values = {"bulk", "all", "group", "combined", "single", "전체", "전체발송"}
    individual_values = {"individual", "separate", "each", "personal", "개별", "개별발송"}

    if text in bulk_values:
        return "bulk"
    if text in individual_values:
        return "individual"

    logger.warning("⚠️ 알 수 없는 email_send_mode=%s 값입니다. 기본값 %s를 사용합니다.", value, default)
    return default


def apply_exclude_keywords(news_list, exclude_keywords, section_name):
    """
    AI 선별 전에 명백히 제외할 키워드를 포함한 뉴스를 제거한다.
    제목/설명/언론사/키워드 필드에서 단순 포함 여부만 본다.
    """
    exclude_keywords = normalize_exclude_keywords(exclude_keywords)
    if not exclude_keywords:
        return news_list, 0

    filtered = []
    excluded_count = 0

    for news in news_list or []:
        haystack = " ".join([
            str(news.get("title") or ""),
            str(news.get("description") or ""),
            str(news.get("source") or ""),
            str(news.get("keyword") or ""),
        ]).lower()

        matched_keyword = None
        for keyword in exclude_keywords:
            if keyword.lower() in haystack:
                matched_keyword = keyword
                break

        if matched_keyword:
            excluded_count += 1
            continue

        filtered.append(news)

    logger.info(
        f"🧹 [{section_name}] 제외 키워드 필터 완료: "
        f"{len(news_list or [])}개 → {len(filtered)}개 / 제외 {excluded_count}개"
    )
    return filtered, excluded_count

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
    issue_history_days=DEFAULT_ISSUE_HISTORY_DAYS,
    issue_history_file_path=issue_history.HISTORY_FILE_PATH,
    unmapped_press_domains_file_path=None,
    exclude_keywords=None,
    grouping_max_groups=None,
    grouping_exclude_low_quality=True,
    selector_candidate_group_limit=DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT,
    summary_max_length=DEFAULT_SUMMARY_MAX_LENGTH,
    email_insight_ai=DEFAULT_EMAIL_INSIGHT_AI
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
        issue_history_file_path: 설정 파일별로 분리된 이슈 히스토리 파일 경로
        unmapped_press_domains_file_path: 설정 파일별로 분리된 미매핑 언론사 도메인 파일 경로
        exclude_keywords: 섹션별 제외 키워드 리스트
        grouping_max_groups: 그룹화 후 OpenAI 선별에 넘길 최대 그룹 수
        grouping_exclude_low_quality: 사진/저품질 그룹을 선별 후보에서 제외할지 여부
        selector_candidate_group_limit: OpenAI 그룹 선별 프롬프트에 실제 포함할 후보 그룹 수
        summary_max_length: 뉴스별 요약 최대 글자 수
        email_insight_ai: 메일 상단 핵심 3줄을 별도 AI 호출로 생성할지 여부

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

    logger.info(
        "📰 [%s] 시작: 키워드 %s개 / 최근 %s시간 / 수집상한 %s개 / "
        "AI후보 %s개 / 선별 %s개",
        section_name,
        len(keywords or []),
        recent_hours,
        max_total_news,
        selector_candidate_group_limit,
        select_limit,
    )

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
        max_total_news=max_total_news,
        unmapped_press_domains_file_path=unmapped_press_domains_file_path
    )

    # naver_news_scraper.py에서 마지막 수집 통계를 가져온다.
    # 이 시점의 news_list는 URL/시간 필터까지만 거친 전체 후보다.
    scrape_stats = naver_news_scraper.get_last_scrape_stats()
    scrape_stats["pre_issue_filter_candidate_count"] = len(news_list)
    scrape_stats["selector_candidate_group_limit"] = int(selector_candidate_group_limit or 0)
    scrape_stats["summary_max_length"] = int(summary_max_length or DEFAULT_SUMMARY_MAX_LENGTH)
    scrape_stats["email_insight_ai"] = bool(email_insight_ai)

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
    issue_filter_result = issue_history.filter_seen_issues_with_llm(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        candidate_news=news_list,
        days=issue_history_days,
        file_path=issue_history_file_path,
        remove_internal_duplicates=False
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
    scrape_stats["issue_filter_internal_duplicate_filter_enabled"] = issue_filter_result.get(
        "internal_duplicate_filter_enabled",
        False,
    )

    issue_token_stats = issue_filter_result.get("token_stats", {}) or {}
    scrape_stats["issue_key_tokens"] = issue_token_stats.get("issue_key_tokens", 0)
    scrape_stats["issue_duplicate_tokens"] = issue_token_stats.get("llm_duplicate_tokens", 0)

    # ====================================
    # 섹션별 제외 키워드 필터
    # ====================================
    before_exclude_keyword_count = len(news_list)
    news_list, exclude_keyword_count = apply_exclude_keywords(
        news_list=news_list,
        exclude_keywords=exclude_keywords,
        section_name=section_name,
    )
    after_exclude_keyword_count = len(news_list)

    scrape_stats["exclude_keyword_before_count"] = before_exclude_keyword_count
    scrape_stats["exclude_keyword_after_count"] = after_exclude_keyword_count
    scrape_stats["exclude_keyword_excluded_count"] = exclude_keyword_count

    if not news_list:
        logger.error(f"❌ [{section_name}] 제외 키워드 필터 후 남은 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": 0,
            "selected_count": 0,
            "scrape_stats": scrape_stats
        }

    # ====================================
    # Python 규칙 기반 사건 그룹화
    # ====================================
    grouping_max_groups = int(grouping_max_groups or max_total_news)

    grouping_result = news_grouper.build_grouping_result(
        news_list=news_list,
        max_groups=grouping_max_groups,
        exclude_low_quality=bool(grouping_exclude_low_quality),
    )

    grouped_representative_news = grouping_result.get("representative_news", [])

    grouping_news_count = int(grouping_result.get("news_count", len(news_list)) or 0)
    grouping_group_count = int(grouping_result.get("group_count", 0) or 0)
    grouping_low_quality_group_count = int(grouping_result.get("low_quality_group_count", 0) or 0)
    grouping_low_quality_article_count = int(grouping_result.get("low_quality_article_count", 0) or 0)
    grouping_duplicate_article_count = int(grouping_result.get("duplicate_article_count", 0) or 0)
    grouping_selection_group_count = int(grouping_result.get("selection_group_count", 0) or 0)
    grouping_selection_article_count = int(grouping_result.get("selection_article_count", 0) or 0)

    scrape_stats["grouping_news_count"] = grouping_news_count
    scrape_stats["grouping_group_count"] = grouping_group_count
    scrape_stats["grouping_multi_article_group_count"] = grouping_result.get("multi_article_group_count", 0)
    scrape_stats["grouping_low_quality_group_count"] = grouping_low_quality_group_count
    scrape_stats["grouping_low_quality_article_count"] = grouping_low_quality_article_count
    scrape_stats["grouping_duplicate_article_count"] = grouping_duplicate_article_count
    scrape_stats["grouping_selection_group_count"] = grouping_selection_group_count
    scrape_stats["grouping_selection_article_count"] = grouping_selection_article_count
    scrape_stats["grouping_exclude_low_quality"] = bool(grouping_exclude_low_quality)

    # 메일 대시보드용 집계
    # - 전체검색수: total_seen_count
    # - 24시간초과제외: old_news_count
    # - 반복이슈제외(3일): issue_filter_excluded_count
    # - 규칙기반제외: URL 중복, 제외 키워드, 그룹화 중복 대표화, 저품질/사진성 그룹 제외
    #   반복이슈제외는 별도 표시하므로 규칙기반제외에 중복 집계하지 않는다.
    rule_based_excluded_count = (
        int(scrape_stats.get("duplicate_count", 0) or 0)
        + int(scrape_stats.get("exclude_keyword_excluded_count", 0) or 0)
        + grouping_duplicate_article_count
        + grouping_low_quality_article_count
    )
    scrape_stats["grouping_duplicate_excluded_count"] = grouping_duplicate_article_count
    scrape_stats["rule_based_excluded_count"] = rule_based_excluded_count
    # 기존 요약/로그 호환용 키. 이제 반복이슈제외는 포함하지 않는다.
    scrape_stats["code_rule_excluded_count"] = rule_based_excluded_count
    scrape_stats["ai_duplicate_excluded_count"] = 0

    logger.info(
        f"🧩 [{section_name}] 그룹화 결과: "
        f"기사 {grouping_result.get('news_count', len(news_list))}개 → "
        f"그룹 {grouping_result.get('group_count', 0)}개 / "
        f"2건 이상 그룹 {grouping_result.get('multi_article_group_count', 0)}개 / "
        f"중복 대표화 제외 {grouping_result.get('duplicate_article_count', 0)}개 / "
        f"저품질 제외 {grouping_result.get('low_quality_group_count', 0)}그룹·{grouping_result.get('low_quality_article_count', 0)}기사 / "
        f"AI 선별 후보 그룹 {grouping_result.get('selection_group_count', 0)}개"
    )

    if not grouped_representative_news:
        logger.warning(f"⚠️ [{section_name}] 그룹화 후 AI 후보가 없어 원본 후보에서 시간대 샘플링 fallback을 사용합니다.")
        before_sampling_after_filter_count = len(news_list)
        grouped_representative_news = naver_news_scraper.sample_news_by_time_bucket(
            news_list=news_list,
            bucket_hours=4,
            max_total_news=max_total_news,
            min_per_bucket=0,
            recent_hours=recent_hours,
        )
        after_sampling_count = len(grouped_representative_news)
    else:
        before_sampling_after_filter_count = len(news_list)
        after_sampling_count = len(grouped_representative_news)

    naver_news_scraper.update_post_issue_filter_sampling_stats(
        before_issue_filter_count=before_issue_filter_count,
        after_issue_filter_count=after_issue_filter_count,
        before_sampling_count=before_sampling_after_filter_count,
        after_sampling_count=after_sampling_count,
    )

    updated_scrape_stats = naver_news_scraper.get_last_scrape_stats()
    updated_scrape_stats.update(scrape_stats)
    scrape_stats = updated_scrape_stats

    logger.info(
        f"🧺 [{section_name}] AI 선별 후보 준비: "
        f"반복이슈/제외어 필터 후 {before_sampling_after_filter_count}개 → "
        f"그룹 대표 후보 {after_sampling_count}개 "
        f"(max_total_news={max_total_news})"
    )

    if not grouped_representative_news:
        logger.error(f"❌ [{section_name}] 그룹화/샘플링 후 남은 뉴스가 없습니다.")
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
    selected_news = news_selector.select_important_news_groups(
        group_list=grouping_result.get("selection_groups", []),
        fallback_news_list=grouped_representative_news,
        topic_name=section_name,
        topic_description=topic_description,
        limit=select_limit,
        candidate_group_limit=selector_candidate_group_limit
    )

    selection_stats = news_selector.get_last_selection_stats()
    scrape_stats["selection_tokens"] = selection_stats.get("selection_tokens", 0)
    scrape_stats["event_group_tokens"] = selection_stats.get("event_group_tokens", 0)
    scrape_stats["ai_selected_before_final_dedup_count"] = selection_stats.get("selected_before_final_dedup_count", len(selected_news or []))
    scrape_stats["ai_duplicate_excluded_count"] = selection_stats.get("final_duplicate_excluded_count", 0)
    scrape_stats["ai_selected_after_final_dedup_count"] = selection_stats.get("selected_after_final_dedup_count", len(selected_news or []))

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
    for news in selected_news:
        news["content"] = news.get("description", "")

    summaries = summarizer.summarize_batch(
        selected_news,
        max_length=summary_max_length
    )
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
    reset_openai_usage_totals()

    config_path = args.config
    config = load_config(config_path)
    state_paths = get_state_file_paths(config_path)
    issue_history_file_path = state_paths["issue_history_file_path"]
    unmapped_press_domains_file_path = state_paths["unmapped_press_domains_file_path"]

    briefing_name = config["briefing_name"]
    subject_prefix = config["subject_prefix"]
    sections = config["sections"]
    receiver_env = config.get("receiver_env", "EMAIL_RECEIVER")

    display_per_keyword = int(config.get("display_per_keyword", DEFAULT_DISPLAY_PER_KEYWORD))
    pages_per_keyword = int(config.get("pages_per_keyword", DEFAULT_PAGES_PER_KEYWORD))
    sorts = normalize_sorts(config.get("sorts", DEFAULT_SORTS))
    max_total_news = int(config.get("max_total_news", DEFAULT_MAX_TOTAL_NEWS))
    select_limit = int(config.get("select_limit", DEFAULT_SELECT_LIMIT))
    selector_candidate_group_limit = int(
        config.get("selector_candidate_group_limit", DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT)
    )
    summary_max_length = int(config.get("summary_max_length", DEFAULT_SUMMARY_MAX_LENGTH))
    email_insight_ai = normalize_bool(
        config.get("email_insight_ai", DEFAULT_EMAIL_INSIGHT_AI),
        DEFAULT_EMAIL_INSIGHT_AI
    )
    email_send_mode = normalize_email_send_mode(
        config.get("email_send_mode", DEFAULT_EMAIL_SEND_MODE),
        DEFAULT_EMAIL_SEND_MODE
    )
    save_issue_history = normalize_bool(
        config.get("save_issue_history", DEFAULT_SAVE_ISSUE_HISTORY),
        DEFAULT_SAVE_ISSUE_HISTORY
    )
    if args.no_save_history:
        save_issue_history = False

    recent_hours = int(config.get("recent_hours", DEFAULT_RECENT_HOURS))
    issue_history_days = int(config.get("issue_history_days", DEFAULT_ISSUE_HISTORY_DAYS))

    config_exclude_keywords = normalize_exclude_keywords(config.get("exclude_keywords", []))
    logger.info(
        "🚀 뉴스 스크래퍼 시작: %s / 섹션 %s개 / selector=%s / summary=%s / config=%s",
        briefing_name,
        len(sections),
        getattr(news_selector, "SELECTOR_MODEL", "확인불가"),
        getattr(summarizer, "MODEL", "확인불가"),
        config_path,
    )
    logger.info(
        "⚙️ 기본값: recent=%sh / issue_history=%s일 / max_total=%s / select=%s / ai_groups=%s / send_mode=%s / save_history=%s",
        recent_hours,
        issue_history_days,
        max_total_news,
        select_limit,
        selector_candidate_group_limit,
        email_send_mode,
        save_issue_history,
    )

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
        section_selector_candidate_group_limit = int(
            section.get("selector_candidate_group_limit", selector_candidate_group_limit)
        )
        section_summary_max_length = int(
            section.get("summary_max_length", summary_max_length)
        )
        section_email_insight_ai = normalize_bool(
            section.get("email_insight_ai", email_insight_ai),
            email_insight_ai
        )
        section_recent_hours = int(
            section.get("recent_hours", recent_hours)
        )
        section_issue_history_days = int(
            section.get("issue_history_days", issue_history_days)
        )
        section_exclude_keywords = normalize_exclude_keywords(
            section.get("exclude_keywords", config.get("exclude_keywords", []))
        )
        section_grouping_max_groups = int(
            section.get("grouping_max_groups", config.get("grouping_max_groups", section_max_total_news))
        )
        section_grouping_exclude_low_quality = normalize_bool(
            section.get("grouping_exclude_low_quality", config.get("grouping_exclude_low_quality", True)),
            True
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
            issue_history_days=section_issue_history_days,
            issue_history_file_path=issue_history_file_path,
            unmapped_press_domains_file_path=unmapped_press_domains_file_path,
            exclude_keywords=section_exclude_keywords,
            grouping_max_groups=section_grouping_max_groups,
            grouping_exclude_low_quality=section_grouping_exclude_low_quality,
            selector_candidate_group_limit=section_selector_candidate_group_limit,
            summary_max_length=section_summary_max_length,
            email_insight_ai=section_email_insight_ai
        )

        section_results.append(section_result)

    # ====================================
    # 메일 발송 직전 전체 섹션 최종 중복 제거
    # - 이미 요약된 결과만 비교하므로 OpenAI 호출/토큰 사용량은 늘지 않는다.
    # - 섹션이 달라져도 같은 사건이면 하나만 남긴다.
    # ====================================
    final_dedup_result = issue_history.deduplicate_section_results(section_results)
    section_results = final_dedup_result.get("section_results", section_results)
    if final_dedup_result.get("excluded_count", 0):
        logger.info(
            "🧹 메일 최종 중복 제거: %s개 → %s개 / 제외 %s개",
            final_dedup_result.get("before_count", 0),
            final_dedup_result.get("after_count", 0),
            final_dedup_result.get("excluded_count", 0),
        )

    # ====================================
    # 이메일 발송
    # ====================================
    logger.debug("📧 이메일 발송 시작: receiver_env=%s", receiver_env)

    result = email_sender.send_email(
        briefing_name=briefing_name,
        subject_prefix=subject_prefix,
        section_results=section_results,
        receiver_env_name=receiver_env,
        send_mode=email_send_mode
    )

    if result["success"]:
        logger.info(f"✅ {result['message']}")

        if save_issue_history:
            history_result = issue_history.append_sent_issues(
                briefing_name=briefing_name,
                subject_prefix=subject_prefix,
                receiver_env=receiver_env,
                section_results=section_results,
                file_path=issue_history_file_path,
                keep_days=issue_history_days
            )

            logger.info(
                f"🗂️ 이슈 히스토리 저장 완료: "
                f"신규 {history_result['saved_count']}개 / "
                f"오래된 이슈 삭제 {history_result.get('pruned_count', 0)}개 / "
                f"누적 {history_result['total_count']}개"
            )
        else:
            logger.info("🧪 테스트 실행 옵션으로 이슈 히스토리 저장을 건너뜁니다: %s", issue_history_file_path)

    else:
        logger.error(f"❌ {result['message']}")

    # ====================================
    # 미매핑 언론사 도메인 저장
    # ====================================
    if hasattr(naver_news_scraper, "save_unmapped_press_domains"):
        try:
            naver_news_scraper.save_unmapped_press_domains(
                filename=unmapped_press_domains_file_path
            )
            logger.debug(f"🗂️ 미매핑 언론사 도메인 저장 완료: {unmapped_press_domains_file_path}")
        except Exception as e:
            logger.warning(f"⚠️ 미매핑 언론사 도메인 저장 실패: {e}")
    else:
        logger.debug("미매핑 언론사 도메인 저장 함수 없음: 건너뜀")

    # ====================================
    # 최종 결과
    # ====================================
    logger.info("📊 작업 완료 요약 생성")

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

        total_seen_count = scrape_stats.get("total_seen_count", issue_filter_before_count)
        exclude_keyword_excluded_count = scrape_stats.get("exclude_keyword_excluded_count", 0)
        grouping_duplicate_article_count = scrape_stats.get("grouping_duplicate_excluded_count", 0)
        grouping_low_quality_article_count = scrape_stats.get("grouping_low_quality_article_count", 0)
        logger.info(
            "📊 [%s] 처리: 검색 %s개 → 후보 %s개 → 반복필터 %s개 → "
            "AI후보 %s개 → 선별 %s개 → 요약 %s개 / 토큰 %s",
            section_name,
            total_seen_count,
            pre_sampling_count,
            issue_filter_after_count,
            final_candidate_count,
            selected_count,
            summary_count,
            f"{section_total_tokens:,}",
        )
        logger.info(
            "🧹 [%s] 제외: 시간초과 %s / URL중복 %s / 반복 %s(과거 %s건 비교) / "
            "제외어 %s / 그룹중복 %s / 저품질 %s / AI중복 %s",
            section_name,
            old_news_count,
            duplicate_count,
            issue_filter_excluded_count,
            issue_filter_past_issue_count,
            exclude_keyword_excluded_count,
            grouping_duplicate_article_count,
            grouping_low_quality_article_count,
            scrape_stats.get("ai_duplicate_excluded_count", 0),
        )

    logger.info(
        "📊 전체 처리: 후보 %s개 / 선별 %s개 / 요약 %s개 / 발송 %s",
        total_raw_count,
        total_selected_count,
        total_summary_count,
        result["success"],
    )
    logger.info(
        "🧾 전체 AI 토큰: 선별 %s / 사건그룹 %s / 요약 %s / 메일3줄 %s / 총 %s",
        f"{total_selection_tokens:,}",
        f"{total_event_group_tokens:,}",
        f"{total_summary_tokens:,}",
        f"{total_insight_tokens:,}",
        f"{total_tokens:,}",
    )
    log_openai_usage_summary(logger)
    logger.info("✅ 모든 작업 완료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ 사용자가 작업을 중단했습니다.")
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
