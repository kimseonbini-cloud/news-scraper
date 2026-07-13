# =============================================================================
# [파일 설명]
# - 수행 기능: 설정 파일을 기준으로 뉴스 수집, 중복 이슈 제거, 그룹화, AI 선별, 요약, 관련보도 페이지 생성, 이메일 발송, 히스토리 저장을 총괄합니다.
# - 프로세스: CLI 인자/설정 로드 -> 섹션별 수집 및 선별/요약 -> 전체 중복 제거 -> 관련보도 페이지 연결 -> 이메일 발송 -> 이슈 히스토리/사용량 기록
# - 호출하는 곳: 직접 실행 또는 수동 import
# - 주요 파라미터/입력: 명령행 --config, configs/*.json, 환경변수, 네이버/OpenAI/SMTP 설정
# - 리턴값/출력: main()은 정수 종료 코드 대신 로그와 파일/메일 발송 부수 효과를 남깁니다.
# =============================================================================

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
import related_pages
from openai_usage import reset_openai_usage_totals, log_openai_usage_summary


# ====================================
# 기본 설정
# ====================================
DEFAULT_CONFIG_PATH = "configs/company_briefing.json"     # 기본설정파일경로

# 네이버 뉴스 API는 1회 호출 display 최대 100개.
# 기본값: 키워드당 관련도순(sim) 100개 × 3페이지 = 최대 300개 조회.
# 최신성 비중을 높이고 싶으면 설정 파일에서 sorts를 ["date", "sim"] 또는 ["date"]로 지정한다.
DEFAULT_DISPLAY_PER_KEYWORD = 100                         # 키워드별페이지당수집수
DEFAULT_PAGES_PER_KEYWORD = 3                             # 키워드별조회페이지수
DEFAULT_SORTS = ["sim"]                                   # 기본검색정렬방식

# 최근 몇 시간 이내 뉴스만 사용할지
DEFAULT_RECENT_HOURS = 24                                 # 기본최근뉴스시간범위

# 시간대별 샘플링 후 AI 선별 단계로 넘길 최종 후보 최대 개수
DEFAULT_MAX_TOTAL_NEWS = 100                              # 기본AI전달후보상한

# OpenAI 최종 선별 개수
DEFAULT_SELECT_LIMIT = 10                                 # 기본최종선별개수

# OpenAI 그룹 선별에 실제로 전달할 후보 그룹 수
# 로컬 그룹화는 넓게 하되, AI 입력은 설정별로 압축해 토큰을 줄인다.
# 45→35로 줄인 대신 모든 후보에 짧은 설명을 포함해 프롬프트 토큰은 기존 수준을 유지한다.
DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT = 35               # 기본AI후보그룹상한

# AI가 부여한 중요도가 이 값 미만인 그룹은 메일에 싣지 않는다.
DEFAULT_SELECTOR_MIN_IMPORTANCE = 3                       # 기본최소중요도점수

# AI가 select_limit보다 적게 골랐을 때 미선택 후보로 강제 보충할지 여부.
# 기본은 끄기: 관련성 낮은 뉴스가 채워지는 것보다 적게 보내는 쪽을 택한다.
DEFAULT_SELECTOR_FILL_TO_LIMIT = False                    # 기본부족분강제보충여부

# 요약 최대 길이. 설정 파일/섹션별로 조절 가능하다.
DEFAULT_SUMMARY_MAX_LENGTH = 360                          # 기본뉴스요약최대길이

# 메일 상단 3줄 흐름 요약을 별도 AI 호출로 만들지 여부.
# 기본은 꺼서 섹션당 1회씩 추가되는 AI 호출을 줄인다.
DEFAULT_EMAIL_INSIGHT_AI = False                          # 기본메일상단AI요약사용여부

# 이메일 발송 방식: individual=수신자별 개별 발송, bulk=수신자 전체에게 1회 발송
DEFAULT_EMAIL_SEND_MODE = "individual"                    # 기본메일발송방식

# 최근 며칠간 이미 다룬 이슈를 비교할지
DEFAULT_ISSUE_HISTORY_DAYS = 3                            # 기본반복이슈비교일수

# 발송 성공 후 seen_issues.json에 저장할지 여부
DEFAULT_SAVE_ISSUE_HISTORY = True                         # 기본이슈히스토리저장여부

# 관련보도 상세 페이지 생성 설정
DEFAULT_RELATED_PAGE_ENABLED = True                       # 기본관련보도페이지생성여부
DEFAULT_RELATED_PAGE_KEEP_DAYS = 7                        # 기본관련보도페이지보존일수


# ====================================
# 로깅 설정
# - GitHub Actions 콘솔 로그만 남김
# - logs/scraper.log 파일 저장 안 함
# ====================================
logging.basicConfig(
    level=logging.INFO,  # level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[  # handlers
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)  # 모듈로거
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# [코드 이해 주석]
# - 역할: 실행 인자 파싱.
# - 호출하는 곳: main.main, main.parse_args
# - 파라미터: 없음
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 문자열/설정을 읽습니다 -> 가능한 형식으로 변환을 시도합니다 -> 실패 시 안전한 기본값을 반환합니다.
def parse_args():
    """
    실행 인자 파싱
    """
    parser = argparse.ArgumentParser(  # 실행인자파서
        description="뉴스 브리핑 스크래퍼"
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,  # 기본값
        help=f"브리핑 설정 JSON 파일 경로. 기본값: {DEFAULT_CONFIG_PATH}"
    )

    parser.add_argument(
        "--no-save-history",
        action="store_true",
        help="테스트 실행용. 이메일 발송 성공 후 seen_issues.json 저장을 건너뜁니다."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "로컬 테스트 실행용. seen_issues.json 저장, 관련보도 HTML 파일(docs/briefings) 기록, "
            "unmapped_press_domains.json 저장을 모두 건너뜁니다. "
            "메일은 관련보도 링크 포함 운영과 동일하게 발송됩니다(링크 클릭 시 404는 정상)."
        )
    )

    return parser.parse_args()


# [코드 이해 주석]
# - 역할: 브리핑 설정 파일 로드.
# - 호출하는 곳: main.main
# - 파라미터: config_path: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 파일 경로를 확인합니다 -> JSON/환경 값을 읽습니다 -> 없거나 깨진 값은 기본 구조로 보정합니다.
def load_config(config_path):
    """
    브리핑 설정 파일 로드
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:  # 파일객체
        config = json.load(f)  # 브리핑설정데이터

    validate_config(config, config_path)

    return config


# [코드 이해 주석]
# - 역할: 설정 파일 필수값 검증.
# - 호출하는 곳: main.load_config
# - 파라미터: config: Any, config_path: Any
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def validate_config(config, config_path):
    """
    설정 파일 필수값 검증
    """
    required_top_keys = ["briefing_name", "subject_prefix", "receiver_env", "sections"]  # 필수상위설정키

    for key in required_top_keys:  # 키
        if key not in config:
            raise ValueError(f"{config_path} 설정 파일에 '{key}' 값이 없습니다.")

    if not isinstance(config["sections"], list) or not config["sections"]:
        raise ValueError(f"{config_path} 설정 파일의 'sections'는 비어 있지 않은 리스트여야 합니다.")

    for idx, section in enumerate(config["sections"], 1):  # 섹션순번,섹션설정
        for key in ["section_name", "keywords", "topic_description"]:  # 키
            if key not in section:
                raise ValueError(f"{config_path} sections[{idx}]에 '{key}' 값이 없습니다.")

        if not isinstance(section["keywords"], list) or not section["keywords"]:
            raise ValueError(f"{config_path} sections[{idx}].keywords는 비어 있지 않은 리스트여야 합니다.")


# [코드 이해 주석]
# - 역할: 파일명에 안전한 문자열 생성.
# - 호출하는 곳: main.get_config_slug
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_safe_filename(value):
    """
    파일명에 안전한 문자열 생성
    한글은 유지하고, 특수문자/공백은 _로 치환
    """
    value = str(value).strip()  # 값
    value = re.sub(r"[^\w가-힣]+", "_", value)  # 값
    value = value.strip("_")  # 값

    if not value:
        return "section"

    return value


# [코드 이해 주석]
# - 역할: 설정 파일 경로에서 상태 파일 구분용 slug를 생성한다.
# - 호출하는 곳: main.get_state_file_paths
# - 파라미터: config_path: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_config_slug(config_path):
    """
    설정 파일 경로에서 상태 파일 구분용 slug를 생성한다.

    예:
    - configs/company_briefing.json -> company_briefing
    - configs/economy_briefing.json -> economy_briefing
    - configs/inohas_briefing.json -> inohas_briefing

    config 파일이 추가되어도 파일명 기준으로 자동 분리된다.
    """
    config_base_name = os.path.splitext(os.path.basename(config_path))[0]  # 설정파일명
    return make_safe_filename(config_base_name)


# [코드 이해 주석]
# - 역할: 설정 파일 기준으로 상태 파일 경로를 동적으로 생성한다.
# - 호출하는 곳: main.main
# - 파라미터: config_path: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_state_file_paths(config_path):
    """
    설정 파일 기준으로 상태 파일 경로를 동적으로 생성한다.

    같은 저장소에서 여러 브리핑 workflow가 실행되더라도
    브리핑별 상태 파일을 따로 사용해 Git 충돌 가능성을 낮춘다.
    """
    config_slug = get_config_slug(config_path)              # 설정구분슬러그
    state_dir = os.path.join("data", "state", config_slug)  # 상태파일디렉터리
    os.makedirs(state_dir, exist_ok=True)  # existok

    return {
        "config_slug": config_slug,                                                                    # 설정구분슬러그
        "state_dir": state_dir,                                                                        # 상태파일디렉터리
        "issue_history_file_path": os.path.join(state_dir, "seen_issues.json"),                        # 이슈히스토리파일경로
        "unmapped_press_domains_file_path": os.path.join(state_dir, "unmapped_press_domains.json"),    # 미매핑언론사파일경로
    }


# [코드 이해 주석]
# - 역할: 설정 파일의 sorts 값을 안전하게 리스트로 변환한다.
# - 호출하는 곳: main.collect_select_and_summarize, main.main
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
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
        cleaned = [str(item).strip() for item in value if str(item).strip()]  # 정리된정렬목록
        return cleaned or DEFAULT_SORTS

    if isinstance(value, str):
        value = value.strip()  # 값
        if not value:
            return DEFAULT_SORTS
        return [value]

    return DEFAULT_SORTS




# [코드 이해 주석]
# - 역할: 설정 파일의 exclude_keywords 값을 안전하게 리스트로 변환한다.
# - 호출하는 곳: main.apply_exclude_keywords, main.main
# - 파라미터: value: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
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


# [코드 이해 주석]
# - 역할: 설정 파일의 bool 값을 안전하게 변환한다.
# - 호출하는 곳: main.main
# - 파라미터: value: Any, default: Any = False
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
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

    text = str(value).strip().lower()  # 텍스트
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    return bool(default)


# [코드 이해 주석]
# - 역할: 설정 파일의 이메일 발송 방식을 안전하게 변환한다.
# - 호출하는 곳: main.main
# - 파라미터: value: Any, default: Any = DEFAULT_EMAIL_SEND_MODE
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_email_send_mode(value, default=DEFAULT_EMAIL_SEND_MODE):
    """
    설정 파일의 이메일 발송 방식을 안전하게 변환한다.

    지원:
    - individual: 수신자별 개별 발송
    - bulk: 전체 수신자를 To에 넣어 한 번에 발송
    """
    default = str(default or DEFAULT_EMAIL_SEND_MODE).strip().lower()  # 기본값
    if default not in {"individual", "bulk"}:
        default = DEFAULT_EMAIL_SEND_MODE  # 기본값

    text = str(value or default).strip().lower()  # 텍스트
    bulk_values = {"bulk", "all", "group", "combined", "single", "전체", "전체발송"}               # 전체발송허용값
    individual_values = {"individual", "separate", "each", "personal", "개별", "개별발송"}          # 개별발송허용값

    if text in bulk_values:
        return "bulk"
    if text in individual_values:
        return "individual"

    logger.warning("⚠️ 알 수 없는 email_send_mode=%s 값입니다. 기본값 %s를 사용합니다.", value, default)
    return default


# [코드 이해 주석]
# - 역할: AI 선별 전에 명백히 제외할 키워드를 포함한 뉴스를 제거한다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: news_list: Any, exclude_keywords: Any, section_name: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def apply_exclude_keywords(news_list, exclude_keywords, section_name):
    """
    AI 선별 전에 명백히 제외할 키워드를 포함한 뉴스를 제거한다.
    제목/설명/언론사/키워드 필드에서 단순 포함 여부만 본다.
    """
    exclude_keywords = normalize_exclude_keywords(exclude_keywords)  # 제외키워드목록
    if not exclude_keywords:
        return news_list, 0

    filtered = []        # 제외키워드통과뉴스
    excluded_count = 0   # 제외키워드제외건수

    for news in news_list or []:  # 후보뉴스
        haystack = " ".join([  # haystack
            str(news.get("title") or ""),
            str(news.get("description") or ""),
            str(news.get("source") or ""),
            str(news.get("keyword") or ""),
        ]).lower()  # 제외키워드검색문자열

        matched_keyword = None         # 매칭된제외키워드
        for keyword in exclude_keywords:  # 제외키워드
            if keyword.lower() in haystack:
                matched_keyword = keyword  # 매칭된제외키워드
                break

        if matched_keyword:
            excluded_count += 1  # 처리값
            continue

        filtered.append(news)

    logger.info(
        f"🧹 [{section_name}] 제외 키워드 필터 완료: "
        f"{len(news_list or [])}개 → {len(filtered)}개 / 제외 {excluded_count}개"
    )
    return filtered, excluded_count

# [코드 이해 주석]
# - 역할: 섹션별 뉴스 수집 → 최근 반복 이슈 제거 → OpenAI 선별 → 요약 처리.
# - 호출하는 곳: main.main
# - 파라미터: briefing_name: Any, receiver_env: Any, section_name: Any, keywords: Any, topic_description: Any,
# display_per_keyword: Any = DEFAULT_DISPLAY_PER_KEYWORD, pages_per_keyword: Any = DEFAULT_PAGES_PER_KEYWORD, sorts:
# Any = None, max_total_news: Any = DEFAULT_MAX_TOTAL_NEWS, select_limit: Any = DEFAULT_SELECT_LIMIT, recent_hours:
# Any = DEFAULT_RECENT_HOURS, issue_history_days: Any = DEFAULT_ISSUE_HISTORY_DAYS, issue_history_file_path: Any =
# issue_history.HISTORY_FILE_PATH, unmapped_press_domains_file_path: Any = None, exclude_keywords: Any = None,
# grouping_max_groups: Any = None, grouping_exclude_low_quality: Any = True, selector_candidate_group_limit: Any =
# DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT, summary_max_length: Any = DEFAULT_SUMMARY_MAX_LENGTH, email_insight_ai: Any
# = DEFAULT_EMAIL_INSIGHT_AI
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 목록을 순회합니다 -> 조건에 맞는 항목을 모읍니다 -> 후속 단계가 사용할 목록/통계를 반환합니다.
def collect_select_and_summarize(
    briefing_name,
    receiver_env,
    section_name,
    keywords,
    topic_description,
    display_per_keyword=DEFAULT_DISPLAY_PER_KEYWORD,  # 표시건수기사별키워드
    pages_per_keyword=DEFAULT_PAGES_PER_KEYWORD,  # 페이지목록기사별키워드
    sorts=None,  # 정렬목록
    max_total_news=DEFAULT_MAX_TOTAL_NEWS,  # 최대전체뉴스
    select_limit=DEFAULT_SELECT_LIMIT,  # select상한
    recent_hours=DEFAULT_RECENT_HOURS,  # recent시간수
    issue_history_days=DEFAULT_ISSUE_HISTORY_DAYS,  # 이슈히스토리일수
    issue_history_file_path=issue_history.HISTORY_FILE_PATH,  # 이슈히스토리파일경로
    unmapped_press_domains_file_path=None,  # unmapped언론사domains파일경로
    exclude_keywords=None,  # 제외키워드목록
    grouping_max_groups=None,  # grouping최대그룹목록
    grouping_exclude_low_quality=True,  # grouping제외lowquality
    selector_candidate_group_limit=DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT,  # selector후보그룹상한
    selector_min_importance=DEFAULT_SELECTOR_MIN_IMPORTANCE,  # selector최소중요도
    selector_fill_to_limit=DEFAULT_SELECTOR_FILL_TO_LIMIT,  # selector부족분강제보충여부
    summary_max_length=DEFAULT_SUMMARY_MAX_LENGTH,  # 요약최대length
    email_insight_ai=DEFAULT_EMAIL_INSIGHT_AI  # 이메일주소insightAI
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
        selector_min_importance: AI 중요도가 이 값 미만인 그룹은 메일에서 제외
        selector_fill_to_limit: AI가 적게 골랐을 때 미선택 후보로 강제 보충할지 여부
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
    sorts = normalize_sorts(sorts)  # 정규화된검색정렬방식

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
    news_list = naver_news_scraper.search_multiple_keywords(  # 수집후보뉴스목록
        keywords=keywords,  # 키워드목록
        display_per_keyword=display_per_keyword,  # 표시건수기사별키워드
        recent_hours=recent_hours,  # recent시간수
        sorts=sorts,  # 정렬목록
        pages_per_keyword=pages_per_keyword,  # 페이지목록기사별키워드
        enable_time_bucket_sampling=False,  # enable시간bucketsampling
        max_total_news=max_total_news,  # 최대전체뉴스
        unmapped_press_domains_file_path=unmapped_press_domains_file_path  # unmapped언론사domains파일경로
    )

    # naver_news_scraper.py에서 마지막 수집 통계를 가져온다.
    # 이 시점의 news_list는 URL/시간 필터까지만 거친 전체 후보다.
    scrape_stats = naver_news_scraper.get_last_scrape_stats()  # 수집통계데이터

    # 수집 단계 이후부터는 여러 모듈의 처리 통계를 scrape_stats 한 곳에 누적한다.
    # 1) collect_select_and_summarize()의 지역변수는 함수가 끝나면 main()/email_sender.py에서 직접 볼 수 없다.
    # 2) 그래서 섹션별 처리 결과를 반환할 때 scrape_stats에 "수집→필터→그룹화→선별→요약" 숫자를 함께 실어 보낸다.
    # 3) main() 마지막 실행 로그와 email_sender.py의 메일 운영 대시보드는 이 dict를 읽어 단계별 제외 건수를 표시한다.
    scrape_stats["pre_issue_filter_candidate_count"] = len(news_list)                                # 반복필터전수집후보수
    scrape_stats["selector_candidate_group_limit"] = int(selector_candidate_group_limit or 0)         # AI선별프롬프트그룹상한
    scrape_stats["summary_max_length"] = int(summary_max_length or DEFAULT_SUMMARY_MAX_LENGTH)        # 요약최대길이설정값
    scrape_stats["email_insight_ai"] = bool(email_insight_ai)                                         # 메일상단AI요약설정값

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

    # 2) 최근 발송 히스토리와 비교해 이미 메일에 나간 사건을 제거한다.
    #    입력 상태: news_list는 네이버 API에서 온 "이번 실행 후보 전체"다.
    #    출력 상태: filtered_news는 같은 수신자/브리핑 범위에서 최근 N일간 다루지 않은 새 후보만 남는다.
    #    이 단계를 AI 선별보다 먼저 두는 이유는, 이미 보낸 사건에 토큰과 선별 슬롯을 쓰지 않기 위해서다.
    issue_filter_result = issue_history.filter_seen_issues_with_llm(  # 이슈필터결과
        briefing_name=briefing_name,  # briefing이름
        receiver_env=receiver_env,  # 수신자env
        section_name=section_name,  # 섹션이름
        candidate_news=news_list,  # 후보뉴스
        days=issue_history_days,  # 일수
        file_path=issue_history_file_path,  # 파일경로
        remove_internal_duplicates=False  # removeinternal중복목록
    )

    before_issue_filter_count = len(news_list)                                  # 반복이슈필터전후보수
    news_list = issue_filter_result.get("filtered_news", news_list)             # 반복이슈필터통과뉴스
    after_issue_filter_count = len(news_list)                                   # 반복이슈필터후후보수

    logger.info(
        f"🧹 [{section_name}] 반복 이슈 제거: "
        f"후보 {before_issue_filter_count}개 중 "
        f"{issue_filter_result.get('excluded_count', 0)}개 제외, "
        f"{after_issue_filter_count}개 유지 "
        f"(최근 {issue_history_days}일 이슈 {issue_filter_result.get('past_issue_count', 0)}개 비교)"
    )

    # 반복 이슈 필터의 판단 결과를 scrape_stats에 저장한다.
    # 1) news_list만 교체하면 후보가 줄어든 이유가 사라지므로, 전/후/제외 사유를 별도 key로 남긴다.
    # 2) 메일 대시보드에서는 issue_filter_excluded_count로 "반복이슈 제외" 수를 보여준다.
    # 3) 실패 메시지와 세부 제외 방식도 남겨두면 Actions 로그에서 어떤 필터가 작동했는지 추적할 수 있다.
    scrape_stats["issue_filter_before_count"] = before_issue_filter_count                                           # 반복이슈필터전후보수
    scrape_stats["issue_filter_after_count"] = after_issue_filter_count                                             # 반복이슈필터후후보수
    scrape_stats["issue_filter_excluded_count"] = issue_filter_result.get("excluded_count", 0)                      # 반복이슈제외건수
    scrape_stats["issue_filter_past_issue_count"] = issue_filter_result.get("past_issue_count", 0)                  # 비교대상과거이슈수
    scrape_stats["issue_filter_days"] = issue_history_days                                                          # 반복이슈비교기간일수
    scrape_stats["issue_filter_success"] = issue_filter_result.get("success", False)                                # 반복이슈필터성공여부
    scrape_stats["issue_filter_message"] = issue_filter_result.get("message", "")                                   # 반복이슈필터상태메시지
    scrape_stats["issue_filter_url_excluded_count"] = issue_filter_result.get("url_excluded_count", 0)              # URL기준반복제외수
    scrape_stats["issue_filter_title_excluded_count"] = issue_filter_result.get("title_excluded_count", 0)          # 제목기준반복제외수
    # issue_history.py는 이제 LLM/issue_key를 쓰지 않고 규칙 기반 fingerprint 비교만 수행한다.
    scrape_stats["issue_filter_core_key_excluded_count"] = issue_filter_result.get("core_key_excluded_count", 0)    # 핵심키기준반복제외수
    scrape_stats["issue_filter_llm_excluded_count"] = issue_filter_result.get("llm_excluded_count", 0)              # LLM판정반복제외수
    scrape_stats["issue_filter_internal_duplicate_count"] = issue_filter_result.get("internal_duplicate_count", 0)  # 후보내부중복제외수
    scrape_stats["issue_filter_text_excluded_count"] = issue_filter_result.get("text_excluded_count", 0)            # 본문유사도기준제외수
    scrape_stats["issue_filter_token_overlap_excluded_count"] = issue_filter_result.get("token_overlap_excluded_count", 0)  # 토큰겹침기준제외수
    scrape_stats["issue_filter_simhash_excluded_count"] = issue_filter_result.get("simhash_excluded_count", 0)      # SimHash기준제외수
    scrape_stats["issue_filter_internal_duplicate_filter_enabled"] = issue_filter_result.get(                       # 후보내부중복필터사용여부
        "internal_duplicate_filter_enabled",
        False,
    )

    issue_token_stats = issue_filter_result.get("token_stats", {}) or {}        # 반복이슈필터토큰통계
    scrape_stats["issue_key_tokens"] = issue_token_stats.get("issue_key_tokens", 0)                                  # 반복이슈키생성토큰수
    scrape_stats["issue_duplicate_tokens"] = issue_token_stats.get("llm_duplicate_tokens", 0)                        # 반복이슈중복판정토큰수

    # 3) 설정 파일의 제외 키워드를 적용한다.
    #    입력 상태: 반복 이슈 제거를 통과한 신규 후보.
    #    출력 상태: 섹션 목적과 맞지 않는 키워드가 포함된 기사를 제거한 후보.
    #    예: 특정 브랜드/인물/광고성 키워드를 섹션별로 빠르게 걷어내는 운영용 안전장치다.
    before_exclude_keyword_count = len(news_list)                               # 제외키워드필터전후보수
    news_list, exclude_keyword_count = apply_exclude_keywords(  # 뉴스list,제외키워드건수
        news_list=news_list,  # 뉴스list
        exclude_keywords=exclude_keywords,  # 제외키워드목록
        section_name=section_name,  # 섹션이름
    )                                                                           # 제외키워드필터결과
    after_exclude_keyword_count = len(news_list)                                # 제외키워드필터후후보수

    # 제외 키워드 필터 결과도 scrape_stats에 남긴다.
    # 1) 이 필터는 AI가 아니라 설정 파일의 단순 규칙으로 빠진 기사라서 "반복이슈 제외"와 따로 집계한다.
    # 2) email_sender.py는 이 값을 rule_based_excluded_count 계산에 더해 운영자가 규칙 기반 제외 규모를 보게 한다.
    scrape_stats["exclude_keyword_before_count"] = before_exclude_keyword_count                    # 제외키워드필터전후보수
    scrape_stats["exclude_keyword_after_count"] = after_exclude_keyword_count                      # 제외키워드필터후후보수
    scrape_stats["exclude_keyword_excluded_count"] = exclude_keyword_count                         # 제외키워드제외건수

    if not news_list:
        logger.error(f"❌ [{section_name}] 제외 키워드 필터 후 남은 뉴스가 없습니다.")
        return {
            "section_name": section_name,
            "summaries": [],
            "raw_count": 0,
            "selected_count": 0,
            "scrape_stats": scrape_stats
        }

    # 4) 같은 사건을 보도한 여러 기사를 하나의 그룹으로 묶는다.
    #    입력 상태: URL 중복, 반복 이슈, 제외 키워드를 통과한 기사 목록.
    #    출력 상태:
    #    - representative_news: 각 사건 그룹에서 메일 대표 기사로 쓸 후보
    #    - selection_groups: OpenAI가 읽을 그룹 단위 후보와 관련보도 목록
    #    이 단계가 있어야 "같은 사건 제목만 다른 기사"가 메일 10개 슬롯을 중복 점유하지 않는다.
    grouping_max_groups = int(grouping_max_groups or max_total_news)            # 그룹화최대그룹수

    grouping_result = news_grouper.build_grouping_result(                       # 사건그룹화결과
        news_list=news_list,  # 뉴스list
        max_groups=grouping_max_groups,  # 최대그룹목록
        exclude_low_quality=bool(grouping_exclude_low_quality),  # 제외lowquality
    )

    grouped_representative_news = grouping_result.get("representative_news", [])                       # 그룹대표뉴스후보목록

    grouping_news_count = int(grouping_result.get("news_count", len(news_list)) or 0)                   # 그룹화대상기사수
    grouping_group_count = int(grouping_result.get("group_count", 0) or 0)                              # 생성된사건그룹수
    grouping_low_quality_group_count = int(grouping_result.get("low_quality_group_count", 0) or 0)      # 저품질그룹수
    grouping_low_quality_article_count = int(grouping_result.get("low_quality_article_count", 0) or 0)  # 저품질기사수
    grouping_duplicate_article_count = int(grouping_result.get("duplicate_article_count", 0) or 0)      # 그룹중복대표화제외기사수
    grouping_selection_group_count = int(grouping_result.get("selection_group_count", 0) or 0)          # AI선별후보그룹수
    grouping_selection_article_count = int(grouping_result.get("selection_article_count", 0) or 0)      # AI선별후보기사수

    # 그룹화 결과를 scrape_stats에 복사한다.
    # 1) grouping_result는 여기서만 쓰이는 임시 결과라 그대로 두면 main() 최종 로그와 email_sender.py가 볼 수 없다.
    # 2) 아래 key들은 "여러 언론사가 같은 사건을 보도해서 대표 기사 하나로 합쳐진 수"를 추적하기 위해 필요하다.
    # 3) email_sender.py는 grouping_low_quality_article_count/grouping_duplicate_excluded_count를 규칙 기반 제외 수에 포함한다.
    scrape_stats["grouping_news_count"] = grouping_news_count                                                        # 그룹화대상기사수
    scrape_stats["grouping_group_count"] = grouping_group_count                                                      # 생성된사건그룹수
    scrape_stats["grouping_multi_article_group_count"] = grouping_result.get("multi_article_group_count", 0)         # 복수기사사건그룹수
    scrape_stats["grouping_low_quality_group_count"] = grouping_low_quality_group_count                              # 저품질제외그룹수
    scrape_stats["grouping_low_quality_article_count"] = grouping_low_quality_article_count                          # 저품질제외기사수
    scrape_stats["grouping_duplicate_article_count"] = grouping_duplicate_article_count                              # 대표기사외중복기사수
    scrape_stats["grouping_selection_group_count"] = grouping_selection_group_count                                  # AI선별전달그룹수
    scrape_stats["grouping_selection_article_count"] = grouping_selection_article_count                              # AI선별전달기사수
    scrape_stats["grouping_exclude_low_quality"] = bool(grouping_exclude_low_quality)                                # 저품질그룹제외설정값

    # 메일 대시보드용 집계
    # - 전체검색수: total_seen_count
    # - 24시간초과제외: old_news_count
    # - 반복이슈제외(3일): issue_filter_excluded_count
    # - 규칙기반제외: URL 중복, 제외 키워드, 그룹화 중복 대표화, 저품질/사진성 그룹 제외
    #   반복이슈제외는 별도 표시하므로 규칙기반제외에 중복 집계하지 않는다.
    rule_based_excluded_count = (                                                   # 코드규칙제외총건수
        int(scrape_stats.get("duplicate_count", 0) or 0)
        + int(scrape_stats.get("exclude_keyword_excluded_count", 0) or 0)
        + grouping_duplicate_article_count
        + grouping_low_quality_article_count
    )
    # 규칙 기반 제외 총합을 별도 key로 저장한다.
    # 1) 반복이슈 제외는 과거 발송 이력과의 비교 결과라서 이 합계에 넣지 않는다.
    # 2) 이 값은 메일 대시보드의 "규칙기반 제외" 숫자와 main() 마지막 제외 로그의 기준값이 된다.
    scrape_stats["grouping_duplicate_excluded_count"] = grouping_duplicate_article_count                             # 그룹중복제외기사수
    scrape_stats["rule_based_excluded_count"] = rule_based_excluded_count                                            # 규칙기반제외총건수
    # 기존 요약/로그 호환용 키. 이제 반복이슈제외는 포함하지 않는다.
    scrape_stats["code_rule_excluded_count"] = rule_based_excluded_count                                             # 기존호환용코드규칙제외수
    scrape_stats["ai_duplicate_excluded_count"] = 0                                                                 # AI후보중복제외초기값

    logger.info(
        f"🧩 [{section_name}] 그룹화 결과: "
        f"기사 {grouping_result.get('news_count', len(news_list))}개 → "
        f"그룹 {grouping_result.get('group_count', 0)}개 / "
        f"2건 이상 그룹 {grouping_result.get('multi_article_group_count', 0)}개 / "
        f"중복 대표화 제외 {grouping_result.get('duplicate_article_count', 0)}개 / "
        f"저품질 제외 {grouping_result.get('low_quality_group_count', 0)}그룹·{grouping_result.get('low_quality_article_count', 0)}기사 / "
        f"AI 선별 후보 그룹 {grouping_result.get('selection_group_count', 0)}개"
    )

    # 5) 그룹화 결과가 비어 있으면 원본 후보에서 시간대 샘플링으로 복구한다.
    #    보통은 그룹 대표 후보를 쓰지만, 모든 그룹이 저품질/사진성으로 걸러지는 설정에서는 메일이 비어버릴 수 있다.
    #    fallback은 "최근 시간대가 한쪽으로 쏠리지 않게" 후보를 다시 골라 최소한의 브리핑 생성을 보장한다.
    if not grouped_representative_news:
        logger.warning(f"⚠️ [{section_name}] 그룹화 후 AI 후보가 없어 원본 후보에서 시간대 샘플링 fallback을 사용합니다.")
        before_sampling_after_filter_count = len(news_list)                         # 샘플링전후보수
        grouped_representative_news = naver_news_scraper.sample_news_by_time_bucket(     # 시간대샘플링대표후보
            news_list=news_list,  # 뉴스list
            bucket_hours=4,  # bucket시간수
            max_total_news=max_total_news,  # 최대전체뉴스
            min_per_bucket=0,  # 최소기사별bucket
            recent_hours=recent_hours,  # recent시간수
        )
        after_sampling_count = len(grouped_representative_news)                     # 샘플링후후보수
    else:
        before_sampling_after_filter_count = len(news_list)                         # 샘플링전후보수
        after_sampling_count = len(grouped_representative_news)                     # 샘플링후후보수

    naver_news_scraper.update_post_issue_filter_sampling_stats(
        before_issue_filter_count=before_issue_filter_count,  # before이슈filter건수
        after_issue_filter_count=after_issue_filter_count,  # after이슈filter건수
        before_sampling_count=before_sampling_after_filter_count,  # beforesampling건수
        after_sampling_count=after_sampling_count,  # aftersampling건수
    )

    # 시간대 샘플링 후 naver_news_scraper.py의 마지막 수집 통계를 다시 가져와 누적 통계와 합친다.
    # 1) updated_scrape_stats에는 샘플링 전/후 후보 수처럼 scraper가 마지막에 갱신한 값이 들어 있다.
    # 2) scrape_stats에는 반복이슈/제외키워드/그룹화처럼 이 함수에서 추가한 값이 들어 있다.
    # 3) 둘을 합쳐야 최종 반환값 하나로 메일 대시보드와 실행 로그가 전체 처리 흐름을 모두 읽을 수 있다.
    updated_scrape_stats = naver_news_scraper.get_last_scrape_stats()               # 샘플링반영수집통계
    updated_scrape_stats.update(scrape_stats)                                       # 수집통계와필터통계병합
    scrape_stats = updated_scrape_stats                                             # 최종섹션처리통계

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

    # 6) 그룹 단위 후보를 OpenAI에 넘겨 최종 뉴스 슬롯을 고른다.
    #    group_list는 "관련보도/언론사 수/기사 수" 같은 사건 단위 맥락을 담고,
    #    fallback_news_list는 OpenAI 실패나 부족분 보충 때 사용할 대표 기사 목록이다.
    #    선택 결과 selected_news는 아직 요약 전이며, title/url/source/description/group_* 메타를 유지한다.
    selected_news = news_selector.select_important_news_groups(                     # AI최종선별뉴스
        group_list=grouping_result.get("selection_groups", []),
        fallback_news_list=grouped_representative_news,  # 대체뉴스list
        topic_name=section_name,  # topic이름
        topic_description=topic_description,  # topic설명
        limit=select_limit,  # 상한
        candidate_group_limit=selector_candidate_group_limit,  # 후보그룹상한
        min_importance_score=selector_min_importance,  # 최소중요도점수
        fill_to_limit=selector_fill_to_limit  # 부족분강제보충여부
    )

    selection_stats = news_selector.get_last_selection_stats()                      # AI선별통계

    # AI 선별 결과를 scrape_stats에 붙인다.
    # 1) 선별 단계에서 쓴 토큰은 최종 비용 로그에 합산된다.
    # 2) AI가 고른 뒤 마지막 중복 제거로 빠진 건수는 ai_duplicate_excluded_count에 기록해 그룹화 중복과 구분한다.
    scrape_stats["selection_tokens"] = selection_stats.get("selection_tokens", 0)                                                     # AI뉴스선별토큰수
    scrape_stats["event_group_tokens"] = selection_stats.get("event_group_tokens", 0)                                                 # AI사건그룹판단토큰수
    scrape_stats["ai_selected_before_final_dedup_count"] = selection_stats.get("selected_before_final_dedup_count", len(selected_news or []))  # AI최종중복제거전선별수
    scrape_stats["ai_duplicate_excluded_count"] = selection_stats.get("final_duplicate_excluded_count", 0)                            # AI최종중복제외수
    scrape_stats["ai_selected_after_final_dedup_count"] = selection_stats.get("selected_after_final_dedup_count", len(selected_news or []))    # AI최종중복제거후선별수

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

    # 7) 선별된 대표 기사만 요약한다.
    #    입력 상태: selected_news는 메일에 실릴 최종 기사 후보이며, content에는 요약에 사용할 description을 넣는다.
    #    출력 상태: summaries는 메일/히스토리/관련보도 페이지가 공통으로 쓰는 표준 결과 dict다.
    #    요약을 마지막에 하는 이유는, 제거될 기사까지 OpenAI 요약 비용을 쓰지 않기 위해서다.
    for news in selected_news:                                                      # 선별뉴스
        news["content"] = news.get("description", "")  # 처리값

    summaries = summarizer.summarize_batch(                                        # 요약결과목록
        selected_news,
        max_length=summary_max_length  # 최대length
    )

    # 요약 단계 토큰을 scrape_stats에 저장한다.
    # 1) 요약은 선별된 기사에만 수행되므로, summary_tokens가 실제 메일 본문 생성 비용이다.
    # 2) main() 마지막 전체 AI 토큰 로그에서 selection/event_group/summary/insight 토큰을 구분해 보여준다.
    scrape_stats["summary_tokens"] = sum(summary.get("tokens_used", 0) for summary in summaries or [])  # 뉴스요약토큰수

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

    # 8) 이 섹션의 산출물을 main()으로 돌려준다.
    #    main()은 모든 섹션 결과를 모은 뒤 섹션 간 최종 중복 제거, 관련보도 페이지 생성, 이메일 발송을 이어간다.
    return {
        "section_name": section_name,
        "summaries": summaries,
        "raw_count": len(news_list),
        "selected_count": len(selected_news),
        "scrape_stats": scrape_stats
    }


# [코드 이해 주석]
# - 역할: 메인 실행 함수.
# - 호출하는 곳: python main.py 실행 시 __main__ 블록에서 호출합니다.
# - 파라미터: 없음
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 인자/설정을 읽습니다 -> 섹션별 파이프라인을 실행합니다 -> 발송/저장/사용량 로그를 마무리합니다.
def main():
    """
    메인 실행 함수
    """
    # 1) 실행 설정을 확정한다.
    #    config_path는 어떤 브리핑 설정을 실행할지 결정하고,
    #    state_paths는 해당 설정만의 seen_issues/unmapped_press_domains 파일 위치를 만든다.
    #    설정별 상태 파일을 분리해야 회사/경제/이노하스 브리핑의 반복 이슈가 서로 섞이지 않는다.
    args = parse_args()                                                                 # 실행인자
    reset_openai_usage_totals()

    config_path = args.config                                                           # 설정파일경로
    config = load_config(config_path)                                                   # 브리핑설정데이터
    state_paths = get_state_file_paths(config_path)                                     # 상태파일경로모음
    issue_history_file_path = state_paths["issue_history_file_path"]                    # 이슈히스토리파일경로
    unmapped_press_domains_file_path = state_paths["unmapped_press_domains_file_path"]  # 미매핑언론사파일경로

    briefing_name = config["briefing_name"]                                             # 브리핑명
    subject_prefix = config["subject_prefix"]                                           # 메일제목접두어
    sections = config["sections"]                                                       # 섹션설정목록
    receiver_env = config.get("receiver_env", "EMAIL_RECEIVER")                         # 수신자환경변수명

    # 2) 전역 기본값을 config와 환경변수 기준으로 정규화한다.
    #    아래 값들은 섹션별 설정이 없을 때 사용하는 공통 기본값이고,
    #    각 section dict에서 같은 키를 지정하면 그 섹션만 별도로 덮어쓴다.
    display_per_keyword = int(config.get("display_per_keyword", DEFAULT_DISPLAY_PER_KEYWORD))   # 기본키워드별수집수
    pages_per_keyword = int(config.get("pages_per_keyword", DEFAULT_PAGES_PER_KEYWORD))         # 기본키워드별페이지수
    sorts = normalize_sorts(config.get("sorts", DEFAULT_SORTS))                                 # 기본검색정렬방식
    max_total_news = int(config.get("max_total_news", DEFAULT_MAX_TOTAL_NEWS))                  # 기본AI후보상한
    select_limit = int(config.get("select_limit", DEFAULT_SELECT_LIMIT))                        # 기본최종선별개수
    selector_candidate_group_limit = int(                                                       # 기본AI후보그룹상한
        config.get("selector_candidate_group_limit", DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT)
    )
    selector_min_importance = int(                                                              # 기본최소중요도점수
        config.get("selector_min_importance", DEFAULT_SELECTOR_MIN_IMPORTANCE)
    )
    selector_fill_to_limit = normalize_bool(                                                    # 기본부족분강제보충여부
        config.get("selector_fill_to_limit", DEFAULT_SELECTOR_FILL_TO_LIMIT),
        DEFAULT_SELECTOR_FILL_TO_LIMIT
    )
    summary_max_length = int(config.get("summary_max_length", DEFAULT_SUMMARY_MAX_LENGTH))      # 기본요약최대길이
    email_insight_ai = normalize_bool(                                                          # 기본메일상단AI요약여부
        config.get("email_insight_ai", DEFAULT_EMAIL_INSIGHT_AI),
        DEFAULT_EMAIL_INSIGHT_AI
    )
    email_send_mode = normalize_email_send_mode(                                                # 메일발송방식
        config.get("email_send_mode", DEFAULT_EMAIL_SEND_MODE),
        DEFAULT_EMAIL_SEND_MODE
    )
    save_issue_history = normalize_bool(                                                        # 이슈히스토리저장여부
        config.get("save_issue_history", DEFAULT_SAVE_ISSUE_HISTORY),
        DEFAULT_SAVE_ISSUE_HISTORY
    )
    if args.no_save_history:
        save_issue_history = False  # 이슈히스토리저장여부

    related_page_enabled = normalize_bool(                                                      # 관련보도페이지생성여부
        config.get("related_page_enabled", DEFAULT_RELATED_PAGE_ENABLED),
        DEFAULT_RELATED_PAGE_ENABLED
    )

    # --test: 로컬 테스트 실행. 상태/로그성 파일을 일절 남기지 않는다.
    # 1) seen_issues.json 저장 안 함 (--no-save-history와 동일 효과)
    # 2) 관련보도 페이지는 dry_run으로 처리: 메일에 "관련보도 N건 보기" 링크는 운영과 똑같이 붙이되
    #    docs/briefings/<슬러그>/YYYY-MM-DD-HHMMSS.html 파일 기록만 생략한다 (링크 클릭 시 404는 허용).
    # 3) unmapped_press_domains.json 저장 안 함 (아래 발송 이후 단계에서 args.test로 건너뜀)
    if args.test:
        save_issue_history = False        # 이슈히스토리저장여부
        logger.info("🧪 --test 모드: seen_issues/관련보도 HTML/unmapped 도메인 파일을 저장하지 않습니다. (메일 링크는 정상 표시)")
    related_page_keep_days = int(                                                               # 관련보도페이지보존일수
        config.get("related_page_keep_days", DEFAULT_RELATED_PAGE_KEEP_DAYS)
    )

    recent_hours = int(config.get("recent_hours", DEFAULT_RECENT_HOURS))                        # 기본최근뉴스시간범위
    issue_history_days = int(config.get("issue_history_days", DEFAULT_ISSUE_HISTORY_DAYS))      # 기본반복이슈비교일수

    config_exclude_keywords = normalize_exclude_keywords(config.get("exclude_keywords", []))    # 설정공통제외키워드
    logger.info(
        "🚀 뉴스 스크래퍼 시작: %s / 섹션 %s개 / selector=%s / summary=%s / config=%s",
        briefing_name,
        len(sections),
        getattr(news_selector, "SELECTOR_MODEL", "확인불가"),
        getattr(summarizer, "MODEL", "확인불가"),
        config_path,
    )
    logger.info(
        "⚙️ 기본값: recent=%sh / issue_history=%s일 / max_total=%s / select=%s / ai_groups=%s / min_imp=%s / fill_to_limit=%s / send_mode=%s / save_history=%s / related_page=%s(%s일)",
        recent_hours,
        issue_history_days,
        max_total_news,
        select_limit,
        selector_candidate_group_limit,
        selector_min_importance,
        selector_fill_to_limit,
        email_send_mode,
        save_issue_history,
        related_page_enabled,
        related_page_keep_days,
    )

    section_results = []                                                                        # 섹션별처리결과목록

    # 3) 섹션별 파이프라인을 순서대로 실행한다.
    #    section_result 하나가 "메일의 한 섹션"이 되며, summaries에는 실제 카드로 렌더링할 뉴스가 들어간다.
    #    각 섹션은 키워드/최근시간/선별개수/요약길이를 별도로 가질 수 있어 브리핑 안에서도 운영 정책을 달리할 수 있다.
    for section in sections:                                                                    # 섹션설정
        section_name = section["section_name"]                                                  # 섹션명
        keywords = section["keywords"]                                                          # 섹션검색키워드
        topic_description = section["topic_description"]                                        # 섹션선별기준설명

        section_display_per_keyword = int(                                                      # 섹션키워드별수집수
            section.get("display_per_keyword", display_per_keyword)
        )
        section_pages_per_keyword = int(                                                        # 섹션키워드별페이지수
            section.get("pages_per_keyword", pages_per_keyword)
        )
        section_sorts = normalize_sorts(                                                        # 섹션검색정렬방식
            section.get("sorts", sorts)
        )
        section_max_total_news = int(                                                           # 섹션AI후보상한
            section.get("max_total_news", max_total_news)
        )
        section_select_limit = int(                                                             # 섹션최종선별개수
            section.get("select_limit", select_limit)
        )
        section_selector_candidate_group_limit = int(                                           # 섹션AI후보그룹상한
            section.get("selector_candidate_group_limit", selector_candidate_group_limit)
        )
        section_selector_min_importance = int(                                                  # 섹션최소중요도점수
            section.get("selector_min_importance", selector_min_importance)
        )
        section_selector_fill_to_limit = normalize_bool(                                        # 섹션부족분강제보충여부
            section.get("selector_fill_to_limit", selector_fill_to_limit),
            selector_fill_to_limit
        )
        section_summary_max_length = int(                                                       # 섹션요약최대길이
            section.get("summary_max_length", summary_max_length)
        )
        section_email_insight_ai = normalize_bool(                                              # 섹션메일상단AI요약여부
            section.get("email_insight_ai", email_insight_ai),
            email_insight_ai
        )
        section_recent_hours = int(                                                             # 섹션최근뉴스시간범위
            section.get("recent_hours", recent_hours)
        )
        section_issue_history_days = int(                                                       # 섹션반복이슈비교일수
            section.get("issue_history_days", issue_history_days)
        )
        section_exclude_keywords = normalize_exclude_keywords(                                  # 섹션제외키워드목록
            section.get("exclude_keywords", config.get("exclude_keywords", []))
        )
        section_grouping_max_groups = int(                                                      # 섹션그룹화최대그룹수
            section.get("grouping_max_groups", config.get("grouping_max_groups", section_max_total_news))
        )
        section_grouping_exclude_low_quality = normalize_bool(                                  # 섹션저품질그룹제외여부
            section.get("grouping_exclude_low_quality", config.get("grouping_exclude_low_quality", True)),
            True
        )

        section_result = collect_select_and_summarize(                                          # 섹션처리결과
            briefing_name=briefing_name,  # briefing이름
            receiver_env=receiver_env,  # 수신자env
            section_name=section_name,  # 섹션이름
            keywords=keywords,  # 키워드목록
            topic_description=topic_description,  # topic설명
            display_per_keyword=section_display_per_keyword,  # 표시건수기사별키워드
            pages_per_keyword=section_pages_per_keyword,  # 페이지목록기사별키워드
            sorts=section_sorts,  # 정렬목록
            max_total_news=section_max_total_news,  # 최대전체뉴스
            select_limit=section_select_limit,  # select상한
            recent_hours=section_recent_hours,  # recent시간수
            issue_history_days=section_issue_history_days,  # 이슈히스토리일수
            issue_history_file_path=issue_history_file_path,  # 이슈히스토리파일경로
            unmapped_press_domains_file_path=unmapped_press_domains_file_path,  # unmapped언론사domains파일경로
            exclude_keywords=section_exclude_keywords,  # 제외키워드목록
            grouping_max_groups=section_grouping_max_groups,  # grouping최대그룹목록
            grouping_exclude_low_quality=section_grouping_exclude_low_quality,  # grouping제외lowquality
            selector_candidate_group_limit=section_selector_candidate_group_limit,  # selector후보그룹상한
            selector_min_importance=section_selector_min_importance,  # selector최소중요도
            selector_fill_to_limit=section_selector_fill_to_limit,  # selector부족분강제보충여부
            summary_max_length=section_summary_max_length,  # 요약최대length
            email_insight_ai=section_email_insight_ai  # 이메일주소insightAI
        )

        section_results.append(section_result)

    # 4) 메일 발송 직전에 섹션 간 중복을 한 번 더 제거한다.
    #    섹션별로는 중복 제거가 끝났더라도, 같은 사건이 "IT"와 "롯데"처럼 다른 섹션 키워드에서 동시에 잡힐 수 있다.
    #    이미 요약된 결과끼리 비교하므로 OpenAI 호출/토큰 비용은 늘지 않고, 최종 메일 카드만 정리된다.
    final_dedup_result = issue_history.deduplicate_section_results(section_results)  # 섹션간최종중복제거결과
    section_results = final_dedup_result.get("section_results", section_results)     # 최종메일섹션결과
    if final_dedup_result.get("excluded_count", 0):
        logger.info(
            "🧹 메일 최종 중복 제거: %s개 → %s개 / 제외 %s개",
            final_dedup_result.get("before_count", 0),
            final_dedup_result.get("after_count", 0),
            final_dedup_result.get("excluded_count", 0),
        )

    # 5) 관련보도 상세 페이지를 만들고, 각 뉴스 요약 dict에 related_reports_url을 붙인다.
    #    이 함수는 section_results를 직접 갱신하므로, 반드시 이메일 HTML 생성 전에 실행해야 메일에 "관련보도 N건 보기" 링크가 들어간다.
    if related_page_enabled:
        related_page_result = related_pages.generate_related_page(                    # 관련보도페이지생성결과
            config=config,  # 설정
            config_slug=state_paths["config_slug"],
            briefing_name=briefing_name,  # briefing이름
            subject_prefix=subject_prefix,  # 메일제목prefix
            section_results=section_results,  # 섹션결과목록
            keep_days=related_page_keep_days,  # 보존일수
            dry_run=args.test,  # 테스트실행파일기록생략여부
        )
        if related_page_result.get("generated"):
            logger.info(
                "🔗 관련보도 상세 페이지 생성 완료: %s / 연결 뉴스 %s개 / 오래된 페이지 삭제 %s개",
                related_page_result.get("path"),
                related_page_result.get("linked_count", 0),
                related_page_result.get("removed_count", 0),
            )
        else:
            logger.info(
                "🔗 관련보도 상세 페이지 생성 건너뜀: %s / 오래된 페이지 삭제 %s개",
                related_page_result.get("reason"),
                related_page_result.get("removed_count", 0),
            )

    # 6) 최종 section_results를 HTML 메일로 렌더링하고 SMTP로 발송한다.
    #    이 시점의 section_results에는 섹션 간 중복 제거와 관련보도 URL 연결이 모두 반영되어 있다.
    logger.debug("📧 이메일 발송 시작: receiver_env=%s", receiver_env)

    result = email_sender.send_email(                                                # 이메일발송결과
        briefing_name=briefing_name,  # briefing이름
        subject_prefix=subject_prefix,  # 메일제목prefix
        section_results=section_results,  # 섹션결과목록
        receiver_env_name=receiver_env,  # 수신자env이름
        send_mode=email_send_mode  # send방식
    )

    if result["success"]:
        logger.info(f"✅ {result['message']}")

        # 7) 발송 성공 후에만 seen_issues.json을 갱신한다.
        #    실패한 메일의 뉴스를 히스토리에 저장하면, 실제로 받지 못한 이슈가 다음 실행에서 "이미 보낸 이슈"로 제외될 수 있다.
        if save_issue_history:
            history_result = issue_history.append_sent_issues(                       # 이슈히스토리저장결과
                briefing_name=briefing_name,  # briefing이름
                subject_prefix=subject_prefix,  # 메일제목prefix
                receiver_env=receiver_env,  # 수신자env
                section_results=section_results,  # 섹션결과목록
                file_path=issue_history_file_path,  # 파일경로
                keep_days=issue_history_days  # 보존일수
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

    # 8) 언론사명 매핑에 실패한 도메인을 저장한다.
    #    이 파일은 press_map.py 보강 후보를 모으는 운영용 자료이며, 메일 발송 성공 여부와 무관하게 마지막에 기록한다.
    #    --test 모드에서는 로컬에 상태 파일을 남기지 않기 위해 건너뛴다.
    if args.test:
        logger.info("🧪 --test 모드: 미매핑 언론사 도메인 저장을 건너뜁니다.")
    elif hasattr(naver_news_scraper, "save_unmapped_press_domains"):
        try:
            naver_news_scraper.save_unmapped_press_domains(
                filename=unmapped_press_domains_file_path  # filename
            )
            logger.debug(f"🗂️ 미매핑 언론사 도메인 저장 완료: {unmapped_press_domains_file_path}")
        except Exception as e:  # 예외객체
            logger.warning(f"⚠️ 미매핑 언론사 도메인 저장 실패: {e}")
    else:
        logger.debug("미매핑 언론사 도메인 저장 함수 없음: 건너뜀")

    # 9) 섹션별 scrape_stats를 합산해 실행 로그용 최종 요약을 만든다.
    #    이 집계는 메일 내용이 아니라 운영자가 Actions 로그에서 수집/제외/토큰 흐름을 확인하기 위한 것이다.
    logger.info("📊 작업 완료 요약 생성")

    total_raw_count = 0                 # 전체후보뉴스수
    total_selected_count = 0            # 전체선별뉴스수
    total_summary_count = 0             # 전체요약뉴스수
    total_tokens = 0                    # 전체AI토큰수
    total_issue_key_tokens = 0          # 전체이슈키토큰수
    total_issue_duplicate_tokens = 0    # 전체이슈중복판정토큰수
    total_selection_tokens = 0          # 전체뉴스선별토큰수
    total_event_group_tokens = 0        # 전체사건그룹토큰수
    total_summary_tokens = 0            # 전체요약토큰수
    total_insight_tokens = 0            # 전체메일3줄요약토큰수

    for section_result in section_results:                                      # 섹션처리결과
        section_name = section_result["section_name"]                           # 섹션명
        summaries = section_result["summaries"]                                 # 섹션요약뉴스목록
        scrape_stats = section_result.get("scrape_stats", {})                   # 섹션처리통계

        raw_count = section_result["raw_count"]                                 # 섹션후보뉴스수
        selected_count = section_result["selected_count"]                       # 섹션선별뉴스수
        summary_count = len(summaries)                                          # 섹션요약뉴스수

        duplicate_count = scrape_stats.get("duplicate_count", 0)                # URL중복제외수
        old_news_count = scrape_stats.get("old_news_count", 0)                  # 시간초과제외수
        pre_sampling_count = scrape_stats.get("pre_sampling_count", raw_count)  # 샘플링전후보수
        final_candidate_count = scrape_stats.get("final_candidate_count", raw_count)  # 최종AI후보수

        issue_filter_before_count = scrape_stats.get("issue_filter_before_count", final_candidate_count)  # 반복필터전후보수
        issue_filter_after_count = scrape_stats.get("issue_filter_after_count", raw_count)                # 반복필터후후보수
        issue_filter_excluded_count = scrape_stats.get("issue_filter_excluded_count", 0)                  # 반복이슈제외수
        issue_filter_past_issue_count = scrape_stats.get("issue_filter_past_issue_count", 0)              # 비교한과거이슈수

        total_raw_count += raw_count  # 처리값
        total_selected_count += selected_count  # 처리값
        total_summary_count += summary_count  # 처리값
        issue_key_tokens = scrape_stats.get("issue_key_tokens", 0)                                      # 섹션이슈키토큰수
        issue_duplicate_tokens = scrape_stats.get("issue_duplicate_tokens", 0)                          # 섹션이슈중복판정토큰수
        selection_tokens = scrape_stats.get("selection_tokens", 0)                                      # 섹션뉴스선별토큰수
        event_group_tokens = scrape_stats.get("event_group_tokens", 0)                                  # 섹션사건그룹토큰수
        summary_tokens = scrape_stats.get("summary_tokens", sum(summary.get("tokens_used", 0) for summary in summaries))  # 섹션요약토큰수
        insight_tokens = scrape_stats.get("insight_tokens", 0)                                          # 섹션메일3줄요약토큰수

        section_total_tokens = (                                                                        # 섹션전체AI토큰수
            issue_key_tokens
            + issue_duplicate_tokens
            + selection_tokens
            + event_group_tokens
            + summary_tokens
            + insight_tokens
        )

        total_issue_key_tokens += issue_key_tokens  # 처리값
        total_issue_duplicate_tokens += issue_duplicate_tokens  # 처리값
        total_selection_tokens += selection_tokens  # 처리값
        total_event_group_tokens += event_group_tokens  # 처리값
        total_summary_tokens += summary_tokens  # 처리값
        total_insight_tokens += insight_tokens  # 처리값
        total_tokens += section_total_tokens  # 처리값

        total_seen_count = scrape_stats.get("total_seen_count", issue_filter_before_count)              # 전체검색확인수
        exclude_keyword_excluded_count = scrape_stats.get("exclude_keyword_excluded_count", 0)          # 제외키워드제외수
        grouping_duplicate_article_count = scrape_stats.get("grouping_duplicate_excluded_count", 0)     # 그룹중복제외기사수
        grouping_low_quality_article_count = scrape_stats.get("grouping_low_quality_article_count", 0)  # 저품질제외기사수
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
    except Exception as e:  # 예외객체
        logger.error(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
