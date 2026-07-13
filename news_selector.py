# =============================================================================
# [파일 설명]
# - 수행 기능: 그룹화된 후보 또는 원본 후보 중 메일에 보낼 핵심 뉴스를 OpenAI와 로컬 규칙으로 선별합니다.
# - 프로세스: 후보 텍스트 구성 -> OpenAI 선별 요청 -> 사건 단위 중복 제거 -> 부족분 보충 -> 선택 통계 기록
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 후보 뉴스/그룹 목록, 섹션명, 선별 개수, OpenAI 모델/토큰 설정
# - 리턴값/출력: 요약 단계로 넘길 대표 뉴스 dict 목록과 LAST_SELECTION_STATS 통계를 제공합니다.
# =============================================================================

"""
OpenAI API를 사용한 뉴스 선별 모듈

역할:
- news_grouper.py가 만든 사건 그룹 후보 중
- 주제 적합성과 중요도를 기준으로 OpenAI가 최종 뉴스 그룹을 선택한다.
- 선택된 뉴스에 중요도 점수를 함께 부여한다.
- AI 선택 결과 안에서 남은 유사 사건은 로컬 규칙으로 한 번 더 제거한다.
"""

import os
import json
import logging
import re
import hashlib
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # OpenAI클래스
from dotenv import load_dotenv
from openai_usage import (
    create_chat_completion,
    record_openai_usage,
    openai_token_limit_kwargs,
    openai_temperature_kwargs,
    openai_reasoning_effort_kwargs,
    openai_json_response_format_kwargs,
    is_gpt5_model,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈로거

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                                # OpenAI인증키

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)  # OpenAI클라이언트
else:
    client = None  # OpenAI클라이언트

# 뉴스 선별 모델
# - SELECTOR_MODEL: 그룹 후보 중 최종 뉴스를 고르는 그룹 단위 선별용
#
# 중요:
# 기본값을 GPT-5 nano로 둔다.
# 더 높은 품질이 필요하면 env에서 OPENAI_SELECTOR_MODEL만 gpt-5-mini 또는 gpt-5.4-nano 등으로 올리면 된다.
SELECTOR_MODEL = os.getenv("OPENAI_SELECTOR_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))  # 그룹선별모델명

LAST_SELECTION_STATS = {  # 마지막선별통계
    "selection_tokens": 0,                                                   # 뉴스선별토큰수
    "event_group_tokens": 0,                                                 # LLM사건그룹화토큰수
    "final_duplicate_excluded_count": 0,                                     # 최종중복제외건수
    "selected_before_final_dedup_count": 0,                                  # 최종중복제거전선별수
    "selected_after_final_dedup_count": 0,                                   # 최종중복제거후선별수
}                                                                           # 마지막선별통계


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: 현재 모듈 내부 전용 보조 함수입니다. 정적 직접 호출이 없으면 조건부 흐름에서 사용될 수 있습니다.
# - 파라미터: name: str, default: int, min_value: int = 1, max_value: Optional[int] = None
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _env_int(name: str, default: int, min_value: int = 1, max_value: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))  # 값
    except Exception:
        value = int(default)  # 값

    value = max(int(min_value), value)  # 값
    if max_value is not None:
        value = min(int(max_value), value)  # 값
    return value


# 후보를 45개(설명은 15개만)에서 35개(전원 설명 포함)로 바꿨다.
# 후보 수를 줄인 만큼 모든 후보에 짧은 설명을 실어도 프롬프트 토큰은 기존과 비슷하며,
# AI가 제목만 보고 관련성을 추측하는 구간이 없어진다.
DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT = _env_int("SELECTOR_CANDIDATE_GROUP_LIMIT", 35, 10, 100)  # 기본AI전달그룹상한
SELECTOR_MAX_COMPLETION_TOKENS = _env_int("SELECTOR_MAX_COMPLETION_TOKENS", 700, 256, 2048)       # 선별응답토큰상한
SELECTOR_DETAILED_CANDIDATE_COUNT = _env_int("SELECTOR_DETAILED_CANDIDATE_COUNT", 100, 0, 100)    # 상세설명포함후보수
GROUP_CANDIDATE_TITLE_CHARS = _env_int("SELECTOR_GROUP_TITLE_CHARS", 65, 35, 120)                 # 후보제목최대문자수
GROUP_CANDIDATE_DESCRIPTION_CHARS = _env_int("SELECTOR_GROUP_DESCRIPTION_CHARS", 50, 0, 160)      # 후보설명최대문자수
GROUP_CANDIDATE_SOURCES_CHARS = _env_int("SELECTOR_GROUP_SOURCES_CHARS", 30, 20, 100)             # 후보언론사목록최대문자수
GROUP_CANDIDATE_KEYWORDS_CHARS = _env_int("SELECTOR_GROUP_KEYWORDS_CHARS", 30, 10, 100)           # 후보키워드목록최대문자수


# [코드 이해 주석]
# - 역할: 누적 통계나 상태 값을 초기 상태로 되돌립니다.
# - 호출하는 곳: news_selector.select_important_news, news_selector.select_important_news_groups
# - 파라미터: 없음
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def reset_selection_stats():
    global LAST_SELECTION_STATS
    # main.py는 선별이 끝난 뒤 get_last_selection_stats()로 이 값을 가져와 scrape_stats에 붙인다.
    # 섹션별 토큰/중복 통계가 섞이지 않도록 선별 함수 진입마다 새 dict로 초기화한다.
    LAST_SELECTION_STATS = {  # 마지막선별통계
        "selection_tokens": 0,                                               # 뉴스선별토큰수
        "event_group_tokens": 0,                                             # LLM사건그룹화토큰수
        "final_duplicate_excluded_count": 0,                                 # 최종중복제외건수
        "selected_before_final_dedup_count": 0,                              # 최종중복제거전선별수
        "selected_after_final_dedup_count": 0,                               # 최종중복제거후선별수
    }


# [코드 이해 주석]
# - 역할: 누적 통계나 그룹에 새 값을 더합니다.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups
# - 파라미터: key: str, value: int
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def add_selection_tokens(key: str, value: int):
    try:
        token_count = int(value or 0)                                        # 추가할토큰수
    except Exception:
        token_count = 0  # 토큰건수
    # selection_tokens/event_group_tokens 모두 main.py 최종 AI 토큰 요약에 합산된다.
    # 호출 단계가 늘어나도 key 이름만 맞추면 같은 누적 dict로 전달할 수 있게 한다.
    LAST_SELECTION_STATS[key] = int(LAST_SELECTION_STATS.get(key, 0)) + token_count  # 처리값


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: 없음
# - 리턴값: Dict[str, int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_last_selection_stats() -> Dict[str, int]:
    return dict(LAST_SELECTION_STATS)


# [코드 이해 주석]
# - 역할: None 방지용 문자열 변환.
# - 호출하는 곳: news_selector._build_candidate_text, news_selector._build_event_dedup_text,
# news_selector._build_final_dedup_payload, news_selector._build_group_candidate_text, news_selector._clip_text,
# news_selector._extract_final_anchor_tokens 외 13곳
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _safe_text(value: Any) -> str:
    """
    None 방지용 문자열 변환
    """
    if value is None:
        return ""
    return str(value).strip()


# [코드 이해 주석]
# - 역할: 중요도 점수 안전 변환.
# - 호출하는 곳: news_selector._estimate_importance_score_from_news, news_selector._prepare_selected_news
# - 파라미터: value: Any, default: int = 3, min_value: int = 1, max_value: int = 5
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _safe_int(value: Any, default: int = 3, min_value: int = 1, max_value: int = 5) -> int:
    """
    중요도 점수 안전 변환
    """
    try:
        number = int(value)  # 숫자값
    except Exception:
        number = default  # 숫자값

    if number < min_value:
        return min_value

    if number > max_value:
        return max_value

    return number


# [코드 이해 주석]
# - 역할: LLM 입력 토큰 절감을 위해 긴 설명을 적정 길이로 자른다.
# - 호출하는 곳: news_selector._build_candidate_text, news_selector._build_event_dedup_text,
# news_selector._build_group_candidate_text
# - 파라미터: value: Any, limit: int = 180
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _clip_text(value: Any, limit: int = 180) -> str:
    """
    LLM 입력 토큰 절감을 위해 긴 설명을 적정 길이로 자른다.
    """
    text = _safe_text(value)  # 텍스트
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."




# [코드 이해 주석]
# - 역할: OpenAI 응답에서 JSON 파싱.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups
# - 파라미터: content: str
# - 리턴값: Dict 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_json(content: str) -> Dict:
    """
    OpenAI 응답에서 JSON 파싱.
    원칙적으로 JSON만 오게 하지만, 혹시 코드블록이 섞이는 경우를 대비한다.
    """
    content = _safe_text(content)  # 본문

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()  # 본문

    return json.loads(content)


# [코드 이해 주석]
# - 역할: 모델이 JSON 배열이나 문자열을 반환해도 호출부에서 AttributeError가 나지 않게 한다.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups
# - 파라미터: result: Any
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _ensure_json_object(result: Any) -> Dict[str, Any]:
    """
    모델이 JSON 배열이나 문자열을 반환해도 호출부에서 AttributeError가 나지 않게 한다.
    """
    if isinstance(result, dict):
        return result
    return {}


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups
# - 파라미터: value: Any
# - 리턴값: List[Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _ensure_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


# [코드 이해 주석]
# - 역할: 요약 단계로 넘기기 전에 필요한 필드 보강.
# - 호출하는 곳: news_selector._fallback_select, news_selector._fallback_select_groups,
# news_selector._supplement_after_dedup, news_selector._supplement_final_news_after_dedup,
# news_selector.select_important_news, news_selector.select_important_news_groups
# - 파라미터: news: Dict, importance_score: Any = 3
# - 리턴값: Dict 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _prepare_selected_news(
    news: Dict,
    importance_score: Any = 3  # 중요도점수
) -> Dict:
    """
    요약 단계로 넘기기 전에 필요한 필드 보강
    """
    news["importance_score"] = _safe_int(importance_score)  # 처리값
    news["content"] = news.get("description", "")  # 처리값

    return news


# ====================================
# 최종 AI 선별 결과 중복 제거 유틸
# ====================================
_FINAL_DEDUP_STOPWORDS = {  # 최종중복제거불용어
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
}


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화하는 내부 보조 함수입니다.
# - 호출하는 곳: news_selector._build_final_dedup_payload, news_selector._extract_final_dedup_tokens,
# news_selector._extract_final_title_tokens, news_selector._normalize_final_dedup_title
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def _normalize_final_dedup_text(value: Any) -> str:
    text = _safe_text(value).lower()  # 텍스트
    text = re.sub(r"<.*?>", " ", text)  # 텍스트
    text = text.replace("&quot;", '"').replace("&amp;", "&")  # 텍스트
    text = text.replace("&lt;", "<").replace("&gt;", ">")  # 텍스트
    text = text.replace("&#39;", "'")  # 텍스트
    text = text.replace("美", "미국").replace("韓", "한국").replace("中", "중국")  # 텍스트
    text = text.replace("日", "일본").replace("李", "이").replace("金", "김")  # 텍스트
    text = re.sub(r"\[[^\]]*\]|【[^】]*】", " ", text)  # 텍스트
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)  # 텍스트
    text = re.sub(r"\s+", " ", text)  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화하는 내부 보조 함수입니다.
# - 호출하는 곳: news_selector._build_final_dedup_payload
# - 파라미터: value: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def _normalize_final_dedup_title(value: Any) -> str:
    text = _normalize_final_dedup_text(value)  # 텍스트
    text = re.sub(r"\s+", "", text)  # 텍스트
    text = re.sub(r"[^0-9a-z가-힣]", "", text)  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._is_final_duplicate_news
# - 파라미터: value: Any
# - 리턴값: set 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_final_number_tokens(value: Any) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", _safe_text(value)))


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._build_final_dedup_payload
# - 파라미터: news: Dict[str, Any], max_tokens: int = 90
# - 리턴값: List[str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_final_dedup_tokens(news: Dict[str, Any], max_tokens: int = 90) -> List[str]:
    title = _safe_text(news.get("title"))  # 제목
    description = _safe_text(news.get("description") or news.get("summary") or news.get("content"))  # 설명
    text = _normalize_final_dedup_text(f"{title} {description}")  # 텍스트
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)  # 원본토큰수

    tokens = []  # 토큰수
    seen = set()  # 확인된
    for token in raw_tokens:  # 토큰
        token = _normalize_final_token(token)  # 토큰
        if not token or token in _FINAL_DEDUP_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._is_final_duplicate_news
# - 파라미터: tokens_a: List[str], tokens_b: List[str]
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _final_token_overlap(tokens_a: List[str], tokens_b: List[str]):
    set_a = set(tokens_a or [])  # seta
    set_b = set(tokens_b or [])  # setb
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b  # common
    denominator = max(1, min(len(set_a), len(set_b)))  # denominator
    return len(common) / denominator, len(common)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._make_final_simhash
# - 파라미터: token: str
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _stable_final_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # digest
    return int(digest[:16], 16)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._build_final_dedup_payload
# - 파라미터: tokens: List[str]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _make_final_simhash(tokens: List[str]) -> str:
    if not tokens:
        return ""
    vector = [0] * 64  # vector
    for token in tokens:  # 토큰
        value = _stable_final_token_hash(token)  # 값
        for bit in range(64):  # bit
            if value & (1 << bit):
                vector[bit] += 1  # vector
            else:
                vector[bit] -= 1  # vector
    fingerprint = 0  # fingerprint
    for bit, score in enumerate(vector):  # 비트,점수
        if score >= 0:
            fingerprint |= (1 << bit)  # 처리값
    return f"{fingerprint:016x}"


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._is_final_duplicate_news
# - 파라미터: a: str, b: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _final_simhash_distance(a: str, b: str):
    if not a or not b:
        return None
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None




# [코드 이해 주석]
# - 역할: 외부 형태소 분석기 없이 조사/어미 차이만 가볍게 줄인다.
# - 호출하는 곳: news_selector._extract_final_dedup_tokens, news_selector._extract_final_title_tokens
# - 파라미터: token: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def _normalize_final_token(token: str) -> str:
    """
    외부 형태소 분석기 없이 조사/어미 차이만 가볍게 줄인다.
    특정 주제 단어가 아니라 한국어 문장 공통 형태를 다루는 규칙이다.
    """
    token = _safe_text(token).lower().strip()  # 토큰
    if not token:
        return ""

    # 흔한 한자 약칭을 앞단에서 이미 바꿨지만, 토큰 단위에서도 한 번 더 방어한다.
    token = token.replace("美", "미국").replace("韓", "한국").replace("中", "중국")  # 토큰
    token = token.replace("日", "일본").replace("李", "이").replace("金", "김")  # 토큰

    # 한국어 조사/어미 일부 제거: 발언/발언에, 코스피/코스피는, 김용범/김용범의 등을 맞춘다.
    suffixes = [  # suffixes
        "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
        "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
        "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
    ]
    for suffix in suffixes:  # suffix
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]  # 토큰
            break

    return token.strip()

# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._build_final_dedup_payload
# - 파라미터: title: Any
# - 리턴값: List[str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_final_title_tokens(title: Any) -> List[str]:
    normalized = _normalize_final_dedup_text(title)  # 정규화
    tokens = []  # 토큰수
    seen = set()  # 확인된
    for token in re.findall(r"[가-힣a-zA-Z0-9]{2,}", normalized):  # 토큰
        token = _normalize_final_token(token)  # 토큰
        if not token or token in _FINAL_DEDUP_STOPWORDS or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


# [코드 이해 주석]
# - 역할: 최종 선별 후 중복 제거용 핵심 토큰.
# - 호출하는 곳: news_selector._build_final_dedup_payload
# - 파라미터: tokens: List[str]
# - 리턴값: List[str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_final_anchor_tokens(tokens: List[str]) -> List[str]:
    """
    최종 선별 후 중복 제거용 핵심 토큰.

    특정 브리핑 주제 단어를 코드에 박지 않고, 길이와 형태만으로 정보량이 큰 토큰을 고른다.
    - 영문 약어/혼합 토큰: KDI, AI, EMR 같은 기관·기술명 후보
    - 3자 이상 한글/영문 토큰: 사건을 구분할 가능성이 높은 단어
    """
    anchors = []  # anchors
    for token in tokens or []:  # 토큰
        token = _safe_text(token).lower()  # 토큰
        if not token or token in _FINAL_DEDUP_STOPWORDS:
            continue
        has_alpha = bool(re.search(r"[a-zA-Z]", token))  # hasalpha
        has_korean = bool(re.search(r"[가-힣]", token))  # haskorean
        if has_alpha or len(token) >= 3 or (has_korean and len(token) >= 3):
            anchors.append(token)
    return anchors


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 내부에서 사용할 출력 구조를 만드는 보조 함수입니다.
# - 호출하는 곳: news_selector._deduplicate_final_selected_news, news_selector._find_final_duplicate_info,
# news_selector._is_final_duplicate_news, news_selector._supplement_final_news_after_dedup
# - 파라미터: news: Dict[str, Any]
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 내부용 문자열/dict 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def _build_final_dedup_payload(news: Dict[str, Any]) -> Dict[str, Any]:
    title = _safe_text(news.get("title"))  # 제목
    description = _safe_text(news.get("description") or news.get("summary") or news.get("content"))  # 설명
    normalized_title = _normalize_final_dedup_title(title)  # 정규화제목
    normalized_text = _normalize_final_dedup_text(f"{title} {description}")  # 정규화텍스트
    compact_text = re.sub(r"\s+", "", normalized_text)  # compact텍스트
    tokens = _extract_final_dedup_tokens(news)  # 토큰수
    title_tokens = _extract_final_title_tokens(title)  # 제목토큰수
    anchor_tokens = _extract_final_anchor_tokens(title_tokens or tokens)  # 앵커토큰수
    return {
        "title": title,
        "normalized_title": normalized_title,
        "normalized_text": compact_text,
        "tokens": tokens,
        "title_tokens": title_tokens,
        "anchor_tokens": anchor_tokens,
        "simhash": _make_final_simhash(tokens),
    }


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._is_final_duplicate_news
# - 파라미터: cand_payload: Dict[str, Any], kept_payload: Dict[str, Any]
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _has_shared_final_anchor(cand_payload: Dict[str, Any], kept_payload: Dict[str, Any]) -> bool:
    cand_anchors = set(cand_payload.get("anchor_tokens") or [])  # candanchors
    kept_anchors = set(kept_payload.get("anchor_tokens") or [])  # keptanchors
    if not cand_anchors or not kept_anchors:
        return False
    return bool(cand_anchors & kept_anchors)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._find_final_duplicate_info
# - 파라미터: candidate: Dict[str, Any], kept: Dict[str, Any]
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _is_final_duplicate_news(candidate: Dict[str, Any], kept: Dict[str, Any]):
    cand_payload = candidate.get("_final_dedup_payload") or _build_final_dedup_payload(candidate)  # 후보데이터
    kept_payload = kept.get("_final_dedup_payload") or _build_final_dedup_payload(kept)  # kept데이터

    cand_title = cand_payload.get("normalized_title") or ""  # 후보제목
    kept_title = kept_payload.get("normalized_title") or ""  # kept제목
    cand_numbers = _extract_final_number_tokens(cand_payload.get("title") or cand_title)  # 후보numbers
    kept_numbers = _extract_final_number_tokens(kept_payload.get("title") or kept_title)  # keptnumbers
    number_conflict = bool(cand_numbers and kept_numbers and cand_numbers != kept_numbers)  # 숫자값conflict

    if cand_title and kept_title and cand_title == kept_title:
        return True, "title_exact", 1.0

    # 한쪽 제목이 다른 쪽 제목을 대부분 포함하면 같은 사건으로 본다.
    # 예: 제목 뒤에 '(종합)', 부제, 수치 설명이 붙은 변형 기사.
    if cand_title and kept_title and min(len(cand_title), len(kept_title)) >= 10 and not number_conflict:
        shorter, longer = sorted([cand_title, kept_title], key=len)  # 짧은값,긴값
        if shorter in longer:
            return True, "title_contains", len(shorter) / max(1, len(longer))

    title_overlap, title_common_count = _final_token_overlap(  # 제목overlap,제목공통건수
        cand_payload.get("title_tokens", []),
        kept_payload.get("title_tokens", []),
    )

    title_similarity = 0.0  # 제목유사도
    if cand_title and kept_title:
        title_similarity = SequenceMatcher(None, cand_title, kept_title).ratio()  # 제목유사도
        if title_similarity >= 0.88 and not number_conflict:
            return True, "title_similarity", title_similarity
        if (
            title_similarity >= 0.76
            and title_common_count >= 3
            and _has_shared_final_anchor(cand_payload, kept_payload)
            and not number_conflict
        ):
            return True, f"title_similarity_anchor_common_{title_common_count}", title_similarity

    if title_overlap >= 0.45 and title_common_count >= 3 and not number_conflict:
        return True, f"title_token_overlap_common_{title_common_count}", title_overlap

    # 최종 10개 안에서는 제목 핵심어 3개 이상이 겹치고 anchor도 공유하면
    # 표현이 달라도 같은 사건일 가능성이 높다.
    if title_common_count >= 3 and _has_shared_final_anchor(cand_payload, kept_payload) and not number_conflict:
        return True, f"title_common_anchor_{title_common_count}", title_overlap

    overlap, common_count = _final_token_overlap(  # overlap,공통건수
        cand_payload.get("tokens", []),
        kept_payload.get("tokens", []),
    )
    shared_anchor = _has_shared_final_anchor(cand_payload, kept_payload)  # shared앵커

    cand_text = cand_payload.get("normalized_text") or ""  # 후보텍스트
    kept_text = kept_payload.get("normalized_text") or ""  # kept텍스트
    text_similarity = 0.0  # 텍스트유사도
    if cand_text and kept_text:
        text_similarity = SequenceMatcher(None, cand_text, kept_text).ratio()  # 텍스트유사도
        if text_similarity >= 0.92 and (common_count >= 5 or shared_anchor) and not number_conflict:
            return True, "text_similarity", text_similarity
        if text_similarity >= 0.72 and common_count >= 4 and shared_anchor and not number_conflict:
            return True, f"text_similarity_anchor_common_{common_count}", text_similarity

    # 최종 후보는 이미 AI가 고른 10개 안쪽이므로, 여기서는 중복 제거를 조금 더 적극적으로 적용한다.
    # 단, 단순히 흔한 단어만 겹쳐서 지워지는 것을 막기 위해 공통 토큰 수와 anchor 공유를 함께 본다.
    if overlap >= 0.48 and common_count >= 4 and shared_anchor and not number_conflict:
        return True, f"token_overlap_common_{common_count}", overlap

    distance = _final_simhash_distance(cand_payload.get("simhash"), kept_payload.get("simhash"))  # distance
    if distance is not None and distance <= 8 and common_count >= 4 and shared_anchor and not number_conflict:
        return True, f"simhash_distance_{distance}", 1.0 - (distance / 64)

    return False, "", max(title_similarity, text_similarity, overlap, title_overlap)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._deduplicate_final_selected_news, news_selector._supplement_final_news_after_dedup
# - 파라미터: candidate: Dict[str, Any], kept_news: List[Dict[str, Any]]
# - 리턴값: Optional[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _find_final_duplicate_info(
    candidate: Dict[str, Any],
    kept_news: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidate["_final_dedup_payload"] = (  # 처리값
        candidate.get("_final_dedup_payload")
        or _build_final_dedup_payload(candidate)
    )

    for kept in kept_news:  # kept
        is_duplicate, method, score = _is_final_duplicate_news(candidate, kept)  # is중복,method,점수
        if is_duplicate:
            return {
                "method": method,
                "score": score,
                "kept_title": kept.get("title", ""),
            }

    return None


# [코드 이해 주석]
# - 역할: OpenAI가 고른 최종 후보 안에서만 코드 규칙으로 중복을 제거한다.
# - 호출하는 곳: news_selector._fallback_select_groups, news_selector.select_important_news_groups
# - 파라미터: news_list: List[Dict[str, Any]]
# - 리턴값: List[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _deduplicate_final_selected_news(news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OpenAI가 고른 최종 후보 안에서만 코드 규칙으로 중복을 제거한다.

    의도:
    - AI가 10개를 골랐더라도 같은 사건이 섞이면 제거한다.
    - 제거 수는 LAST_SELECTION_STATS["final_duplicate_excluded_count"]에 저장해
      이메일 대시보드의 "AI 중복제외"에 표시한다.
    """
    kept_news = []                                                  # 최종유지뉴스목록
    excluded_count = 0                                              # 최종중복제외건수

    for news in news_list or []:                                    # AI선택뉴스
        candidate = dict(news)                                      # 중복검사용뉴스복사본
        candidate["_final_dedup_payload"] = _build_final_dedup_payload(candidate)  # 최종중복판정payload

        duplicate_info = _find_final_duplicate_info(candidate, kept_news)  # 이미유지한뉴스와의중복정보

        if duplicate_info:
            excluded_count += 1  # 처리값
            continue

        candidate.pop("_final_dedup_payload", None)
        kept_news.append(candidate)

    # 이 통계는 main.py가 scrape_stats로 가져가고, 실행 로그에서 AI 선별 후 최종 중복 제거 규모를 보여준다.
    LAST_SELECTION_STATS["selected_before_final_dedup_count"] = len(news_list or [])  # 최종중복제거전선별수
    LAST_SELECTION_STATS["final_duplicate_excluded_count"] = excluded_count           # 최종중복제외건수
    LAST_SELECTION_STATS["selected_after_final_dedup_count"] = len(kept_news)         # 최종중복제거후선별수

    logger.info(
        f"🧹 AI 선별 후 최종 중복 제거 완료: "
        f"{len(news_list or [])}개 → {len(kept_news)}개 "
        f"(중복 제외 {excluded_count}개)"
    )

    return kept_news


# [코드 이해 주석]
# - 역할: 최종 중복 제거 후 limit보다 적으면 남은 후보에서 중복이 아닌 뉴스만 보충한다.
# - 호출하는 곳: news_selector._fallback_select_groups, news_selector.select_important_news_groups
# - 파라미터: selected_news: List[Dict[str, Any]], candidate_news: List[Dict[str, Any]], limit: int, used_group_ids:
# Optional[set] = None
# - 리턴값: List[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _supplement_final_news_after_dedup(
    selected_news: List[Dict[str, Any]],
    candidate_news: List[Dict[str, Any]],
    limit: int,
    used_group_ids: Optional[set] = None,  # used그룹ids
) -> List[Dict[str, Any]]:
    """
    최종 중복 제거 후 limit보다 적으면 남은 후보에서 중복이 아닌 뉴스만 보충한다.
    중복 방지가 우선이므로, 후보가 있어도 같은 사건이면 채우지 않는다.
    """
    final_news = [dict(news) for news in (selected_news or [])]       # 보충전최종뉴스목록
    if len(final_news) >= limit:
        return final_news[:limit]

    used_group_ids = set(used_group_ids or set())                     # 이미사용한그룹ID목록
    for news in final_news:  # 뉴스
        group_id = _safe_text(news.get("group_id"))                  # 유지뉴스그룹ID
        if group_id:
            used_group_ids.add(group_id)

    added_count = 0                                                   # 보충추가건수
    duplicate_skip_count = 0                                          # 보충중중복제외건수

    for news in candidate_news or []:  # 뉴스
        if len(final_news) >= limit:
            break

        group_id = _safe_text(news.get("group_id"))                  # 보충후보그룹ID
        if group_id and group_id in used_group_ids:
            continue

        candidate = dict(news)                                        # 보충후보복사본
        _prepare_selected_news(
            candidate,
            importance_score=candidate.get("importance_score", 3),
        )
        candidate["_final_dedup_payload"] = _build_final_dedup_payload(candidate)  # 처리값
        duplicate_info = _find_final_duplicate_info(candidate, final_news)  # 보충후보중복정보

        if duplicate_info:
            duplicate_skip_count += 1  # 처리값
            continue

        candidate.pop("_final_dedup_payload", None)
        final_news.append(candidate)
        added_count += 1  # 처리값
        if group_id:
            used_group_ids.add(group_id)

    if duplicate_skip_count:
        # 보충 과정에서 빠진 중복도 최종 AI 중복 제외로 합산한다.
        # 그래야 main.py의 scrape_stats와 메일 대시보드가 "AI 선택 이후 실제로 빠진 수"를 놓치지 않는다.
        LAST_SELECTION_STATS["final_duplicate_excluded_count"] = (  # 처리값
            int(LAST_SELECTION_STATS.get("final_duplicate_excluded_count", 0))
            + duplicate_skip_count
        )

    LAST_SELECTION_STATS["selected_after_final_dedup_count"] = len(final_news)  # 보충후최종선별수

    logger.info(
        f"➕ 최종 뉴스 보충 완료: "
        f"{len(selected_news or [])}개 → {len(final_news)}개 / "
        f"추가 {added_count}개 / 보충 중복 제외 {duplicate_skip_count}개"
    )

    return final_news[:limit]


# [코드 이해 주석]
# - 역할: 그룹의 검색 키워드가 대표 기사 제목/설명에 실제로 등장하는지 점수화한다.
# - 호출하는 곳: news_selector._group_sort_key
# - 파라미터: group: Dict[str, Any]
# - 리턴값: float 타입 값을 반환합니다.
# - 프로세스 흐름: 키워드를 어절 단위로 나눕니다 -> 제목/설명 포함 여부를 확인합니다 -> 가점을 합산해 반환합니다.
def _topic_match_bonus(group: Dict[str, Any]) -> float:
    """
    네이버 검색은 본문에 키워드가 스치기만 해도 기사를 돌려주므로,
    키워드가 제목에 없는 그룹은 주제와 무관할 가능성이 높다.
    제목 매치 +6, 설명 매치 +2 가점으로 로컬 정렬에서 주제 밀접 그룹을 앞세운다.
    """
    rep = group.get("representative") or {}  # 대표기사
    title = _safe_text(rep.get("title")).lower()          # 대표기사제목
    description = _safe_text(rep.get("description")).lower()  # 대표기사설명

    if not title and not description:
        return 0.0

    title_hit = False        # 키워드제목매치여부
    description_hit = False  # 키워드설명매치여부

    for keyword in group.get("keywords") or []:  # 검색키워드
        # "정보통신기술 ICT"처럼 복합 키워드는 어절 단위로 나눠 하나라도 맞으면 매치로 본다.
        for word in _safe_text(keyword).lower().split():  # 키워드어절
            if len(word) < 2:
                continue
            if word in title:
                title_hit = True  # 키워드제목매치여부
            elif word in description:
                description_hit = True  # 키워드설명매치여부
        if title_hit:
            break

    return (6.0 if title_hit else 0.0) + (2.0 if description_hit else 0.0)


# [코드 이해 주석]
# - 역할: 그룹 로컬 우선순위 정렬 키를 만든다. 주제 키워드 매치 가점을 포함한다.
# - 호출하는 곳: news_selector._shortlist_groups_for_ai, news_selector._fallback_select_groups
# - 파라미터: group: Dict[str, Any]
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _group_sort_key(group: Dict[str, Any]):
    rep = group.get("representative") or {}  # rep
    return (
        float(group.get("priority_score") or 0) + _topic_match_bonus(group),
        int(group.get("source_count") or 0),
        int(group.get("article_count") or 0),
        _safe_text(rep.get("published_at_kst") or rep.get("published_at")),
    )


# [코드 이해 주석]
# - 역할: 로컬 점수로 넓게 정렬하되, AI에는 설정된 수만 넘긴다.
# - 호출하는 곳: news_selector.select_important_news_groups
# - 파라미터: group_list: List[Dict[str, Any]], final_limit: int, candidate_group_limit: Optional[int] = None
# - 리턴값: List[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _shortlist_groups_for_ai(
    group_list: List[Dict[str, Any]],
    final_limit: int,
    candidate_group_limit: Optional[int] = None,  # 후보그룹상한
) -> List[Dict[str, Any]]:
    """
    로컬 점수로 넓게 정렬하되, AI에는 설정된 수만 넘긴다.
    단순 상위 N개만 자르면 특정 키워드에 쏠릴 수 있어 키워드별 대표와 최신 그룹을 섞는다.
    """
    # 1) 전체 그룹을 로컬 우선순위로 먼저 정렬한다.
    #    이 정렬은 기사 수, 언론사 수, 대표 기사 품질, 최신성을 종합한 1차 후보 순서다.
    sorted_groups = sorted(group_list or [], key=_group_sort_key, reverse=True)  # sorted그룹목록
    if not sorted_groups:
        return []

    # 2) OpenAI에 넘길 그룹 수를 확정한다.
    #    최종 선별 개수(final_limit)보다 작게 자르면 AI가 고를 선택지가 부족하므로 최소 final_limit 이상은 보장한다.
    target = int(candidate_group_limit or DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT)  # target
    target = max(int(final_limit or 1), target)  # target
    target = min(len(sorted_groups), target)  # target

    if len(sorted_groups) <= target:
        return sorted_groups

    selected: List[Dict[str, Any]] = []  # any
    seen_group_ids = set()  # 확인된그룹ids

    # [코드 이해 주석]
    # - 역할: 누적 통계나 그룹에 새 값을 더합니다.
    # - 호출하는 곳: news_selector._shortlist_groups_for_ai
    # - 파라미터: group: Dict[str, Any]
    # - 리턴값: bool 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    def add_group(group: Dict[str, Any]) -> bool:
        if len(selected) >= target:
            return False
        group_id = _safe_text(group.get("group_id"))  # 그룹id
        if not group_id or group_id in seen_group_ids:
            return False
        seen_group_ids.add(group_id)
        selected.append(group)
        return True

    # 3) 상위 로컬 점수 그룹을 먼저 넣어 중요한 사건이 후보 압축 과정에서 빠지지 않게 한다.
    priority_seed_count = min(len(sorted_groups), max(final_limit * 3, target // 2))  # priorityseed건수
    for group in sorted_groups[:priority_seed_count]:  # 그룹
        add_group(group)

    # 4) 키워드별 bucket을 만들어 후보 다양성을 보강한다.
    #    상위 점수만 자르면 특정 키워드/기업 뉴스가 몰릴 수 있어, 키워드마다 일정량을 다시 섞는다.
    keyword_buckets: Dict[str, List[Dict[str, Any]]] = {}  # any
    for group in sorted_groups:  # 그룹
        keywords = group.get("keywords") or ["__unknown__"]  # 키워드목록
        for keyword in keywords:  # 키워드
            keyword = _safe_text(keyword) or "__unknown__"  # 키워드
            keyword_buckets.setdefault(keyword, []).append(group)

    per_keyword_quota = max(2, final_limit // 2)  # 기사별키워드quota
    for keyword in sorted(keyword_buckets):  # 키워드
        added_for_keyword = 0  # 추가for키워드
        for group in keyword_buckets[keyword]:  # 그룹
            if add_group(group):
                added_for_keyword += 1  # 처리값
            if len(selected) >= target or added_for_keyword >= per_keyword_quota:
                break
        if len(selected) >= target:
            break

    # 5) 최신 그룹을 추가로 섞는다.
    #    로컬 점수는 높지 않아도 방금 발생한 사건은 브리핑 가치가 있을 수 있기 때문이다.
    recent_groups = sorted(  # recent그룹목록
        sorted_groups,
        key=lambda group: _safe_text((group.get("representative") or {}).get("published_at_kst")),
        reverse=True,  # reverse
    )
    for group in recent_groups[:max(final_limit * 2, 10)]:  # 그룹
        add_group(group)

    # 6) 그래도 슬롯이 남으면 원래 우선순위 순서대로 채운다.
    #    반환되는 selected는 "AI에게 보여줄 압축 후보"이지 최종 메일 뉴스는 아니다.
    for group in sorted_groups:  # 그룹
        if not add_group(group) and len(selected) >= target:
            break

    return selected[:target]


# [코드 이해 주석]
# - 역할: Python 그룹화 결과를 OpenAI가 읽기 쉬운 짧은 후보 목록으로 변환한다.
# - 호출하는 곳: news_selector.select_important_news_groups
# - 파라미터: group_list: List[Dict[str, Any]]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 내부용 문자열/dict 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def _build_group_candidate_text(group_list: List[Dict[str, Any]]) -> str:
    """
    Python 그룹화 결과를 OpenAI가 읽기 쉬운 짧은 후보 목록으로 변환한다.
    기사 원문 전체가 아니라 그룹 대표 정보와 그룹 통계만 전달해 토큰을 줄인다.
    """
    lines = []  # lines
    # 1) 각 그룹을 한 블록짜리 프롬프트 후보로 압축한다.
    #    n/src/score/flags는 AI가 "많이 보도된 사건인지, 품질 경고가 있는지" 판단하는 신호다.
    for idx, group in enumerate(group_list or [], 1):  # 순번,그룹
        rep = group.get("representative") or {}  # rep
        group_id = _safe_text(group.get("group_id") or f"G{idx:03d}")  # 그룹id
        title = _clip_text(rep.get("title"), GROUP_CANDIDATE_TITLE_CHARS)  # 제목
        description = ""  # 설명
        if idx <= SELECTOR_DETAILED_CANDIDATE_COUNT:
            description = _clip_text(rep.get("description"), GROUP_CANDIDATE_DESCRIPTION_CHARS)  # 설명
        source = _clip_text(rep.get("source"), 30)  # 출처
        sources = ", ".join(group.get("sources") or [])[:GROUP_CANDIDATE_SOURCES_CHARS]  # 출처목록
        keywords = ", ".join(group.get("keywords") or [])[:GROUP_CANDIDATE_KEYWORDS_CHARS]  # 키워드목록
        article_count = int(group.get("article_count") or 1)  # 기사건수
        source_count = int(group.get("source_count") or 1)  # 출처건수
        priority_score = _safe_text(group.get("priority_score"))  # priority점수
        quality_flags = ",".join(group.get("quality_flags") or []) or "-"  # qualityflags
        description_part = f" | desc={description}" if description else ""  # 설명part

        lines.append(
            f"[{idx}] id={group_id} | n={article_count} src={source_count} score={priority_score} "
            f"flags={quality_flags} | press={sources or source} | kw={keywords} | title={title}"
            f"{description_part}"
        )
    return "\n\n".join(lines)


# [코드 이해 주석]
# - 역할: AI 점수가 없는 fallback/보충 후보의 중요도를 로컬 그룹 신호로 추정한다.
# - 호출하는 곳: news_selector._estimate_importance_score_from_news, news_selector._fallback_select_groups,
# news_selector._representative_news_from_group, news_selector.select_important_news_groups
# - 파라미터: group: Dict[str, Any]
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _estimate_importance_score_from_group(group: Dict[str, Any]) -> int:
    """
    AI 점수가 없는 fallback/보충 후보의 중요도를 로컬 그룹 신호로 추정한다.
    중복 보충 과정에서 모든 뉴스가 3점으로 표시되는 것을 막기 위한 안전망이다.
    """
    try:
        priority_score = float(group.get("priority_score") or 0)  # priority점수
    except Exception:
        priority_score = 0.0  # priority점수

    article_count = int(group.get("article_count") or 1)  # 기사건수
    source_count = int(group.get("source_count") or 1)  # 출처건수
    flags = set(group.get("quality_flags") or [])  # flags

    if "low_representative_score" in flags or "photo_like_representative" in flags:
        return 2

    if priority_score >= 24 or (source_count >= 5 and article_count >= 8):
        return 5

    if source_count >= 3 or article_count >= 4 or priority_score >= 13:
        return 4

    if priority_score < 4:
        return 2

    return 3


# [코드 이해 주석]
# - 역할: 그룹 dict가 아닌 대표 기사 dict만 있을 때의 fallback 중요도 추정.
# - 호출하는 곳: news_selector._fallback_select_groups, news_selector.select_important_news_groups
# - 파라미터: news: Dict[str, Any]
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _estimate_importance_score_from_news(news: Dict[str, Any]) -> int:
    """
    그룹 dict가 아닌 대표 기사 dict만 있을 때의 fallback 중요도 추정.
    """
    if news.get("importance_score") not in (None, ""):
        return _safe_int(news.get("importance_score"), default=3)

    try:
        priority_score = float(news.get("group_priority_score") or 0)  # priority점수
    except Exception:
        priority_score = 0.0  # priority점수

    article_count = int(news.get("group_article_count") or 1)  # 기사건수
    source_count = int(news.get("group_source_count") or 1)  # 출처건수
    flags = set(news.get("group_quality_flags") or [])  # flags

    return _estimate_importance_score_from_group({
        "priority_score": priority_score,
        "article_count": article_count,
        "source_count": source_count,
        "quality_flags": list(flags),
    })


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_selector._fallback_select_groups, news_selector.select_important_news_groups
# - 파라미터: group: Dict[str, Any]
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _representative_news_from_group(group: Dict[str, Any]) -> Dict[str, Any]:
    # 그룹 선별의 출력은 "그룹"이지만, 요약/메일 단계는 여전히 "뉴스 dict" 하나를 기대한다.
    # 따라서 대표 기사 dict를 복사하고 group_* 메타를 붙여, 대표 뉴스 1건 안에 관련보도 정보까지 싣는다.
    rep = dict(group.get("representative") or {})                 # 대표기사dict복사본
    articles = group.get("articles") or []                        # 그룹소속기사목록
    related_articles = [                                          # 관련보도표시대상기사목록
        article
        for article in articles[:12]  # 기사
        if _safe_text(article.get("title")) and _safe_text(article.get("url"))
    ]

    # 그룹화 결과의 대표 기사에는 published_at_kst만 있는 경우가 있다.
    # 이후 요약/메일 단계는 published_at을 우선 사용하므로 여기서 호환 필드를 보강한다.
    if not rep.get("published_at") and rep.get("published_at_kst"):
        rep["published_at"] = rep.get("published_at_kst")  # 처리값
    if not rep.get("published_at_kst") and rep.get("published_at"):
        rep["published_at_kst"] = rep.get("published_at")  # 처리값

    # group_* 필드는 summarizer가 결과에 보존하고, email_sender/related_pages가 관련보도 건수와 상세 링크를 만들 때 읽는다.
    rep["group_id"] = group.get("group_id")                       # 사건그룹ID
    rep["group_article_count"] = group.get("article_count", 1)    # 그룹기사수
    rep["group_source_count"] = group.get("source_count", 1)      # 그룹언론사수
    rep["group_sources"] = group.get("sources", [])               # 그룹언론사목록
    rep["group_keywords"] = group.get("keywords", [])             # 그룹키워드목록
    rep["group_quality_flags"] = group.get("quality_flags", [])   # 그룹품질플래그
    rep["group_priority_score"] = group.get("priority_score", 0)  # 그룹로컬우선순위점수
    rep["group_article_titles"] = [  # 처리값
        _safe_text(article.get("title"))
        for article in related_articles  # 기사
    ]                                                             # 관련보도제목목록
    rep["group_article_urls"] = [  # 처리값
        _safe_text(article.get("url"))
        for article in related_articles  # 기사
    ]                                                             # 관련보도URL목록
    rep["group_article_sources"] = [  # 처리값
        _safe_text(article.get("source"))
        for article in related_articles  # 기사
    ]                                                             # 관련보도언론사목록
    rep["local_importance_score"] = _estimate_importance_score_from_group(group)  # 로컬추정중요도
    if rep.get("importance_score") in (None, ""):
        rep["importance_score"] = rep["local_importance_score"]  # 처리값
    rep["content"] = rep.get("description", "")  # 처리값
    return rep


# [코드 이해 주석]
# - 역할: 그룹 단위 OpenAI 선별 실패 시 로컬 우선순위 순서대로 대표 기사 사용.
# - 호출하는 곳: news_selector.select_important_news_groups
# - 파라미터: group_list: List[Dict[str, Any]], fallback_news_list: List[Dict[str, Any]], limit: int
# - 리턴값: List[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _fallback_select_groups(
    group_list: List[Dict[str, Any]],
    fallback_news_list: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """
    그룹 단위 OpenAI 선별 실패 시 로컬 우선순위 순서대로 대표 기사 사용.
    """
    selected_news = []  # 선별뉴스

    if group_list:
        sorted_groups = sorted(  # sorted그룹목록
            group_list,
            key=_group_sort_key,  # 키
            reverse=True,  # reverse
        )
        candidate_news = []  # 후보뉴스
        for group in sorted_groups:  # 그룹
            news = _representative_news_from_group(group)  # 뉴스
            _prepare_selected_news(
                news,
                importance_score=_estimate_importance_score_from_group(group),  # 중요도점수
            )
            candidate_news.append(news)

        selected_news = _deduplicate_final_selected_news(candidate_news[:limit])  # 선별뉴스
        return _supplement_final_news_after_dedup(
            selected_news=selected_news,  # 선별뉴스
            candidate_news=candidate_news,  # 후보뉴스
            limit=limit,  # 상한
        )

    candidate_news = []  # 후보뉴스
    for news in fallback_news_list or []:  # 뉴스
        news = dict(news)  # 뉴스
        _prepare_selected_news(
            news,
            importance_score=_estimate_importance_score_from_news(news),  # 중요도점수
        )
        candidate_news.append(news)

    selected_news = _deduplicate_final_selected_news(candidate_news[:limit])  # 선별뉴스
    return _supplement_final_news_after_dedup(
        selected_news=selected_news,  # 선별뉴스
        candidate_news=candidate_news,  # 후보뉴스
        limit=limit,  # 상한
    )


# [코드 이해 주석]
# - 역할: Python 규칙 기반으로 묶인 사건 그룹 중 OpenAI가 중요한 그룹만 선택한다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: group_list: List[Dict[str, Any]], fallback_news_list: List[Dict[str, Any]], topic_name: str,
# topic_description: str, limit: int = 10, candidate_group_limit: Optional[int] = None
# - 리턴값: List[Dict] 타입 값을 반환합니다.
# - 프로세스 흐름: 후보를 압축해 입력 텍스트를 만듭니다 -> AI/규칙으로 고릅니다 -> 중복 제거와 보충을 수행합니다.
def select_important_news_groups(
    group_list: List[Dict[str, Any]],
    fallback_news_list: List[Dict[str, Any]],
    topic_name: str,
    topic_description: str,
    limit: int = 10,  # 상한
    candidate_group_limit: Optional[int] = None,  # 후보그룹상한
    min_importance_score: int = 3,  # 최소중요도점수
    fill_to_limit: bool = False  # 부족분강제보충여부
) -> List[Dict]:
    """
    Python 규칙 기반으로 묶인 사건 그룹 중 OpenAI가 중요한 그룹만 선택한다.

    기존 기사 단위 선별과 달리:
    - 입력은 기사 전체가 아니라 사건 그룹 대표 정보다.
    - 같은 사건 중복 제거는 이미 Python 그룹화 단계에서 수행한다.
    - 출력은 group_id 기준으로 받는다.

    관련성 정책:
    - min_importance_score 미만으로 평가된 그룹은 메일에 싣지 않는다.
    - fill_to_limit=False(기본)면 AI가 limit보다 적게 골라도 그대로 존중한다.
      AI가 안 뽑은 그룹을 로컬 점수 순으로 다시 채우면 관련성 낮은 뉴스가
      메일에 섞이는 주원인이 되므로, 강제 보충은 명시적으로 켠 경우에만 한다.
    """
    reset_selection_stats()

    # 1) 입력 후보 상태를 확인한다.
    #    group_list는 Python 그룹화가 만든 사건 단위 후보이고,
    #    fallback_news_list는 OpenAI 호출 실패나 보충 상황에서 사용할 대표 기사 후보다.
    if not group_list and not fallback_news_list:
        logger.warning("선택할 뉴스 그룹 후보가 없습니다.")
        return []

    logger.info(
        f"🧠 [{topic_name}] 그룹 선별 시작: 후보 {len(group_list or [])}개 / 목표 {limit}개"
    )

    # 2) OpenAI 클라이언트가 없으면 로컬 우선순위 기반 fallback으로 전환한다.
    #    운영 환경변수 문제로 브리핑 전체가 실패하는 것보다, 품질은 낮아도 발송 가능한 결과를 만드는 쪽을 선택한다.
    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 그룹 fallback 선별을 사용합니다.")
        return _fallback_select_groups(group_list, fallback_news_list, limit)

    # 3) 전체 그룹을 그대로 보내지 않고 candidate_group_limit까지 압축한다.
    #    토큰을 줄이면서도 로컬 점수, 키워드 분산, 최신성을 섞어 AI 선택지가 한쪽으로 치우치지 않게 한다.
    prepared_groups = _shortlist_groups_for_ai(                          # AI전달압축그룹목록
        group_list=group_list or [],  # 그룹list
        final_limit=limit,  # 최종상한
        candidate_group_limit=candidate_group_limit,  # 후보그룹상한
    )

    if not prepared_groups:
        logger.warning("⚠️ OpenAI에 전달할 그룹이 없어 fallback 선별을 사용합니다.")
        return _fallback_select_groups(group_list, fallback_news_list, limit)

    # 4) OpenAI가 읽을 후보 텍스트를 만든다.
    #    여기에는 기사 전문이 아니라 그룹 id, 기사 수, 언론사 수, 대표 제목/설명만 들어간다.
    group_text = _build_group_candidate_text(prepared_groups)            # AI프롬프트후보텍스트
    selection_limit = min(len(prepared_groups), max(limit, 1))           # AI가선택할최대그룹수
    completion_limit = SELECTOR_MAX_COMPLETION_TOKENS if is_gpt5_model(SELECTOR_MODEL) else min(900, SELECTOR_MAX_COMPLETION_TOKENS)  # 선별응답토큰상한

    logger.info(
        "🧺 그룹 선별 후보 압축: 전체 %s개 → AI 전달 %s개 "
        "(로컬점수+키워드분산+최신 혼합, candidate_group_limit=%s, completion_limit=%s)",
        len(group_list or []),
        len(prepared_groups),
        candidate_group_limit or DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT,
        completion_limit,
    )

    prompt = f"""
작업: "{topic_name}" 브리핑에 넣을 뉴스 그룹을 고릅니다.

주제:
{topic_description}

후보 필드:
- id: 선택할 때 반드시 사용할 group_id
- n: 같은 사건으로 묶인 기사 수
- src: 서로 다른 언론사 수
- score: 로컬 우선순위 점수
- flags: 품질 경고
- title/desc: 대표 기사 제목/짧은 설명

선택 기준:
1. 위 주제와 직접 관련 있는 사건만 선택합니다. 키워드만 스치듯 언급되고 사건 자체는 주제와 무관하면 선택하지 마세요.
2. 주제와 직접 관련이 확실한 사건 중 정책/규제/실적/투자/계약/시장 변화, src·n·score가 높은 사건을 우선합니다.
3. 최대 {selection_limit}개까지 서로 다른 사건으로 고르되, 확신이 없는 후보로 개수를 채우지 마세요. 적합한 사건이 적으면 적게 고르는 것이 정답입니다.
4. 홍보성 기사, 단순 행사/수상/기념, photo_like_representative, low_representative_score, overgroup_risk_token_time_span 플래그는 제외합니다.
5. 같은 group_id를 두 번 쓰지 말고, 후보에 없는 group_id를 만들지 마세요.

importance_score:
- 5: 주제 핵심이며 영향이 큰 정책/규제/실적/투자/대형계약/시장 변화
- 4: 주요 기업/기관의 전략, 서비스, 기술, 제휴, 수급 변화
- 3: 주제와 직접 관련 있는 일반 뉴스
주제와의 관련성이 3점 기준에 못 미치는 후보는 selected에 아예 넣지 마세요.
모든 항목을 기계적으로 3점으로 주지 말고 후보 신호에 따라 차등화하세요.

출력은 JSON 객체 하나만:
{{"selected":[{{"group_id":"G001","importance_score":5}}]}}

후보:
{group_text}
"""

    try:
        # 5) AI는 group_id와 importance_score만 돌려준다.
        #    실제 뉴스 dict 복원은 아래에서 group_id를 기준으로 하므로, 후보에 없는 id를 만드는 응답은 무시한다.
        response = create_chat_completion(  # 응답
            client,
            logger,
            model=SELECTOR_MODEL,  # 모델
            messages=[  # messages
                {
                    "role": "system",
                    "content": (
                        "뉴스 편집 데스크입니다. 후보 group_id 중에서만 선택하고, "
                        "중복 사건을 피하며, 중요도 점수를 1~5로 차등 부여합니다. "
                        "응답은 JSON 객체 하나만 출력합니다."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            **openai_temperature_kwargs(SELECTOR_MODEL, 0.1),
            **openai_reasoning_effort_kwargs(SELECTOR_MODEL),
            **openai_json_response_format_kwargs(),
            **openai_token_limit_kwargs(SELECTOR_MODEL, completion_limit)
        )

        content = response.choices[0].message.content.strip()  # 본문
        usage_info = record_openai_usage(  # usageinfo
            logger,
            "그룹 단위 뉴스 선별",
            SELECTOR_MODEL,
            response.usage,
        )
        tokens_used = usage_info["total_tokens"]                         # 그룹선별사용토큰수
        add_selection_tokens("selection_tokens", tokens_used)
        logger.debug(f"🧾 그룹 단위 뉴스 선별 토큰 사용량: {tokens_used}")

        try:
            result = _ensure_json_object(_extract_json(content))  # 결과
        except Exception:
            logger.error("❌ OpenAI 그룹 선별 응답 JSON 파싱 실패")
            logger.error("응답 미리보기: %s", content[:300])
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        selected_items = _ensure_json_list(result.get("selected"))       # AI선택그룹ID목록
        if not selected_items:
            logger.warning("⚠️ OpenAI가 선택한 그룹이 없습니다. fallback을 사용합니다.")
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        # 6) AI 응답을 검증하며 대표 뉴스 dict로 복원한다.
        #    중복 group_id, 후보에 없는 group_id, dict가 아닌 항목은 모두 버려 최종 메일 데이터가 깨지지 않게 한다.
        group_by_id = {str(group.get("group_id")): group for group in prepared_groups}  # group_id별후보그룹
        selected_news = []                                             # AI선택복원뉴스목록
        used_group_ids = set()                                         # 중복선택방지그룹ID

        invalid_item_count = 0                                         # 무시한AI응답항목수
        low_importance_count = 0                                       # 중요도미달제외건수
        for item in selected_items:  # 항목
            if not isinstance(item, dict):
                invalid_item_count += 1  # 처리값
                continue

            group_id = _safe_text(item.get("group_id"))                # AI가선택한그룹ID
            if not group_id or group_id in used_group_ids:
                invalid_item_count += 1  # 처리값
                continue

            group = group_by_id.get(group_id)                          # 선택그룹원본데이터
            if not group:
                invalid_item_count += 1  # 처리값
                continue

            importance_score = _safe_int(item.get("importance_score", 3))  # AI부여중요도점수
            # 프롬프트가 3점 미만은 선택하지 말라고 지시하지만,
            # 모델이 그래도 낮은 점수를 붙여 보내면 여기서 한 번 더 걸러 관련성 낮은 뉴스가 메일에 실리지 않게 한다.
            if importance_score < int(min_importance_score or 0):
                low_importance_count += 1  # 처리값
                continue

            news = _representative_news_from_group(group)               # 대표기사뉴스dict
            _prepare_selected_news(
                news=news,  # 뉴스
                importance_score=importance_score
            )
            selected_news.append(news)
            used_group_ids.add(group_id)

            if len(selected_news) >= limit:
                break

        if invalid_item_count:
            logger.debug("그룹 선별 응답 무시 항목: %s개", invalid_item_count)
        if low_importance_count:
            logger.info(
                "🧹 중요도 %s점 미만 그룹 제외: %s개",
                min_importance_score,
                low_importance_count,
            )

        if not selected_news:
            # AI가 응답은 했지만 전부 중요도 미달로 걸러진 경우는 "이번 시간대에 실을 뉴스가 없다"는 판단이므로
            # fallback으로 관련성 낮은 뉴스를 다시 채우지 않고 빈 결과를 존중한다.
            if low_importance_count:
                logger.info("🧹 모든 AI 선택 그룹이 중요도 미달로 제외되어 이번 섹션은 빈 결과를 반환합니다.")
                return []
            logger.warning("⚠️ 유효하게 선택된 그룹이 없습니다. fallback을 사용합니다.")
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        # 7) AI가 고른 결과 안에서도 한 번 더 최종 중복 제거를 한다.
        #    Python 그룹화가 놓친 유사 사건이나 AI가 중복 성격의 그룹을 같이 고른 경우를 마지막으로 걸러낸다.
        before_final_dedup_count = len(selected_news)                   # 최종중복제거전선별수
        selected_news = _deduplicate_final_selected_news(selected_news)  # 선별뉴스

        # 8) fill_to_limit=True일 때만, 준비된 그룹과 fallback 후보에서 부족분을 보충한다.
        #    기본(False)에서는 AI가 "이 정도만 관련 있다"고 판단한 결과를 그대로 존중한다.
        #    과거에는 항상 limit까지 채웠는데, 이때 AI가 뽑지 않은 그룹이 로컬 점수 순으로
        #    다시 들어와 관련성 낮은 뉴스가 메일에 섞이는 주원인이었다.
        if fill_to_limit and len(selected_news) < limit:
            supplement_candidates = []                                  # 부족분보충후보뉴스목록

            for group in prepared_groups:  # 그룹
                news = _representative_news_from_group(group)  # 뉴스
                _prepare_selected_news(
                    news,
                    importance_score=_estimate_importance_score_from_group(group),  # 중요도점수
                )
                supplement_candidates.append(news)

            for news in fallback_news_list or []:  # 뉴스
                news = dict(news)  # 뉴스
                _prepare_selected_news(
                    news,
                    importance_score=_estimate_importance_score_from_news(news),  # 중요도점수
                )
                supplement_candidates.append(news)

            selected_news = _supplement_final_news_after_dedup(  # 선별뉴스
                selected_news=selected_news,  # 선별뉴스
                candidate_news=supplement_candidates,  # 후보뉴스
                limit=limit,  # 상한
                used_group_ids=used_group_ids,  # used그룹ids
            )

        logger.info(
            f"✅ 그룹 단위 뉴스 선별 완료: "
            f"AI 선택 {before_final_dedup_count}개 → 최종 {len(selected_news)}개 "
            f"(중복 제외 {LAST_SELECTION_STATS.get('final_duplicate_excluded_count', 0)}개)"
        )
        return selected_news

    except Exception as e:  # 예외객체
        logger.error(f"❌ OpenAI 그룹 단위 뉴스 선별 실패: {e}")
        return _fallback_select_groups(group_list, fallback_news_list, limit)
