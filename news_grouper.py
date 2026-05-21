# =============================================================================
# [파일 설명]
# - 수행 기능: 수집된 뉴스 후보를 제목, 본문, 토큰, URL, 시간 정보를 기준으로 같은 사건 그룹으로 묶습니다.
# - 프로세스: 기사 payload 생성 -> 유사도/토큰/시간 비교 -> 그룹 대표 기사 선정 -> 품질 플래그/우선순위 계산 -> 직렬화
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 뉴스 dict 목록과 그룹화 임계값 설정
# - 리턴값/출력: 그룹 목록, 대표 기사 후보, 그룹 품질/우선순위 통계를 포함한 dict를 반환합니다.
# =============================================================================

"""
뉴스 후보 그룹화 모듈

역할:
- 네이버 뉴스 수집 후보를 같은 사건 단위로 규칙 기반 그룹화한다.
- 사진/속보/짧은 기사보다 요약하기 좋은 대표 기사를 고른다.
- OpenAI 선별 단계에는 기사 전체가 아니라 그룹 대표 정보만 넘기도록 돕는다.

주의:
- 외부 AI 호출 없음.
- 운영 파일에서 import해서 사용한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse



# ====================================
# 로깅 설정
# ====================================
logging.basicConfig(
    level=logging.INFO,  # level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # handlers
)
logger = logging.getLogger(__name__)  # 모듈로거


# ====================================
# 기본값
# ====================================
DEFAULT_DISPLAY_PER_KEYWORD = 100                              # 테스트용키워드별수집수
DEFAULT_PAGES_PER_KEYWORD = 3                                  # 테스트용키워드별페이지수
DEFAULT_SORTS = ["sim"]                                        # 테스트용검색정렬방식
DEFAULT_RECENT_HOURS = 24                                      # 테스트용최근뉴스시간범위
DEFAULT_MAX_TOTAL_NEWS = 300                                   # 테스트용최대후보수
DEFAULT_OUTPUT_PATH = "data/grouping_test/news_grouping_result.json"  # 테스트결과저장경로

# 그룹화 기준값. 테스트하면서 조정하기 쉽게 CLI 인자로도 받는다.
DEFAULT_TITLE_SIMILARITY_THRESHOLD = 0.78                      # 제목유사도그룹화기준
DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.70                       # 본문유사도그룹화기준
DEFAULT_TOKEN_OVERLAP_THRESHOLD = 0.65                         # 토큰겹침그룹화기준
DEFAULT_SIMHASH_DISTANCE_THRESHOLD = 6                         # SimHash거리그룹화기준
DEFAULT_MIN_COMMON_TOKEN_COUNT = 5                             # 최소공통토큰수

# 너무 흔해서 사건 구분에 도움이 약한 단어들
STOPWORDS = {  # STOPWORDS
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
    "한국", "국내", "글로벌", "최신", "주요", "확인", "가능", "기준",
}

TOKEN_SUFFIXES = (  # 토큰정규화제거접미사
    "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
    "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
)

# 대표 기사로 쓰기 애매한 패턴. 그룹에는 남기되 대표 선정 점수에서 감점한다.
PHOTO_URL_PATTERNS = ("/photos/", "/photo/", "NISI")  # 사진기사URL패턴
PHOTO_TITLE_PATTERNS = (  # 사진기사제목패턴
    "참석하는", "주재하는", "발언하는", "모두 발언", "모두발언", "기념촬영",
    "손드는", "이동하는", "입장하는", "대화하는", "자료 살펴보", "브리핑하는",
)
BREAKING_TITLE_PATTERNS = ("[속보]", "[1보]", "[2보]", "[외환]", "[달러·원]")  # 속보성제목패턴



# ====================================
# 텍스트/URL 정규화 유틸
# ====================================
# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.normalize_compare_text, news_grouper.normalize_title
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def strip_html_entities(text: str) -> str:
    if text is None:
        return ""
    text = str(text)  # 텍스트
    text = re.sub(r"<.*?>", " ", text)  # 텍스트
    text = text.replace("&quot;", '"').replace("&amp;", "&")  # 텍스트
    text = text.replace("&lt;", "<").replace("&gt;", ">")  # 텍스트
    text = text.replace("&#39;", "'")  # 텍스트
    text = text.replace("…", " ").replace("...", " ")  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: title: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = strip_html_entities(title).lower().strip()  # 텍스트
    text = re.sub(r"\[[^\]]*\]|【[^】]*】|\([^)]*\)", " ", text)  # 텍스트
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)  # 텍스트
    text = re.sub(r"\s+", "", text)  # 텍스트
    text = re.sub(r"[^0-9a-z가-힣]", "", text)  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: news_grouper.compact_compare_text, news_grouper.extract_tokens
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_compare_text(text: str) -> str:
    if not text:
        return ""
    text = strip_html_entities(text).lower()  # 텍스트
    text = re.sub(r"\[[^\]]*\]|【[^】]*】", " ", text)  # 텍스트
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)  # 텍스트
    text = re.sub(r"\b\w+@\w+(?:\.\w+)+\b", " ", text)  # 텍스트
    text = re.sub(r"[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일", " ", text)  # 텍스트
    text = re.sub(r"\s+", " ", text)  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def compact_compare_text(text: str) -> str:
    text = normalize_compare_text(text)  # 텍스트
    text = re.sub(r"\s+", "", text)  # 텍스트
    return text


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: news_grouper.extract_tokens
# - 파라미터: token: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_token(token: str) -> str:
    token = str(token or "").lower().strip()  # 토큰
    if not token:
        return ""

    token = token.replace("美", "미국").replace("韓", "한국").replace("中", "중국")  # 토큰
    token = token.replace("日", "일본").replace("李", "이").replace("金", "김")  # 토큰

    for suffix in TOKEN_SUFFIXES:  # suffix
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]  # 토큰
            break

    return token.strip()


# [코드 이해 주석]
# - 역할: 입력 데이터에서 필요한 토큰, URL, 날짜, 사용량 같은 핵심 값을 추출합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: text: str, max_tokens: int = 80
# - 리턴값: List[str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_tokens(text: str, max_tokens: int = 80) -> List[str]:
    text = normalize_compare_text(text)  # 텍스트
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)  # 원본토큰수

    tokens: List[str] = []  # 토큰수
    seen = set()  # 확인된
    for token in raw_tokens:  # 토큰
        token = normalize_token(token)  # 토큰
        if not token or token in STOPWORDS or token.isdigit() or len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: url: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())  # parsed
        domain = parsed.netloc.lower().strip()  # 도메인
        path = parsed.path.strip()  # 경로
        if domain.startswith("www."):
            domain = domain[4:]  # 도메인
        return f"{domain}{path}".rstrip("/")
    except Exception:
        return str(url).lower().strip().rstrip("/")


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.make_simhash
# - 파라미터: token: str
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def stable_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # digest
    return int(digest[:16], 16)


# [코드 이해 주석]
# - 역할: 여러 입력 값을 조합해 식별자, 해시, 키 같은 파생 값을 만듭니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: tokens: Iterable[str]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_simhash(tokens: Iterable[str]) -> str:
    tokens = list(tokens or [])  # 토큰수
    if not tokens:
        return ""

    vector = [0] * 64  # vector
    for token in tokens:  # 토큰
        value = stable_token_hash(token)  # 값
        for bit in range(64):  # bit
            if value & (1 << bit):
                vector[bit] += 1  # vector
            else:
                vector[bit] -= 1  # vector

    fingerprint = 0  # fingerprint
    for bit, score in enumerate(vector):  # 비트,점수
        if score >= 0:
            fingerprint |= 1 << bit  # 처리값

    return f"{fingerprint:016x}"


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.compare_payloads
# - 파라미터: a: str, b: str
# - 리턴값: Optional[int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def simhash_distance(a: str, b: str) -> Optional[int]:
    if not a or not b:
        return None
    try:
        x = int(str(a), 16) ^ int(str(b), 16)  # x
        return x.bit_count()
    except Exception:
        return None


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.compare_payloads
# - 파라미터: tokens_a: Iterable[str], tokens_b: Iterable[str]
# - 리턴값: Tuple[float, int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def token_overlap_score(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> Tuple[float, int]:
    set_a = set(tokens_a or [])  # seta
    set_b = set(tokens_b or [])  # setb
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b  # common
    denominator = max(1, min(len(set_a), len(set_b)))  # denominator
    return len(common) / denominator, len(common)


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: news: Dict[str, Any]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_title(news: Dict[str, Any]) -> str:
    return str(news.get("title") or "").strip()


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: news: Dict[str, Any]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_description(news: Dict[str, Any]) -> str:
    return str(news.get("description") or news.get("summary") or news.get("content") or "").strip()


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: news: Dict[str, Any]
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_url(news: Dict[str, Any]) -> str:
    return str(news.get("originallink") or news.get("url") or news.get("link") or "").strip()


# [코드 이해 주석]
# - 역할: 로그/정렬용 문자열. 실패하면 빈 문자열.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: value: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 문자열/설정을 읽습니다 -> 가능한 형식으로 변환을 시도합니다 -> 실패 시 안전한 기본값을 반환합니다.
def parse_iso_datetime(value: str) -> str:
    """로그/정렬용 문자열. 실패하면 빈 문자열."""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except Exception:
        return str(value)


# [코드 이해 주석]
# - 역할: 문자열이나 설정 값을 프로그램에서 다루기 쉬운 값으로 파싱합니다.
# - 호출하는 곳: news_grouper.group_time_span_hours
# - 파라미터: value: str
# - 리턴값: Optional[datetime] 타입 값을 반환합니다.
# - 프로세스 흐름: 문자열/설정을 읽습니다 -> 가능한 형식으로 변환을 시도합니다 -> 실패 시 안전한 기본값을 반환합니다.
def parse_dt_for_compare(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


# [코드 이해 주석]
# - 역할: 입력값이 특정 조건을 만족하는지 bool로 판정합니다.
# - 호출하는 곳: news_grouper.representative_quality_score
# - 파라미터: payload: 'NewsPayload'
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_photo_like(payload: "NewsPayload") -> bool:
    title = payload.title or ""  # 제목
    url = payload.url or ""  # URL
    if any(pattern in url for pattern in PHOTO_URL_PATTERNS):
        return True
    if any(pattern in title for pattern in PHOTO_TITLE_PATTERNS):
        return True
    return False


# [코드 이해 주석]
# - 역할: 입력값이 특정 조건을 만족하는지 bool로 판정합니다.
# - 호출하는 곳: news_grouper.representative_quality_score
# - 파라미터: payload: 'NewsPayload'
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_breaking_like(payload: "NewsPayload") -> bool:
    title = payload.title or ""  # 제목
    return any(pattern in title for pattern in BREAKING_TITLE_PATTERNS)


# [코드 이해 주석]
# - 역할: 그룹 대표 기사 후보 점수.
# - 호출하는 곳: news_grouper.group_quality_flags, news_grouper.refresh_group_representative, news_grouper.serialize_group
# - 파라미터: payload: 'NewsPayload', group: Optional['NewsGroup'] = None
# - 리턴값: Tuple[float, List[str]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def representative_quality_score(payload: "NewsPayload", group: Optional["NewsGroup"] = None) -> Tuple[float, List[str]]:
    """
    그룹 대표 기사 후보 점수.
    높은 점수일수록 메일 대표 기사로 쓰기 좋다.
    """
    score = 0.0  # 점수
    flags: List[str] = []  # flags

    title_len = len(payload.title or "")  # 제목len
    desc_len = len(payload.description or "")  # desclen

    score += min(title_len, 70) / 10.0  # 처리값
    score += min(desc_len, 220) / 35.0  # 처리값

    if payload.normalized_url:
        score += 1.0  # 처리값
    if payload.source:
        score += 0.5  # 처리값

    if group is not None:
        score += min(group.source_count, 6) * 0.4  # 처리값
        score += min(group.article_count, 8) * 0.2  # 처리값

    if is_photo_like(payload):
        score -= 6.0  # 처리값
        flags.append("photo_like")

    if is_breaking_like(payload):
        score -= 2.0  # 처리값
        flags.append("breaking_like")

    if desc_len == 0:
        score -= 3.0  # 처리값
        flags.append("empty_description")
    elif desc_len < 40:
        score -= 1.5  # 처리값
        flags.append("short_description")

    if title_len < 16:
        score -= 2.0  # 처리값
        flags.append("short_title")

    if len(payload.tokens) < 5:
        score -= 1.5  # 처리값
        flags.append("few_tokens")

    return score, flags


# [코드 이해 주석]
# - 역할: 그룹 안에서 메일 대표 기사로 가장 좋은 후보를 다시 고른다.
# - 호출하는 곳: news_grouper.group_news
# - 파라미터: group: 'NewsGroup'
# - 리턴값: None 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def refresh_group_representative(group: "NewsGroup") -> None:
    """그룹 안에서 메일 대표 기사로 가장 좋은 후보를 다시 고른다."""
    if not group.items:
        return
    scored = []  # scored
    for item in group.items:  # 항목
        score, flags = representative_quality_score(item, group)  # 점수,flags
        scored.append((score, item.published_at_kst or "", item, flags))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    group.representative = scored[0][2]  # 처리값


# [코드 이해 주석]
# - 역할: 여러 뉴스 항목을 사건 단위 그룹이나 그룹 통계로 처리합니다.
# - 호출하는 곳: news_grouper.group_quality_flags, news_grouper.serialize_group
# - 파라미터: group: 'NewsGroup'
# - 리턴값: float 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def group_time_span_hours(group: "NewsGroup") -> float:
    dates = [parse_dt_for_compare(item.published_at_kst) for item in group.items]  # dates
    dates = [d for d in dates if d is not None]  # dates
    if len(dates) < 2:
        return 0.0
    return (max(dates) - min(dates)).total_seconds() / 3600.0


# [코드 이해 주석]
# - 역할: 여러 뉴스 항목을 사건 단위 그룹이나 그룹 통계로 처리합니다.
# - 호출하는 곳: news_grouper.serialize_group
# - 파라미터: group: 'NewsGroup'
# - 리턴값: List[str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def group_quality_flags(group: "NewsGroup") -> List[str]:
    flags: List[str] = []  # flags
    method_counts = Counter(reason.get("method", "unknown") for reason in group.match_reasons)  # method건수목록
    non_rep_count = max(group.article_count - 1, 1)  # non대표기사건수
    token_ratio = method_counts.get("token_overlap", 0) / non_rep_count  # 토큰ratio
    span_hours = group_time_span_hours(group)  # spanhours
    rep_score, rep_flags = representative_quality_score(group.representative, group)  # 대표기사품질점수,대표기사품질플래그

    if token_ratio >= 0.6 and span_hours >= 6:
        flags.append("overgroup_risk_token_time_span")
    if token_ratio >= 0.75 and group.article_count >= 5:
        flags.append("overgroup_risk_token_dominant")
    if rep_score < 2.0:
        flags.append("low_representative_score")
    if "photo_like" in rep_flags:
        flags.append("photo_like_representative")
    if group.article_count >= 5 and group.source_count <= 2:
        flags.append("many_articles_few_sources")

    return flags


# [코드 이해 주석]
# - 역할: 관련 데이터와 동작을 묶어 후속 처리에서 쓰기 쉽게 만드는 클래스입니다.
# - 호출하는 곳: news_grouper.build_payload
# - 파라미터: 클래스 속성과 생성 인자는 dataclass/본문의 필드 정의를 기준으로 사용합니다.
# - 리턴값: 클래스 정의 자체를 제공하며, 인스턴스는 필드/메서드 조합으로 사용합니다.
# - 프로세스 흐름: 필드와 메서드를 한 곳에 묶습니다 -> 다른 함수가 인스턴스를 만들고 값을 갱신합니다 -> 직렬화/비교 단계에서 사용됩니다.
@dataclass
class NewsPayload:
    index: int                                                  # 원본후보순번
    news: Dict[str, Any]                                       # 원본뉴스dict
    title: str                                                  # 표시용제목
    description: str                                            # 표시용설명
    source: str                                                 # 언론사명
    keyword: str                                                # 수집키워드
    published_at_kst: str                                       # KST발행시각문자열
    url: str                                                    # 기사URL
    normalized_url: str                                         # 중복비교용URL
    normalized_title: str                                       # 중복비교용제목
    normalized_text: str                                        # 중복비교용본문
    tokens: List[str]                                           # 본문비교토큰목록
    title_tokens: List[str]                                     # 제목비교토큰목록
    fingerprint: str                                            # SimHash지문


# [코드 이해 주석]
# - 역할: 관련 데이터와 동작을 묶어 후속 처리에서 쓰기 쉽게 만드는 클래스입니다.
# - 호출하는 곳: news_grouper.group_news
# - 파라미터: 클래스 속성과 생성 인자는 dataclass/본문의 필드 정의를 기준으로 사용합니다.
# - 리턴값: 클래스 정의 자체를 제공하며, 인스턴스는 필드/메서드 조합으로 사용합니다.
# - 프로세스 흐름: 필드와 메서드를 한 곳에 묶습니다 -> 다른 함수가 인스턴스를 만들고 값을 갱신합니다 -> 직렬화/비교 단계에서 사용됩니다.
@dataclass
class NewsGroup:
    group_id: str                                               # 사건그룹ID
    representative: NewsPayload                                 # 메일대표기사후보
    items: List[NewsPayload] = field(default_factory=list)       # 그룹소속기사목록
    match_reasons: List[Dict[str, Any]] = field(default_factory=list)  # 기사별그룹편입근거

    # [코드 이해 주석]
    # - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
    # - 호출하는 곳: news_grouper.extract_tokens, news_grouper.group_news
    # - 파라미터: self(현재 인스턴스), payload: NewsPayload, reason: Dict[str, Any]
    # - 리턴값: None 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    def add(self, payload: NewsPayload, reason: Dict[str, Any]) -> None:
        self.items.append(payload)
        self.match_reasons.append(reason)

    # [코드 이해 주석]
    # - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
    # - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
    # - 파라미터: self(현재 인스턴스)
    # - 리턴값: List[str] 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    @property
    def sources(self) -> List[str]:
        return sorted({item.source for item in self.items if item.source})

    # [코드 이해 주석]
    # - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
    # - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
    # - 파라미터: self(현재 인스턴스)
    # - 리턴값: List[str] 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    @property
    def keywords(self) -> List[str]:
        return sorted({item.keyword for item in self.items if item.keyword})

    # [코드 이해 주석]
    # - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
    # - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
    # - 파라미터: self(현재 인스턴스)
    # - 리턴값: int 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    @property
    def article_count(self) -> int:
        return len(self.items)

    # [코드 이해 주석]
    # - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
    # - 호출하는 곳: 외부 모듈에서 import해 호출할 수 있는 공개 함수입니다. 정적 직접 호출은 없습니다.
    # - 파라미터: self(현재 인스턴스)
    # - 리턴값: int 타입 값을 반환합니다.
    # - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
    @property
    def source_count(self) -> int:
        return len(self.sources)


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: news_grouper.group_news
# - 파라미터: news: Dict[str, Any], index: int
# - 리턴값: NewsPayload 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_payload(news: Dict[str, Any], index: int) -> NewsPayload:
    title = get_news_title(news)                                # 표시용제목
    description = get_news_description(news)                    # 표시용설명
    compare_text = f"{title} {description}"                     # 제목본문결합비교문자열
    tokens = extract_tokens(compare_text)                       # 본문비교토큰목록
    title_tokens = extract_tokens(title, max_tokens=30)         # 제목비교토큰목록
    return NewsPayload(
        index=index,  # 순번
        news=news,  # 뉴스
        title=title,  # 제목
        description=description,  # 설명
        source=str(news.get("source") or "").strip(),
        keyword=str(news.get("keyword") or "").strip(),
        published_at_kst=parse_iso_datetime(str(news.get("published_at_kst") or "")),
        url=get_news_url(news),  # URL
        normalized_url=normalize_url(get_news_url(news)),  # 정규화URL
        normalized_title=normalize_title(title),  # 정규화제목
        normalized_text=compact_compare_text(compare_text),  # 정규화텍스트
        tokens=tokens,  # 토큰수
        title_tokens=title_tokens,  # 제목토큰수
        fingerprint=make_simhash(tokens),  # fingerprint
    )


# [코드 이해 주석]
# - 역할: 두 기사가 같은 사건으로 보이는지 판단한다.
# - 호출하는 곳: news_grouper.group_news
# - 파라미터: candidate: NewsPayload, representative: NewsPayload, title_threshold: float, text_threshold: float,
# token_threshold: float, simhash_threshold: int, min_common_tokens: int
# - 리턴값: Tuple[bool, Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def compare_payloads(
    candidate: NewsPayload,
    representative: NewsPayload,
    title_threshold: float,
    text_threshold: float,
    token_threshold: float,
    simhash_threshold: int,
    min_common_tokens: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    두 기사가 같은 사건으로 보이는지 판단한다.
    반환: (같은그룹여부, 판단근거 dict)
    """
    # 1) 가장 강한 신호부터 본다: URL/정규화 제목이 같으면 같은 기사 또는 같은 사건으로 바로 묶는다.
    #    보통 URL 중복은 스크래퍼에서 제거되지만, 여러 키워드/링크 필드 조합 때문에 남을 수 있어 안전망으로 둔다.
    if candidate.normalized_url and candidate.normalized_url == representative.normalized_url:
        return True, {"method": "url", "detail": "정규화 URL 동일", "score": 1.0}

    if candidate.normalized_title and candidate.normalized_title == representative.normalized_title:
        return True, {"method": "title_exact", "detail": "정규화 제목 동일", "score": 1.0}

    # 2) 제목 기반 비교는 "짧고 강한" 신호다.
    #    제목 포함/유사도/제목 토큰 겹침 순서로 보며, 같은 보도자료를 제목만 바꿔 쓴 경우를 잡는다.
    title_score = 0.0  # 제목점수
    if candidate.normalized_title and representative.normalized_title:
        if min(len(candidate.normalized_title), len(representative.normalized_title)) >= 10:
            shorter, longer = sorted(  # 짧은값,긴값
                [candidate.normalized_title, representative.normalized_title],
                key=len,  # 키
            )
            if shorter in longer:
                return True, {
                    "method": "title_contains",
                    "detail": "한쪽 제목이 다른 쪽 제목을 포함",
                    "score": len(shorter) / max(1, len(longer)),
                }

        title_score = SequenceMatcher(None, candidate.normalized_title, representative.normalized_title).ratio()  # 제목점수
        if title_score >= title_threshold:
            return True, {
                "method": "title_similarity",
                "detail": f"제목 유사도 {title_score:.2f}",
                "score": title_score,
            }

    title_overlap_score, title_common_count = token_overlap_score(  # 제목overlap점수,제목공통건수
        candidate.title_tokens,
        representative.title_tokens,
    )
    if title_common_count >= 3 and title_overlap_score >= 0.55:
        return True, {
            "method": "title_token_overlap",
            "detail": f"제목 핵심 토큰 겹침률 {title_overlap_score:.2f}, 공통 {title_common_count}개",
            "score": title_overlap_score,
            "common_count": title_common_count,
        }

    # 3) 설명문 전체 유사도는 제목보다 느슨한 신호다.
    #    제목은 달라도 요약 설명이 거의 같으면 같은 사건일 가능성이 높다.
    text_score = 0.0  # 텍스트점수
    if candidate.normalized_text and representative.normalized_text:
        text_score = SequenceMatcher(None, candidate.normalized_text, representative.normalized_text).ratio()  # 텍스트점수
        if text_score >= text_threshold:
            return True, {
                "method": "text_similarity",
                "detail": f"본문 유사도 {text_score:.2f}",
                "score": text_score,
            }

    # 4) 토큰 겹침과 SimHash는 표현이 조금 달라도 핵심 명사/수치가 겹치는 사건을 잡기 위한 보완 신호다.
    #    공통 토큰 수 조건을 함께 두어, 너무 일반적인 단어 몇 개만으로 과하게 묶이는 것을 막는다.
    overlap_score, common_count = token_overlap_score(candidate.tokens, representative.tokens)  # overlap점수,공통건수
    if common_count >= min_common_tokens and overlap_score >= token_threshold:
        return True, {
            "method": "token_overlap",
            "detail": f"토큰 겹침률 {overlap_score:.2f}, 공통 {common_count}개",
            "score": overlap_score,
            "common_count": common_count,
        }

    distance = simhash_distance(candidate.fingerprint, representative.fingerprint)  # distance
    if distance is not None and distance <= simhash_threshold and common_count >= max(2, min_common_tokens - 1):
        return True, {
            "method": "simhash",
            "detail": f"SimHash 거리 {distance}, 공통 토큰 {common_count}개",
            "score": 1.0 - (distance / 64.0),
            "distance": distance,
            "common_count": common_count,
        }

    return False, {
        "method": "none",
        "detail": "그룹 기준 미충족",
        "title_score": round(title_score, 3),
        "text_score": round(text_score, 3),
        "token_overlap": round(overlap_score, 3),
        "common_count": common_count,
        "simhash_distance": distance,
    }


# [코드 이해 주석]
# - 역할: 여러 뉴스 항목을 사건 단위 그룹이나 그룹 통계로 처리합니다.
# - 호출하는 곳: news_grouper.build_grouping_result
# - 파라미터: news_list: List[Dict[str, Any]], title_threshold: float = DEFAULT_TITLE_SIMILARITY_THRESHOLD,
# text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD, token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
# simhash_threshold: int = DEFAULT_SIMHASH_DISTANCE_THRESHOLD, min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT
# - 리턴값: List[NewsGroup] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def group_news(
    news_list: List[Dict[str, Any]],
    title_threshold: float = DEFAULT_TITLE_SIMILARITY_THRESHOLD,  # 제목기준값
    text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,  # 텍스트기준값
    token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,  # 토큰기준값
    simhash_threshold: int = DEFAULT_SIMHASH_DISTANCE_THRESHOLD,  # SimHash기준값
    min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT,  # 최소공통토큰수
) -> List[NewsGroup]:
    # 1) 원본 뉴스 dict를 비교 전용 NewsPayload로 변환한다.
    #    Payload에는 정규화 제목, 비교 텍스트, 토큰, URL fingerprint가 미리 들어 있어 이후 비교가 단순해진다.
    payloads = [build_payload(news, idx + 1) for idx, news in enumerate(news_list or [])]  # payloads

    # 2) 최신 기사 우선으로 정렬한다.
    #    먼저 들어온 payload가 새 그룹의 최초 대표가 되므로, 기본 대표가 오래된 기사로 잡히지 않게 한다.
    payloads.sort(key=lambda p: p.published_at_kst or "", reverse=True)

    groups: List[NewsGroup] = []  # 그룹목록

    # 3) 각 기사를 기존 그룹들과 비교해 가장 점수가 높은 그룹에 붙인다.
    #    대표 기사만 보지 않고 그룹 내 최신 일부 기사도 비교해, 대표가 독특한 제목일 때 생기는 미매칭을 줄인다.
    for payload in payloads:  # 데이터
        best_group: Optional[NewsGroup] = None  # 최적그룹
        best_reason: Optional[Dict[str, Any]] = None  # any
        best_score = -1.0  # 최적점수

        for group in groups:  # 그룹
            # 대표 기사뿐 아니라 그룹 내 기사 일부와도 비교한다.
            # 단, 속도와 로그 안정성을 위해 최신 5개까지만 비교한다.
            compare_targets = [group.representative] + group.items[:5]  # comparetargets

            for target in compare_targets:  # target
                is_match, reason = compare_payloads(  # is매칭,사유
                    candidate=payload,  # 후보
                    representative=target,  # 대표기사
                    title_threshold=title_threshold,  # 제목기준값
                    text_threshold=text_threshold,  # 텍스트기준값
                    token_threshold=token_threshold,  # 토큰기준값
                    simhash_threshold=simhash_threshold,  # SimHash기준값
                    min_common_tokens=min_common_tokens,  # 최소공통토큰수
                )
                if is_match:
                    score = float(reason.get("score", 0.0) or 0.0)  # 점수
                    if score > best_score:
                        best_group = group  # 최적그룹
                        best_reason = {  # 최적판정사유
                            **reason,
                            "matched_with_index": target.index,
                            "matched_with_title": target.title,
                        }
                        best_score = score  # 최적점수

        # 4) 어느 그룹과도 기준을 넘지 못하면 새 사건 그룹을 만든다.
        #    기준을 넘은 그룹이 있으면 match_reason을 저장해 나중에 왜 묶였는지 추적할 수 있게 한다.
        if best_group is None:
            group_id = f"G{len(groups) + 1:03d}"  # 그룹id
            new_group = NewsGroup(  # new그룹
                group_id=group_id,  # 그룹id
                representative=payload,  # 대표기사
                items=[payload],  # 항목목록
                match_reasons=[{"method": "representative", "detail": "그룹 대표 기사"}],
            )
            groups.append(new_group)
        else:
            assert best_reason is not None
            best_group.add(payload, best_reason)

    # 5) 그룹이 완성된 뒤 대표 기사를 다시 고른다.
    #    최초 대표는 최신순 기준이지만, 메일에는 사진/속보/짧은 기사보다 설명이 풍부한 기사가 더 적합하다.
    for group in groups:  # 그룹
        refresh_group_representative(group)

    # 6) 큰 그룹, 언론사 다양한 그룹, 최신 그룹 순으로 정렬한다.
    #    이 순서가 뒤의 OpenAI 후보 압축과 로컬 fallback 우선순위에 영향을 준다.
    groups.sort(
        key=lambda g: (  # 키
            g.article_count,
            g.source_count,
            g.representative.published_at_kst or "",
        ),
        reverse=True,  # reverse
    )

    return groups


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.serialize_groups
# - 파라미터: group: NewsGroup
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def serialize_group(group: NewsGroup) -> Dict[str, Any]:
    # NewsGroup 객체는 Python 내부 처리에는 편하지만, 뒤 단계의 OpenAI 프롬프트/로그/테스트 파일에는 dict가 필요하다.
    # 여기서 그룹 품질, 대표 기사 점수, 그룹 편입 근거를 한 구조로 묶어 build_grouping_result()가 그대로 재사용하게 한다.
    method_counts = Counter(reason.get("method", "unknown") for reason in group.match_reasons)  # 그룹편입방식별건수
    rep_score, rep_flags = representative_quality_score(group.representative, group)             # 대표기사품질점수와플래그
    flags = group_quality_flags(group)                                                           # 그룹전체품질플래그

    return {
        "group_id": group.group_id,                                                              # 사건그룹ID
        "article_count": group.article_count,                                                     # 그룹소속기사수
        "source_count": group.source_count,                                                       # 그룹언론사수
        "sources": group.sources,                                                                 # 그룹언론사목록
        "keywords": group.keywords,                                                               # 그룹수집키워드목록
        "time_span_hours": round(group_time_span_hours(group), 2),                                # 그룹발행시간폭
        "quality_flags": flags,                                                                   # 그룹품질플래그
        "representative_score": round(rep_score, 2),                                              # 대표기사품질점수
        "representative_flags": rep_flags,                                                        # 대표기사품질플래그
        "representative": {
            "index": group.representative.index,                                                  # 대표기사원본순번
            "title": group.representative.title,                                                  # 대표기사제목
            "description": group.representative.description,                                      # 대표기사설명
            "source": group.representative.source,                                                # 대표기사언론사
            "keyword": group.representative.keyword,                                              # 대표기사수집키워드
            "published_at_kst": group.representative.published_at_kst,                            # 대표기사발행시각
            "url": group.representative.url,                                                      # 대표기사URL
        },
        "match_method_counts": dict(method_counts),                                               # 그룹편입방식통계
        "articles": [
            {
                "index": item.index,                                                             # 기사원본순번
                "title": item.title,                                                             # 기사제목
                "source": item.source,                                                           # 기사언론사
                "keyword": item.keyword,                                                         # 기사수집키워드
                "published_at_kst": item.published_at_kst,                                       # 기사발행시각
                "url": item.url,                                                                 # 기사URL
                "representative_candidate_score": round(representative_quality_score(item, group)[0], 2),  # 대표후보품질점수
                "representative_candidate_flags": representative_quality_score(item, group)[1],   # 대표후보품질플래그
                "match_reason": group.match_reasons[i] if i < len(group.match_reasons) else {},  # 그룹편입근거
            }
            for i, item in enumerate(group.items)  # i,항목
        ],
    }



# ====================================
# 운영용 헬퍼
# ====================================
# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.build_grouping_result
# - 파라미터: groups: List[NewsGroup]
# - 리턴값: List[Dict[str, Any]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def serialize_groups(groups: List[NewsGroup]) -> List[Dict[str, Any]]:
    return [serialize_group(group) for group in groups]


# [코드 이해 주석]
# - 역할: 입력값이 특정 조건을 만족하는지 bool로 판정합니다.
# - 호출하는 곳: news_grouper.build_grouping_result
# - 파라미터: serialized_group: Dict[str, Any]
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_low_quality_group(serialized_group: Dict[str, Any]) -> bool:
    flags = set(serialized_group.get("quality_flags") or [])  # flags
    rep_flags = set(serialized_group.get("representative_flags") or [])  # 대표기사품질플래그
    rep_score = float(serialized_group.get("representative_score") or 0)  # 대표기사품질점수

    if "overgroup_risk_token_time_span" in flags:
        return True
    if "low_representative_score" in flags:
        return True
    if "photo_like_representative" in flags and serialized_group.get("source_count", 0) <= 2:
        return True
    if rep_score < 2.0:
        return True
    if "photo_like" in rep_flags and serialized_group.get("source_count", 0) <= 2:
        return True
    return False


# [코드 이해 주석]
# - 역할: AI 선별 후보로 넘길 그룹의 로컬 우선순위 점수.
# - 호출하는 곳: news_grouper.build_grouping_result
# - 파라미터: serialized_group: Dict[str, Any]
# - 리턴값: float 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def group_priority_score(serialized_group: Dict[str, Any]) -> float:
    """AI 선별 후보로 넘길 그룹의 로컬 우선순위 점수."""
    article_count = int(serialized_group.get("article_count") or 0)      # 그룹기사수
    source_count = int(serialized_group.get("source_count") or 0)        # 그룹언론사수
    rep_score = float(serialized_group.get("representative_score") or 0)  # 대표기사품질점수
    flags = set(serialized_group.get("quality_flags") or [])             # 그룹품질플래그

    # 점수는 AI 후보 압축 전 1차 정렬용이다.
    # 여러 언론사가 반복 보도한 사건을 올리고, 사진성/과묶음 위험 그룹은 낮춰 AI 입력 슬롯을 아낀다.
    score = 0.0                                                          # 그룹우선순위점수
    score += min(source_count, 8) * 2.2  # 처리값
    score += min(article_count, 10) * 0.9  # 처리값
    score += rep_score * 0.8  # 처리값

    if source_count >= 4:
        score += 2.0  # 처리값
    if article_count >= 4:
        score += 1.0  # 처리값
    if "many_articles_few_sources" in flags:
        score -= 2.5  # 처리값
    if "photo_like_representative" in flags:
        score -= 4.0  # 처리값
    if "overgroup_risk_token_dominant" in flags:
        score -= 2.0  # 처리값
    if "overgroup_risk_token_time_span" in flags:
        score -= 6.0  # 처리값
    if "low_representative_score" in flags:
        score -= 5.0  # 처리값

    return round(score, 3)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: news_grouper.build_grouping_result
# - 파라미터: serialized_group: Dict[str, Any]
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def representative_news_from_group(serialized_group: Dict[str, Any]) -> Dict[str, Any]:
    # 1) selection_groups의 대표 기사 dict를 복사해 후속 단계가 수정해도 원본 그룹 직렬화 결과가 흔들리지 않게 한다.
    rep = dict(serialized_group.get("representative") or {})  # rep
    articles = serialized_group.get("articles") or []  # 기사목록
    # 2) 관련보도 페이지와 메일 링크에 쓸 기사 목록을 만든다.
    #    제목과 URL이 모두 있는 항목만 같은 순서로 남겨 title/url/source 배열의 인덱스가 서로 맞게 유지한다.
    related_articles = [  # 관련보도기사목록
        article
        for article in articles[:12]  # 기사
        if str(article.get("title") or "").strip()
        and str(article.get("url") or "").strip()
    ]
    # 3) group_* 필드는 news_selector/summarizer/email_sender/related_pages가 공유하는 그룹 메타데이터다.
    #    대표 기사 하나만 메일에 실려도, 관련보도 수와 언론사 다양성을 같이 표시할 수 있게 보존한다.
    rep["group_id"] = serialized_group.get("group_id")  # 처리값
    rep["group_article_count"] = serialized_group.get("article_count", 1)  # 처리값
    rep["group_source_count"] = serialized_group.get("source_count", 1)  # 처리값
    rep["group_sources"] = serialized_group.get("sources", [])  # 처리값
    rep["group_keywords"] = serialized_group.get("keywords", [])  # 처리값
    rep["group_quality_flags"] = serialized_group.get("quality_flags", [])  # 처리값
    rep["group_representative_score"] = serialized_group.get("representative_score", 0)  # 처리값
    rep["group_priority_score"] = serialized_group.get("priority_score", 0)  # 처리값
    rep["group_article_titles"] = [  # 처리값
        str(article.get("title") or "").strip()
        for article in related_articles  # 기사
    ]
    rep["group_article_urls"] = [  # 처리값
        str(article.get("url") or "").strip()
        for article in related_articles  # 기사
    ]
    rep["group_article_sources"] = [  # 처리값
        str(article.get("source") or "").strip()
        for article in related_articles  # 기사
    ]
    rep["description"] = rep.get("description", "")  # 처리값
    rep["content"] = rep.get("description", "")  # 처리값
    return rep


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: news_list: List[Dict[str, Any]], max_groups: int = 100, exclude_low_quality: bool = True, title_threshold:
# float = DEFAULT_TITLE_SIMILARITY_THRESHOLD, text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
# token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD, simhash_threshold: int =
# DEFAULT_SIMHASH_DISTANCE_THRESHOLD, min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_grouping_result(
    news_list: List[Dict[str, Any]],
    max_groups: int = 100,  # 최대그룹목록
    exclude_low_quality: bool = True,  # 제외lowquality
    title_threshold: float = DEFAULT_TITLE_SIMILARITY_THRESHOLD,  # 제목기준값
    text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,  # 텍스트기준값
    token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,  # 토큰기준값
    simhash_threshold: int = DEFAULT_SIMHASH_DISTANCE_THRESHOLD,  # SimHash기준값
    min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT,  # 최소공통토큰수
) -> Dict[str, Any]:
    # 1) 원본 뉴스 후보를 사건 단위 그룹으로 묶는다.
    #    이 시점의 news_list는 main.py에서 반복 이슈/제외 키워드를 통과한 후보이므로, 여기서는 "이번 실행 내부 중복"에 집중한다.
    groups = group_news(  # 그룹목록
        news_list=news_list,  # 뉴스list
        title_threshold=title_threshold,  # 제목기준값
        text_threshold=text_threshold,  # 텍스트기준값
        token_threshold=token_threshold,  # 토큰기준값
        simhash_threshold=simhash_threshold,  # SimHash기준값
        min_common_tokens=min_common_tokens,  # 최소공통토큰수
    )

    # 2) NewsGroup 객체를 JSON 친화적인 dict로 바꾸고, OpenAI 선별에 쓸 우선순위/품질 플래그를 계산한다.
    #    객체 상태를 그대로 넘기지 않는 이유는 이후 로그, 메일, 테스트에서 같은 구조를 쉽게 읽기 위해서다.
    serialized_groups = serialize_groups(groups)  # serialized그룹목록
    for group in serialized_groups:  # 그룹
        group["priority_score"] = group_priority_score(group)  # 처리값
        group["low_quality_excluded"] = is_low_quality_group(group)  # 처리값

    # 3) selection_groups는 실제 AI 후보로 넘길 그룹이다.
    #    저품질 제외 옵션이 켜져 있으면 사진성/대표품질 낮은 그룹은 통계에는 남기되 AI 후보에서는 뺀다.
    selection_groups = [  # selection그룹목록
        group for group in serialized_groups
        if not (exclude_low_quality and group.get("low_quality_excluded"))
    ]
    selection_groups.sort(
        key=lambda g: (  # 키
            float(g.get("priority_score") or 0),
            int(g.get("source_count") or 0),
            int(g.get("article_count") or 0),
            str((g.get("representative") or {}).get("published_at_kst") or ""),
        ),
        reverse=True,  # reverse
    )
    selection_groups = selection_groups[:max_groups]  # selection그룹목록

    multi_article_groups = [group for group in serialized_groups if int(group.get("article_count") or 0) >= 2]  # 복수기사사건그룹목록
    low_quality_groups = [group for group in serialized_groups if group.get("low_quality_excluded")]           # 저품질제외그룹목록

    # 4) 메일 대시보드용 기사 단위 통계를 계산한다.
    #    - duplicate_article_count: 같은 사건으로 묶이며 대표 1건만 남아 AI 후보에서 빠진 기사 수
    #    - low_quality_article_count: 사진성/저품질 그룹으로 판단되어 AI 후보에서 제외된 기사 수
    #    그룹 수가 아니라 기사 수로 세야 "몇 건을 줄였는지"가 운영자가 보기에 직관적이다.
    duplicate_article_count = sum(                                           # 대표기사외중복기사수
        max(int(group.get("article_count") or 0) - 1, 0)
        for group in serialized_groups  # 그룹
    )
    low_quality_article_count = sum(                                         # 저품질제외기사수
        int(group.get("article_count") or 0)
        for group in low_quality_groups  # 그룹
    )
    selection_article_count = sum(                                           # AI선별전달기사수
        int(group.get("article_count") or 0)
        for group in selection_groups  # 그룹
    )

    # 5) 반환값은 두 갈래로 쓰인다.
    #    main.py는 selection_groups를 OpenAI 선별에 넘기고,
    #    representative_news는 OpenAI 실패/fallback 또는 관련보도 메타 보존에 사용한다.
    #    통계 key는 main.py에서 scrape_stats로 복사되어 email_sender.py 대시보드의 그룹중복/저품질 제외 수가 된다.
    return {
        "news_count": len(news_list or []),                                 # 그룹화대상기사수
        "group_count": len(serialized_groups),                              # 생성된사건그룹수
        "multi_article_group_count": len(multi_article_groups),             # 복수기사사건그룹수
        "low_quality_group_count": len(low_quality_groups),                 # 저품질제외그룹수
        "low_quality_article_count": low_quality_article_count,             # 저품질제외기사수
        "duplicate_article_count": duplicate_article_count,                 # 대표기사외중복기사수
        "selection_group_count": len(selection_groups),                     # AI선별전달그룹수
        "selection_article_count": selection_article_count,                 # AI선별전달기사수
        "parameters": {
            "title_threshold": title_threshold,                             # 제목유사도기준
            "text_threshold": text_threshold,                               # 본문유사도기준
            "token_threshold": token_threshold,                             # 토큰겹침기준
            "simhash_threshold": simhash_threshold,                         # SimHash거리기준
            "min_common_tokens": min_common_tokens,                         # 최소공통토큰수
            "exclude_low_quality": exclude_low_quality,                     # 저품질그룹제외여부
            "max_groups": max_groups,                                       # AI전달최대그룹수
        },
        "groups": serialized_groups,                                        # 전체직렬화그룹목록
        "selection_groups": selection_groups,                               # AI선별후보그룹목록
        "representative_news": [representative_news_from_group(group) for group in selection_groups],  # 그룹대표뉴스목록
    }
