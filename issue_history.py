# =============================================================================
# [파일 설명]
# - 수행 기능: 최근 발송 이슈를 저장하고, 새 후보가 이미 다룬 사건인지 규칙/LLM으로 판정합니다.
# - 프로세스: 텍스트/URL 정규화 -> 사건 서명 생성 -> 과거 이슈 색인 구성 -> 중복 판정 -> 히스토리 저장/정리
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 요약 뉴스, seen_issues.json, 브리핑/수신자/섹션 범위, 비교 일수
# - 리턴값/출력: 필터링된 뉴스/제외 목록/히스토리 dict와 중복 제거 통계를 반환합니다.
# =============================================================================

"""
뉴스 이슈 히스토리 관리

목적:
- 메일에 실제로 발송된 최종 뉴스만 히스토리에 저장한다.
- 이후 실행에서 최근 N일 발송 이력과 후보 뉴스를 비교해 반복 이슈를 제거한다.
- 비용과 실행 시간을 줄이기 위해 히스토리 비교에는 OpenAI/LLM을 사용하지 않는다.
- issue_key/core_issue_key 생성도 사용하지 않는다.

반복 이슈 제거 방식:
1. URL 정규화 완전 일치
2. 정규화 제목 완전 일치
3. 제목 유사도
4. 제목+설명 텍스트 유사도
5. 주요 토큰 겹침률
6. SimHash 거리

주의:
- 함수명 filter_seen_issues_with_llm은 기존 main.py 호환을 위해 유지한다.
- 실제 동작은 LLM 없는 규칙 기반 필터다.
"""

import json
import os
import re
import hashlib
import logging
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qsl, urlencode

import pytz


# ====================================
# 기본 설정
# ====================================
HISTORY_FILE_PATH = "data/seen_issues.json"  # 이슈히스토리파일경로
KST = pytz.timezone("Asia/Seoul")  # 한국시간대

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈로거

# 제목 유사도 기준
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("HISTORY_TITLE_SIMILARITY_THRESHOLD", "0.88"))  # 제목유사도기준값

# 제목+설명 전체 텍스트 유사도 기준
TEXT_SIMILARITY_THRESHOLD = float(os.getenv("HISTORY_TEXT_SIMILARITY_THRESHOLD", "0.82"))  # 텍스트유사도기준값

# 주요 토큰 겹침률 기준
TOKEN_OVERLAP_THRESHOLD = float(os.getenv("HISTORY_TOKEN_OVERLAP_THRESHOLD", "0.65"))  # 토큰overlap기준값

# SimHash 거리 기준. 작을수록 거의 같은 문서다.
SIMHASH_DISTANCE_THRESHOLD = int(os.getenv("HISTORY_SIMHASH_DISTANCE_THRESHOLD", "6"))  # SimHash거리기준값

# 너무 짧은 기사끼리 토큰 겹침만으로 중복 처리되는 것을 막기 위한 최소 공통 토큰 수
MIN_COMMON_TOKEN_COUNT = int(os.getenv("HISTORY_MIN_COMMON_TOKEN_COUNT", "4"))  # 최소공통토큰건수

# URL 정규화 시 유지할 의미 있는 query parameter
MEANINGFUL_QUERY_PARAMS = {  # URL정규화유지쿼리파라미터
    "no", "idxno", "article_no", "articleid", "article_id",
    "newsid", "news_id", "aid", "oid", "sid", "id", "seq", "num",
}

# 토큰 사용량 통계. 이 파일은 LLM을 쓰지 않으므로 항상 0이다.
LAST_TOKEN_STATS = {  # 마지막토큰통계
    "issue_key_tokens": 0,
    "llm_duplicate_tokens": 0,
}

# 한국어 뉴스에서 반복적으로 등장하지만 중복 판단에는 도움이 적은 단어들
STOPWORDS = {  # STOPWORDS
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
    "한국", "국내", "글로벌", "최신", "주요", "확인", "가능", "기준",
    "기반", "활용", "도입", "사용", "운용", "참여", "소개", "제공",
}

TOKEN_SUFFIXES = (  # 토큰정규화제거접미사
    "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
    "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
)

ANCHOR_STOPWORDS = STOPWORDS | {  # 앵커토큰불용어
    "ai", "it", "ict", "dx", "si", "emr", "시스템", "기술", "산업", "시장",
    "사업", "서비스", "플랫폼", "솔루션", "정보", "디지털",
}

# 히스토리 비교/저장 범위. 기본값은 briefing으로 두어 같은 메일 안에서
# 섹션만 달라진 반복 이슈도 제거한다. 필요하면 환경변수로 section/receiver로 조정한다.
HISTORY_MATCH_SCOPE = os.getenv("HISTORY_MATCH_SCOPE", "briefing").strip().lower()  # 히스토리매칭범위
HISTORY_SAVE_SCOPE = os.getenv("HISTORY_SAVE_SCOPE", HISTORY_MATCH_SCOPE).strip().lower()  # 히스토리save범위

# 반복 이슈 제외 내역 디버그 JSON 저장.
# 값은 노출 가능한 제목/간단 근거 중심으로만 저장한다.

# 외부 AI 호출 없이 표기 차이를 줄이기 위한 최소 별칭 사전.
# 값은 비교용 텍스트에 들어갈 안정 토큰이다.
ENTITY_ALIASES = {  # ENTITYALIASES
    "sk ax": "skax",
    "sk㈜ c&c": "skax",
    "sk c&c": "skax",
    "에스케이 ax": "skax",
    "에스케이 씨앤씨": "skax",
    "오픈ai": "openai",
    "오픈 ai": "openai",
    "open ai": "openai",
    "챗gpt": "chatgpt",
    "chat gpt": "chatgpt",
    "경복대학교": "경복대",
    "롯데 홈쇼핑": "롯데홈쇼핑",
    "태광 산업": "태광산업",
    "메가존 클라우드": "메가존클라우드",
    "아이티센 글로벌": "아이티센글로벌",
    "차 바이오텍": "차바이오텍",
    "ai디지털 헬스케어": "ai디지털헬스케어",
    "ai 디지털 헬스케어": "ai디지털헬스케어",
    "경기도": "gyeonggi",
    "풍수해대책": "풍수해",
    "재난대응체계": "대응체계",
    "첨단 재난대응체계": "대응체계",
}

ACTION_KEYWORDS = {  # 사건행위키워드
    "협력", "제휴", "파트너십", "동맹", "계약", "공급", "선정", "수주",
    "인수", "매각", "투자", "출시", "개설", "신설", "구축", "도입",
    "갱신", "인증", "해임", "부결", "갈등", "분쟁", "실적", "영업이익",
    "영업손실", "매출", "수출", "대응", "점검", "재편", "개최",
    "동맹", "합류", "가동", "대비", "진행", "선보",
}

ACTION_ALIASES = {  # 행위키워드별칭
    "파트너십": "협력",
    "제휴": "협력",
    "동맹": "협력",
    "협력": "협력",
    "합류": "협력",
    "계약": "협력",
    "진행": "개최",
    "선보": "출시",
    "개설": "신설",
    "신설": "신설",
    "선정": "선정",
    "공급": "공급",
    "수주": "수주",
    "구축": "대응",
    "대응": "대응",
    "가동": "대응",
    "대비": "대응",
    "해임안": "해임",
    "해임": "해임",
    "부결": "부결",
    "갈등": "갈등",
    "분쟁": "갈등",
    "실적": "실적",
    "영업이익": "실적",
    "영업손실": "실적",
    "매출": "실적",
}


# 중복 판단에서 단독 근거로 쓰기 약한 범용 앵커/행위어.
# 완전히 버리지는 않고, strong 조건을 계산할 때 가중치를 낮춘다.
WEAK_ANCHOR_TOKENS = ANCHOR_STOPWORDS | {  # 약한앵커토큰
    "경제", "경제학상", "노벨경제학상", "대통령", "정부", "부총리", "장관",
    "부동산", "부동산시장", "주택시장", "증시", "뉴욕증시", "코스피", "코스닥",
    "나스닥", "다우", "다우지수", "정상회담", "미중회담", "트럼프", "시진핑",
    "15일", "14일", "현지시간", "연합뉴스", "시장", "주식", "투자자",
}

WEAK_ACTION_TOKENS = {  # 약한행위토큰
    "협력", "투자", "대응", "공급", "점검", "개최", "실적", "수출", "출시"
}

STRONG_ACTION_TOKENS = {  # 강한행위토큰
    "인수", "매각", "해임", "부결", "갈등", "분쟁", "수주", "선정", "신설",
    "갱신", "인증", "재편", "영업이익", "영업손실", "매출"
}

# [코드 이해 주석]
# - 역할: 누적 통계나 상태 값을 초기 상태로 되돌립니다.
# - 호출하는 곳: issue_history.filter_seen_issues_with_llm
# - 파라미터: 없음
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def reset_token_stats():
    LAST_TOKEN_STATS["issue_key_tokens"] = 0  # 처리값
    LAST_TOKEN_STATS["llm_duplicate_tokens"] = 0  # 처리값


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.filter_seen_issues_with_llm
# - 파라미터: 없음
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_last_token_stats():
    return dict(LAST_TOKEN_STATS)


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.build_issue_record,
# issue_history.get_recent_issues_for_section, issue_history.prune_old_issues
# - 파라미터: 없음
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_now_kst():
    return datetime.now(KST)


# ====================================
# 정규화 유틸
# ====================================
# [코드 이해 주석]
# - 역할: 기사 URL 비교용 정규화.
# - 호출하는 곳: issue_history.build_issue_record, issue_history.deduplicate_section_results,
# issue_history.filter_seen_issues_with_llm, issue_history.get_issue_normalized_url,
# issue_history.get_news_alias_urls, issue_history.make_issue_id
# - 파라미터: url: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_url(url: str) -> str:
    """
    기사 URL 비교용 정규화.
    tracking query는 제거하고, 기사 식별에 의미 있는 query만 유지한다.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())  # parsed
        domain = parsed.netloc.lower().strip()  # 도메인
        path = parsed.path.strip()  # 경로
        if domain.startswith("www."):
            domain = domain[4:]  # 도메인

        meaningful_params = []  # 의미있는파라미터
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):  # 키,값
            key_lower = str(key).lower().strip()  # 키lower
            if key_lower in MEANINGFUL_QUERY_PARAMS and str(value).strip():
                meaningful_params.append((key_lower, str(value).strip()))
        meaningful_params.sort()

        normalized = f"{domain}{path}".rstrip("/")  # 정규화
        if meaningful_params:
            normalized += "?" + urlencode(meaningful_params)  # 처리값
        return normalized
    except Exception:
        return str(url).lower().strip().rstrip("/")


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.normalize_compare_text, issue_history.normalize_title
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
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.build_event_signature,
# issue_history.get_issue_compare_payload
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
# - 역할: 같은 기관/서비스의 표기 차이를 규칙 기반으로 정규화한다.
# - 호출하는 곳: issue_history.normalize_compare_text
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def apply_entity_aliases(text: str) -> str:
    """
    같은 기관/서비스의 표기 차이를 규칙 기반으로 정규화한다.
    LLM을 사용하지 않으므로 토큰 비용은 늘지 않는다.
    """
    value = str(text or "").lower()  # 값
    if not value:
        return ""

    # 붙여 쓴 영문/한글 표기도 먼저 보강한다.
    value = value.replace("openai", " openai ")  # 값
    value = value.replace("skax", " skax ")  # 값

    for alias, canonical in ENTITY_ALIASES.items():  # 별칭,표준값
        value = value.replace(alias.lower(), f" {canonical.lower()} ")  # 값
    return value


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.compact_compare_text, issue_history.extract_tokens
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_compare_text(text: str) -> str:
    if not text:
        return ""
    text = apply_entity_aliases(strip_html_entities(text)).lower()  # 텍스트
    text = re.sub(r"\[[^\]]*\]|【[^】]*】", " ", text)  # 텍스트
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)  # 텍스트
    text = re.sub(r"\b\w+@\w+(?:\.\w+)+\b", " ", text)  # 텍스트
    text = re.sub(r"[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일", " ", text)  # 텍스트
    text = re.sub(r"\s+", " ", text)  # 텍스트
    return text.strip()


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.build_compare_payload
# - 파라미터: text: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def compact_compare_text(text: str) -> str:
    text = normalize_compare_text(text)  # 텍스트
    text = re.sub(r"\s+", "", text)  # 텍스트
    return text


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: issue_history.extract_anchor_tokens, issue_history.extract_tokens, issue_history.is_strong_anchor_token,
# issue_history.is_weak_anchor_token, issue_history.normalize_action_token, issue_history.soft_common_tokens 외 1곳
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
# - 역할: 외부 형태소 분석기 없이 동작하는 간단 토큰 추출.
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.build_event_signature,
# issue_history.extract_title_tokens
# - 파라미터: text: str, max_tokens: int = 80
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_tokens(text: str, max_tokens: int = 80):
    """
    외부 형태소 분석기 없이 동작하는 간단 토큰 추출.
    한국어/영문/숫자 2자 이상 토큰만 사용한다.
    """
    text = normalize_compare_text(text)  # 텍스트
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)  # 원본토큰수

    tokens = []  # 토큰수
    seen = set()  # 확인된
    for token in raw_tokens:  # 토큰
        token = normalize_token(token)  # 토큰
        if not token:
            continue
        # 행위어는 중복 판단에 중요하므로 STOPWORDS에 있더라도 유지한다.
        if token in STOPWORDS and normalize_action_token(token) == "":
            continue
        # 순수 숫자 토큰은 단독으로 중복 판단에 취약하므로 제외
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


# [코드 이해 주석]
# - 역할: 입력 데이터에서 필요한 토큰, URL, 날짜, 사용량 같은 핵심 값을 추출합니다.
# - 호출하는 곳: issue_history.build_compare_payload
# - 파라미터: title: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_title_tokens(title: str):
    return extract_tokens(title, max_tokens=30)


# [코드 이해 주석]
# - 역할: 입력 데이터에서 필요한 토큰, URL, 날짜, 사용량 같은 핵심 값을 추출합니다.
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.build_event_signature
# - 파라미터: tokens: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_anchor_tokens(tokens):
    anchors = []  # anchors
    seen = set()  # 확인된
    for token in tokens or []:  # 토큰
        token = normalize_token(token)  # 토큰
        if not token or token in ANCHOR_STOPWORDS:
            continue
        has_alpha = bool(re.search(r"[a-zA-Z]", token))  # hasalpha
        has_korean = bool(re.search(r"[가-힣]", token))  # haskorean
        if has_alpha or (has_korean and len(token) >= 3) or len(token) >= 4:
            if token not in seen:
                seen.add(token)
                anchors.append(token)
    return anchors[:20]


# [코드 이해 주석]
# - 역할: 입력 데이터에서 필요한 토큰, URL, 날짜, 사용량 같은 핵심 값을 추출합니다.
# - 호출하는 곳: issue_history.build_compare_payload
# - 파라미터: value: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_number_tokens(value: str):
    return set(re.findall(r"\d+(?:\.\d+)?", str(value or "")))


# [코드 이해 주석]
# - 역할: 비교와 저장에 일관되게 사용할 수 있도록 값을 표준 형태로 정규화합니다.
# - 호출하는 곳: issue_history.extract_action_tokens, issue_history.extract_tokens, issue_history.has_specific_action,
# issue_history.soft_common_tokens
# - 파라미터: token: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_action_token(token: str) -> str:
    token = normalize_token(token)  # 토큰
    if not token:
        return ""
    for keyword, canonical in ACTION_ALIASES.items():  # 키워드,표준값
        if keyword in token:
            return canonical
    if token in ACTION_KEYWORDS:
        return token
    return ""


# [코드 이해 주석]
# - 역할: 입력 데이터에서 필요한 토큰, URL, 날짜, 사용량 같은 핵심 값을 추출합니다.
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.build_event_signature,
# issue_history.build_rule_event_key_from_payload, issue_history.get_issue_compare_payload
# - 파라미터: tokens: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_action_tokens(tokens):
    actions = []  # actions
    seen = set()  # 확인된
    for token in tokens or []:  # 토큰
        action = normalize_action_token(token)  # 행위
        if not action or action in seen:
            continue
        seen.add(action)
        actions.append(action)
    return actions[:6]


# [코드 이해 주석]
# - 역할: 입력값이 특정 조건을 만족하는지 bool로 판정합니다.
# - 호출하는 곳: issue_history.is_strong_anchor_token
# - 파라미터: token: str
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_weak_anchor_token(token: str) -> bool:
    token = normalize_token(token)  # 토큰
    if not token:
        return True
    if token in WEAK_ANCHOR_TOKENS:
        return True
    # 날짜/시각/단순 숫자성 토큰은 단독 앵커로 약하다.
    if re.fullmatch(r"\d{1,4}(?:년|월|일|선|대|개|명|억|조|%|p|pt)?", token):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return True
    return False


# [코드 이해 주석]
# - 역할: 입력값이 특정 조건을 만족하는지 bool로 판정합니다.
# - 호출하는 곳: issue_history.strong_common_anchor_tokens
# - 파라미터: token: str
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_strong_anchor_token(token: str) -> bool:
    token = normalize_token(token)  # 토큰
    if not token or is_weak_anchor_token(token):
        return False
    has_alpha = bool(re.search(r"[a-zA-Z]", token))  # hasalpha
    has_korean = bool(re.search(r"[가-힣]", token))  # haskorean
    return has_alpha or (has_korean and len(token) >= 3) or len(token) >= 5


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.build_rule_event_key_from_payload, issue_history.judge_duplicate_by_payload
# - 파라미터: tokens: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def strong_common_anchor_tokens(tokens):
    return unique_nonempty([token for token in tokens or [] if is_strong_anchor_token(token)], limit=20)


# [코드 이해 주석]
# - 역할: 입력 데이터에 특정 특징이나 조건이 있는지 bool로 판정합니다.
# - 호출하는 곳: issue_history.judge_duplicate_by_payload
# - 파라미터: action_tokens: Any
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 목록/토큰을 검사합니다 -> 조건 충족 여부를 계산합니다 -> True/False를 반환합니다.
def has_specific_action(action_tokens) -> bool:
    for action in action_tokens or []:  # 행위
        action = normalize_action_token(action) or str(action or "").strip()  # 행위
        if action and action not in WEAK_ACTION_TOKENS:
            return True
    return False


# [코드 이해 주석]
# - 역할: 기관/핵심 앵커 + 행위어 + 숫자 일부로 사건 키를 만든다.
# - 호출하는 곳: issue_history.build_compare_payload, issue_history.build_event_signature,
# issue_history.get_issue_compare_payload
# - 파라미터: payload: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_rule_event_key_from_payload(payload):
    """
    기관/핵심 앵커 + 행위어 + 숫자 일부로 사건 키를 만든다.
    키가 너무 일반적이면 빈 문자열을 반환해 오탐을 줄인다.
    """
    tokens = payload.get("tokens") or []  # 토큰수
    title_tokens = payload.get("title_tokens") or []  # 제목토큰수
    anchors = payload.get("anchor_tokens") or []  # anchors
    numbers = payload.get("number_tokens") or []  # numbers

    actions = payload.get("action_tokens") or extract_action_tokens(title_tokens + tokens)  # actions
    strong_anchors = strong_common_anchor_tokens(anchors)[:4]  # 강한anchors
    specific_actions = [action for action in actions if action not in WEAK_ACTION_TOKENS]  # specificactions

    # 범용 앵커/행위어만으로 만든 사건 키는 오탐이 많다.
    # 고유 앵커 2개 이상, 또는 고유 앵커 1개 + 구체 행위어 1개 이상일 때만 키를 만든다.
    if len(strong_anchors) < 2 and not (strong_anchors and specific_actions):
        return ""

    key_parts = strong_anchors[:3] + (specific_actions or actions)[:2] + list(numbers)[:2]  # 키부분목록
    return "|".join(unique_nonempty(key_parts, limit=7))


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.build_event_signature,
# issue_history.build_rule_event_key_from_payload, issue_history.get_issue_compare_payload,
# issue_history.get_news_alias_titles, issue_history.get_news_alias_urls 외 4곳
# - 파라미터: values: Any, limit: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def unique_nonempty(values, limit=None):
    seen = set()  # 확인된
    result = []  # 결과
    for value in values or []:  # 값
        text = str(value or "").strip()  # 텍스트
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_event_signature
# - 파라미터: news: dict
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_alias_titles(news: dict):
    titles = [get_news_title(news)]  # titles
    titles.extend(news.get("group_article_titles") or [])
    return unique_nonempty(titles, limit=16)


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_event_signature
# - 파라미터: news: dict
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_alias_urls(news: dict):
    urls = [get_news_url(news)]  # URL목록
    urls.extend(news.get("group_article_urls") or [])
    return unique_nonempty([normalize_url(url) for url in urls], limit=16)


# [코드 이해 주석]
# - 역할: 여러 입력 값을 조합해 식별자, 해시, 키 같은 파생 값을 만듭니다.
# - 호출하는 곳: issue_history.make_issue_id
# - 파라미터: value: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.make_simhash
# - 파라미터: token: str
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def stable_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # digest
    return int(digest[:16], 16)


# [코드 이해 주석]
# - 역할: 64비트 SimHash를 16자리 hex 문자열로 반환한다.
# - 호출하는 곳: issue_history.build_compare_payload
# - 파라미터: tokens: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_simhash(tokens) -> str:
    """
    64비트 SimHash를 16자리 hex 문자열로 반환한다.
    """
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
            fingerprint |= (1 << bit)  # 처리값

    return f"{fingerprint:016x}"


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.judge_duplicate_by_payload
# - 파라미터: a: str, b: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def simhash_distance(a: str, b: str):
    if not a or not b:
        return None
    try:
        x = int(str(a), 16) ^ int(str(b), 16)  # x
        return x.bit_count()
    except Exception:
        return None


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.judge_duplicate_by_payload
# - 파라미터: tokens_a: Any, tokens_b: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def token_overlap_score(tokens_a, tokens_b):
    set_a = set(tokens_a or [])  # seta
    set_b = set(tokens_b or [])  # setb
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b  # common
    # 짧은 쪽 기준 겹침률. 같은 보도자료의 제목/설명 변형에 더 민감하다.
    denominator = max(1, min(len(set_a), len(set_b)))  # denominator
    return len(common) / denominator, len(common)


# [코드 이해 주석]
# - 역할: 완전 일치가 아니어도 같은 고유명사/행사명 변형으로 볼 수 있는지 판단한다.
# - 호출하는 곳: issue_history.soft_common_tokens
# - 파라미터: token_a: Any, token_b: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def tokens_are_related(token_a, token_b):
    """
    완전 일치가 아니어도 같은 고유명사/행사명 변형으로 볼 수 있는지 판단한다.
    특정 회사명·행사명을 하드코딩하지 않고, 부분 문자열/유사도만 사용한다.
    """
    a = normalize_token(token_a)  # a
    b = normalize_token(token_b)  # b
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = sorted([a, b], key=len)  # 짧은값,긴값
    if len(shorter) >= 3 and shorter in longer:
        return True

    # 긴 고유명사에서 조사/띄어쓰기/축약 차이로 토큰이 조금 달라지는 경우를 보완한다.
    if min(len(a), len(b)) >= 4:
        return SequenceMatcher(None, a, b).ratio() >= 0.82

    return False


# [코드 이해 주석]
# - 역할: 토큰 목록 간 완전 일치 + 부분 일치 기반 공통 토큰을 계산한다.
# - 호출하는 곳: issue_history.judge_duplicate_by_payload, issue_history.soft_token_overlap_score
# - 파라미터: tokens_a: Any, tokens_b: Any, ignore_stopwords: Any = True
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def soft_common_tokens(tokens_a, tokens_b, ignore_stopwords=True):
    """
    토큰 목록 간 완전 일치 + 부분 일치 기반 공통 토큰을 계산한다.
    예: '메이플스토리'와 '메이플', '오픈ai'와 'openai'처럼 제목 표현이 달라도
    같은 사건의 핵심 앵커로 볼 수 있는 경우를 잡기 위한 규칙 기반 보완이다.
    """
    left = []  # left
    for token in tokens_a or []:  # 토큰
        token = normalize_token(token)  # 토큰
        if not token:
            continue
        if ignore_stopwords and token in ANCHOR_STOPWORDS and normalize_action_token(token) == "":
            continue
        left.append(token)

    right = []  # right
    for token in tokens_b or []:  # 토큰
        token = normalize_token(token)  # 토큰
        if not token:
            continue
        if ignore_stopwords and token in ANCHOR_STOPWORDS and normalize_action_token(token) == "":
            continue
        right.append(token)

    used_right = set()  # usedright
    common = []  # common
    for a in left:  # a
        for idx, b in enumerate(right):  # 순번,b
            if idx in used_right:
                continue
            if tokens_are_related(a, b):
                used_right.add(idx)
                common.append(a if len(a) <= len(b) else b)
                break

    return unique_nonempty(common)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: issue_history.judge_duplicate_by_payload
# - 파라미터: tokens_a: Any, tokens_b: Any, ignore_stopwords: Any = True
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def soft_token_overlap_score(tokens_a, tokens_b, ignore_stopwords=True):
    common = soft_common_tokens(tokens_a, tokens_b, ignore_stopwords=ignore_stopwords)  # common
    len_a = len(unique_nonempty(tokens_a or []))  # lena
    len_b = len(unique_nonempty(tokens_b or []))  # lenb
    if not len_a or not len_b:
        return 0.0, 0, []
    denominator = max(1, min(len_a, len_b))  # denominator
    return len(common) / denominator, len(common), common


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_issue_record, issue_history.deduplicate_section_results,
# issue_history.filter_seen_issues_with_llm, issue_history.get_news_alias_urls, issue_history.make_excluded_item
# - 파라미터: news: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_url(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return (
        str(news.get("url") or "").strip()
        or str(news.get("originallink") or "").strip()
        or str(news.get("link") or "").strip()
    )


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_event_signature, issue_history.build_issue_record,
# issue_history.deduplicate_section_results, issue_history.filter_seen_issues_with_llm,
# issue_history.get_news_alias_titles, issue_history.make_excluded_item
# - 파라미터: news: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_title(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return str(news.get("title") or "").strip()


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_event_signature, issue_history.build_issue_record,
# issue_history.filter_seen_issues_with_llm
# - 파라미터: news: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_summary_or_description(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return (
        str(news.get("summary") or "").strip()
        or str(news.get("description") or "").strip()
        or str(news.get("content") or "").strip()
    )


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.build_event_signature, issue_history.build_issue_record
# - 파라미터: news: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_news_compare_text(news: dict) -> str:
    if not isinstance(news, dict):
        return ""

    parts = [  # parts
        str(news.get("summary") or "").strip(),
        str(news.get("description") or "").strip(),
        str(news.get("content") or "").strip(),
    ]

    for keyword in news.get("group_keywords") or []:  # 키워드
        parts.append(str(keyword).strip())

    keyword = str(news.get("keyword") or "").strip()  # 키워드
    if keyword:
        parts.append(keyword)

    seen = set()  # 확인된
    cleaned = []  # cleaned
    for part in parts:  # part
        if not part or part in seen:
            continue
        seen.add(part)
        cleaned.append(part)

    return " ".join(cleaned)


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: issue_history.build_event_signature, issue_history.get_issue_compare_payload, issue_history.make_issue_id
# - 파라미터: title: str, summary: str
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_compare_payload(title: str, summary: str):
    # 반복 이슈 판정은 원문 문장 그대로가 아니라 "비교 가능한 압축 신호"끼리 비교한다.
    # 이 payload 구조를 과거 이슈 저장, 당일 후보 필터, 메일 발송 직전 최종 중복 제거가 공통으로 사용한다.
    compare_text = normalize_compare_text(f"{title} {summary}")    # 정규화비교문자열
    compact_text = compact_compare_text(compare_text)              # 공백제거압축문자열
    tokens = extract_tokens(compare_text)                          # 본문비교토큰목록
    title_tokens = extract_title_tokens(title)                     # 제목비교토큰목록
    payload = {  # 데이터
        "normalized_title": normalize_title(title),                # 정규화제목
        "normalized_text": compact_text,                           # 정규화본문
        "tokens": tokens,                                          # 본문비교토큰목록
        "title_tokens": title_tokens,                              # 제목비교토큰목록
        "anchor_tokens": extract_anchor_tokens(list(title_tokens or []) + list(tokens or [])),  # 사건앵커토큰목록
        "number_tokens": sorted(extract_number_tokens(f"{title} {summary}")),                   # 숫자토큰목록
        "fingerprint": make_simhash(tokens),                       # SimHash지문
    }
    payload["action_tokens"] = extract_action_tokens(title_tokens + tokens)  # 행위토큰목록
    payload["event_key"] = build_rule_event_key_from_payload(payload)        # 강한규칙사건키
    return payload


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: issue_history.build_issue_record, issue_history.deduplicate_section_results,
# issue_history.filter_seen_issues_with_llm
# - 파라미터: news: dict
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_event_signature(news: dict):
    # 1) 후보 뉴스에서 비교에 쓸 대표 텍스트를 모은다.
    #    제목/요약뿐 아니라 group_article_titles까지 alias로 포함해야, 같은 사건의 관련보도 제목도 과거 이슈와 매칭된다.
    title = get_news_title(news)  # 제목
    compare_text = get_news_compare_text(news) or get_news_summary_or_description(news)  # 비교텍스트텍스트
    raw_alias_titles = get_news_alias_titles(news)  # 원본별칭titles
    alias_titles = [  # 별칭titles
        normalize_title(alias_title)
        for alias_title in raw_alias_titles  # 별칭제목
        if normalize_title(alias_title)
    ]
    alias_titles = unique_nonempty(alias_titles, limit=16)  # 별칭titles
    alias_urls = get_news_alias_urls(news)  # 별칭URL목록

    event_text_parts = [title, compare_text]  # 사건텍스트부분목록
    event_text_parts.extend(raw_alias_titles)
    event_text = " ".join(unique_nonempty(event_text_parts))  # 사건텍스트
    # 2) build_compare_payload()가 제목/본문 정규화, 토큰, 숫자, SimHash fingerprint를 만든다.
    #    이 payload는 반복 이슈 필터와 메일 발송 직전 최종 중복 제거에서 같은 형식으로 재사용된다.
    payload = build_compare_payload(title, event_text)  # 데이터
    alias_title_tokens = extract_tokens(" ".join(raw_alias_titles), max_tokens=60)  # 별칭제목토큰수
    payload["title_tokens"] = unique_nonempty(  # 처리값
        list(payload.get("title_tokens") or []) + alias_title_tokens,
        limit=80,  # 상한
    )
    payload["anchor_tokens"] = extract_anchor_tokens(  # 처리값
        list(payload.get("title_tokens") or []) + list(payload.get("tokens") or [])
    )
    payload["action_tokens"] = extract_action_tokens(  # 처리값
        payload.get("title_tokens", []) + payload.get("tokens", [])
    )
    # 3) event_key는 고유 앵커와 구체 행위어가 충분할 때만 만들어지는 강한 사건 키다.
    #    범용 단어만으로 만든 키는 오탐이 커서 build_rule_event_key_from_payload() 내부에서 빈 값으로 버린다.
    payload["event_key"] = build_rule_event_key_from_payload(payload)  # 처리값
    payload["alias_titles"] = alias_titles  # 처리값
    payload["alias_urls"] = alias_urls  # 처리값
    return payload


# ====================================
# 파일 입출력
# ====================================
# [코드 이해 주석]
# - 역할: 파일이나 환경에서 데이터를 읽어 기본 구조로 적재합니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.get_recent_issues_for_section
# - 파라미터: file_path: Any = HISTORY_FILE_PATH
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 파일 경로를 확인합니다 -> JSON/환경 값을 읽습니다 -> 없거나 깨진 값은 기본 구조로 보정합니다.
def load_issue_history(file_path=HISTORY_FILE_PATH):
    if not os.path.exists(file_path):
        return {"version": 3, "issues": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # 파일객체
            data = json.load(f)  # 데이터
        if not isinstance(data, dict):
            return {"version": 3, "issues": []}
        if "issues" not in data or not isinstance(data["issues"], list):
            data["issues"] = []  # 처리값
        data["version"] = max(int(data.get("version", 1) or 1), 3)  # 처리값
        return data
    except Exception as e:  # 예외객체
        logger.warning(f"⚠️ 이슈 히스토리 읽기 실패, 빈 히스토리로 시작합니다: {e}")
        return {"version": 3, "issues": []}


# [코드 이해 주석]
# - 역할: 메모리의 데이터를 파일이나 외부 저장소에 기록합니다.
# - 호출하는 곳: issue_history.append_sent_issues
# - 파라미터: data: Any, file_path: Any = HISTORY_FILE_PATH
# - 리턴값: 명시 반환값은 없으며 None 또는 내부 상태 변경/부수 효과를 사용합니다.
# - 프로세스 흐름: 저장할 구조를 준비합니다 -> 대상 파일에 기록합니다 -> 실패 시 로그/예외 흐름에 맡깁니다.
def save_issue_history(data, file_path=HISTORY_FILE_PATH):
    dirname = os.path.dirname(file_path)  # dirname
    if dirname:
        os.makedirs(dirname, exist_ok=True)  # existok
    with open(file_path, "w", encoding="utf-8") as f:  # 파일객체
        json.dump(data, f, ensure_ascii=False, indent=2)  # 파일객체,ensureascii






# [코드 이해 주석]
# - 역할: 보존 기간이 지난 오래된 데이터를 제거합니다.
# - 호출하는 곳: issue_history.append_sent_issues
# - 파라미터: history: Any, days: Any = 3
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def prune_old_issues(history, days=3):
    if not isinstance(history, dict):
        return {"version": 3, "issues": []}, 0

    issues = history.get("issues", [])  # 이슈목록
    if not isinstance(issues, list):
        history["issues"] = []  # 처리값
        return history, 0

    today = get_now_kst().date()  # today
    kept_issues = []  # kept이슈목록
    removed_count = 0  # 삭제건수

    for issue in issues:  # 이슈
        saved_date_text = issue.get("saved_date")  # 저장날짜텍스트
        if not saved_date_text:
            removed_count += 1  # 처리값
            continue
        try:
            saved_date = datetime.strptime(saved_date_text, "%Y-%m-%d").date()  # 저장날짜
        except Exception:
            removed_count += 1  # 처리값
            continue

        day_diff = (today - saved_date).days  # daydiff
        if 0 <= day_diff < days:
            kept_issues.append(issue)
        else:
            removed_count += 1  # 처리값

    history["issues"] = kept_issues  # 처리값
    history["version"] = max(int(history.get("version", 3) or 3), 3)  # 처리값
    return history, removed_count


# ====================================
# 히스토리 record 생성/조회
# ====================================
# [코드 이해 주석]
# - 역할: 여러 입력 값을 조합해 식별자, 해시, 키 같은 파생 값을 만듭니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.get_recent_issues_for_section, issue_history.make_issue_id
# - 파라미터: briefing_name: Any, receiver_env: Any, section_name: Any = None, scope_mode: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_history_scope(briefing_name, receiver_env, section_name=None, scope_mode=None):
    mode = str(scope_mode or HISTORY_MATCH_SCOPE or "briefing").strip().lower()  # 방식
    if mode == "receiver":
        return "|".join([str(receiver_env or "").strip()])
    if mode == "section":
        return "|".join([
            str(briefing_name or "").strip(),
            str(receiver_env or "").strip(),
            str(section_name or "").strip(),
        ])
    # 기본: 같은 브리핑/수신자 안에서는 섹션을 넘나들며 반복 이슈로 본다.
    return "|".join([
        str(briefing_name or "").strip(),
        str(receiver_env or "").strip(),
    ])


# [코드 이해 주석]
# - 역할: 여러 입력 값을 조합해 식별자, 해시, 키 같은 파생 값을 만듭니다.
# - 호출하는 곳: issue_history.build_issue_record
# - 파라미터: briefing_name: Any, receiver_env: Any, section_name: Any, title: Any, summary: Any, url: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_issue_id(briefing_name, receiver_env, section_name, title, summary, url=None):
    normalized_url = normalize_url(url)  # 정규화URL
    scope = make_history_scope(  # 범위
        briefing_name=briefing_name,  # briefing이름
        receiver_env=receiver_env,  # 수신자env
        section_name=section_name,  # 섹션이름
        scope_mode=HISTORY_SAVE_SCOPE,  # 범위방식
    )

    if normalized_url:
        return make_hash(f"url|{scope}|{normalized_url}")

    payload = build_compare_payload(title, summary)  # 데이터
    raw = "|".join([  # 원본
        "text",
        scope,
        payload.get("normalized_title", ""),
        payload.get("fingerprint", ""),
        payload.get("normalized_text", "")[:160],
    ])
    return make_hash(raw)


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: issue_history.append_sent_issues
# - 파라미터: briefing_name: Any, subject_prefix: Any, receiver_env: Any, section_name: Any, news: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_issue_record(
    briefing_name, subject_prefix, receiver_env, section_name, news
):
    title = get_news_title(news)  # 제목
    summary = get_news_summary_or_description(news)  # 요약
    compare_text = get_news_compare_text(news) or summary  # 비교텍스트텍스트
    source = str(news.get("source") or "").strip()  # 출처
    url = get_news_url(news)  # URL
    published_at = str(news.get("published_at") or "").strip()  # 발행at
    importance_score = news.get("importance_score")  # 중요도점수

    payload = build_event_signature(news)  # 데이터
    normalized_url = normalize_url(url)  # 정규화URL
    issue_id = make_issue_id(  # 이슈id
        briefing_name=briefing_name,  # briefing이름
        receiver_env=receiver_env,  # 수신자env
        section_name=section_name,  # 섹션이름
        title=title,  # 제목
        summary=compare_text,  # 요약
        url=url,  # URL
    )

    now_kst = get_now_kst()  # 현재한국시간

    return {
        "issue_id": issue_id,
        "saved_at": now_kst.isoformat(),
        "saved_date": now_kst.strftime("%Y-%m-%d"),
        "event_signature_version": 3,
        "briefing_name": briefing_name,
        "subject_prefix": subject_prefix,
        "receiver_env": receiver_env,
        "section_name": section_name,
        "title": title,
        "summary": summary,
        "description": str(news.get("description") or "").strip(),
        "compare_text": compare_text[:1000],
        "source": source,
        "url": url,
        "normalized_url": normalized_url,
        "alias_urls": payload["alias_urls"],
        "alias_titles": payload["alias_titles"],
        "normalized_title": payload["normalized_title"],
        "normalized_text": payload["normalized_text"],
        "content_fingerprint": payload["fingerprint"],
        "content_tokens": payload["tokens"],
        "title_tokens": payload["title_tokens"],
        "anchor_tokens": payload["anchor_tokens"],
        "number_tokens": payload["number_tokens"],
        "event_normalized_text": payload["normalized_text"],
        "event_fingerprint": payload["fingerprint"],
        "event_tokens": payload["tokens"],
        "event_title_tokens": payload["title_tokens"],
        "event_anchor_tokens": payload["anchor_tokens"],
        "event_number_tokens": payload["number_tokens"],
        "event_action_tokens": payload.get("action_tokens", []),
        "event_key": payload.get("event_key", ""),
        "published_at": published_at,
        "importance_score": importance_score,
        # 이전 버전 히스토리와의 호환용 필드. 새 로직에서는 사용하지 않는다.
        "issue_key": "",
        "issue_label": title[:80] if title else "",
        "core_issue_key": "",
        "normalized_issue_key": "",
        "normalized_core_issue_key": "",
    }


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.build_past_issue_indexes,
# issue_history.get_issue_compare_payload
# - 파라미터: issue: dict
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_issue_normalized_url(issue: dict) -> str:
    if not isinstance(issue, dict):
        return ""
    return str(issue.get("normalized_url") or "").strip() or normalize_url(issue.get("url") or "")


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: issue_history.append_sent_issues, issue_history.build_past_issue_indexes
# - 파라미터: issue: dict
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_issue_compare_payload(issue: dict):
    if not isinstance(issue, dict):
        issue = {}  # 이슈

    title = str(issue.get("title") or "")  # 제목
    summary = " ".join([  # 요약
        str(issue.get("summary") or "").strip(),
        str(issue.get("description") or "").strip(),
        str(issue.get("compare_text") or "").strip(),
        str(issue.get("content") or "").strip(),
    ]).strip()

    normalized_title = str(issue.get("normalized_title") or "").strip() or normalize_title(title)  # 정규화제목
    normalized_text = str(issue.get("event_normalized_text") or issue.get("normalized_text") or "").strip()  # 정규화텍스트
    tokens = issue.get("event_tokens") or issue.get("content_tokens")  # 토큰수
    title_tokens = issue.get("event_title_tokens") or issue.get("title_tokens")  # 제목토큰수
    anchor_tokens = issue.get("event_anchor_tokens") or issue.get("anchor_tokens")  # 앵커토큰수
    number_tokens = issue.get("event_number_tokens") or issue.get("number_tokens")  # 숫자값토큰수
    action_tokens = issue.get("event_action_tokens") or issue.get("action_tokens")  # 행위토큰수
    fingerprint = str(issue.get("event_fingerprint") or issue.get("content_fingerprint") or "").strip()  # fingerprint
    event_key = str(issue.get("event_key") or "").strip()  # 사건키
    alias_titles = issue.get("alias_titles")  # 별칭titles
    alias_urls = issue.get("alias_urls")  # 별칭URL목록

    if (
        not normalized_text
        or not isinstance(tokens, list)
        or not isinstance(title_tokens, list)
        or not isinstance(anchor_tokens, list)
        or not isinstance(number_tokens, list)
        or not isinstance(action_tokens, list)
        or not isinstance(alias_titles, list)
        or not isinstance(alias_urls, list)
        or not fingerprint
    ):
        payload = build_compare_payload(title, summary)  # 데이터
        normalized_text = normalized_text or payload["normalized_text"]  # 정규화텍스트
        tokens = tokens if isinstance(tokens, list) and tokens else payload["tokens"]  # 토큰수
        title_tokens = (  # 제목토큰수
            title_tokens
            if isinstance(title_tokens, list) and title_tokens
            else payload["title_tokens"]
        )
        anchor_tokens = (  # 앵커토큰수
            anchor_tokens
            if isinstance(anchor_tokens, list) and anchor_tokens
            else payload["anchor_tokens"]
        )
        number_tokens = (  # 숫자값토큰수
            number_tokens
            if isinstance(number_tokens, list)
            else payload["number_tokens"]
        )
        action_tokens = (  # 행위토큰수
            action_tokens
            if isinstance(action_tokens, list) and action_tokens
            else payload.get("action_tokens", [])
        )
        alias_titles = (  # 별칭titles
            alias_titles
            if isinstance(alias_titles, list) and alias_titles
            else unique_nonempty([normalized_title], limit=16)  # 상한
        )
        alias_urls = (  # 별칭URL목록
            alias_urls
            if isinstance(alias_urls, list)
            else unique_nonempty([get_issue_normalized_url(issue)], limit=16)  # 상한
        )
        fingerprint = fingerprint or payload["fingerprint"]  # fingerprint

    if not event_key:
        event_key = build_rule_event_key_from_payload({  # 사건키
            "tokens": tokens or [],
            "title_tokens": title_tokens or [],
            "anchor_tokens": anchor_tokens or [],
            "number_tokens": number_tokens or [],
            "action_tokens": action_tokens or [],
        })

    if not isinstance(action_tokens, list) or not action_tokens:
        action_tokens = extract_action_tokens((title_tokens or []) + (tokens or []))  # 행위토큰수

    return {
        "normalized_title": normalized_title,
        "normalized_text": normalized_text,
        "tokens": tokens or [],
        "title_tokens": title_tokens or [],
        "anchor_tokens": anchor_tokens or [],
        "number_tokens": number_tokens or [],
        "action_tokens": action_tokens or [],
        "alias_titles": alias_titles or [],
        "alias_urls": alias_urls or [],
        "fingerprint": fingerprint,
        "event_key": event_key,
    }


# [코드 이해 주석]
# - 역할: 메일에 실제 발송된 section_results의 summaries를 이슈 히스토리에 저장한다.
# - 호출하는 곳: main.main
# - 파라미터: briefing_name: Any, subject_prefix: Any, receiver_env: Any, section_results: Any, file_path: Any =
# HISTORY_FILE_PATH, keep_days: Any = 3
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def append_sent_issues(
    briefing_name, subject_prefix, receiver_env,
    section_results, file_path=HISTORY_FILE_PATH, keep_days=3  # 섹션결과목록,파일경로
):
    """
    메일에 실제 발송된 section_results의 summaries를 이슈 히스토리에 저장한다.
    LLM/issue_key 생성 없이 비교용 fingerprint만 저장한다.
    """
    # 발송 성공 후에만 호출되므로, 여기서 저장되는 record는 다음 실행의 반복 이슈 필터 기준이 된다.
    # 같은 메일 안에서 중복 record가 여러 번 저장되면 다음 실행에서 불필요하게 많은 후보가 제외될 수 있어 먼저 색인을 만든다.
    history = load_issue_history(file_path)                        # 기존이슈히스토리데이터
    history, pruned_count = prune_old_issues(history, days=keep_days)  # 보존기간정리후히스토리와삭제수

    existing_issue_ids = set()                                     # 현재저장대상이슈ID집합
    existing_scope_url_keys = set()                                # 현재저장대상URL키집합
    existing_scope_title_keys = set()                              # 현재저장대상제목키집합
    existing_scope_event_keys = set()                              # 현재저장대상사건키집합

    for item in history.get("issues", []):                         # 기존히스토리이슈
        issue_id = item.get("issue_id")                            # 저장된이슈ID
        if issue_id:
            existing_issue_ids.add(str(issue_id))

        scope = make_history_scope(  # 범위
            briefing_name=item.get("briefing_name"),
            receiver_env=item.get("receiver_env"),
            section_name=item.get("section_name"),
            scope_mode=HISTORY_SAVE_SCOPE,  # 범위방식
        )
        item_url = get_issue_normalized_url(item)                  # 기존이슈정규화URL
        item_payload = get_issue_compare_payload(item)             # 기존이슈비교payload
        item_title = item_payload.get("normalized_title", "")      # 기존이슈정규화제목
        item_event_key = item_payload.get("event_key", "")         # 기존이슈사건키

        if item_event_key:
            existing_scope_event_keys.add(f"{scope}|{item_event_key}")
        for alias_url in unique_nonempty([item_url] + (item_payload.get("alias_urls") or [])):  # aliasURL
            existing_scope_url_keys.add(f"{scope}|{alias_url}")
        for alias_title in unique_nonempty([item_title] + (item_payload.get("alias_titles") or [])):  # 별칭제목
            existing_scope_title_keys.add(f"{scope}|{alias_title}")

    # historical_*는 "실행 전부터 있던 이슈"를 구분하기 위한 스냅샷이다.
    # existing_*는 이번 실행에서 새로 추가하는 record까지 계속 갱신해, 같은 발송 결과 안의 중복 저장도 막는다.
    historical_issue_ids = set(existing_issue_ids)                 # 실행전이슈ID집합
    historical_scope_url_keys = set(existing_scope_url_keys)       # 실행전URL키집합
    historical_scope_title_keys = set(existing_scope_title_keys)   # 실행전제목키집합
    historical_scope_event_keys = set(existing_scope_event_keys)   # 실행전사건키집합

    new_records = []                                               # 이번실행신규저장이슈목록
    skipped_duplicate_count = 0                                    # 중복으로저장건너뛴수

    for section_result in section_results or []:                   # 발송된섹션결과
        section_name = section_result.get("section_name", "뉴스 섹션")  # 섹션명
        summaries = section_result.get("summaries", []) or []     # 실제발송뉴스요약목록

        for news in summaries:                                     # 발송뉴스요약
            record = build_issue_record(                           # 저장할이슈record
                briefing_name=briefing_name,  # briefing이름
                subject_prefix=subject_prefix,  # 메일제목prefix
                receiver_env=receiver_env,  # 수신자env
                section_name=section_name,  # 섹션이름
                news=news,  # 뉴스
            )

            scope = make_history_scope(  # 범위
                briefing_name=briefing_name,  # briefing이름
                receiver_env=receiver_env,  # 수신자env
                section_name=section_name,  # 섹션이름
                scope_mode=HISTORY_SAVE_SCOPE,  # 범위방식
            )
            url_key = f"{scope}|{record.get('normalized_url', '')}"      # scope포함URL중복키
            title_key = f"{scope}|{record.get('normalized_title', '')}"  # scope포함제목중복키
            event_key = str(record.get("event_key") or "").strip()       # record사건키
            event_key_full = f"{scope}|{event_key}" if event_key else ""  # scope포함사건중복키
            alias_url_keys = [  # 별칭URL키목록
                f"{scope}|{alias_url}"
                for alias_url in unique_nonempty(record.get("alias_urls") or [])
            ]
            alias_title_keys = [  # 별칭제목키목록
                f"{scope}|{alias_title}"
                for alias_title in unique_nonempty(record.get("alias_titles") or [])
            ]

            if record["issue_id"] in historical_issue_ids:
                skipped_duplicate_count += 1  # 처리값
                continue
            if record["issue_id"] in existing_issue_ids:
                skipped_duplicate_count += 1  # 처리값
                continue
            if record.get("normalized_url") and url_key in historical_scope_url_keys:
                skipped_duplicate_count += 1  # 처리값
                continue
            if any(key in historical_scope_url_keys for key in alias_url_keys):
                skipped_duplicate_count += 1  # 처리값
                continue
            if record.get("normalized_title") and title_key in historical_scope_title_keys:
                skipped_duplicate_count += 1  # 처리값
                continue
            if any(key in historical_scope_title_keys for key in alias_title_keys):
                skipped_duplicate_count += 1  # 처리값
                continue
            if event_key_full and event_key_full in historical_scope_event_keys:
                skipped_duplicate_count += 1  # 처리값
                continue

            existing_issue_ids.add(record["issue_id"])
            if record.get("normalized_url"):
                existing_scope_url_keys.add(url_key)
            if record.get("normalized_title"):
                existing_scope_title_keys.add(title_key)
            if event_key_full:
                existing_scope_event_keys.add(event_key_full)
            for key in alias_url_keys:  # 키
                existing_scope_url_keys.add(key)
            for key in alias_title_keys:  # 키
                existing_scope_title_keys.add(key)

            new_records.append(record)

    if new_records:
        history["issues"].extend(new_records)

    history["last_updated_at"] = get_now_kst().isoformat()  # 처리값
    history["version"] = max(int(history.get("version", 3) or 3), 3)  # 처리값
    save_issue_history(history, file_path)

    return {
        "success": True,
        "saved_count": len(new_records),
        "skipped_duplicate_count": skipped_duplicate_count,
        "pruned_count": pruned_count,
        "total_count": len(history.get("issues", [])),
        "file_path": file_path,
    }


# ====================================
# 최근 이슈 조회
# ====================================
# [코드 이해 주석]
# - 역할: 최근 발송 이슈 조회.
# - 호출하는 곳: issue_history.filter_seen_issues_with_llm
# - 파라미터: briefing_name: Any, receiver_env: Any, section_name: Any, days: Any = 3, file_path: Any = HISTORY_FILE_PATH,
# scope_mode: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_recent_issues_for_section(
    briefing_name, receiver_env, section_name,
    days=3, file_path=HISTORY_FILE_PATH, scope_mode=None  # 일수
):
    """
    최근 발송 이슈 조회.
    기본은 briefing 범위라서 같은 메일 안에서 섹션이 달라져도 반복 이슈로 비교한다.
    HISTORY_MATCH_SCOPE=section 으로 두면 기존 섹션 단위 동작으로 되돌릴 수 있다.
    """
    history = load_issue_history(file_path)  # 히스토리
    issues = history.get("issues", [])  # 이슈목록
    today = get_now_kst().date()  # today
    recent_issues = []  # recent이슈목록

    target_scope = make_history_scope(  # target범위
        briefing_name=briefing_name,  # briefing이름
        receiver_env=receiver_env,  # 수신자env
        section_name=section_name,  # 섹션이름
        scope_mode=scope_mode or HISTORY_MATCH_SCOPE,  # 범위방식
    )

    for issue in issues:  # 이슈
        issue_scope = make_history_scope(  # 이슈범위
            briefing_name=issue.get("briefing_name"),
            receiver_env=issue.get("receiver_env"),
            section_name=issue.get("section_name"),
            scope_mode=scope_mode or HISTORY_MATCH_SCOPE,  # 범위방식
        )
        if issue_scope != target_scope:
            continue

        saved_date_text = issue.get("saved_date")  # 저장날짜텍스트
        if not saved_date_text:
            continue
        try:
            saved_date = datetime.strptime(saved_date_text, "%Y-%m-%d").date()  # 저장날짜
        except Exception:
            continue

        if 0 <= (today - saved_date).days < days:
            recent_issues.append(issue)

    return recent_issues


# ====================================
# 반복 이슈 필터
# ====================================
# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 HTML, payload, 메시지, 결과 dict 같은 출력 구조를 만듭니다.
# - 호출하는 곳: issue_history.filter_seen_issues_with_llm
# - 파라미터: past_issues: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 문자열/dict/HTML 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def build_past_issue_indexes(past_issues):
    past_by_url = {}  # past기준별URL
    past_payloads = []  # pastpayloads

    # 1) 과거 이슈를 URL 색인과 payload 목록 두 형태로 준비한다.
    #    URL은 가장 빠른 exact match에 쓰고, payload는 제목/토큰/SimHash 기반 의미 비교에 쓴다.
    for issue in past_issues or []:  # 이슈
        issue_url = get_issue_normalized_url(issue)  # 이슈URL
        payload = get_issue_compare_payload(issue)  # 데이터

        if issue_url and issue_url not in past_by_url:
            past_by_url[issue_url] = issue  # 처리값
        for alias_url in payload.get("alias_urls") or []:  # aliasURL
            if alias_url and alias_url not in past_by_url:
                past_by_url[alias_url] = issue  # 처리값

        # 2) issue 원문 전체를 매번 다시 파싱하지 않도록 비교에 필요한 필드만 뽑아 둔다.
        #    filter_seen_issues_with_llm()의 후보 루프는 이 목록과 반복 비교하므로, 여기서 구조를 고정해둔다.
        past_payloads.append({
            "issue": issue,
            "title": issue.get("title", ""),
            "normalized_title": payload.get("normalized_title", ""),
            "normalized_text": payload.get("normalized_text", ""),
            "tokens": payload.get("tokens", []),
            "title_tokens": payload.get("title_tokens", []),
            "anchor_tokens": payload.get("anchor_tokens", []),
            "number_tokens": payload.get("number_tokens", []),
            "action_tokens": payload.get("action_tokens", []),
            "alias_titles": payload.get("alias_titles", []),
            "alias_urls": payload.get("alias_urls", []),
            "fingerprint": payload.get("fingerprint", ""),
            "event_key": payload.get("event_key", ""),
        })

    return {
        "past_by_url": past_by_url,
        "past_payloads": past_payloads,
    }


# [코드 이해 주석]
# - 역할: 후보와 과거/오늘 유지 후보가 같은 반복 이슈인지 규칙 기반으로 판단한다.
# - 호출하는 곳: issue_history.find_matching_payload
# - 파라미터: candidate_payload: Any, past_payload: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def judge_duplicate_by_payload(candidate_payload, past_payload):
    """
    후보와 과거/오늘 유지 후보가 같은 반복 이슈인지 규칙 기반으로 판단한다.

    과제외를 줄이기 위해 판정 강도를 나눈다.
    - exact: URL/제목/event_key처럼 매우 강한 근거
    - strong: 고유 앵커 여러 개 + 제목/행위/숫자 보조 근거
    - soft: 범용 앵커나 일반 행위어 중심. soft만으로는 제외하지 않는다.

    Returns: (is_duplicate, method, score_text)
    """
    cand_title = candidate_payload.get("normalized_title", "")  # 후보제목
    past_title = past_payload.get("normalized_title", "")  # past제목
    candidate_titles = unique_nonempty(  # 후보titles
        [cand_title] + (candidate_payload.get("alias_titles") or []),
        limit=16,  # 상한
    )
    past_titles = unique_nonempty(  # pasttitles
        [past_title] + (past_payload.get("alias_titles") or []),
        limit=16,  # 상한
    )

    cand_numbers = set(candidate_payload.get("number_tokens") or [])  # 후보numbers
    past_numbers = set(past_payload.get("number_tokens") or [])  # pastnumbers
    number_conflict = bool(cand_numbers and past_numbers and not (cand_numbers & past_numbers))  # 숫자값conflict

    cand_event_key = str(candidate_payload.get("event_key") or "").strip()  # 후보사건키
    past_event_key = str(past_payload.get("event_key") or "").strip()  # past사건키
    if cand_event_key and past_event_key and cand_event_key == past_event_key and not number_conflict:
        return True, "event_key", f"규칙 기반 사건 키 동일({cand_event_key}) | strength=exact"

    shared_anchor_tokens = (  # shared앵커토큰수
        set(candidate_payload.get("anchor_tokens") or [])
        & set(past_payload.get("anchor_tokens") or [])
    )
    shared_action_tokens = (  # shared행위토큰수
        set(candidate_payload.get("action_tokens") or [])
        & set(past_payload.get("action_tokens") or [])
    )
    soft_shared_anchor_tokens = soft_common_tokens(  # softshared앵커토큰수
        candidate_payload.get("anchor_tokens") or [],
        past_payload.get("anchor_tokens") or [],
    )
    all_shared_anchor_tokens = unique_nonempty(list(shared_anchor_tokens) + list(soft_shared_anchor_tokens), limit=30)  # 전체shared앵커토큰수
    strong_shared_anchor_tokens = strong_common_anchor_tokens(all_shared_anchor_tokens)  # 강한shared앵커토큰수

    shared_anchor_count = len(all_shared_anchor_tokens)  # shared앵커건수
    strong_shared_anchor_count = len(strong_shared_anchor_tokens)  # 강한shared앵커건수
    shared_anchor = shared_anchor_count > 0  # shared앵커
    shared_action = bool(shared_action_tokens)  # shared행위
    specific_shared_action = has_specific_action(shared_action_tokens)  # specificshared행위

    title_overlap, title_common_count = token_overlap_score(  # 제목overlap,제목공통건수
        candidate_payload.get("title_tokens", []),
        past_payload.get("title_tokens", []),
    )
    soft_title_overlap, soft_title_common_count, soft_title_common_tokens = soft_token_overlap_score(  # soft제목overlap,soft제목공통건수,soft제목공통토큰수
        candidate_payload.get("title_tokens", []),
        past_payload.get("title_tokens", []),
    )
    title_common_best = max(title_common_count, soft_title_common_count)  # 제목공통최적
    title_overlap_best = max(title_overlap, soft_title_overlap)  # 제목overlap최적

    # 숫자 불일치는 기본적으로 오탐 방지 신호다.
    # 다만 고유 앵커가 충분히 겹치고 제목/구체 행위가 받쳐주면 같은 사건으로 본다.
    if number_conflict and (
        (strong_shared_anchor_count >= 3 and title_common_best >= 3)
        or (strong_shared_anchor_count >= 2 and specific_shared_action and title_common_best >= 2)
        or strong_shared_anchor_count >= 4
    ):
        number_conflict = False  # 숫자값conflict

    best_title_score = 0.0  # 최적제목점수
    for candidate_title in candidate_titles:  # 후보제목
        for past_alias_title in past_titles:  # past별칭제목
            if not candidate_title or not past_alias_title:
                continue

            if candidate_title == past_alias_title:
                return True, "title_exact", "정규화 제목/별칭 제목 동일 | strength=exact"

            if min(len(candidate_title), len(past_alias_title)) >= 14:
                shorter, longer = sorted([candidate_title, past_alias_title], key=len)  # 짧은값,긴값
                if shorter in longer and not number_conflict:
                    # 짧은 제목이 너무 일반적인 경우를 막기 위해 제목 토큰도 확인한다.
                    if title_common_best >= 3 or strong_shared_anchor_count >= 2:
                        return True, "title_contains", "후보 제목과 과거 별칭 제목이 포함 관계 | strength=exact"

            title_score = SequenceMatcher(None, candidate_title, past_alias_title).ratio()  # 제목점수
            if title_score > best_title_score:
                best_title_score = title_score  # 최적제목점수

    if best_title_score >= TITLE_SIMILARITY_THRESHOLD and not number_conflict:
        return True, "title_similarity", f"제목/별칭 제목 유사도 {best_title_score:.2f} | strength=strong"

    if (
        best_title_score >= 0.80
        and title_common_best >= 3
        and strong_shared_anchor_count >= 1
        and not number_conflict
    ):
        return True, "title_similarity_anchor", (
            f"제목/별칭 제목 유사도 {best_title_score:.2f}, 제목 공통 토큰 {title_common_best}개, "
            f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
        )

    # 같은 행사/실적/협력 발표처럼 제목 표현은 달라도 고유 앵커가 충분히 겹치는 경우.
    # 단, 경제/부동산/증권처럼 범용어가 많은 섹션에서 과제외가 커지므로 고유 앵커와 보조 조건을 함께 요구한다.
    if (
        strong_shared_anchor_count >= 4
        and title_common_best >= 2
        and not number_conflict
    ):
        return True, "anchor_bundle", (
            f"공통 고유 앵커 {strong_shared_anchor_count}개({','.join(strong_shared_anchor_tokens[:5])}), "
            f"제목 공통 토큰 {title_common_best}개 | strength=strong"
        )

    if (
        strong_shared_anchor_count >= 3
        and specific_shared_action
        and title_common_best >= 2
        and not number_conflict
    ):
        return True, "anchor_action", (
            f"공통 고유 앵커 {strong_shared_anchor_count}개, 구체 행위 {','.join(sorted(shared_action_tokens))}, "
            f"제목 공통 토큰 {title_common_best}개 | strength=strong"
        )

    if (
        strong_shared_anchor_count >= 3
        and shared_action
        and title_common_best >= 3
        and not number_conflict
    ):
        return True, "anchor_action_title", (
            f"공통 고유 앵커 {strong_shared_anchor_count}개, 공통 행위 {','.join(sorted(shared_action_tokens))}, "
            f"제목 공통 토큰 {title_common_best}개 | strength=strong"
        )

    if (
        strong_shared_anchor_count >= 3
        and title_common_best >= 4
        and not number_conflict
    ):
        return True, "anchor_title_bundle", (
            f"공통 고유 앵커 {strong_shared_anchor_count}개, 제목 공통 토큰 {title_common_best}개 | strength=strong"
        )

    # 기존 distinctive_anchor_pair는 공통 앵커 2개만으로 과제외가 컸다.
    # 이제는 고유 앵커 3개 이상 + 제목 공통 3개 이상일 때만 보조적으로 사용한다.
    if (
        strong_shared_anchor_count >= 3
        and title_common_best >= 3
        and not number_conflict
    ):
        return True, "distinctive_anchor_pair", (
            f"고유 앵커 {strong_shared_anchor_count}개({','.join(strong_shared_anchor_tokens[:5])}), "
            f"제목 공통 토큰 {title_common_best}개 | strength=strong"
        )

    if (
        title_overlap_best >= 0.58
        and title_common_best >= 4
        and strong_shared_anchor_count >= 1
        and not number_conflict
    ):
        return True, "title_token_overlap", (
            f"제목 토큰 겹침률 {title_overlap_best:.2f}, 공통 {title_common_best}개, "
            f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
        )

    overlap_score, common_count = token_overlap_score(  # overlap점수,공통건수
        candidate_payload.get("tokens", []),
        past_payload.get("tokens", []),
    )
    soft_overlap_score, soft_common_count, soft_common_token_list = soft_token_overlap_score(  # softoverlap점수,soft공통건수,soft공통토큰list
        candidate_payload.get("tokens", []),
        past_payload.get("tokens", []),
    )
    overlap_score = max(overlap_score, soft_overlap_score)  # overlap점수
    common_count = max(common_count, soft_common_count)  # 공통건수

    cand_text = candidate_payload.get("normalized_text", "")  # 후보텍스트
    past_text = past_payload.get("normalized_text", "")  # past텍스트
    if cand_text and past_text:
        text_score = SequenceMatcher(None, cand_text, past_text).ratio()  # 텍스트점수
        if (
            text_score >= TEXT_SIMILARITY_THRESHOLD
            and not number_conflict
            and common_count >= MIN_COMMON_TOKEN_COUNT
            and strong_shared_anchor_count >= 1
        ):
            return True, "text_similarity", (
                f"본문 유사도 {text_score:.2f}, 공통 토큰 {common_count}개, "
                f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
            )

        if (
            text_score >= 0.76
            and common_count >= MIN_COMMON_TOKEN_COUNT + 1
            and strong_shared_anchor_count >= 2
            and not number_conflict
        ):
            return True, "text_similarity_anchor", (
                f"본문 유사도 {text_score:.2f}, 공통 토큰 {common_count}개, "
                f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
            )

    if (
        common_count >= MIN_COMMON_TOKEN_COUNT + 1
        and overlap_score >= max(TOKEN_OVERLAP_THRESHOLD, 0.70)
        and not number_conflict
        and strong_shared_anchor_count >= 2
    ):
        return True, "token_overlap", (
            f"토큰 겹침률 {overlap_score:.2f}, 공통 {common_count}개, "
            f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
        )

    distance = simhash_distance(  # distance
        candidate_payload.get("fingerprint", ""),
        past_payload.get("fingerprint", ""),
    )
    if distance is not None and distance <= SIMHASH_DISTANCE_THRESHOLD:
        # SimHash만으로 과하게 지워지는 것을 막기 위해 고유 앵커/토큰 공통 조건을 강화한다.
        if common_count >= max(4, MIN_COMMON_TOKEN_COUNT) and strong_shared_anchor_count >= 2 and not number_conflict:
            return True, "simhash", (
                f"SimHash 거리 {distance}, 공통 토큰 {common_count}개, "
                f"고유 앵커 {strong_shared_anchor_count}개 | strength=strong"
            )

    return False, "", ""

# [코드 이해 주석]
# - 역할: 목록 안에서 조건에 맞는 payload나 항목을 찾습니다.
# - 호출하는 곳: issue_history.deduplicate_section_results, issue_history.filter_seen_issues_with_llm
# - 파라미터: candidate_payload: Any, past_payloads: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def find_matching_payload(candidate_payload, past_payloads):
    for past_payload in past_payloads or []:  # past데이터
        is_dup, method, detail = judge_duplicate_by_payload(candidate_payload, past_payload)  # isdup,method,detail
        if is_dup:
            return past_payload, method, detail
    return None, "", ""


# [코드 이해 주석]
# - 역할: 반복/내부 중복으로 제외된 후보의 진단 정보를 표준 형태로 만든다.
# - 호출하는 곳: issue_history.filter_seen_issues_with_llm
# - 파라미터: index: Any, news: Any, method: Any, reason: Any, matched_title: Any = '', detail: Any = '',
# candidate_payload: Any = None
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def make_excluded_item(
    index,
    news,
    method,
    reason,
    matched_title="",
    detail="",
    candidate_payload=None,  # 후보데이터
):
    """
    반복/내부 중복으로 제외된 후보의 진단 정보를 표준 형태로 만든다.
    실행 흐름에는 쓰지 않고, 로그/대시보드/추후 디버깅용으로 반환한다.
    """
    if not isinstance(news, dict):
        news = {}  # 뉴스
    candidate_payload = candidate_payload if isinstance(candidate_payload, dict) else {}  # 후보데이터

    return {
        "index": index,
        "title": get_news_title(news),
        "source": str(news.get("source") or "").strip(),
        "url": get_news_url(news),
        "method": str(method or "").strip(),
        "reason": str(reason or "").strip(),
        "matched_title": str(matched_title or "").strip(),
        "detail": str(detail or "").strip(),
        "normalized_title": candidate_payload.get("normalized_title", ""),
        "event_key": candidate_payload.get("event_key", ""),
        "anchor_tokens": list(candidate_payload.get("anchor_tokens") or [])[:12],
        "action_tokens": list(candidate_payload.get("action_tokens") or [])[:8],
        "number_tokens": list(candidate_payload.get("number_tokens") or [])[:8],
    }


# [코드 이해 주석]
# - 역할: 기존 함수명은 유지하지만 LLM을 사용하지 않는다.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: briefing_name: Any, receiver_env: Any, section_name: Any, candidate_news: Any, days: Any = 3, file_path: Any
# = HISTORY_FILE_PATH, remove_internal_duplicates: Any = True
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 후보를 순회합니다 -> 제외/중복 기준을 적용합니다 -> 유지 목록과 제외 사유를 반환합니다.
def filter_seen_issues_with_llm(
    briefing_name, receiver_env, section_name,
    candidate_news, days=3, file_path=HISTORY_FILE_PATH,  # 후보뉴스,일수
    remove_internal_duplicates=True  # removeinternal중복목록
):
    """
    기존 함수명은 유지하지만 LLM을 사용하지 않는다.
    최근 N일간 이미 메일에 보낸 이슈와 후보 뉴스를 규칙 기반으로 비교해 제외한다.
    remove_internal_duplicates=False이면 당일 후보끼리의 의미 중복은 보존한다.
    이 경우 뒤의 news_grouper.py가 같은 사건을 묶어 관련보도 건수를 계산한다.
    """
    reset_token_stats()

    # 1) 후보가 없으면 히스토리를 읽지 않고 빈 결과를 반환한다.
    #    main.py는 이 dict의 filtered_news와 통계 키를 그대로 scrape_stats에 반영한다.
    if not candidate_news:
        return {
            "success": True,                                              # 필터성공여부
            "message": "후보 뉴스 없음",                                  # 필터상태메시지
            "filtered_news": [],                                           # 필터통과뉴스목록
            "excluded_count": 0,                                           # 전체제외건수
            "excluded_items": [],                                          # 제외상세목록
            "past_issue_count": 0,                                         # 비교대상과거이슈수
            "prefilter_excluded_count": 0,                                 # 과거이슈규칙제외수
            "llm_excluded_count": 0,                                       # LLM제외수호환키
            "core_key_excluded_count": 0,                                  # 핵심키제외수호환키
            "internal_duplicate_count": 0,                                 # 후보내부중복제외수
            "url_excluded_count": 0,                                       # URL기준제외수
            "title_excluded_count": 0,                                     # 제목기준제외수
            "text_excluded_count": 0,                                      # 본문유사도제외수
            "token_overlap_excluded_count": 0,                             # 토큰겹침제외수
            "simhash_excluded_count": 0,                                   # SimHash제외수
            "internal_duplicate_filter_enabled": bool(remove_internal_duplicates),  # 후보내부중복필터사용여부
            "token_stats": get_last_token_stats(),                         # 반복이슈토큰통계
        }

    # 2) 같은 브리핑/수신자 범위의 최근 발송 이슈만 불러온다.
    #    scope 설정에 따라 섹션을 넘나들며 반복 이슈로 볼지, 섹션별로만 볼지가 결정된다.
    past_issues = get_recent_issues_for_section(  # past이슈목록
        briefing_name=briefing_name,  # briefing이름
        receiver_env=receiver_env,  # 수신자env
        section_name=section_name,  # 섹션이름
        days=days,  # 일수
        file_path=file_path,  # 파일경로
    )

    # 3) 과거 이슈를 빠른 URL 색인과 의미 비교 payload 목록으로 변환한다.
    #    후보마다 과거 이슈 JSON 전체를 다시 정규화하지 않기 위한 전처리 단계다.
    past_indexes = build_past_issue_indexes(past_issues)  # pastindexes
    past_by_url = past_indexes["past_by_url"]  # past기준별URL
    past_payloads = past_indexes["past_payloads"]  # pastpayloads

    excluded_items = []                                            # 제외후보상세목록
    filtered_news = []                                             # 반복이슈필터통과뉴스목록

    url_excluded_count = 0                                         # URL기준제외수
    title_excluded_count = 0                                       # 제목기준제외수
    text_excluded_count = 0                                        # 본문유사도제외수
    token_overlap_excluded_count = 0                               # 토큰겹침제외수
    simhash_excluded_count = 0                                     # SimHash제외수
    internal_duplicate_count = 0                                   # 후보내부중복제외수

    # 4) 오늘 후보 내부 중복은 선택적으로만 제거한다.
    #    메인 브리핑 흐름에서는 False로 두어 뒤의 news_grouper.py가 같은 사건을 묶고 관련보도 건수를 계산하게 한다.
    #    테스트/단독 사용에서 True로 두면 히스토리 필터 단계에서 오늘 후보끼리도 의미 중복을 제거한다.
    seen_today_payloads = []                                       # 오늘유지후보비교payload목록
    seen_today_urls = set()                                        # 오늘유지후보URL집합

    # 5) 후보를 하나씩 과거 이슈와 비교한다.
    #    유지되는 후보는 filtered_news에 남고, 제외되는 후보는 excluded_items에 이유와 매칭 근거를 남긴다.
    for idx, news in enumerate(candidate_news or []):  # 순번,뉴스
        news_title = get_news_title(news)                          # 후보제목
        news_summary = get_news_summary_or_description(news)       # 후보요약또는설명
        news_url = normalize_url(get_news_url(news))               # 후보정규화URL
        candidate_payload = build_event_signature(news)            # 후보비교payload

        # 5-1) 과거 발송 URL 완전 일치.
        #      같은 URL은 같은 기사로 볼 수 있으므로 가장 먼저 빠르게 제외한다.
        if news_url and news_url in past_by_url:
            matched = past_by_url[news_url]                        # URL일치과거이슈
            excluded_items.append(make_excluded_item(
                index=idx,  # 순번
                news=news,  # 뉴스
                method="url",
                reason="같은 URL의 기사가 최근 발송 이력에 있음",
                matched_title=matched.get("title", ""),
                detail="정규화 URL 완전 일치",
                candidate_payload=candidate_payload,  # 후보데이터
            ))
            url_excluded_count += 1  # 처리값
            continue

        # 5-2) 과거 발송 텍스트/fingerprint 비교.
        #      URL은 달라도 제목/본문/토큰/SimHash가 같은 사건으로 판단되면 반복 이슈로 제외한다.
        matched_payload, method, detail = find_matching_payload(candidate_payload, past_payloads)  # 과거이슈의미중복판정결과
        if matched_payload:
            matched_issue = matched_payload.get("issue", {})       # 매칭된과거이슈원본
            excluded_items.append(make_excluded_item(
                index=idx,  # 순번
                news=news,  # 뉴스
                method=method,  # method
                reason="최근 발송 이력과 유사함",
                matched_title=matched_issue.get("title", ""),
                detail=detail,  # detail
                candidate_payload=candidate_payload,  # 후보데이터
            ))
            if method.startswith("title_"):
                title_excluded_count += 1  # 처리값
            elif method.startswith("text_"):
                text_excluded_count += 1  # 처리값
            elif method == "token_overlap":
                token_overlap_excluded_count += 1  # 처리값
            elif method == "simhash":
                simhash_excluded_count += 1  # 처리값
            continue

        if remove_internal_duplicates:
            # 5-3) 오늘 후보 내부 URL 중복.
            #      remove_internal_duplicates=True인 호출에서만 적용한다.
            if news_url and news_url in seen_today_urls:
                excluded_items.append(make_excluded_item(
                    index=idx,  # 순번
                    news=news,  # 뉴스
                    method="internal_url",
                    reason="오늘 후보 내부에서 같은 URL이 이미 유지됨",
                    matched_title="오늘 후보 내부 중복 URL",
                    detail="정규화 URL 완전 일치",
                    candidate_payload=candidate_payload,  # 후보데이터
                ))
                internal_duplicate_count += 1  # 처리값
                continue

            # 5-4) 오늘 후보 내부 텍스트/fingerprint 중복.
            #      main.py에서는 관련보도 그룹화를 위해 꺼두지만, 독립 필터 용도로 쓸 때는 여기서 제거할 수 있다.
            matched_today, today_method, today_detail = find_matching_payload(candidate_payload, seen_today_payloads)  # 오늘후보내부중복판정결과
            if matched_today:
                excluded_items.append(make_excluded_item(
                    index=idx,  # 순번
                    news=news,  # 뉴스
                    method=f"internal_{today_method}",
                    reason="오늘 후보 내부에서 이미 유지한 기사와 유사함",
                    matched_title=matched_today.get("title", ""),
                    detail=today_detail,  # detail
                    candidate_payload=candidate_payload,  # 후보데이터
                ))
                internal_duplicate_count += 1  # 처리값
                continue

        # 6) 여기까지 온 후보는 최근 발송 이슈와 겹치지 않는 신규 후보로 본다.
        #    내부 중복 제거가 켜진 경우에는 이후 후보가 이 기사와도 비교할 수 있도록 seen_today_*에 등록한다.
        filtered_news.append(news)
        if remove_internal_duplicates:
            if news_url:
                seen_today_urls.add(news_url)
            seen_today_payloads.append({
                "title": news_title,
                "normalized_title": candidate_payload.get("normalized_title", ""),
                "normalized_text": candidate_payload.get("normalized_text", ""),
                "tokens": candidate_payload.get("tokens", []),
                "title_tokens": candidate_payload.get("title_tokens", []),
                "anchor_tokens": candidate_payload.get("anchor_tokens", []),
                "number_tokens": candidate_payload.get("number_tokens", []),
                "action_tokens": candidate_payload.get("action_tokens", []),
                "alias_titles": candidate_payload.get("alias_titles", []),
                "alias_urls": candidate_payload.get("alias_urls", []),
                "fingerprint": candidate_payload.get("fingerprint", ""),
                "event_key": candidate_payload.get("event_key", ""),
            })

    excluded_count = len(excluded_items)                           # 전체제외건수

    logger.info(
        f"🧹 [{section_name}] 규칙 기반 반복 이슈 필터 완료: "
        f"후보 {len(candidate_news)}개 → {len(filtered_news)}개 / "
        f"제외 {excluded_count}개 / 과거 이슈 {len(past_issues)}개 / "
        f"LLM 토큰 0"
    )

    return {
        "success": True,                                           # 필터성공여부
        "message": "규칙 기반 반복 이슈 필터 완료(LLM 미사용)",    # 필터상태메시지
        "filtered_news": filtered_news,                            # 반복이슈필터통과뉴스목록
        "excluded_count": excluded_count,                          # 전체제외건수
        "excluded_items": excluded_items,                          # 제외후보상세목록
        "past_issue_count": len(past_issues),                      # 비교대상과거이슈수
        "prefilter_excluded_count": url_excluded_count + title_excluded_count + text_excluded_count + token_overlap_excluded_count + simhash_excluded_count,  # 과거이슈규칙제외수
        "llm_excluded_count": 0,                                   # LLM제외수호환키
        "core_key_excluded_count": 0,                              # 핵심키제외수호환키
        "internal_duplicate_count": internal_duplicate_count,       # 후보내부중복제외수
        "url_excluded_count": url_excluded_count,                  # URL기준제외수
        "title_excluded_count": title_excluded_count,              # 제목기준제외수
        "text_excluded_count": text_excluded_count,                # 본문유사도제외수
        "token_overlap_excluded_count": token_overlap_excluded_count,  # 토큰겹침제외수
        "simhash_excluded_count": simhash_excluded_count,          # SimHash제외수
        "internal_duplicate_filter_enabled": bool(remove_internal_duplicates),  # 후보내부중복필터사용여부
        "token_stats": get_last_token_stats(),                     # 반복이슈토큰통계
    }


# [코드 이해 주석]
# - 역할: 메일 발송 직전 전체 섹션의 최종 요약 결과를 다시 한 번 사건 단위로 중복 제거한다.
# - 호출하는 곳: main.main
# - 파라미터: section_results: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 각 항목의 비교 payload를 만듭니다 -> 이미 유지한 항목과 비교합니다 -> 대표 항목만 남깁니다.
def deduplicate_section_results(section_results):
    """
    메일 발송 직전 전체 섹션의 최종 요약 결과를 다시 한 번 사건 단위로 중복 제거한다.
    이미 요약된 결과만 대상으로 하므로 OpenAI 호출/토큰 사용이 늘지 않는다.
    """
    kept_payloads = []  # keptpayloads
    kept_urls = set()  # keptURL목록
    excluded_items = []  # 제외항목목록
    total_before = 0  # 전체before
    total_after = 0  # 전체after

    # 1) 섹션 순서를 유지한 채 앞에서부터 대표 뉴스를 확정한다.
    #    먼저 나온 섹션/뉴스를 대표로 남기고, 뒤에서 같은 사건이 나오면 제외한다.
    for section_result in section_results or []:  # 섹션결과
        section_name = section_result.get("section_name", "뉴스 섹션")  # 섹션이름
        summaries = section_result.get("summaries", []) or []  # 요약목록
        total_before += len(summaries)  # 처리값
        deduped = []  # 중복제거

        # 2) summaries는 이미 요약이 끝난 메일 카드 후보이므로, 여기서 제외하면 실제 메일에서 사라진다.
        #    OpenAI 호출 없이 build_event_signature() 기반 규칙으로만 비교한다.
        for idx, news in enumerate(summaries):  # 순번,뉴스
            news_title = get_news_title(news)  # 뉴스제목
            news_url = normalize_url(get_news_url(news))  # 뉴스URL
            candidate_payload = build_event_signature(news)  # 후보데이터

            # 2-1) URL이 이미 유지된 뉴스와 같으면 즉시 제외한다.
            if news_url and news_url in kept_urls:
                excluded_items.append({
                    "section_name": section_name,
                    "title": news_title,
                    "matched_title": "메일 전체 내부 중복 URL",
                    "method": "final_internal_url",
                })
                continue

            # 2-2) URL은 달라도 같은 사건으로 판단되면 섹션 간 중복으로 제외한다.
            matched_payload, method, detail = find_matching_payload(candidate_payload, kept_payloads)  # 매칭된데이터,method,detail
            if matched_payload:
                excluded_items.append({
                    "section_name": section_name,
                    "title": news_title,
                    "matched_title": matched_payload.get("title", ""),
                    "method": f"final_{method}",
                    "detail": detail,
                })
                continue

            # 2-3) 중복이 아니면 이 뉴스를 최종 메일 카드로 유지하고, 이후 뉴스들이 비교할 기준 payload로 등록한다.
            deduped.append(news)
            if news_url:
                kept_urls.add(news_url)
            kept_payloads.append({
                "title": news_title,
                "normalized_title": candidate_payload.get("normalized_title", ""),
                "normalized_text": candidate_payload.get("normalized_text", ""),
                "tokens": candidate_payload.get("tokens", []),
                "title_tokens": candidate_payload.get("title_tokens", []),
                "anchor_tokens": candidate_payload.get("anchor_tokens", []),
                "number_tokens": candidate_payload.get("number_tokens", []),
                "action_tokens": candidate_payload.get("action_tokens", []),
                "alias_titles": candidate_payload.get("alias_titles", []),
                "alias_urls": candidate_payload.get("alias_urls", []),
                "fingerprint": candidate_payload.get("fingerprint", ""),
                "event_key": candidate_payload.get("event_key", ""),
            })

        section_result["summaries"] = deduped  # 처리값
        section_result["selected_count"] = len(deduped)  # 처리값
        # 섹션 간 최종 중복 제거 결과를 scrape_stats에도 남긴다.
        # main.py는 이 값으로 발송 직전 실제 메일 카드가 얼마나 줄었는지 추적할 수 있다.
        scrape_stats = section_result.setdefault("scrape_stats", {})  # 섹션처리통계
        scrape_stats["final_mail_dedup_before"] = len(summaries)      # 메일최종중복제거전수
        scrape_stats["final_mail_dedup_after"] = len(deduped)         # 메일최종중복제거후수
        scrape_stats["final_mail_dedup_excluded"] = len(summaries) - len(deduped)  # 메일최종중복제외수
        total_after += len(deduped)  # 처리값

    logger.info(
        "🧹 메일 발송 직전 전체 섹션 중복 제거 완료: %s개 → %s개 / 제외 %s개 / LLM 토큰 0",
        total_before,
        total_after,
        total_before - total_after,
    )

    return {
        "success": True,                                      # 최종중복제거성공여부
        "before_count": total_before,                         # 중복제거전전체메일카드수
        "after_count": total_after,                           # 중복제거후전체메일카드수
        "excluded_count": total_before - total_after,         # 최종중복제외수
        "excluded_items": excluded_items,                     # 최종중복제외상세목록
        "section_results": section_results,                   # 중복제거반영섹션결과
    }


if __name__ == "__main__":
    history = load_issue_history()  # 히스토리
    print(f"현재 저장된 이슈 수: {len(history.get('issues', []))}")
