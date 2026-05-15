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
HISTORY_FILE_PATH = "data/seen_issues.json"
KST = pytz.timezone("Asia/Seoul")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 제목 유사도 기준
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("HISTORY_TITLE_SIMILARITY_THRESHOLD", "0.88"))

# 제목+설명 전체 텍스트 유사도 기준
TEXT_SIMILARITY_THRESHOLD = float(os.getenv("HISTORY_TEXT_SIMILARITY_THRESHOLD", "0.82"))

# 주요 토큰 겹침률 기준
TOKEN_OVERLAP_THRESHOLD = float(os.getenv("HISTORY_TOKEN_OVERLAP_THRESHOLD", "0.65"))

# SimHash 거리 기준. 작을수록 거의 같은 문서다.
SIMHASH_DISTANCE_THRESHOLD = int(os.getenv("HISTORY_SIMHASH_DISTANCE_THRESHOLD", "6"))

# 너무 짧은 기사끼리 토큰 겹침만으로 중복 처리되는 것을 막기 위한 최소 공통 토큰 수
MIN_COMMON_TOKEN_COUNT = int(os.getenv("HISTORY_MIN_COMMON_TOKEN_COUNT", "4"))

# URL 정규화 시 유지할 의미 있는 query parameter
MEANINGFUL_QUERY_PARAMS = {
    "no", "idxno", "article_no", "articleid", "article_id",
    "newsid", "news_id", "aid", "oid", "sid", "id", "seq", "num",
}

# 토큰 사용량 통계. 이 파일은 LLM을 쓰지 않으므로 항상 0이다.
LAST_TOKEN_STATS = {
    "issue_key_tokens": 0,
    "llm_duplicate_tokens": 0,
}

# 한국어 뉴스에서 반복적으로 등장하지만 중복 판단에는 도움이 적은 단어들
STOPWORDS = {
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
    "한국", "국내", "글로벌", "최신", "주요", "확인", "가능", "기준",
    "기반", "활용", "도입", "사용", "운용", "참여", "소개", "제공",
}

TOKEN_SUFFIXES = (
    "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
    "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
)

ANCHOR_STOPWORDS = STOPWORDS | {
    "ai", "it", "ict", "dx", "si", "emr", "시스템", "기술", "산업", "시장",
    "사업", "서비스", "플랫폼", "솔루션", "정보", "디지털",
}

# 히스토리 비교/저장 범위. 기본값은 briefing으로 두어 같은 메일 안에서
# 섹션만 달라진 반복 이슈도 제거한다. 필요하면 환경변수로 section/receiver로 조정한다.
HISTORY_MATCH_SCOPE = os.getenv("HISTORY_MATCH_SCOPE", "briefing").strip().lower()
HISTORY_SAVE_SCOPE = os.getenv("HISTORY_SAVE_SCOPE", HISTORY_MATCH_SCOPE).strip().lower()

# 반복 이슈 제외 내역 디버그 JSON 저장.
# 값은 노출 가능한 제목/간단 근거 중심으로만 저장한다.

# 외부 AI 호출 없이 표기 차이를 줄이기 위한 최소 별칭 사전.
# 값은 비교용 텍스트에 들어갈 안정 토큰이다.
ENTITY_ALIASES = {
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

ACTION_KEYWORDS = {
    "협력", "제휴", "파트너십", "동맹", "계약", "공급", "선정", "수주",
    "인수", "매각", "투자", "출시", "개설", "신설", "구축", "도입",
    "갱신", "인증", "해임", "부결", "갈등", "분쟁", "실적", "영업이익",
    "영업손실", "매출", "수출", "대응", "점검", "재편", "개최",
    "동맹", "합류", "가동", "대비", "진행", "선보",
}

ACTION_ALIASES = {
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
WEAK_ANCHOR_TOKENS = ANCHOR_STOPWORDS | {
    "경제", "경제학상", "노벨경제학상", "대통령", "정부", "부총리", "장관",
    "부동산", "부동산시장", "주택시장", "증시", "뉴욕증시", "코스피", "코스닥",
    "나스닥", "다우", "다우지수", "정상회담", "미중회담", "트럼프", "시진핑",
    "15일", "14일", "현지시간", "연합뉴스", "시장", "주식", "투자자",
}

WEAK_ACTION_TOKENS = {
    "협력", "투자", "대응", "공급", "점검", "개최", "실적", "수출", "출시"
}

STRONG_ACTION_TOKENS = {
    "인수", "매각", "해임", "부결", "갈등", "분쟁", "수주", "선정", "신설",
    "갱신", "인증", "재편", "영업이익", "영업손실", "매출"
}

def reset_token_stats():
    LAST_TOKEN_STATS["issue_key_tokens"] = 0
    LAST_TOKEN_STATS["llm_duplicate_tokens"] = 0


def get_last_token_stats():
    return dict(LAST_TOKEN_STATS)


def get_now_kst():
    return datetime.now(KST)


# ====================================
# 정규화 유틸
# ====================================
def normalize_url(url: str) -> str:
    """
    기사 URL 비교용 정규화.
    tracking query는 제거하고, 기사 식별에 의미 있는 query만 유지한다.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())
        domain = parsed.netloc.lower().strip()
        path = parsed.path.strip()
        if domain.startswith("www."):
            domain = domain[4:]

        meaningful_params = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            key_lower = str(key).lower().strip()
            if key_lower in MEANINGFUL_QUERY_PARAMS and str(value).strip():
                meaningful_params.append((key_lower, str(value).strip()))
        meaningful_params.sort()

        normalized = f"{domain}{path}".rstrip("/")
        if meaningful_params:
            normalized += "?" + urlencode(meaningful_params)
        return normalized
    except Exception:
        return str(url).lower().strip().rstrip("/")


def strip_html_entities(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    text = text.replace("…", " ").replace("...", " ")
    return text.strip()


def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = strip_html_entities(title).lower().strip()
    text = re.sub(r"\[[^\]]*\]|【[^】]*】|\([^)]*\)", " ", text)
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text.strip()


def apply_entity_aliases(text: str) -> str:
    """
    같은 기관/서비스의 표기 차이를 규칙 기반으로 정규화한다.
    LLM을 사용하지 않으므로 토큰 비용은 늘지 않는다.
    """
    value = str(text or "").lower()
    if not value:
        return ""

    # 붙여 쓴 영문/한글 표기도 먼저 보강한다.
    value = value.replace("openai", " openai ")
    value = value.replace("skax", " skax ")

    for alias, canonical in ENTITY_ALIASES.items():
        value = value.replace(alias.lower(), f" {canonical.lower()} ")
    return value


def normalize_compare_text(text: str) -> str:
    if not text:
        return ""
    text = apply_entity_aliases(strip_html_entities(text)).lower()
    text = re.sub(r"\[[^\]]*\]|【[^】]*】", " ", text)
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)
    text = re.sub(r"\b\w+@\w+(?:\.\w+)+\b", " ", text)
    text = re.sub(r"[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_compare_text(text: str) -> str:
    text = normalize_compare_text(text)
    text = re.sub(r"\s+", "", text)
    return text


def normalize_token(token: str) -> str:
    token = str(token or "").lower().strip()
    if not token:
        return ""

    token = token.replace("美", "미국").replace("韓", "한국").replace("中", "중국")
    token = token.replace("日", "일본").replace("李", "이").replace("金", "김")

    for suffix in TOKEN_SUFFIXES:
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break

    return token.strip()


def extract_tokens(text: str, max_tokens: int = 80):
    """
    외부 형태소 분석기 없이 동작하는 간단 토큰 추출.
    한국어/영문/숫자 2자 이상 토큰만 사용한다.
    """
    text = normalize_compare_text(text)
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)

    tokens = []
    seen = set()
    for token in raw_tokens:
        token = normalize_token(token)
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


def extract_title_tokens(title: str):
    return extract_tokens(title, max_tokens=30)


def extract_anchor_tokens(tokens):
    anchors = []
    seen = set()
    for token in tokens or []:
        token = normalize_token(token)
        if not token or token in ANCHOR_STOPWORDS:
            continue
        has_alpha = bool(re.search(r"[a-zA-Z]", token))
        has_korean = bool(re.search(r"[가-힣]", token))
        if has_alpha or (has_korean and len(token) >= 3) or len(token) >= 4:
            if token not in seen:
                seen.add(token)
                anchors.append(token)
    return anchors[:20]


def extract_number_tokens(value: str):
    return set(re.findall(r"\d+(?:\.\d+)?", str(value or "")))


def normalize_action_token(token: str) -> str:
    token = normalize_token(token)
    if not token:
        return ""
    for keyword, canonical in ACTION_ALIASES.items():
        if keyword in token:
            return canonical
    if token in ACTION_KEYWORDS:
        return token
    return ""


def extract_action_tokens(tokens):
    actions = []
    seen = set()
    for token in tokens or []:
        action = normalize_action_token(token)
        if not action or action in seen:
            continue
        seen.add(action)
        actions.append(action)
    return actions[:6]


def is_weak_anchor_token(token: str) -> bool:
    token = normalize_token(token)
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


def is_strong_anchor_token(token: str) -> bool:
    token = normalize_token(token)
    if not token or is_weak_anchor_token(token):
        return False
    has_alpha = bool(re.search(r"[a-zA-Z]", token))
    has_korean = bool(re.search(r"[가-힣]", token))
    return has_alpha or (has_korean and len(token) >= 3) or len(token) >= 5


def strong_common_anchor_tokens(tokens):
    return unique_nonempty([token for token in tokens or [] if is_strong_anchor_token(token)], limit=20)


def has_specific_action(action_tokens) -> bool:
    for action in action_tokens or []:
        action = normalize_action_token(action) or str(action or "").strip()
        if action and action not in WEAK_ACTION_TOKENS:
            return True
    return False


def build_rule_event_key_from_payload(payload):
    """
    기관/핵심 앵커 + 행위어 + 숫자 일부로 사건 키를 만든다.
    키가 너무 일반적이면 빈 문자열을 반환해 오탐을 줄인다.
    """
    tokens = payload.get("tokens") or []
    title_tokens = payload.get("title_tokens") or []
    anchors = payload.get("anchor_tokens") or []
    numbers = payload.get("number_tokens") or []

    actions = payload.get("action_tokens") or extract_action_tokens(title_tokens + tokens)
    strong_anchors = strong_common_anchor_tokens(anchors)[:4]
    specific_actions = [action for action in actions if action not in WEAK_ACTION_TOKENS]

    # 범용 앵커/행위어만으로 만든 사건 키는 오탐이 많다.
    # 고유 앵커 2개 이상, 또는 고유 앵커 1개 + 구체 행위어 1개 이상일 때만 키를 만든다.
    if len(strong_anchors) < 2 and not (strong_anchors and specific_actions):
        return ""

    key_parts = strong_anchors[:3] + (specific_actions or actions)[:2] + list(numbers)[:2]
    return "|".join(unique_nonempty(key_parts, limit=7))


def unique_nonempty(values, limit=None):
    seen = set()
    result = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def get_news_alias_titles(news: dict):
    titles = [get_news_title(news)]
    titles.extend(news.get("group_article_titles") or [])
    return unique_nonempty(titles, limit=16)


def get_news_alias_urls(news: dict):
    urls = [get_news_url(news)]
    urls.extend(news.get("group_article_urls") or [])
    return unique_nonempty([normalize_url(url) for url in urls], limit=16)


def make_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def stable_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def make_simhash(tokens) -> str:
    """
    64비트 SimHash를 16자리 hex 문자열로 반환한다.
    """
    if not tokens:
        return ""

    vector = [0] * 64
    for token in tokens:
        value = stable_token_hash(token)
        for bit in range(64):
            if value & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1

    fingerprint = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            fingerprint |= (1 << bit)

    return f"{fingerprint:016x}"


def simhash_distance(a: str, b: str):
    if not a or not b:
        return None
    try:
        x = int(str(a), 16) ^ int(str(b), 16)
        return x.bit_count()
    except Exception:
        return None


def token_overlap_score(tokens_a, tokens_b):
    set_a = set(tokens_a or [])
    set_b = set(tokens_b or [])
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b
    # 짧은 쪽 기준 겹침률. 같은 보도자료의 제목/설명 변형에 더 민감하다.
    denominator = max(1, min(len(set_a), len(set_b)))
    return len(common) / denominator, len(common)


def tokens_are_related(token_a, token_b):
    """
    완전 일치가 아니어도 같은 고유명사/행사명 변형으로 볼 수 있는지 판단한다.
    특정 회사명·행사명을 하드코딩하지 않고, 부분 문자열/유사도만 사용한다.
    """
    a = normalize_token(token_a)
    b = normalize_token(token_b)
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = sorted([a, b], key=len)
    if len(shorter) >= 3 and shorter in longer:
        return True

    # 긴 고유명사에서 조사/띄어쓰기/축약 차이로 토큰이 조금 달라지는 경우를 보완한다.
    if min(len(a), len(b)) >= 4:
        return SequenceMatcher(None, a, b).ratio() >= 0.82

    return False


def soft_common_tokens(tokens_a, tokens_b, ignore_stopwords=True):
    """
    토큰 목록 간 완전 일치 + 부분 일치 기반 공통 토큰을 계산한다.
    예: '메이플스토리'와 '메이플', '오픈ai'와 'openai'처럼 제목 표현이 달라도
    같은 사건의 핵심 앵커로 볼 수 있는 경우를 잡기 위한 규칙 기반 보완이다.
    """
    left = []
    for token in tokens_a or []:
        token = normalize_token(token)
        if not token:
            continue
        if ignore_stopwords and token in ANCHOR_STOPWORDS and normalize_action_token(token) == "":
            continue
        left.append(token)

    right = []
    for token in tokens_b or []:
        token = normalize_token(token)
        if not token:
            continue
        if ignore_stopwords and token in ANCHOR_STOPWORDS and normalize_action_token(token) == "":
            continue
        right.append(token)

    used_right = set()
    common = []
    for a in left:
        for idx, b in enumerate(right):
            if idx in used_right:
                continue
            if tokens_are_related(a, b):
                used_right.add(idx)
                common.append(a if len(a) <= len(b) else b)
                break

    return unique_nonempty(common)


def soft_token_overlap_score(tokens_a, tokens_b, ignore_stopwords=True):
    common = soft_common_tokens(tokens_a, tokens_b, ignore_stopwords=ignore_stopwords)
    len_a = len(unique_nonempty(tokens_a or []))
    len_b = len(unique_nonempty(tokens_b or []))
    if not len_a or not len_b:
        return 0.0, 0, []
    denominator = max(1, min(len_a, len_b))
    return len(common) / denominator, len(common), common


def get_news_url(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return (
        str(news.get("url") or "").strip()
        or str(news.get("originallink") or "").strip()
        or str(news.get("link") or "").strip()
    )


def get_news_title(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return str(news.get("title") or "").strip()


def get_news_summary_or_description(news: dict) -> str:
    if not isinstance(news, dict):
        return ""
    return (
        str(news.get("summary") or "").strip()
        or str(news.get("description") or "").strip()
        or str(news.get("content") or "").strip()
    )


def get_news_compare_text(news: dict) -> str:
    if not isinstance(news, dict):
        return ""

    parts = [
        str(news.get("summary") or "").strip(),
        str(news.get("description") or "").strip(),
        str(news.get("content") or "").strip(),
    ]

    for keyword in news.get("group_keywords") or []:
        parts.append(str(keyword).strip())

    keyword = str(news.get("keyword") or "").strip()
    if keyword:
        parts.append(keyword)

    seen = set()
    cleaned = []
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        cleaned.append(part)

    return " ".join(cleaned)


def build_compare_payload(title: str, summary: str):
    compare_text = normalize_compare_text(f"{title} {summary}")
    compact_text = compact_compare_text(compare_text)
    tokens = extract_tokens(compare_text)
    title_tokens = extract_title_tokens(title)
    payload = {
        "normalized_title": normalize_title(title),
        "normalized_text": compact_text,
        "tokens": tokens,
        "title_tokens": title_tokens,
        "anchor_tokens": extract_anchor_tokens(list(title_tokens or []) + list(tokens or [])),
        "number_tokens": sorted(extract_number_tokens(f"{title} {summary}")),
        "fingerprint": make_simhash(tokens),
    }
    payload["action_tokens"] = extract_action_tokens(title_tokens + tokens)
    payload["event_key"] = build_rule_event_key_from_payload(payload)
    return payload


def build_event_signature(news: dict):
    title = get_news_title(news)
    compare_text = get_news_compare_text(news) or get_news_summary_or_description(news)
    raw_alias_titles = get_news_alias_titles(news)
    alias_titles = [
        normalize_title(alias_title)
        for alias_title in raw_alias_titles
        if normalize_title(alias_title)
    ]
    alias_titles = unique_nonempty(alias_titles, limit=16)
    alias_urls = get_news_alias_urls(news)

    event_text_parts = [title, compare_text]
    event_text_parts.extend(raw_alias_titles)
    event_text = " ".join(unique_nonempty(event_text_parts))
    payload = build_compare_payload(title, event_text)
    alias_title_tokens = extract_tokens(" ".join(raw_alias_titles), max_tokens=60)
    payload["title_tokens"] = unique_nonempty(
        list(payload.get("title_tokens") or []) + alias_title_tokens,
        limit=80,
    )
    payload["anchor_tokens"] = extract_anchor_tokens(
        list(payload.get("title_tokens") or []) + list(payload.get("tokens") or [])
    )
    payload["action_tokens"] = extract_action_tokens(
        payload.get("title_tokens", []) + payload.get("tokens", [])
    )
    payload["event_key"] = build_rule_event_key_from_payload(payload)
    payload["alias_titles"] = alias_titles
    payload["alias_urls"] = alias_urls
    return payload


# ====================================
# 파일 입출력
# ====================================
def load_issue_history(file_path=HISTORY_FILE_PATH):
    if not os.path.exists(file_path):
        return {"version": 3, "issues": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 3, "issues": []}
        if "issues" not in data or not isinstance(data["issues"], list):
            data["issues"] = []
        data["version"] = max(int(data.get("version", 1) or 1), 3)
        return data
    except Exception as e:
        logger.warning(f"⚠️ 이슈 히스토리 읽기 실패, 빈 히스토리로 시작합니다: {e}")
        return {"version": 3, "issues": []}


def save_issue_history(data, file_path=HISTORY_FILE_PATH):
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)






def prune_old_issues(history, days=3):
    if not isinstance(history, dict):
        return {"version": 3, "issues": []}, 0

    issues = history.get("issues", [])
    if not isinstance(issues, list):
        history["issues"] = []
        return history, 0

    today = get_now_kst().date()
    kept_issues = []
    removed_count = 0

    for issue in issues:
        saved_date_text = issue.get("saved_date")
        if not saved_date_text:
            removed_count += 1
            continue
        try:
            saved_date = datetime.strptime(saved_date_text, "%Y-%m-%d").date()
        except Exception:
            removed_count += 1
            continue

        day_diff = (today - saved_date).days
        if 0 <= day_diff < days:
            kept_issues.append(issue)
        else:
            removed_count += 1

    history["issues"] = kept_issues
    history["version"] = max(int(history.get("version", 3) or 3), 3)
    return history, removed_count


# ====================================
# 히스토리 record 생성/조회
# ====================================
def make_history_scope(briefing_name, receiver_env, section_name=None, scope_mode=None):
    mode = str(scope_mode or HISTORY_MATCH_SCOPE or "briefing").strip().lower()
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


def make_issue_id(briefing_name, receiver_env, section_name, title, summary, url=None):
    normalized_url = normalize_url(url)
    scope = make_history_scope(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        scope_mode=HISTORY_SAVE_SCOPE,
    )

    if normalized_url:
        return make_hash(f"url|{scope}|{normalized_url}")

    payload = build_compare_payload(title, summary)
    raw = "|".join([
        "text",
        scope,
        payload.get("normalized_title", ""),
        payload.get("fingerprint", ""),
        payload.get("normalized_text", "")[:160],
    ])
    return make_hash(raw)


def build_issue_record(
    briefing_name, subject_prefix, receiver_env, section_name, news
):
    title = get_news_title(news)
    summary = get_news_summary_or_description(news)
    compare_text = get_news_compare_text(news) or summary
    source = str(news.get("source") or "").strip()
    url = get_news_url(news)
    published_at = str(news.get("published_at") or "").strip()
    importance_score = news.get("importance_score")

    payload = build_event_signature(news)
    normalized_url = normalize_url(url)
    issue_id = make_issue_id(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        title=title,
        summary=compare_text,
        url=url,
    )

    now_kst = get_now_kst()

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


def get_issue_normalized_url(issue: dict) -> str:
    if not isinstance(issue, dict):
        return ""
    return str(issue.get("normalized_url") or "").strip() or normalize_url(issue.get("url") or "")


def get_issue_compare_payload(issue: dict):
    if not isinstance(issue, dict):
        issue = {}

    title = str(issue.get("title") or "")
    summary = " ".join([
        str(issue.get("summary") or "").strip(),
        str(issue.get("description") or "").strip(),
        str(issue.get("compare_text") or "").strip(),
        str(issue.get("content") or "").strip(),
    ]).strip()

    normalized_title = str(issue.get("normalized_title") or "").strip() or normalize_title(title)
    normalized_text = str(issue.get("event_normalized_text") or issue.get("normalized_text") or "").strip()
    tokens = issue.get("event_tokens") or issue.get("content_tokens")
    title_tokens = issue.get("event_title_tokens") or issue.get("title_tokens")
    anchor_tokens = issue.get("event_anchor_tokens") or issue.get("anchor_tokens")
    number_tokens = issue.get("event_number_tokens") or issue.get("number_tokens")
    action_tokens = issue.get("event_action_tokens") or issue.get("action_tokens")
    fingerprint = str(issue.get("event_fingerprint") or issue.get("content_fingerprint") or "").strip()
    event_key = str(issue.get("event_key") or "").strip()
    alias_titles = issue.get("alias_titles")
    alias_urls = issue.get("alias_urls")

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
        payload = build_compare_payload(title, summary)
        normalized_text = normalized_text or payload["normalized_text"]
        tokens = tokens if isinstance(tokens, list) and tokens else payload["tokens"]
        title_tokens = (
            title_tokens
            if isinstance(title_tokens, list) and title_tokens
            else payload["title_tokens"]
        )
        anchor_tokens = (
            anchor_tokens
            if isinstance(anchor_tokens, list) and anchor_tokens
            else payload["anchor_tokens"]
        )
        number_tokens = (
            number_tokens
            if isinstance(number_tokens, list)
            else payload["number_tokens"]
        )
        action_tokens = (
            action_tokens
            if isinstance(action_tokens, list) and action_tokens
            else payload.get("action_tokens", [])
        )
        alias_titles = (
            alias_titles
            if isinstance(alias_titles, list) and alias_titles
            else unique_nonempty([normalized_title], limit=16)
        )
        alias_urls = (
            alias_urls
            if isinstance(alias_urls, list)
            else unique_nonempty([get_issue_normalized_url(issue)], limit=16)
        )
        fingerprint = fingerprint or payload["fingerprint"]

    if not event_key:
        event_key = build_rule_event_key_from_payload({
            "tokens": tokens or [],
            "title_tokens": title_tokens or [],
            "anchor_tokens": anchor_tokens or [],
            "number_tokens": number_tokens or [],
            "action_tokens": action_tokens or [],
        })

    if not isinstance(action_tokens, list) or not action_tokens:
        action_tokens = extract_action_tokens((title_tokens or []) + (tokens or []))

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


def append_sent_issues(
    briefing_name, subject_prefix, receiver_env,
    section_results, file_path=HISTORY_FILE_PATH, keep_days=3
):
    """
    메일에 실제 발송된 section_results의 summaries를 이슈 히스토리에 저장한다.
    LLM/issue_key 생성 없이 비교용 fingerprint만 저장한다.
    """
    history = load_issue_history(file_path)
    history, pruned_count = prune_old_issues(history, days=keep_days)

    existing_issue_ids = set()
    existing_scope_url_keys = set()
    existing_scope_title_keys = set()
    existing_scope_event_keys = set()

    for item in history.get("issues", []):
        issue_id = item.get("issue_id")
        if issue_id:
            existing_issue_ids.add(str(issue_id))

        scope = make_history_scope(
            briefing_name=item.get("briefing_name"),
            receiver_env=item.get("receiver_env"),
            section_name=item.get("section_name"),
            scope_mode=HISTORY_SAVE_SCOPE,
        )
        item_url = get_issue_normalized_url(item)
        item_payload = get_issue_compare_payload(item)
        item_title = item_payload.get("normalized_title", "")
        item_event_key = item_payload.get("event_key", "")

        if item_event_key:
            existing_scope_event_keys.add(f"{scope}|{item_event_key}")
        for alias_url in unique_nonempty([item_url] + (item_payload.get("alias_urls") or [])):
            existing_scope_url_keys.add(f"{scope}|{alias_url}")
        for alias_title in unique_nonempty([item_title] + (item_payload.get("alias_titles") or [])):
            existing_scope_title_keys.add(f"{scope}|{alias_title}")

    new_records = []
    skipped_duplicate_count = 0

    for section_result in section_results or []:
        section_name = section_result.get("section_name", "뉴스 섹션")
        summaries = section_result.get("summaries", []) or []

        for news in summaries:
            record = build_issue_record(
                briefing_name=briefing_name,
                subject_prefix=subject_prefix,
                receiver_env=receiver_env,
                section_name=section_name,
                news=news,
            )

            scope = make_history_scope(
                briefing_name=briefing_name,
                receiver_env=receiver_env,
                section_name=section_name,
                scope_mode=HISTORY_SAVE_SCOPE,
            )
            url_key = f"{scope}|{record.get('normalized_url', '')}"
            title_key = f"{scope}|{record.get('normalized_title', '')}"
            event_key = str(record.get("event_key") or "").strip()
            event_key_full = f"{scope}|{event_key}" if event_key else ""
            alias_url_keys = [
                f"{scope}|{alias_url}"
                for alias_url in unique_nonempty(record.get("alias_urls") or [])
            ]
            alias_title_keys = [
                f"{scope}|{alias_title}"
                for alias_title in unique_nonempty(record.get("alias_titles") or [])
            ]

            if record["issue_id"] in existing_issue_ids:
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_url") and url_key in existing_scope_url_keys:
                skipped_duplicate_count += 1
                continue
            if any(key in existing_scope_url_keys for key in alias_url_keys):
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_title") and title_key in existing_scope_title_keys:
                skipped_duplicate_count += 1
                continue
            if any(key in existing_scope_title_keys for key in alias_title_keys):
                skipped_duplicate_count += 1
                continue
            if event_key_full and event_key_full in existing_scope_event_keys:
                skipped_duplicate_count += 1
                continue

            existing_issue_ids.add(record["issue_id"])
            if record.get("normalized_url"):
                existing_scope_url_keys.add(url_key)
            if record.get("normalized_title"):
                existing_scope_title_keys.add(title_key)
            if event_key_full:
                existing_scope_event_keys.add(event_key_full)
            for key in alias_url_keys:
                existing_scope_url_keys.add(key)
            for key in alias_title_keys:
                existing_scope_title_keys.add(key)

            new_records.append(record)

    if new_records:
        history["issues"].extend(new_records)

    history["last_updated_at"] = get_now_kst().isoformat()
    history["version"] = max(int(history.get("version", 3) or 3), 3)
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
def get_recent_issues_for_section(
    briefing_name, receiver_env, section_name,
    days=3, file_path=HISTORY_FILE_PATH, scope_mode=None
):
    """
    최근 발송 이슈 조회.
    기본은 briefing 범위라서 같은 메일 안에서 섹션이 달라져도 반복 이슈로 비교한다.
    HISTORY_MATCH_SCOPE=section 으로 두면 기존 섹션 단위 동작으로 되돌릴 수 있다.
    """
    history = load_issue_history(file_path)
    issues = history.get("issues", [])
    today = get_now_kst().date()
    recent_issues = []

    target_scope = make_history_scope(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        scope_mode=scope_mode or HISTORY_MATCH_SCOPE,
    )

    for issue in issues:
        issue_scope = make_history_scope(
            briefing_name=issue.get("briefing_name"),
            receiver_env=issue.get("receiver_env"),
            section_name=issue.get("section_name"),
            scope_mode=scope_mode or HISTORY_MATCH_SCOPE,
        )
        if issue_scope != target_scope:
            continue

        saved_date_text = issue.get("saved_date")
        if not saved_date_text:
            continue
        try:
            saved_date = datetime.strptime(saved_date_text, "%Y-%m-%d").date()
        except Exception:
            continue

        if 0 <= (today - saved_date).days < days:
            recent_issues.append(issue)

    return recent_issues


# ====================================
# 반복 이슈 필터
# ====================================
def build_past_issue_indexes(past_issues):
    past_by_url = {}
    past_payloads = []

    for issue in past_issues or []:
        issue_url = get_issue_normalized_url(issue)
        payload = get_issue_compare_payload(issue)

        if issue_url and issue_url not in past_by_url:
            past_by_url[issue_url] = issue
        for alias_url in payload.get("alias_urls") or []:
            if alias_url and alias_url not in past_by_url:
                past_by_url[alias_url] = issue

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


def judge_duplicate_by_payload(candidate_payload, past_payload):
    """
    후보와 과거/오늘 유지 후보가 같은 반복 이슈인지 규칙 기반으로 판단한다.

    과제외를 줄이기 위해 판정 강도를 나눈다.
    - exact: URL/제목/event_key처럼 매우 강한 근거
    - strong: 고유 앵커 여러 개 + 제목/행위/숫자 보조 근거
    - soft: 범용 앵커나 일반 행위어 중심. soft만으로는 제외하지 않는다.

    Returns: (is_duplicate, method, score_text)
    """
    cand_title = candidate_payload.get("normalized_title", "")
    past_title = past_payload.get("normalized_title", "")
    candidate_titles = unique_nonempty(
        [cand_title] + (candidate_payload.get("alias_titles") or []),
        limit=16,
    )
    past_titles = unique_nonempty(
        [past_title] + (past_payload.get("alias_titles") or []),
        limit=16,
    )

    cand_numbers = set(candidate_payload.get("number_tokens") or [])
    past_numbers = set(past_payload.get("number_tokens") or [])
    number_conflict = bool(cand_numbers and past_numbers and not (cand_numbers & past_numbers))

    cand_event_key = str(candidate_payload.get("event_key") or "").strip()
    past_event_key = str(past_payload.get("event_key") or "").strip()
    if cand_event_key and past_event_key and cand_event_key == past_event_key and not number_conflict:
        return True, "event_key", f"규칙 기반 사건 키 동일({cand_event_key}) | strength=exact"

    shared_anchor_tokens = (
        set(candidate_payload.get("anchor_tokens") or [])
        & set(past_payload.get("anchor_tokens") or [])
    )
    shared_action_tokens = (
        set(candidate_payload.get("action_tokens") or [])
        & set(past_payload.get("action_tokens") or [])
    )
    soft_shared_anchor_tokens = soft_common_tokens(
        candidate_payload.get("anchor_tokens") or [],
        past_payload.get("anchor_tokens") or [],
    )
    all_shared_anchor_tokens = unique_nonempty(list(shared_anchor_tokens) + list(soft_shared_anchor_tokens), limit=30)
    strong_shared_anchor_tokens = strong_common_anchor_tokens(all_shared_anchor_tokens)

    shared_anchor_count = len(all_shared_anchor_tokens)
    strong_shared_anchor_count = len(strong_shared_anchor_tokens)
    shared_anchor = shared_anchor_count > 0
    shared_action = bool(shared_action_tokens)
    specific_shared_action = has_specific_action(shared_action_tokens)

    title_overlap, title_common_count = token_overlap_score(
        candidate_payload.get("title_tokens", []),
        past_payload.get("title_tokens", []),
    )
    soft_title_overlap, soft_title_common_count, soft_title_common_tokens = soft_token_overlap_score(
        candidate_payload.get("title_tokens", []),
        past_payload.get("title_tokens", []),
    )
    title_common_best = max(title_common_count, soft_title_common_count)
    title_overlap_best = max(title_overlap, soft_title_overlap)

    # 숫자 불일치는 기본적으로 오탐 방지 신호다.
    # 다만 고유 앵커가 충분히 겹치고 제목/구체 행위가 받쳐주면 같은 사건으로 본다.
    if number_conflict and (
        (strong_shared_anchor_count >= 3 and title_common_best >= 3)
        or (strong_shared_anchor_count >= 2 and specific_shared_action and title_common_best >= 2)
        or strong_shared_anchor_count >= 4
    ):
        number_conflict = False

    best_title_score = 0.0
    for candidate_title in candidate_titles:
        for past_alias_title in past_titles:
            if not candidate_title or not past_alias_title:
                continue

            if candidate_title == past_alias_title:
                return True, "title_exact", "정규화 제목/별칭 제목 동일 | strength=exact"

            if min(len(candidate_title), len(past_alias_title)) >= 14:
                shorter, longer = sorted([candidate_title, past_alias_title], key=len)
                if shorter in longer and not number_conflict:
                    # 짧은 제목이 너무 일반적인 경우를 막기 위해 제목 토큰도 확인한다.
                    if title_common_best >= 3 or strong_shared_anchor_count >= 2:
                        return True, "title_contains", "후보 제목과 과거 별칭 제목이 포함 관계 | strength=exact"

            title_score = SequenceMatcher(None, candidate_title, past_alias_title).ratio()
            if title_score > best_title_score:
                best_title_score = title_score

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

    overlap_score, common_count = token_overlap_score(
        candidate_payload.get("tokens", []),
        past_payload.get("tokens", []),
    )
    soft_overlap_score, soft_common_count, soft_common_token_list = soft_token_overlap_score(
        candidate_payload.get("tokens", []),
        past_payload.get("tokens", []),
    )
    overlap_score = max(overlap_score, soft_overlap_score)
    common_count = max(common_count, soft_common_count)

    cand_text = candidate_payload.get("normalized_text", "")
    past_text = past_payload.get("normalized_text", "")
    if cand_text and past_text:
        text_score = SequenceMatcher(None, cand_text, past_text).ratio()
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

    distance = simhash_distance(
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

def find_matching_payload(candidate_payload, past_payloads):
    for past_payload in past_payloads or []:
        is_dup, method, detail = judge_duplicate_by_payload(candidate_payload, past_payload)
        if is_dup:
            return past_payload, method, detail
    return None, "", ""


def filter_seen_issues_with_llm(
    briefing_name, receiver_env, section_name,
    candidate_news, days=3, file_path=HISTORY_FILE_PATH
):
    """
    기존 함수명은 유지하지만 LLM을 사용하지 않는다.
    최근 N일간 이미 메일에 보낸 이슈와 후보 뉴스를 규칙 기반으로 비교해 제외한다.
    """
    reset_token_stats()

    if not candidate_news:
        return {
            "success": True,
            "message": "후보 뉴스 없음",
            "filtered_news": [],
            "excluded_count": 0,
            "excluded_items": [],
            "past_issue_count": 0,
            "prefilter_excluded_count": 0,
            "llm_excluded_count": 0,
            "core_key_excluded_count": 0,
            "internal_duplicate_count": 0,
            "url_excluded_count": 0,
            "title_excluded_count": 0,
            "text_excluded_count": 0,
            "token_overlap_excluded_count": 0,
            "simhash_excluded_count": 0,
            "token_stats": get_last_token_stats(),
        }

    past_issues = get_recent_issues_for_section(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        days=days,
        file_path=file_path,
    )

    past_indexes = build_past_issue_indexes(past_issues)
    past_by_url = past_indexes["past_by_url"]
    past_payloads = past_indexes["past_payloads"]

    excluded_items = []
    filtered_news = []

    url_excluded_count = 0
    title_excluded_count = 0
    text_excluded_count = 0
    token_overlap_excluded_count = 0
    simhash_excluded_count = 0
    internal_duplicate_count = 0

    # 오늘 후보 내부 중복도 같은 기준으로 제거한다.
    seen_today_payloads = []
    seen_today_urls = set()

    for idx, news in enumerate(candidate_news or []):
        news_title = get_news_title(news)
        news_summary = get_news_summary_or_description(news)
        news_url = normalize_url(get_news_url(news))
        candidate_payload = build_event_signature(news)

        # 1. 과거 발송 URL 완전 일치
        if news_url and news_url in past_by_url:
            matched = past_by_url[news_url]
            excluded_items.append(make_excluded_item(
                index=idx,
                news=news,
                method="url",
                reason="같은 URL의 기사가 최근 발송 이력에 있음",
                matched_title=matched.get("title", ""),
                detail="정규화 URL 완전 일치",
                candidate_payload=candidate_payload,
            ))
            url_excluded_count += 1
            continue

        # 2. 과거 발송 텍스트/fingerprint 비교
        matched_payload, method, detail = find_matching_payload(candidate_payload, past_payloads)
        if matched_payload:
            matched_issue = matched_payload.get("issue", {})
            excluded_items.append(make_excluded_item(
                index=idx,
                news=news,
                method=method,
                reason="최근 발송 이력과 유사함",
                matched_title=matched_issue.get("title", ""),
                detail=detail,
                candidate_payload=candidate_payload,
            ))
            if method.startswith("title_"):
                title_excluded_count += 1
            elif method.startswith("text_"):
                text_excluded_count += 1
            elif method == "token_overlap":
                token_overlap_excluded_count += 1
            elif method == "simhash":
                simhash_excluded_count += 1
            continue

        # 3. 오늘 후보 내부 URL 중복
        if news_url and news_url in seen_today_urls:
            excluded_items.append(make_excluded_item(
                index=idx,
                news=news,
                method="internal_url",
                reason="오늘 후보 내부에서 같은 URL이 이미 유지됨",
                matched_title="오늘 후보 내부 중복 URL",
                detail="정규화 URL 완전 일치",
                candidate_payload=candidate_payload,
            ))
            internal_duplicate_count += 1
            continue

        # 4. 오늘 후보 내부 텍스트/fingerprint 중복
        matched_today, today_method, today_detail = find_matching_payload(candidate_payload, seen_today_payloads)
        if matched_today:
            excluded_items.append(make_excluded_item(
                index=idx,
                news=news,
                method=f"internal_{today_method}",
                reason="오늘 후보 내부에서 이미 유지한 기사와 유사함",
                matched_title=matched_today.get("title", ""),
                detail=today_detail,
                candidate_payload=candidate_payload,
            ))
            internal_duplicate_count += 1
            continue

        filtered_news.append(news)
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

    excluded_count = len(excluded_items)

    logger.info(
        f"🧹 [{section_name}] 규칙 기반 반복 이슈 필터 완료: "
        f"후보 {len(candidate_news)}개 → {len(filtered_news)}개 / "
        f"제외 {excluded_count}개 / 과거 이슈 {len(past_issues)}개 / "
        f"LLM 토큰 0"
    )

    return {
        "success": True,
        "message": "규칙 기반 반복 이슈 필터 완료(LLM 미사용)",
        "filtered_news": filtered_news,
        "excluded_count": excluded_count,
        "excluded_items": excluded_items,
        "past_issue_count": len(past_issues),
        "prefilter_excluded_count": url_excluded_count + title_excluded_count + text_excluded_count + token_overlap_excluded_count + simhash_excluded_count,
        "llm_excluded_count": 0,
        "core_key_excluded_count": 0,
        "internal_duplicate_count": internal_duplicate_count,
        "url_excluded_count": url_excluded_count,
        "title_excluded_count": title_excluded_count,
        "text_excluded_count": text_excluded_count,
        "token_overlap_excluded_count": token_overlap_excluded_count,
        "simhash_excluded_count": simhash_excluded_count,
        "token_stats": get_last_token_stats(),
    }


def deduplicate_section_results(section_results):
    """
    메일 발송 직전 전체 섹션의 최종 요약 결과를 다시 한 번 사건 단위로 중복 제거한다.
    이미 요약된 결과만 대상으로 하므로 OpenAI 호출/토큰 사용이 늘지 않는다.
    """
    kept_payloads = []
    kept_urls = set()
    excluded_items = []
    total_before = 0
    total_after = 0

    for section_result in section_results or []:
        section_name = section_result.get("section_name", "뉴스 섹션")
        summaries = section_result.get("summaries", []) or []
        total_before += len(summaries)
        deduped = []

        for idx, news in enumerate(summaries):
            news_title = get_news_title(news)
            news_url = normalize_url(get_news_url(news))
            candidate_payload = build_event_signature(news)

            if news_url and news_url in kept_urls:
                excluded_items.append({
                    "section_name": section_name,
                    "title": news_title,
                    "matched_title": "메일 전체 내부 중복 URL",
                    "method": "final_internal_url",
                })
                continue

            matched_payload, method, detail = find_matching_payload(candidate_payload, kept_payloads)
            if matched_payload:
                excluded_items.append({
                    "section_name": section_name,
                    "title": news_title,
                    "matched_title": matched_payload.get("title", ""),
                    "method": f"final_{method}",
                    "detail": detail,
                })
                continue

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

        section_result["summaries"] = deduped
        section_result["selected_count"] = len(deduped)
        scrape_stats = section_result.setdefault("scrape_stats", {})
        scrape_stats["final_mail_dedup_before"] = len(summaries)
        scrape_stats["final_mail_dedup_after"] = len(deduped)
        scrape_stats["final_mail_dedup_excluded"] = len(summaries) - len(deduped)
        total_after += len(deduped)

    logger.info(
        "🧹 메일 발송 직전 전체 섹션 중복 제거 완료: %s개 → %s개 / 제외 %s개 / LLM 토큰 0",
        total_before,
        total_after,
        total_before - total_after,
    )

    return {
        "success": True,
        "before_count": total_before,
        "after_count": total_after,
        "excluded_count": total_before - total_after,
        "excluded_items": excluded_items,
        "section_results": section_results,
    }


if __name__ == "__main__":
    history = load_issue_history()
    print(f"현재 저장된 이슈 수: {len(history.get('issues', []))}")
