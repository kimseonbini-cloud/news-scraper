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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ====================================
# 기본값
# ====================================
DEFAULT_DISPLAY_PER_KEYWORD = 100
DEFAULT_PAGES_PER_KEYWORD = 3
DEFAULT_SORTS = ["sim"]
DEFAULT_RECENT_HOURS = 24
DEFAULT_MAX_TOTAL_NEWS = 300
DEFAULT_OUTPUT_PATH = "data/grouping_test/news_grouping_result.json"

# 그룹화 기준값. 테스트하면서 조정하기 쉽게 CLI 인자로도 받는다.
DEFAULT_TITLE_SIMILARITY_THRESHOLD = 0.78
DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_TOKEN_OVERLAP_THRESHOLD = 0.65
DEFAULT_SIMHASH_DISTANCE_THRESHOLD = 6
DEFAULT_MIN_COMMON_TOKEN_COUNT = 5

# 너무 흔해서 사건 구분에 도움이 약한 단어들
STOPWORDS = {
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
    "한국", "국내", "글로벌", "최신", "주요", "확인", "가능", "기준",
}

TOKEN_SUFFIXES = (
    "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
    "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
)

# 대표 기사로 쓰기 애매한 패턴. 그룹에는 남기되 대표 선정 점수에서 감점한다.
PHOTO_URL_PATTERNS = ("/photos/", "/photo/", "NISI")
PHOTO_TITLE_PATTERNS = (
    "참석하는", "주재하는", "발언하는", "모두 발언", "모두발언", "기념촬영",
    "손드는", "이동하는", "입장하는", "대화하는", "자료 살펴보", "브리핑하는",
)
BREAKING_TITLE_PATTERNS = ("[속보]", "[1보]", "[2보]", "[외환]", "[달러·원]")



# ====================================
# 텍스트/URL 정규화 유틸
# ====================================
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


def normalize_compare_text(text: str) -> str:
    if not text:
        return ""
    text = strip_html_entities(text).lower()
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


def extract_tokens(text: str, max_tokens: int = 80) -> List[str]:
    text = normalize_compare_text(text)
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)

    tokens: List[str] = []
    seen = set()
    for token in raw_tokens:
        token = normalize_token(token)
        if not token or token in STOPWORDS or token.isdigit() or len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())
        domain = parsed.netloc.lower().strip()
        path = parsed.path.strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return f"{domain}{path}".rstrip("/")
    except Exception:
        return str(url).lower().strip().rstrip("/")


def stable_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def make_simhash(tokens: Iterable[str]) -> str:
    tokens = list(tokens or [])
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
            fingerprint |= 1 << bit

    return f"{fingerprint:016x}"


def simhash_distance(a: str, b: str) -> Optional[int]:
    if not a or not b:
        return None
    try:
        x = int(str(a), 16) ^ int(str(b), 16)
        return x.bit_count()
    except Exception:
        return None


def token_overlap_score(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> Tuple[float, int]:
    set_a = set(tokens_a or [])
    set_b = set(tokens_b or [])
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b
    denominator = max(1, min(len(set_a), len(set_b)))
    return len(common) / denominator, len(common)


def get_news_title(news: Dict[str, Any]) -> str:
    return str(news.get("title") or "").strip()


def get_news_description(news: Dict[str, Any]) -> str:
    return str(news.get("description") or news.get("summary") or news.get("content") or "").strip()


def get_news_url(news: Dict[str, Any]) -> str:
    return str(news.get("originallink") or news.get("url") or news.get("link") or "").strip()


def parse_iso_datetime(value: str) -> str:
    """로그/정렬용 문자열. 실패하면 빈 문자열."""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except Exception:
        return str(value)


def parse_dt_for_compare(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def is_photo_like(payload: "NewsPayload") -> bool:
    title = payload.title or ""
    url = payload.url or ""
    if any(pattern in url for pattern in PHOTO_URL_PATTERNS):
        return True
    if any(pattern in title for pattern in PHOTO_TITLE_PATTERNS):
        return True
    return False


def is_breaking_like(payload: "NewsPayload") -> bool:
    title = payload.title or ""
    return any(pattern in title for pattern in BREAKING_TITLE_PATTERNS)


def representative_quality_score(payload: "NewsPayload", group: Optional["NewsGroup"] = None) -> Tuple[float, List[str]]:
    """
    그룹 대표 기사 후보 점수.
    높은 점수일수록 메일 대표 기사로 쓰기 좋다.
    """
    score = 0.0
    flags: List[str] = []

    title_len = len(payload.title or "")
    desc_len = len(payload.description or "")

    score += min(title_len, 70) / 10.0
    score += min(desc_len, 220) / 35.0

    if payload.normalized_url:
        score += 1.0
    if payload.source:
        score += 0.5

    if group is not None:
        score += min(group.source_count, 6) * 0.4
        score += min(group.article_count, 8) * 0.2

    if is_photo_like(payload):
        score -= 6.0
        flags.append("photo_like")

    if is_breaking_like(payload):
        score -= 2.0
        flags.append("breaking_like")

    if desc_len == 0:
        score -= 3.0
        flags.append("empty_description")
    elif desc_len < 40:
        score -= 1.5
        flags.append("short_description")

    if title_len < 16:
        score -= 2.0
        flags.append("short_title")

    if len(payload.tokens) < 5:
        score -= 1.5
        flags.append("few_tokens")

    return score, flags


def refresh_group_representative(group: "NewsGroup") -> None:
    """그룹 안에서 메일 대표 기사로 가장 좋은 후보를 다시 고른다."""
    if not group.items:
        return
    scored = []
    for item in group.items:
        score, flags = representative_quality_score(item, group)
        scored.append((score, item.published_at_kst or "", item, flags))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    group.representative = scored[0][2]


def group_time_span_hours(group: "NewsGroup") -> float:
    dates = [parse_dt_for_compare(item.published_at_kst) for item in group.items]
    dates = [d for d in dates if d is not None]
    if len(dates) < 2:
        return 0.0
    return (max(dates) - min(dates)).total_seconds() / 3600.0


def group_quality_flags(group: "NewsGroup") -> List[str]:
    flags: List[str] = []
    method_counts = Counter(reason.get("method", "unknown") for reason in group.match_reasons)
    non_rep_count = max(group.article_count - 1, 1)
    token_ratio = method_counts.get("token_overlap", 0) / non_rep_count
    span_hours = group_time_span_hours(group)
    rep_score, rep_flags = representative_quality_score(group.representative, group)

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


@dataclass
class NewsPayload:
    index: int
    news: Dict[str, Any]
    title: str
    description: str
    source: str
    keyword: str
    published_at_kst: str
    url: str
    normalized_url: str
    normalized_title: str
    normalized_text: str
    tokens: List[str]
    title_tokens: List[str]
    fingerprint: str


@dataclass
class NewsGroup:
    group_id: str
    representative: NewsPayload
    items: List[NewsPayload] = field(default_factory=list)
    match_reasons: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, payload: NewsPayload, reason: Dict[str, Any]) -> None:
        self.items.append(payload)
        self.match_reasons.append(reason)

    @property
    def sources(self) -> List[str]:
        return sorted({item.source for item in self.items if item.source})

    @property
    def keywords(self) -> List[str]:
        return sorted({item.keyword for item in self.items if item.keyword})

    @property
    def article_count(self) -> int:
        return len(self.items)

    @property
    def source_count(self) -> int:
        return len(self.sources)


def build_payload(news: Dict[str, Any], index: int) -> NewsPayload:
    title = get_news_title(news)
    description = get_news_description(news)
    compare_text = f"{title} {description}"
    tokens = extract_tokens(compare_text)
    title_tokens = extract_tokens(title, max_tokens=30)
    return NewsPayload(
        index=index,
        news=news,
        title=title,
        description=description,
        source=str(news.get("source") or "").strip(),
        keyword=str(news.get("keyword") or "").strip(),
        published_at_kst=parse_iso_datetime(str(news.get("published_at_kst") or "")),
        url=get_news_url(news),
        normalized_url=normalize_url(get_news_url(news)),
        normalized_title=normalize_title(title),
        normalized_text=compact_compare_text(compare_text),
        tokens=tokens,
        title_tokens=title_tokens,
        fingerprint=make_simhash(tokens),
    )


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
    # URL 완전 일치. 보통 스크래퍼에서 이미 제거되지만 안전망으로 둔다.
    if candidate.normalized_url and candidate.normalized_url == representative.normalized_url:
        return True, {"method": "url", "detail": "정규화 URL 동일", "score": 1.0}

    # 제목 완전 일치
    if candidate.normalized_title and candidate.normalized_title == representative.normalized_title:
        return True, {"method": "title_exact", "detail": "정규화 제목 동일", "score": 1.0}

    title_score = 0.0
    if candidate.normalized_title and representative.normalized_title:
        if min(len(candidate.normalized_title), len(representative.normalized_title)) >= 10:
            shorter, longer = sorted(
                [candidate.normalized_title, representative.normalized_title],
                key=len,
            )
            if shorter in longer:
                return True, {
                    "method": "title_contains",
                    "detail": "한쪽 제목이 다른 쪽 제목을 포함",
                    "score": len(shorter) / max(1, len(longer)),
                }

        title_score = SequenceMatcher(None, candidate.normalized_title, representative.normalized_title).ratio()
        if title_score >= title_threshold:
            return True, {
                "method": "title_similarity",
                "detail": f"제목 유사도 {title_score:.2f}",
                "score": title_score,
            }

    title_overlap_score, title_common_count = token_overlap_score(
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

    text_score = 0.0
    if candidate.normalized_text and representative.normalized_text:
        text_score = SequenceMatcher(None, candidate.normalized_text, representative.normalized_text).ratio()
        if text_score >= text_threshold:
            return True, {
                "method": "text_similarity",
                "detail": f"본문 유사도 {text_score:.2f}",
                "score": text_score,
            }

    overlap_score, common_count = token_overlap_score(candidate.tokens, representative.tokens)
    if common_count >= min_common_tokens and overlap_score >= token_threshold:
        return True, {
            "method": "token_overlap",
            "detail": f"토큰 겹침률 {overlap_score:.2f}, 공통 {common_count}개",
            "score": overlap_score,
            "common_count": common_count,
        }

    distance = simhash_distance(candidate.fingerprint, representative.fingerprint)
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


def group_news(
    news_list: List[Dict[str, Any]],
    title_threshold: float = DEFAULT_TITLE_SIMILARITY_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
    simhash_threshold: int = DEFAULT_SIMHASH_DISTANCE_THRESHOLD,
    min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT,
) -> List[NewsGroup]:
    payloads = [build_payload(news, idx + 1) for idx, news in enumerate(news_list or [])]

    # 최신 기사 우선으로 그룹 대표가 잡히도록 정렬한다.
    payloads.sort(key=lambda p: p.published_at_kst or "", reverse=True)

    groups: List[NewsGroup] = []

    for payload in payloads:
        best_group: Optional[NewsGroup] = None
        best_reason: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for group in groups:
            # 대표 기사뿐 아니라 그룹 내 기사 일부와도 비교한다.
            # 단, 속도와 로그 안정성을 위해 최신 5개까지만 비교한다.
            compare_targets = [group.representative] + group.items[:5]

            for target in compare_targets:
                is_match, reason = compare_payloads(
                    candidate=payload,
                    representative=target,
                    title_threshold=title_threshold,
                    text_threshold=text_threshold,
                    token_threshold=token_threshold,
                    simhash_threshold=simhash_threshold,
                    min_common_tokens=min_common_tokens,
                )
                if is_match:
                    score = float(reason.get("score", 0.0) or 0.0)
                    if score > best_score:
                        best_group = group
                        best_reason = {
                            **reason,
                            "matched_with_index": target.index,
                            "matched_with_title": target.title,
                        }
                        best_score = score

        if best_group is None:
            group_id = f"G{len(groups) + 1:03d}"
            new_group = NewsGroup(
                group_id=group_id,
                representative=payload,
                items=[payload],
                match_reasons=[{"method": "representative", "detail": "그룹 대표 기사"}],
            )
            groups.append(new_group)
        else:
            assert best_reason is not None
            best_group.add(payload, best_reason)

    # 그룹이 완성된 뒤, 사진/속보/짧은 기사보다 설명이 풍부한 기사를 대표로 재선정한다.
    for group in groups:
        refresh_group_representative(group)

    # 큰 그룹, 언론사 다양한 그룹, 최신 그룹 순으로 정렬
    groups.sort(
        key=lambda g: (
            g.article_count,
            g.source_count,
            g.representative.published_at_kst or "",
        ),
        reverse=True,
    )

    return groups


def serialize_group(group: NewsGroup) -> Dict[str, Any]:
    method_counts = Counter(reason.get("method", "unknown") for reason in group.match_reasons)
    rep_score, rep_flags = representative_quality_score(group.representative, group)
    flags = group_quality_flags(group)

    return {
        "group_id": group.group_id,
        "article_count": group.article_count,
        "source_count": group.source_count,
        "sources": group.sources,
        "keywords": group.keywords,
        "time_span_hours": round(group_time_span_hours(group), 2),
        "quality_flags": flags,
        "representative_score": round(rep_score, 2),
        "representative_flags": rep_flags,
        "representative": {
            "index": group.representative.index,
            "title": group.representative.title,
            "description": group.representative.description,
            "source": group.representative.source,
            "keyword": group.representative.keyword,
            "published_at_kst": group.representative.published_at_kst,
            "url": group.representative.url,
        },
        "match_method_counts": dict(method_counts),
        "articles": [
            {
                "index": item.index,
                "title": item.title,
                "source": item.source,
                "keyword": item.keyword,
                "published_at_kst": item.published_at_kst,
                "url": item.url,
                "representative_candidate_score": round(representative_quality_score(item, group)[0], 2),
                "representative_candidate_flags": representative_quality_score(item, group)[1],
                "match_reason": group.match_reasons[i] if i < len(group.match_reasons) else {},
            }
            for i, item in enumerate(group.items)
        ],
    }



# ====================================
# 운영용 헬퍼
# ====================================
def serialize_groups(groups: List[NewsGroup]) -> List[Dict[str, Any]]:
    return [serialize_group(group) for group in groups]


def is_low_quality_group(serialized_group: Dict[str, Any]) -> bool:
    flags = set(serialized_group.get("quality_flags") or [])
    rep_flags = set(serialized_group.get("representative_flags") or [])
    rep_score = float(serialized_group.get("representative_score") or 0)

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


def group_priority_score(serialized_group: Dict[str, Any]) -> float:
    """AI 선별 후보로 넘길 그룹의 로컬 우선순위 점수."""
    article_count = int(serialized_group.get("article_count") or 0)
    source_count = int(serialized_group.get("source_count") or 0)
    rep_score = float(serialized_group.get("representative_score") or 0)
    flags = set(serialized_group.get("quality_flags") or [])

    score = 0.0
    score += min(source_count, 8) * 2.2
    score += min(article_count, 10) * 0.9
    score += rep_score * 0.8

    if source_count >= 4:
        score += 2.0
    if article_count >= 4:
        score += 1.0
    if "many_articles_few_sources" in flags:
        score -= 2.5
    if "photo_like_representative" in flags:
        score -= 4.0
    if "overgroup_risk_token_dominant" in flags:
        score -= 2.0
    if "overgroup_risk_token_time_span" in flags:
        score -= 6.0
    if "low_representative_score" in flags:
        score -= 5.0

    return round(score, 3)


def representative_news_from_group(serialized_group: Dict[str, Any]) -> Dict[str, Any]:
    rep = dict(serialized_group.get("representative") or {})
    articles = serialized_group.get("articles") or []
    rep["group_id"] = serialized_group.get("group_id")
    rep["group_article_count"] = serialized_group.get("article_count", 1)
    rep["group_source_count"] = serialized_group.get("source_count", 1)
    rep["group_sources"] = serialized_group.get("sources", [])
    rep["group_keywords"] = serialized_group.get("keywords", [])
    rep["group_quality_flags"] = serialized_group.get("quality_flags", [])
    rep["group_representative_score"] = serialized_group.get("representative_score", 0)
    rep["group_priority_score"] = serialized_group.get("priority_score", 0)
    rep["group_article_titles"] = [
        str(article.get("title") or "").strip()
        for article in articles[:12]
        if str(article.get("title") or "").strip()
    ]
    rep["group_article_urls"] = [
        str(article.get("url") or "").strip()
        for article in articles[:12]
        if str(article.get("url") or "").strip()
    ]
    rep["description"] = rep.get("description", "")
    rep["content"] = rep.get("description", "")
    return rep


def build_grouping_result(
    news_list: List[Dict[str, Any]],
    max_groups: int = 100,
    exclude_low_quality: bool = True,
    title_threshold: float = DEFAULT_TITLE_SIMILARITY_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    token_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
    simhash_threshold: int = DEFAULT_SIMHASH_DISTANCE_THRESHOLD,
    min_common_tokens: int = DEFAULT_MIN_COMMON_TOKEN_COUNT,
) -> Dict[str, Any]:
    groups = group_news(
        news_list=news_list,
        title_threshold=title_threshold,
        text_threshold=text_threshold,
        token_threshold=token_threshold,
        simhash_threshold=simhash_threshold,
        min_common_tokens=min_common_tokens,
    )

    serialized_groups = serialize_groups(groups)
    for group in serialized_groups:
        group["priority_score"] = group_priority_score(group)
        group["low_quality_excluded"] = is_low_quality_group(group)

    selection_groups = [
        group for group in serialized_groups
        if not (exclude_low_quality and group.get("low_quality_excluded"))
    ]
    selection_groups.sort(
        key=lambda g: (
            float(g.get("priority_score") or 0),
            int(g.get("source_count") or 0),
            int(g.get("article_count") or 0),
            str((g.get("representative") or {}).get("published_at_kst") or ""),
        ),
        reverse=True,
    )
    selection_groups = selection_groups[:max_groups]

    multi_article_groups = [group for group in serialized_groups if int(group.get("article_count") or 0) >= 2]
    low_quality_groups = [group for group in serialized_groups if group.get("low_quality_excluded")]

    # 기사 단위 통계
    # - duplicate_article_count: 같은 사건으로 묶이며 대표 1건만 남아 AI 후보에서 빠진 기사 수
    # - low_quality_article_count: 사진성/저품질 그룹으로 판단되어 AI 후보에서 제외된 기사 수
    #   메일 대시보드에서 “AI 중복제외”, “코드규칙제외”를 기사 수 기준으로 보여주기 위해 사용한다.
    duplicate_article_count = sum(
        max(int(group.get("article_count") or 0) - 1, 0)
        for group in serialized_groups
    )
    low_quality_article_count = sum(
        int(group.get("article_count") or 0)
        for group in low_quality_groups
    )
    selection_article_count = sum(
        int(group.get("article_count") or 0)
        for group in selection_groups
    )

    return {
        "news_count": len(news_list or []),
        "group_count": len(serialized_groups),
        "multi_article_group_count": len(multi_article_groups),
        "low_quality_group_count": len(low_quality_groups),
        "low_quality_article_count": low_quality_article_count,
        "duplicate_article_count": duplicate_article_count,
        "selection_group_count": len(selection_groups),
        "selection_article_count": selection_article_count,
        "parameters": {
            "title_threshold": title_threshold,
            "text_threshold": text_threshold,
            "token_threshold": token_threshold,
            "simhash_threshold": simhash_threshold,
            "min_common_tokens": min_common_tokens,
            "exclude_low_quality": exclude_low_quality,
            "max_groups": max_groups,
        },
        "groups": serialized_groups,
        "selection_groups": selection_groups,
        "representative_news": [representative_news_from_group(group) for group in selection_groups],
    }
