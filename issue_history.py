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
        token = token.lower().strip()
        if not token:
            continue
        if token in STOPWORDS:
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


def build_compare_payload(title: str, summary: str):
    compare_text = normalize_compare_text(f"{title} {summary}")
    compact_text = compact_compare_text(compare_text)
    tokens = extract_tokens(compare_text)
    return {
        "normalized_title": normalize_title(title),
        "normalized_text": compact_text,
        "tokens": tokens,
        "fingerprint": make_simhash(tokens),
    }


# ====================================
# 파일 입출력
# ====================================
def load_issue_history(file_path=HISTORY_FILE_PATH):
    if not os.path.exists(file_path):
        return {"version": 2, "issues": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 2, "issues": []}
        if "issues" not in data or not isinstance(data["issues"], list):
            data["issues"] = []
        data["version"] = max(int(data.get("version", 1) or 1), 2)
        return data
    except Exception as e:
        logger.warning(f"⚠️ 이슈 히스토리 읽기 실패, 빈 히스토리로 시작합니다: {e}")
        return {"version": 2, "issues": []}


def save_issue_history(data, file_path=HISTORY_FILE_PATH):
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def prune_old_issues(history, days=3):
    if not isinstance(history, dict):
        return {"version": 2, "issues": []}, 0

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
    history["version"] = max(int(history.get("version", 2) or 2), 2)
    return history, removed_count


# ====================================
# 히스토리 record 생성/조회
# ====================================
def make_issue_id(briefing_name, receiver_env, section_name, title, summary, url=None):
    normalized_url = normalize_url(url)
    scope = "|".join([
        str(briefing_name or "").strip(),
        str(receiver_env or "").strip(),
        str(section_name or "").strip(),
    ])

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
    category = str(news.get("category") or "").strip()
    source = str(news.get("source") or "").strip()
    url = get_news_url(news)
    published_at = str(news.get("published_at") or "").strip()
    importance_score = news.get("importance_score")

    payload = build_compare_payload(title, summary)
    normalized_url = normalize_url(url)
    issue_id = make_issue_id(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        title=title,
        summary=summary,
        url=url,
    )

    now_kst = get_now_kst()

    return {
        "issue_id": issue_id,
        "saved_at": now_kst.isoformat(),
        "saved_date": now_kst.strftime("%Y-%m-%d"),
        "briefing_name": briefing_name,
        "subject_prefix": subject_prefix,
        "receiver_env": receiver_env,
        "section_name": section_name,
        "title": title,
        "summary": summary,
        "category": category,
        "source": source,
        "url": url,
        "normalized_url": normalized_url,
        "normalized_title": payload["normalized_title"],
        "normalized_text": payload["normalized_text"],
        "content_fingerprint": payload["fingerprint"],
        "content_tokens": payload["tokens"],
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
    summary = str(issue.get("summary") or issue.get("description") or issue.get("content") or "")

    normalized_title = str(issue.get("normalized_title") or "").strip() or normalize_title(title)
    normalized_text = str(issue.get("normalized_text") or "").strip()
    tokens = issue.get("content_tokens")
    fingerprint = str(issue.get("content_fingerprint") or "").strip()

    if not normalized_text or not isinstance(tokens, list) or not fingerprint:
        payload = build_compare_payload(title, summary)
        normalized_text = normalized_text or payload["normalized_text"]
        tokens = tokens if isinstance(tokens, list) and tokens else payload["tokens"]
        fingerprint = fingerprint or payload["fingerprint"]

    return {
        "normalized_title": normalized_title,
        "normalized_text": normalized_text,
        "tokens": tokens or [],
        "fingerprint": fingerprint,
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

    for item in history.get("issues", []):
        issue_id = item.get("issue_id")
        if issue_id:
            existing_issue_ids.add(str(issue_id))

        scope = "|".join([
            str(item.get("briefing_name") or ""),
            str(item.get("receiver_env") or ""),
            str(item.get("section_name") or ""),
        ])
        item_url = get_issue_normalized_url(item)
        item_payload = get_issue_compare_payload(item)
        item_title = item_payload.get("normalized_title", "")

        if item_url:
            existing_scope_url_keys.add(f"{scope}|{item_url}")
        if item_title:
            existing_scope_title_keys.add(f"{scope}|{item_title}")

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

            scope = "|".join([
                str(briefing_name or ""),
                str(receiver_env or ""),
                str(section_name or ""),
            ])
            url_key = f"{scope}|{record.get('normalized_url', '')}"
            title_key = f"{scope}|{record.get('normalized_title', '')}"

            if record["issue_id"] in existing_issue_ids:
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_url") and url_key in existing_scope_url_keys:
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_title") and title_key in existing_scope_title_keys:
                skipped_duplicate_count += 1
                continue

            existing_issue_ids.add(record["issue_id"])
            if record.get("normalized_url"):
                existing_scope_url_keys.add(url_key)
            if record.get("normalized_title"):
                existing_scope_title_keys.add(title_key)

            new_records.append(record)

    if new_records:
        history["issues"].extend(new_records)

    history["last_updated_at"] = get_now_kst().isoformat()
    history["version"] = max(int(history.get("version", 2) or 2), 2)
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
    days=3, file_path=HISTORY_FILE_PATH
):
    history = load_issue_history(file_path)
    issues = history.get("issues", [])
    today = get_now_kst().date()
    recent_issues = []

    for issue in issues:
        if issue.get("briefing_name") != briefing_name:
            continue
        if issue.get("receiver_env") != receiver_env:
            continue
        if issue.get("section_name") != section_name:
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

        past_payloads.append({
            "issue": issue,
            "title": issue.get("title", ""),
            "normalized_title": payload.get("normalized_title", ""),
            "normalized_text": payload.get("normalized_text", ""),
            "tokens": payload.get("tokens", []),
            "fingerprint": payload.get("fingerprint", ""),
        })

    return {
        "past_by_url": past_by_url,
        "past_payloads": past_payloads,
    }


def judge_duplicate_by_payload(candidate_payload, past_payload):
    """
    후보와 과거 이슈가 같은 반복 이슈인지 규칙 기반으로 판단한다.
    Returns: (is_duplicate, method, score_text)
    """
    cand_title = candidate_payload.get("normalized_title", "")
    past_title = past_payload.get("normalized_title", "")

    if cand_title and past_title:
        if cand_title == past_title:
            return True, "title_exact", "정규화 제목 동일"

        title_score = SequenceMatcher(None, cand_title, past_title).ratio()
        if title_score >= TITLE_SIMILARITY_THRESHOLD:
            return True, "title_similarity", f"제목 유사도 {title_score:.2f}"

    cand_text = candidate_payload.get("normalized_text", "")
    past_text = past_payload.get("normalized_text", "")
    if cand_text and past_text:
        text_score = SequenceMatcher(None, cand_text, past_text).ratio()
        if text_score >= TEXT_SIMILARITY_THRESHOLD:
            return True, "text_similarity", f"본문 유사도 {text_score:.2f}"

    overlap_score, common_count = token_overlap_score(
        candidate_payload.get("tokens", []),
        past_payload.get("tokens", []),
    )
    if common_count >= MIN_COMMON_TOKEN_COUNT and overlap_score >= TOKEN_OVERLAP_THRESHOLD:
        return True, "token_overlap", f"토큰 겹침률 {overlap_score:.2f}, 공통 {common_count}개"

    distance = simhash_distance(
        candidate_payload.get("fingerprint", ""),
        past_payload.get("fingerprint", ""),
    )
    if distance is not None and distance <= SIMHASH_DISTANCE_THRESHOLD:
        # SimHash만으로 과하게 지워지는 것을 막기 위해 최소 토큰 공통 조건을 추가한다.
        if common_count >= max(2, MIN_COMMON_TOKEN_COUNT - 1):
            return True, "simhash", f"SimHash 거리 {distance}, 공통 토큰 {common_count}개"

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
        candidate_payload = build_compare_payload(news_title, news_summary)

        # 1. 과거 발송 URL 완전 일치
        if news_url and news_url in past_by_url:
            matched = past_by_url[news_url]
            excluded_items.append({
                "index": idx,
                "title": news_title,
                "matched_past_issue": matched.get("title", ""),
                "reason": "같은 URL의 기사가 최근 발송 이력에 있음",
                "method": "url",
            })
            url_excluded_count += 1
            continue

        # 2. 과거 발송 텍스트/fingerprint 비교
        matched_payload, method, detail = find_matching_payload(candidate_payload, past_payloads)
        if matched_payload:
            matched_issue = matched_payload.get("issue", {})
            excluded_items.append({
                "index": idx,
                "title": news_title,
                "matched_past_issue": matched_issue.get("title", ""),
                "reason": f"최근 발송 이력과 유사함 ({detail})",
                "method": method,
            })
            if method in ("title_exact", "title_similarity"):
                title_excluded_count += 1
            elif method == "text_similarity":
                text_excluded_count += 1
            elif method == "token_overlap":
                token_overlap_excluded_count += 1
            elif method == "simhash":
                simhash_excluded_count += 1
            continue

        # 3. 오늘 후보 내부 URL 중복
        if news_url and news_url in seen_today_urls:
            excluded_items.append({
                "index": idx,
                "title": news_title,
                "matched_past_issue": "오늘 후보 내부 중복 URL",
                "reason": "오늘 후보 내부에서 같은 URL이 이미 유지됨",
                "method": "internal_url",
            })
            internal_duplicate_count += 1
            continue

        # 4. 오늘 후보 내부 텍스트/fingerprint 중복
        matched_today, today_method, today_detail = find_matching_payload(candidate_payload, seen_today_payloads)
        if matched_today:
            excluded_items.append({
                "index": idx,
                "title": news_title,
                "matched_past_issue": matched_today.get("title", ""),
                "reason": f"오늘 후보 내부에서 이미 유지한 기사와 유사함 ({today_detail})",
                "method": f"internal_{today_method}",
            })
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
            "fingerprint": candidate_payload.get("fingerprint", ""),
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


if __name__ == "__main__":
    history = load_issue_history()
    print(f"현재 저장된 이슈 수: {len(history.get('issues', []))}")
