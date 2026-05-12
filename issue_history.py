"""
뉴스 이슈 히스토리 관리

목적:
- 메일에 실제로 발송된 선별 뉴스 이슈를 저장한다.
- 이후 실행에서 최근 이슈와 후보 뉴스를 비교해 반복 이슈를 제거한다.
- 히스토리 파일은 자동 삭제하지 않고 계속 누적 저장한다.
- 저장 시에는 최근 keep_days일만 유지한다.
- 비교할 때는 같은 환경설정/브리핑/섹션의 최근 N일 이슈만 사용한다.

[v2 변경사항]
- 하드코딩된 단어 치환/토큰 목록을 전부 제거하고 동적으로 처리한다.
- 규칙 기반 필터를 통과한 후보에 대해 LLM 의미 비교를 추가한다.
  (batch_llm_duplicate_check)
- 유사도 임계값을 소폭 완화한다.
- llm_excluded_count가 실제로 집계되도록 수정한다.
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

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ====================================
# 기본 설정
# ====================================
HISTORY_FILE_PATH = "data/seen_issues.json"
KST = pytz.timezone("Asia/Seoul")

MODEL = os.getenv("ISSUE_HISTORY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 제목 유사도 기준 (v2: 0.88 → 0.80)
TITLE_SIMILARITY_THRESHOLD = 0.80

# core issue key 유사도 기준 (v2: 0.78 → 0.68 / 0.70 → 0.62)
CORE_KEY_SIMILARITY_THRESHOLD = 0.68
CORE_EVENT_SIMILARITY_THRESHOLD = 0.62

# LLM 이슈키 생성 chunk 크기
ISSUE_KEY_CHUNK_SIZE = int(os.getenv("ISSUE_KEY_CHUNK_SIZE", "50"))

# LLM 중복 판단: 후보를 묶어서 한 번에 보낼 배치 크기
LLM_DUPLICATE_BATCH_SIZE = int(os.getenv("LLM_DUPLICATE_BATCH_SIZE", "10"))

# LLM 중복 판단: 비교할 과거 이슈 최대 수 (최신순)
LLM_PAST_ISSUE_LIMIT = int(os.getenv("LLM_PAST_ISSUE_LIMIT", "40"))

# URL 정규화 시 유지할 의미 있는 query parameter
MEANINGFUL_QUERY_PARAMS = {
    "no", "idxno", "article_no", "articleid", "article_id",
    "newsid", "news_id", "aid", "oid", "sid", "id", "seq", "num",
}


def get_now_kst():
    return datetime.now(KST)


# ====================================
# 정규화 유틸
# ====================================
def normalize_url(url: str) -> str:
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


def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
    text = text.replace("…", "").replace("...", "")
    text = re.sub(r"[\[\]【】〈〉《》\u201c\u201d\u2018\u2019\"'`]", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text.strip()


def normalize_text_for_issue(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_issue_key(value: str) -> str:
    if not value:
        return ""
    text = str(value).lower().strip()
    text = text.replace("\uff5c", "|")  # ｜
    text = re.sub(r"[ㆍ·\- _]", "", text)
    text = re.sub(r"[\"'`\u201c\u201d\u2018\u2019]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣|]", "", text)
    return text


def normalize_core_issue_key(value: str) -> str:
    """
    core_issue_key 비교용 정규화.

    v2: 하드코딩된 단어 치환 목록을 제거하고 수치/연도만 제거한다.
    LLM이 일관된 키를 생성하도록 프롬프트로 제어하며,
    여기서는 최소한의 수치 정규화만 수행한다.
    """
    if not value:
        return ""
    text = str(value).lower().strip()
    text = text.replace("\uff5c", "|")  # ｜
    text = re.sub(r"[ㆍ·\- _]", "", text)
    text = re.sub(r"[\"'`\u201c\u201d\u2018\u2019]", "", text)

    # 연도 표기 제거 (이슈가 갈라지는 주 원인)
    text = re.sub(r"\d{4}년", "", text)
    text = re.sub(r"\d{4}", "", text)

    # 세부 수치 제거
    text = re.sub(r"\d+(억|억원|조|조원|만|명|개|%|퍼센트)", "", text)
    text = re.sub(r"\d+\.\d+", "", text)

    text = re.sub(r"[^0-9a-z가-힣|]", "", text)
    return text.strip("|")


def split_core_key(value: str):
    norm = normalize_core_issue_key(value)
    if not norm:
        return "", ""
    parts = [p for p in norm.split("|") if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "".join(parts[1:])


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


def make_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def is_same_core_issue(a: str, b: str) -> bool:
    """
    두 core_issue_key가 같은 이슈인지 판단한다.

    v2: 하드코딩된 토큰 목록을 제거하고 구조/유사도 기반으로만 판단한다.
    """
    a_norm = normalize_core_issue_key(a)
    b_norm = normalize_core_issue_key(b)

    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True

    whole_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    if whole_score >= CORE_KEY_SIMILARITY_THRESHOLD:
        return True

    a_entity, a_event = split_core_key(a_norm)
    b_entity, b_event = split_core_key(b_norm)

    if not a_entity or not b_entity:
        return False

    entity_score = SequenceMatcher(None, a_entity, b_entity).ratio()
    if entity_score < 0.86:
        return False

    if not a_event or not b_event:
        return False

    if a_event in b_event or b_event in a_event:
        return True

    event_score = SequenceMatcher(None, a_event, b_event).ratio()
    if event_score >= CORE_EVENT_SIMILARITY_THRESHOLD:
        return True

    return False


def make_issue_id(
    briefing_name, receiver_env, section_name,
    title, summary, url=None, core_issue_key=None
):
    normalized_core = normalize_core_issue_key(core_issue_key)
    if normalized_core:
        raw = "|".join([
            "core_issue",
            str(briefing_name or "").strip(),
            str(receiver_env or "").strip(),
            str(section_name or "").strip(),
            normalized_core
        ])
        return make_hash(raw)

    normalized_url = normalize_url(url)
    if normalized_url:
        raw = "|".join([
            "url",
            str(briefing_name or "").strip(),
            str(receiver_env or "").strip(),
            str(section_name or "").strip(),
            normalized_url
        ])
        return make_hash(raw)

    raw = "|".join([
        "title_summary",
        str(briefing_name or "").strip(),
        str(receiver_env or "").strip(),
        str(section_name or "").strip(),
        normalize_title(title),
        normalize_text_for_issue(summary)
    ])
    return make_hash(raw)


# ====================================
# 파일 입출력
# ====================================
def load_issue_history(file_path=HISTORY_FILE_PATH):
    if not os.path.exists(file_path):
        return {"version": 1, "issues": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "issues": []}
        if "issues" not in data or not isinstance(data["issues"], list):
            data["issues"] = []
        if "version" not in data:
            data["version"] = 1
        return data
    except Exception as e:
        logger.warning(f"⚠️ 이슈 히스토리 읽기 실패, 빈 히스토리로 시작합니다: {e}")
        return {"version": 1, "issues": []}


def save_issue_history(data, file_path=HISTORY_FILE_PATH):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ====================================
# 오래된 이슈 정리
# ====================================
def prune_old_issues(history, days=3):
    if not isinstance(history, dict):
        return {"version": 1, "issues": []}, 0

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
    return history, removed_count


# ====================================
# LLM 이슈키 생성
# ====================================
def fallback_core_issue_key(news: dict) -> dict:
    title = get_news_title(news)
    summary = get_news_summary_or_description(news)
    base = title or summary
    short_key = normalize_title(base)[:40] if base else "unknown"
    return {
        "issue_key": short_key,
        "issue_label": title[:60] if title else "이슈",
        "core_issue_key": short_key
    }


def build_issue_key_prompt(news_items):
    lines = []
    for idx, news in enumerate(news_items):
        lines.append(
            f"{idx}. index: {idx}\n"
            f"   제목: {get_news_title(news)}\n"
            f"   설명/요약: {get_news_summary_or_description(news)}\n"
            f"   언론사: {news.get('source', '')}\n"
            f"   발행일: {news.get('published_at', '')}\n"
            f"   URL키: {normalize_url(get_news_url(news))}"
        )

    return f"""
뉴스 기사별로 반복 이슈 제거용 키를 생성하세요.

반드시 JSON만 출력하세요.
마크다운, 설명문, 코드블록은 출력하지 마세요.

각 기사에 대해 다음 3개 값을 만드세요.

1. issue_key
- 형식: "주체|핵심사건|핵심세부"
- 비교적 자세한 이슈 키입니다.
- 예: "롯데쇼핑|1분기실적|영업이익증가"

2. core_issue_key
- 형식: "주체|핵심사건"
- 아주 거친 반복 제거용 키입니다.
- 같은 사건/발표/행사/논란이면 언론사, 제목 표현, 세부 수치가 달라도 반드시 같은 core_issue_key를 써야 합니다.
- 연도, 금액, 퍼센트, 인원수 같은 세부 수치는 절대 포함하지 마세요.
- 핵심사건은 최대한 간결하게 (5~8자 이내) 압축하세요.
- 같은 사건이라면 어떤 기사에서든 반드시 동일한 표현을 써야 합니다.

3. issue_label
- 사람이 읽기 쉬운 짧은 이슈명입니다.

출력 형식:
{{
  "items": [
    {{
      "index": 0,
      "issue_key": "주체|핵심사건|핵심세부",
      "core_issue_key": "주체|핵심사건",
      "issue_label": "짧은 이슈명"
    }}
  ]
}}

[뉴스 목록]
{chr(10).join(lines)}
"""


def extract_json_object(text):
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def chunk_list(items, chunk_size):
    if chunk_size <= 0:
        chunk_size = ISSUE_KEY_CHUNK_SIZE
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def enrich_news_with_issue_keys(news_items):
    if not news_items:
        return news_items

    target_indexes = []
    for idx, news in enumerate(news_items):
        if not isinstance(news, dict):
            continue
        if news.get("issue_key") and news.get("core_issue_key"):
            news["normalized_issue_key"] = normalize_issue_key(news.get("issue_key"))
            news["normalized_core_issue_key"] = normalize_core_issue_key(news.get("core_issue_key"))
            continue
        target_indexes.append(idx)

    if not target_indexes:
        return news_items

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 fallback issue_key를 사용합니다.")
        for idx in target_indexes:
            fb = fallback_core_issue_key(news_items[idx])
            news_items[idx].update({
                "issue_key": fb["issue_key"],
                "core_issue_key": fb["core_issue_key"],
                "issue_label": fb["issue_label"],
                "normalized_issue_key": normalize_issue_key(fb["issue_key"]),
                "normalized_core_issue_key": normalize_core_issue_key(fb["core_issue_key"]),
            })
        return news_items

    try:
        for _, chunk_indexes in chunk_list(target_indexes, ISSUE_KEY_CHUNK_SIZE):
            chunk_news = [news_items[idx] for idx in chunk_indexes]
            prompt = build_issue_key_prompt(chunk_news)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "뉴스 기사에서 반복 이슈 제거용 issue_key와 core_issue_key를 생성하는 분류기입니다. "
                            "같은 사건/발표/행사/논란이면 URL과 제목이 달라도 반드시 같은 core_issue_key를 부여하세요. "
                            "core_issue_key에는 연도·금액·수치를 절대 포함하지 마세요. "
                            "핵심사건은 5~8자 이내로 압축하고, 동일 사건은 항상 동일한 표현을 쓰세요. "
                            "반드시 JSON만 출력하세요."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=3500
            )

            content = response.choices[0].message.content.strip()
            parsed = extract_json_object(content)
            items = parsed.get("items", [])

            by_local_index = {}
            for item in items:
                try:
                    local_idx = int(item.get("index"))
                except Exception:
                    continue
                by_local_index[local_idx] = item

            for local_idx, original_idx in enumerate(chunk_indexes):
                generated = by_local_index.get(local_idx) or fallback_core_issue_key(news_items[original_idx])

                issue_key = str(generated.get("issue_key") or "").strip()
                core_issue_key = str(generated.get("core_issue_key") or "").strip()
                issue_label = str(generated.get("issue_label") or "").strip()

                if not issue_key or not core_issue_key:
                    fb = fallback_core_issue_key(news_items[original_idx])
                    issue_key = issue_key or fb["issue_key"]
                    core_issue_key = core_issue_key or fb["core_issue_key"]
                    issue_label = issue_label or fb["issue_label"]

                news_items[original_idx].update({
                    "issue_key": issue_key,
                    "core_issue_key": core_issue_key,
                    "issue_label": issue_label,
                    "normalized_issue_key": normalize_issue_key(issue_key),
                    "normalized_core_issue_key": normalize_core_issue_key(core_issue_key),
                })

    except Exception as e:
        logger.error(f"❌ issue_key 생성 실패, fallback 사용: {e}")
        for idx in target_indexes:
            if news_items[idx].get("issue_key") and news_items[idx].get("core_issue_key"):
                continue
            fb = fallback_core_issue_key(news_items[idx])
            news_items[idx].update({
                "issue_key": fb["issue_key"],
                "core_issue_key": fb["core_issue_key"],
                "issue_label": fb["issue_label"],
                "normalized_issue_key": normalize_issue_key(fb["issue_key"]),
                "normalized_core_issue_key": normalize_core_issue_key(fb["core_issue_key"]),
            })

    return news_items


# ====================================
# [v2] LLM 의미 기반 중복 판단
# ====================================
def build_llm_duplicate_check_prompt(candidates, past_issues):
    """
    후보 뉴스 배치와 과거 이슈 목록을 받아
    각 후보가 과거 이슈와 같은 이슈인지 판단하는 프롬프트를 생성한다.

    candidates: [{"index": int, "title": str, "core_issue_key": str}, ...]
    past_issues: [{"title": str, "core_issue_key": str}, ...]
    """
    past_lines = "\n".join([
        f"  [{i}] core_key={p.get('core_issue_key', '')} | 제목={p.get('title', '')}"
        for i, p in enumerate(past_issues)
    ])

    candidate_lines = "\n".join([
        f"  [index={c['index']}] core_key={c.get('core_issue_key', '')} | 제목={c.get('title', '')}"
        for c in candidates
    ])

    return f"""
아래 [오늘 후보 기사] 각각이 [과거 발송 이슈] 중 하나와 같은 이슈인지 판단하세요.

판단 기준:
- 같은 사건 / 발표 / 행사 / 논란 / 이슈이면 중복입니다.
- 제목 표현, 언론사, 날짜가 달라도 본질적으로 같은 이슈이면 중복입니다.
- 같은 회사/주체라도 서로 다른 주제면 중복이 아닙니다.

반드시 JSON만 출력하세요. 마크다운, 설명문, 코드블록 없이.

출력 형식:
{{
  "results": [
    {{
      "index": <후보 index>,
      "is_duplicate": true 또는 false,
      "matched_past_key": "매칭된 과거 이슈의 core_issue_key, 없으면 빈 문자열"
    }}
  ]
}}

[과거 발송 이슈]
{past_lines}

[오늘 후보 기사]
{candidate_lines}
"""


def batch_llm_duplicate_check(candidates, past_issues, llm_client=None):
    """
    후보 뉴스 배치를 LLM에게 보내 과거 이슈와의 중복 여부를 판단받는다.

    candidates: [{"index": int, "title": str, "core_issue_key": str}, ...]
    past_issues: 과거 이슈 list (title, core_issue_key 포함)

    Returns:
        dict[int, dict]  →  {후보 index: {"is_duplicate": bool, "matched_past_key": str}}
    """
    result = {}

    if not candidates or not past_issues:
        return result

    active_client = llm_client or client
    if active_client is None:
        logger.warning("⚠️ OpenAI 클라이언트 없음 - LLM 중복 판단 건너뜀")
        return result

    # 과거 이슈는 최신순으로 제한
    limited_past = past_issues[-LLM_PAST_ISSUE_LIMIT:]

    try:
        prompt = build_llm_duplicate_check_prompt(candidates, limited_past)

        response = active_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 중복 이슈 판단기입니다. "
                        "오늘 후보 기사가 과거에 이미 발송된 이슈와 같은지 판단하세요. "
                        "같은 사건·발표·행사·논란이면 표현이 달라도 중복입니다. "
                        "반드시 JSON만 출력하세요."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1500
        )

        content = response.choices[0].message.content.strip()
        parsed = extract_json_object(content)

        for item in parsed.get("results", []):
            try:
                idx = int(item.get("index"))
                result[idx] = {
                    "is_duplicate": bool(item.get("is_duplicate", False)),
                    "matched_past_key": str(item.get("matched_past_key") or "").strip()
                }
            except Exception:
                continue

    except Exception as e:
        logger.error(f"❌ LLM 중복 판단 실패: {e}")

    return result


# ====================================
# 히스토리 저장
# ====================================
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

    normalized_url = normalize_url(url)
    normalized_title = normalize_title(title)

    issue_key = str(news.get("issue_key") or "").strip()
    core_issue_key = str(news.get("core_issue_key") or "").strip()
    issue_label = str(news.get("issue_label") or "").strip()

    if not issue_key or not core_issue_key:
        fb = fallback_core_issue_key(news)
        issue_key = issue_key or fb["issue_key"]
        core_issue_key = core_issue_key or fb["core_issue_key"]
        issue_label = issue_label or fb["issue_label"]

    normalized_issue_key = normalize_issue_key(issue_key)
    normalized_core_issue_key = normalize_core_issue_key(core_issue_key)

    issue_id = make_issue_id(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        title=title,
        summary=summary,
        url=url,
        core_issue_key=core_issue_key
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
        "normalized_title": normalized_title,
        "issue_key": issue_key,
        "issue_label": issue_label,
        "core_issue_key": core_issue_key,
        "normalized_issue_key": normalized_issue_key,
        "normalized_core_issue_key": normalized_core_issue_key,
        "published_at": published_at,
        "importance_score": importance_score
    }


def get_issue_normalized_url(issue: dict) -> str:
    if not isinstance(issue, dict):
        return ""
    return (
        str(issue.get("normalized_url") or "").strip()
        or normalize_url(issue.get("url") or "")
    )


def get_issue_normalized_title(issue: dict) -> str:
    if not isinstance(issue, dict):
        return ""
    return (
        str(issue.get("normalized_title") or "").strip()
        or normalize_title(issue.get("title") or "")
    )


def get_issue_normalized_core_key(issue: dict) -> str:
    if not isinstance(issue, dict):
        return ""
    return (
        str(issue.get("normalized_core_issue_key") or "").strip()
        or normalize_core_issue_key(issue.get("core_issue_key") or "")
        or normalize_core_issue_key(issue.get("issue_key") or "")
        or normalize_core_issue_key(issue.get("title") or "")
    )


def append_sent_issues(
    briefing_name, subject_prefix, receiver_env,
    section_results, file_path=HISTORY_FILE_PATH, keep_days=3
):
    """
    메일에 실제 발송된 section_results의 summaries를 이슈 히스토리에 누적 저장한다.
    """
    history = load_issue_history(file_path)
    history, pruned_count = prune_old_issues(history, days=keep_days)

    existing_issue_ids = set()
    existing_scope_url_keys = set()
    existing_scope_title_keys = set()
    existing_scope_core_keys = []

    for item in history.get("issues", []):
        issue_id = item.get("issue_id")
        if issue_id:
            existing_issue_ids.add(str(issue_id))

        scope = "|".join([
            str(item.get("briefing_name") or ""),
            str(item.get("receiver_env") or ""),
            str(item.get("section_name") or "")
        ])
        item_url = get_issue_normalized_url(item)
        item_title = get_issue_normalized_title(item)
        item_core_key = get_issue_normalized_core_key(item)

        if item_url:
            existing_scope_url_keys.add(f"{scope}|{item_url}")
        if item_title:
            existing_scope_title_keys.add(f"{scope}|{item_title}")
        if item_core_key:
            existing_scope_core_keys.append({"scope": scope, "core_key": item_core_key, "issue": item})

    for section_result in section_results or []:
        summaries = section_result.get("summaries", []) or []
        enrich_news_with_issue_keys(summaries)

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
                news=news
            )

            scope = "|".join([
                str(briefing_name or ""),
                str(receiver_env or ""),
                str(section_name or "")
            ])
            url_key = f"{scope}|{record.get('normalized_url', '')}"
            title_key = f"{scope}|{record.get('normalized_title', '')}"
            record_core_key = record.get("normalized_core_issue_key", "")

            if record["issue_id"] in existing_issue_ids:
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_url") and url_key in existing_scope_url_keys:
                skipped_duplicate_count += 1
                continue
            if record.get("normalized_title") and title_key in existing_scope_title_keys:
                skipped_duplicate_count += 1
                continue

            duplicated_by_core = False
            if record_core_key:
                for existing in existing_scope_core_keys:
                    if existing["scope"] != scope:
                        continue
                    if is_same_core_issue(record_core_key, existing["core_key"]):
                        duplicated_by_core = True
                        break

            if duplicated_by_core:
                skipped_duplicate_count += 1
                continue

            existing_issue_ids.add(record["issue_id"])
            if record.get("normalized_url"):
                existing_scope_url_keys.add(url_key)
            if record.get("normalized_title"):
                existing_scope_title_keys.add(title_key)
            if record_core_key:
                existing_scope_core_keys.append({
                    "scope": scope, "core_key": record_core_key, "issue": record
                })

            new_records.append(record)

    if new_records:
        history["issues"].extend(new_records)

    history["last_updated_at"] = get_now_kst().isoformat()
    save_issue_history(history, file_path)

    return {
        "success": True,
        "saved_count": len(new_records),
        "skipped_duplicate_count": skipped_duplicate_count,
        "pruned_count": pruned_count,
        "total_count": len(history.get("issues", [])),
        "file_path": file_path
    }


# ====================================
# 최근 이슈 조회
# ====================================
def get_recent_issues_for_section(
    briefing_name, receiver_env, section_name,
    days=3, file_path=HISTORY_FILE_PATH
):
    """
    같은 환경설정 + 같은 브리핑 + 같은 섹션의 최근 N일 이슈만 가져온다.
    """
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
    past_titles = []
    past_core_keys = []

    for issue in past_issues or []:
        issue_url = get_issue_normalized_url(issue)
        issue_norm_title = get_issue_normalized_title(issue)
        issue_core_key = get_issue_normalized_core_key(issue)

        if issue_url and issue_url not in past_by_url:
            past_by_url[issue_url] = issue
        if issue_norm_title:
            past_titles.append({
                "normalized_title": issue_norm_title,
                "title": issue.get("title", ""),
                "issue": issue
            })
        if issue_core_key:
            past_core_keys.append({"core_key": issue_core_key, "issue": issue})

    return {
        "past_by_url": past_by_url,
        "past_titles": past_titles,
        "past_core_keys": past_core_keys
    }


def find_matching_past_core_issue(news_core_key, past_core_keys):
    if not news_core_key:
        return None
    for item in past_core_keys or []:
        if is_same_core_issue(news_core_key, item.get("core_key", "")):
            return item.get("issue")
    return None


def find_matching_seen_core_issue(news_core_key, seen_core_keys):
    if not news_core_key:
        return None
    for item in seen_core_keys or []:
        if is_same_core_issue(news_core_key, item.get("core_key", "")):
            return item
    return None


def filter_seen_issues_with_llm(
    briefing_name, receiver_env, section_name,
    candidate_news, days=3, file_path=HISTORY_FILE_PATH
):
    """
    최근 N일간 이미 메일에 보낸 이슈와 오늘 후보 뉴스를 비교해
    반복 이슈로 판단된 후보를 제외한다.

    처리 순서:
    1. 같은 환경설정/브리핑/섹션의 최근 N일 히스토리 조회
    2. 오늘 후보 뉴스에 issue_key/core_issue_key 생성
    3. URL / 제목 / core_issue_key 규칙 기반 중복 제거
    4. [v2] 규칙 기반을 통과한 후보를 LLM 의미 비교로 최종 확인
    5. 오늘 후보 내부에서도 같은 core_issue_key 반복 제거
    """
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
        }

    past_issues = get_recent_issues_for_section(
        briefing_name=briefing_name,
        receiver_env=receiver_env,
        section_name=section_name,
        days=days,
        file_path=file_path
    )

    # 후보에 issue_key / core_issue_key 생성
    candidate_news = enrich_news_with_issue_keys(candidate_news)

    past_indexes = build_past_issue_indexes(past_issues)
    past_by_url = past_indexes["past_by_url"]
    past_titles = past_indexes["past_titles"]
    past_core_keys = past_indexes["past_core_keys"]

    excluded_items = []
    url_excluded_count = 0
    title_excluded_count = 0
    core_key_excluded_count = 0
    llm_excluded_count = 0
    internal_duplicate_count = 0

    # ----------------------------------------
    # 1단계: 규칙 기반 필터
    # ----------------------------------------
    rule_passed = []  # {"original_idx": int, "news": dict}

    for idx, news in enumerate(candidate_news or []):
        news_url = normalize_url(get_news_url(news))
        news_title = get_news_title(news)
        news_norm_title = normalize_title(news_title)
        news_core_key = (
            str(news.get("normalized_core_issue_key") or "").strip()
            or normalize_core_issue_key(news.get("core_issue_key") or "")
        )

        # URL 완전 일치
        if news_url and news_url in past_by_url:
            matched = past_by_url[news_url]
            excluded_items.append({
                "index": idx, "title": news_title,
                "matched_past_issue": matched.get("title", ""),
                "reason": "같은 URL의 기사로 이미 발송됨",
                "method": "url"
            })
            url_excluded_count += 1
            continue

        # 제목 완전 일치
        exact_matched = None
        if news_norm_title:
            for pt in past_titles:
                if news_norm_title == pt["normalized_title"]:
                    exact_matched = pt["issue"]
                    break
        if exact_matched:
            excluded_items.append({
                "index": idx, "title": news_title,
                "matched_past_issue": exact_matched.get("title", ""),
                "reason": "정규화 제목이 이미 발송된 기사와 동일함",
                "method": "title_exact"
            })
            title_excluded_count += 1
            continue

        # 제목 유사도
        similar_matched = None
        similar_score = 0.0
        if news_norm_title:
            for pt in past_titles:
                score = SequenceMatcher(None, news_norm_title, pt["normalized_title"]).ratio()
                if score > similar_score:
                    similar_score = score
                    similar_matched = pt["issue"]
        if similar_matched and similar_score >= TITLE_SIMILARITY_THRESHOLD:
            excluded_items.append({
                "index": idx, "title": news_title,
                "matched_past_issue": similar_matched.get("title", ""),
                "reason": f"제목 유사도 {similar_score:.2f}로 이미 발송된 이슈와 유사함",
                "method": "title_similarity"
            })
            title_excluded_count += 1
            continue

        # core_issue_key 규칙 비교
        matched_core = find_matching_past_core_issue(news_core_key, past_core_keys)
        if matched_core:
            excluded_items.append({
                "index": idx, "title": news_title,
                "matched_past_issue": matched_core.get("title", ""),
                "reason": "core_issue_key 기준으로 이미 발송된 이슈와 동일 또는 유사함",
                "method": "core_issue_key"
            })
            core_key_excluded_count += 1
            continue

        rule_passed.append({"original_idx": idx, "news": news})

    # ----------------------------------------
    # 2단계: LLM 의미 비교 (규칙 기반 통과 후보 대상)
    # ----------------------------------------
    llm_excluded_idxs = set()

    if rule_passed and past_issues and client is not None:
        for batch_start in range(0, len(rule_passed), LLM_DUPLICATE_BATCH_SIZE):
            batch = rule_passed[batch_start:batch_start + LLM_DUPLICATE_BATCH_SIZE]

            candidates_for_llm = [
                {
                    "index": item["original_idx"],
                    "title": get_news_title(item["news"]),
                    "core_issue_key": item["news"].get("core_issue_key", "")
                }
                for item in batch
            ]

            llm_results = batch_llm_duplicate_check(
                candidates=candidates_for_llm,
                past_issues=past_issues
            )

            for item in batch:
                orig_idx = item["original_idx"]
                llm_result = llm_results.get(orig_idx)

                if llm_result and llm_result.get("is_duplicate"):
                    matched_key = llm_result.get("matched_past_key", "")
                    excluded_items.append({
                        "index": orig_idx,
                        "title": get_news_title(item["news"]),
                        "matched_past_issue": matched_key,
                        "reason": f"LLM 의미 비교로 이미 발송된 이슈와 동일 판정 (과거 키: {matched_key})",
                        "method": "llm_semantic"
                    })
                    llm_excluded_idxs.add(orig_idx)
                    llm_excluded_count += 1

    # LLM까지 통과한 후보
    after_llm_passed = [
        item for item in rule_passed
        if item["original_idx"] not in llm_excluded_idxs
    ]

    # ----------------------------------------
    # 3단계: 오늘 후보 내부 중복 제거
    # ----------------------------------------
    seen_candidate_core_keys = []
    filtered_news = []

    for item in after_llm_passed:
        idx = item["original_idx"]
        news = item["news"]
        news_title = get_news_title(news)
        news_core_key = (
            str(news.get("normalized_core_issue_key") or "").strip()
            or normalize_core_issue_key(news.get("core_issue_key") or "")
        )

        matched_seen = find_matching_seen_core_issue(news_core_key, seen_candidate_core_keys)
        if matched_seen:
            excluded_items.append({
                "index": idx,
                "title": news_title,
                "matched_past_issue": matched_seen.get("title", ""),
                "reason": "오늘 후보 내부에서 이미 유지한 이슈와 동일 또는 유사함",
                "method": "internal_core_issue_key"
            })
            internal_duplicate_count += 1
            continue

        filtered_news.append(news)

        if news_core_key:
            seen_candidate_core_keys.append({
                "core_key": news_core_key,
                "title": news_title
            })

    return {
        "success": True,
        "message": "반복 이슈 필터 완료",
        "filtered_news": filtered_news,
        "excluded_count": len(excluded_items),
        "excluded_items": excluded_items,
        "past_issue_count": len(past_issues),
        "prefilter_excluded_count": url_excluded_count + title_excluded_count,
        "llm_excluded_count": llm_excluded_count,
        "core_key_excluded_count": core_key_excluded_count,
        "internal_duplicate_count": internal_duplicate_count,
        "url_excluded_count": url_excluded_count,
        "title_excluded_count": title_excluded_count,
    }


# ====================================
# 단독 테스트용
# ====================================
if __name__ == "__main__":
    history = load_issue_history()
    print(f"현재 저장된 이슈 수: {len(history.get('issues', []))}")