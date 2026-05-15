"""
OpenAI API를 사용한 뉴스 선별 모듈

역할:
- 네이버 뉴스 API로 수집된 전체 뉴스 후보 중
- 주제 적합성, 중요도, 중복 여부를 기준으로
- 요약할 뉴스 후보를 먼저 선택한다.
- 선택된 뉴스에 중요도 점수를 함께 부여한다.
- 선택된 뉴스 중 같은 사건을 다룬 중복 기사들은 LLM으로 그룹화하여 1개만 남긴다.
"""

import os
import json
import logging
import re
import hashlib
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from dotenv import load_dotenv
from openai_usage import (
    record_openai_usage,
    openai_token_limit_kwargs,
    openai_temperature_kwargs,
    openai_reasoning_effort_kwargs,
    openai_json_response_format_kwargs,
    is_gpt5_model,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# 뉴스 선별 모델
# - MODEL: 일반 1차 선별/사건 그룹화용
# - SELECTOR_MODEL: 그룹 후보 100개 중 최종 10개를 고르는 그룹 단위 선별용
#
# 중요:
# 기본값을 GPT-5 nano로 둔다.
# 더 높은 품질이 필요하면 env에서 OPENAI_SELECTOR_MODEL만 gpt-5-mini 또는 gpt-5.4-nano 등으로 올리면 된다.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
SELECTOR_MODEL = os.getenv("OPENAI_SELECTOR_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))

LAST_SELECTION_STATS = {
    "selection_tokens": 0,
    "event_group_tokens": 0,
    "final_duplicate_excluded_count": 0,
    "selected_before_final_dedup_count": 0,
    "selected_after_final_dedup_count": 0,
}


def _env_int(name: str, default: int, min_value: int = 1, max_value: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)

    value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT = _env_int("SELECTOR_CANDIDATE_GROUP_LIMIT", 45, 10, 100)
SELECTOR_MAX_COMPLETION_TOKENS = _env_int("SELECTOR_MAX_COMPLETION_TOKENS", 700, 256, 2048)
SELECTOR_DETAILED_CANDIDATE_COUNT = _env_int("SELECTOR_DETAILED_CANDIDATE_COUNT", 15, 0, 50)
GROUP_CANDIDATE_TITLE_CHARS = _env_int("SELECTOR_GROUP_TITLE_CHARS", 65, 35, 120)
GROUP_CANDIDATE_DESCRIPTION_CHARS = _env_int("SELECTOR_GROUP_DESCRIPTION_CHARS", 50, 0, 160)
GROUP_CANDIDATE_SOURCES_CHARS = _env_int("SELECTOR_GROUP_SOURCES_CHARS", 30, 20, 100)
GROUP_CANDIDATE_KEYWORDS_CHARS = _env_int("SELECTOR_GROUP_KEYWORDS_CHARS", 30, 10, 100)


def reset_selection_stats():
    global LAST_SELECTION_STATS
    LAST_SELECTION_STATS = {
        "selection_tokens": 0,
        "event_group_tokens": 0,
        "final_duplicate_excluded_count": 0,
        "selected_before_final_dedup_count": 0,
        "selected_after_final_dedup_count": 0,
    }


def add_selection_tokens(key: str, value: int):
    try:
        token_count = int(value or 0)
    except Exception:
        token_count = 0
    LAST_SELECTION_STATS[key] = int(LAST_SELECTION_STATS.get(key, 0)) + token_count


def get_last_selection_stats() -> Dict[str, int]:
    return dict(LAST_SELECTION_STATS)


def _safe_text(value: Any) -> str:
    """
    None 방지용 문자열 변환
    """
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 3, min_value: int = 1, max_value: int = 5) -> int:
    """
    중요도 점수 안전 변환
    """
    try:
        number = int(value)
    except Exception:
        number = default

    if number < min_value:
        return min_value

    if number > max_value:
        return max_value

    return number


def _clip_text(value: Any, limit: int = 180) -> str:
    """
    LLM 입력 토큰 절감을 위해 긴 설명을 적정 길이로 자른다.
    """
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."




def _openai_token_limit_kwargs(model: str, limit: int) -> Dict[str, int]:
    """
    모델별 출력 토큰 제한 파라미터를 반환한다.

    GPT-5 계열은 Chat Completions에서 max_tokens를 지원하지 않고
    max_completion_tokens를 요구한다.
    기존 GPT-4o 계열은 max_tokens를 그대로 사용한다.
    """
    model_name = _safe_text(model).lower()

    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": int(limit)}

    return {"max_tokens": int(limit)}


def _normalize_url(url: Any) -> str:
    """
    URL 완전 중복 비교용 정규화

    처리:
    - scheme 제거
    - www. 제거
    - query string 제외
    - fragment 제외
    - 마지막 slash 제거
    """
    url = _safe_text(url)

    if not url:
        return ""

    try:
        parsed = urlparse(url)

        domain = parsed.netloc.lower().strip()
        path = parsed.path.strip()

        if domain.startswith("www."):
            domain = domain[4:]

        normalized = f"{domain}{path}".rstrip("/")
        return normalized

    except Exception:
        return url.lower().strip().rstrip("/")


def _deduplicate_by_url(news_list: List[Dict], log_prefix: str = "후보") -> List[Dict]:
    """
    URL이 완전히 같은 기사만 제거한다.

    같은 사건 여부는 LLM이 따로 판단하고,
    여기서는 완전히 같은 링크만 1차 제거한다.
    """
    deduped_news = []
    seen_urls = set()

    for news in news_list:
        url_candidates = [
            _normalize_url(news.get("url")),
            _normalize_url(news.get("link")),
            _normalize_url(news.get("originallink")),
        ]

        url_candidates = [url for url in url_candidates if url]

        duplicated = False

        for url in url_candidates:
            if url in seen_urls:
                duplicated = True
                break

        if duplicated:
            continue

        for url in url_candidates:
            seen_urls.add(url)

        deduped_news.append(news)

    logger.info(
        f"🧹 {log_prefix} URL 중복 제거 완료: {len(news_list)}개 → {len(deduped_news)}개"
    )

    return deduped_news


def _build_candidate_text(news_list: List[Dict]) -> str:
    """
    OpenAI에 전달할 뉴스 후보 목록 텍스트 생성
    """
    lines = []

    for idx, news in enumerate(news_list, 1):
        title = _clip_text(news.get("title"), 120)
        description = _clip_text(news.get("description"), 180)
        keyword = _clip_text(news.get("keyword"), 80)
        source = _clip_text(news.get("source"), 40)
        date = _safe_text(news.get("date"))
        published_at = _safe_text(news.get("published_at"))

        lines.append(
            f"""
[{idx}]
키워드: {keyword}
언론사: {source}
제목: {title}
설명: {description}
날짜: {published_at or date}
""".strip()
        )

    return "\n\n".join(lines)


def _build_event_dedup_text(news_list: List[Dict]) -> str:
    """
    LLM 사건 중복 제거용 뉴스 목록 텍스트 생성
    """
    lines = []

    for idx, news in enumerate(news_list, 1):
        source = _clip_text(news.get("source"), 40)
        importance_score = _safe_text(news.get("importance_score"))
        title = _clip_text(news.get("title"), 120)
        description = _clip_text(
            news.get("description")
            or news.get("summary")
            or news.get("content"),
            160
        )
        published_at = _safe_text(news.get("published_at"))

        lines.append(
            f"""
[{idx}]
언론사: {source}
중요도: {importance_score}
제목: {title}
설명: {description}
발행일: {published_at}
""".strip()
        )

    return "\n\n".join(lines)


def _extract_json(content: str) -> Dict:
    """
    OpenAI 응답에서 JSON 파싱.
    원칙적으로 JSON만 오게 하지만, 혹시 코드블록이 섞이는 경우를 대비한다.
    """
    content = _safe_text(content)

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)


def _ensure_json_object(result: Any) -> Dict[str, Any]:
    """
    모델이 JSON 배열이나 문자열을 반환해도 호출부에서 AttributeError가 나지 않게 한다.
    """
    if isinstance(result, dict):
        return result
    return {}


def _ensure_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _prepare_selected_news(
    news: Dict,
    importance_score: Any = 3
) -> Dict:
    """
    요약 단계로 넘기기 전에 필요한 필드 보강
    """
    news["importance_score"] = _safe_int(importance_score)
    news["content"] = news.get("description", "")

    return news


def _fallback_select(news_list: List[Dict], limit: int) -> List[Dict]:
    """
    OpenAI 선별 실패 시 안전 fallback.
    URL 중복 제거 후 앞에서 limit개를 사용하되,
    요약 단계에 필요한 필드를 기본값으로 넣는다.
    """
    deduped_news = _deduplicate_by_url(news_list, log_prefix="fallback")
    fallback_news = deduped_news[:limit]

    for news in fallback_news:
        _prepare_selected_news(
            news=news,
            importance_score=news.get("importance_score", 3)
        )

    return fallback_news


def _deduplicate_by_llm_event_group(
    news_list: List[Dict],
    limit: int
) -> List[Dict]:
    """
    LLM을 사용해 모든 뉴스를 사건 단위로 그룹화하고,
    각 사건 그룹에서 대표 기사 1개만 남긴다.

    핵심:
    - 중복 그룹만 선택적으로 찾게 하지 않는다.
    - 모든 기사에 반드시 event_group을 배정하게 한다.
    - 같은 사건이면 같은 그룹에 넣는다.
    - 각 그룹에서 representative_index 1개만 최종 유지한다.
    """
    if not news_list:
        return []

    if len(news_list) <= 1:
        return news_list[:limit]

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 LLM 사건 그룹화를 건너뜁니다.")
        return news_list[:limit]

    news_text = _build_event_dedup_text(news_list)

    prompt = f"""
아래 뉴스 목록을 '사건 단위'로 그룹화하세요.

[목표]
뉴스 제목이나 언론사가 달라도 실제로 같은 사건을 다룬 기사라면 같은 event_group으로 묶고,
각 event_group에서 대표 기사 1개만 남기세요.

[반드시 지켜야 할 기준]
1. 모든 뉴스 index는 반드시 정확히 하나의 event_group에 포함되어야 합니다.
2. 같은 회사, 병원, 기관, 학회, 정부기관, 지자체가 같은 발표·공개·론칭·계약·제휴·실적·공시·행사·정책·서비스를 다룬 기사는 같은 사건입니다.
3. 같은 솔루션, 같은 플랫폼, 같은 서비스, 같은 기술, 같은 시스템 공개를 다룬 기사는 제목이 달라도 같은 사건입니다.
4. 여러 언론사가 같은 보도자료를 받아쓴 기사는 같은 사건입니다.
5. 제목 표현, 언론사, 문장 구조, 중요도가 달라도 실제 사건이 같으면 같은 event_group입니다.
6. 단순히 같은 회사나 같은 질병 분야가 언급됐다는 이유만으로는 같은 사건이 아닙니다.
7. 서로 다른 발표, 다른 서비스, 다른 날짜의 별도 사건이면 다른 event_group입니다.
8. 각 event_group의 representative_index는 가장 정보가 구체적이고 요약하기 좋은 기사 1개로 고르세요.
9. 최종적으로 같은 event_group에서는 representative_index만 남기고 나머지는 제거됩니다.

[예시]
- "의료영상 특화 LLM 플랫폼 공개"
- "의료영상 특화 LLM 플랫폼 론칭"
- "LLM 의료영상 플랫폼 공개"
위처럼 같은 회사의 같은 플랫폼 공개를 다룬 기사들은 같은 event_group입니다.

- "온열질환 예측정보 서비스 제공"
- "폭염 대비 온열질환 발생 예측정보 서비스"
- "온열질환 위험 사전 예측"
위처럼 같은 예측정보 서비스 발표를 다룬 기사들은 같은 event_group입니다.

[출력 규칙]
반드시 JSON만 출력하세요.
설명 문장, 마크다운, 코드블록은 쓰지 마세요.

[출력 형식]
{{
  "event_groups": [
    {{
      "event_group": "제이엘케이 의료영상 특화 LLM 플랫폼 공개",
      "representative_index": 2,
      "indexes": [2, 4, 6]
    }},
    {{
      "event_group": "온열질환 예측정보 서비스 제공",
      "representative_index": 8,
      "indexes": [8, 9, 10]
    }},
    {{
      "event_group": "다른 독립 뉴스",
      "representative_index": 1,
      "indexes": [1]
    }}
  ]
}}

[뉴스 목록]
{news_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 뉴스 편집 데스크입니다. "
                        "모든 뉴스를 사건 단위로 그룹화하고, "
                        "각 사건 그룹에서 대표 기사 1개만 남깁니다. "
                        "모든 index는 반드시 하나의 event_group에 포함되어야 합니다. "
                        "반드시 JSON만 출력합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.0),
            **openai_reasoning_effort_kwargs(MODEL),
            **openai_json_response_format_kwargs(),
            **_openai_token_limit_kwargs(MODEL, 4096 if is_gpt5_model(MODEL) else 1200)
        )

        content = response.choices[0].message.content.strip()
        usage_info = record_openai_usage(
            logger,
            "LLM 사건 그룹화",
            MODEL,
            response.usage,
        )
        event_group_tokens = usage_info["total_tokens"]
        add_selection_tokens("event_group_tokens", event_group_tokens)

        logger.debug(f"🧩 LLM 사건 그룹화 응답 수신: 토큰 {event_group_tokens}")

        try:
            result = _ensure_json_object(_extract_json(content))
        except json.JSONDecodeError:
            logger.error("❌ LLM 사건 그룹화 JSON 파싱 실패")
            logger.error("응답 미리보기: %s", content[:300])
            return news_list[:limit]

        event_groups = _ensure_json_list(result.get("event_groups"))

        if not event_groups:
            logger.warning("⚠️ LLM 사건 그룹화 결과가 비어 있어 원본 선택 결과를 유지합니다.")
            return news_list[:limit]

        representative_indexes = []
        covered_indexes = set()

        for group in event_groups:
            if not isinstance(group, dict):
                continue
            indexes = group.get("indexes", [])
            representative_index = group.get("representative_index")

            normalized_indexes = []

            for value in indexes:
                try:
                    idx = int(value)
                except Exception:
                    continue

                if 1 <= idx <= len(news_list):
                    normalized_indexes.append(idx)
                    covered_indexes.add(idx)

            try:
                rep_idx = int(representative_index)
            except Exception:
                rep_idx = None

            # representative_index가 그룹 내부에 없거나 이상하면 그룹 첫 번째 기사로 대체
            if rep_idx not in normalized_indexes:
                rep_idx = normalized_indexes[0] if normalized_indexes else None

            if rep_idx and 1 <= rep_idx <= len(news_list):
                representative_indexes.append(rep_idx)

        # 혹시 LLM이 누락한 index가 있으면 독립 뉴스로 보고 유지
        all_indexes = set(range(1, len(news_list) + 1))
        missing_indexes = sorted(all_indexes - covered_indexes)

        if missing_indexes:
            logger.warning(f"⚠️ LLM 사건 그룹화에서 누락된 index 유지: {missing_indexes}")
            representative_indexes.extend(missing_indexes)

        # 중복 제거 + 원래 뉴스 순서 유지
        representative_index_set = set(representative_indexes)

        deduped_news = []

        for idx, news in enumerate(news_list, 1):
            if idx in representative_index_set:
                deduped_news.append(news)

        deduped_news = deduped_news[:limit]

        logger.info(
            f"🧠 LLM 사건 그룹화 중복 제거 완료: "
            f"{len(news_list)}개 → {len(deduped_news)}개"
        )

        return deduped_news

    except Exception as e:
        logger.error(f"❌ LLM 사건 그룹화 중복 제거 실패: {e}")
        logger.warning("⚠️ 중복 제거 실패로 기존 선택 결과를 유지합니다.")
        return news_list[:limit]

def _supplement_after_dedup(
    selected_news: List[Dict],
    candidate_pool: List[Dict],
    used_indexes: set,
    limit: int
) -> List[Dict]:
    """
    사건 중복 제거 후 뉴스 수가 limit보다 적을 경우,
    후보군에서 아직 사용하지 않은 뉴스를 반복적으로 보충한다.

    핵심:
    - 한 번만 보충하지 않는다.
    - 보충 후 다시 사건 그룹화한다.
    - 그래도 부족하면 다음 후보 묶음을 또 보충한다.
    - limit에 도달하거나 후보가 소진될 때까지 반복한다.
    """
    if len(selected_news) >= limit:
        return selected_news[:limit]

    final_news = list(selected_news)
    batch_size = max(limit, 10)

    while len(final_news) < limit:
        supplement_batch = []

        for idx, news in enumerate(candidate_pool):
            if idx in used_indexes:
                continue

            used_indexes.add(idx)

            prepared_news = _prepare_selected_news(
                news=news,
                importance_score=news.get("importance_score", 3)
            )

            supplement_batch.append(prepared_news)

            if len(supplement_batch) >= batch_size:
                break

        if not supplement_batch:
            logger.warning("⚠️ 보충 가능한 후보가 더 이상 없습니다.")
            break

        before_count = len(final_news)

        combined_news = final_news + supplement_batch

        logger.info(
            f"➕ 부족분 반복 보충: 현재 {len(final_news)}개, "
            f"추가 후보 {len(supplement_batch)}개"
        )

        final_news = _deduplicate_by_llm_event_group(
            combined_news,
            limit=limit
        )

        after_count = len(final_news)

        logger.info(
            f"🔁 보충 후 사건 중복 제거 결과: "
            f"{before_count}개 → {after_count}개"
        )

        # 보충했는데도 개수가 전혀 늘지 않았고, 아직 후보가 남아있을 수 있으므로 계속 돈다.
        # 단, candidate_pool을 다 소진하면 위 supplement_batch가 비어서 종료된다.

    return final_news[:limit]


def select_important_news(
    news_list: List[Dict],
    topic_name: str,
    topic_description: str,
    limit: int = 10
) -> List[Dict]:
    """
    전체 뉴스 후보 중 OpenAI가 중요하고 주제에 맞는 뉴스만 선택

    Args:
        news_list: 네이버 API로 수집된 전체 뉴스 리스트
        topic_name: 섹션명 예: "롯데그룹 의료뉴스브리핑"
        topic_description: 어떤 뉴스를 뽑아야 하는지 설명
        limit: 최종 선택 개수

    Returns:
        선택된 뉴스 리스트.
        각 뉴스에는 아래 필드가 추가된다.
        - importance_score: 중요도 점수, 1~5
        - content: summarizer.py에서 사용할 요약 대상 본문
    """
    reset_selection_stats()

    if not news_list:
        logger.warning("선택할 뉴스 후보가 없습니다.")
        return []

    logger.info(
        "🧠 [%s] 기사 선별 시작: 후보 %s개 / 목표 %s개",
        topic_name,
        len(news_list),
        limit,
    )

    # 1차 URL 중복 제거
    candidate_pool = _deduplicate_by_url(
        news_list,
        log_prefix="OpenAI 전달 전 후보"
    )

    if not candidate_pool:
        logger.warning("URL 중복 제거 후 남은 후보 뉴스가 없습니다.")
        return []

    logger.info(f"🧺 [{topic_name}] 기사 선별 후보: URL중복 후 {len(candidate_pool)}개")

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 fallback 선별을 사용합니다.")
        return _fallback_select(candidate_pool, limit)

    # 중복 제거 후에도 최종 limit개를 확보하기 위해
    # 1차 선별에서는 limit보다 넉넉하게 뽑는다.
    # 비용 절감을 위해 1차 선별 후보를 과하게 넓히지 않는다.
    # 부족분은 아래 보충 로직에서 처리한다.
    candidate_limit = min(len(candidate_pool), max(limit * 2, 20))

    candidate_text = _build_candidate_text(candidate_pool)

    prompt = f"""
작업: "{topic_name}" 브리핑의 뉴스 후보를 고릅니다.

주제:
{topic_description}

선택 기준:
1. 주제와 직접 관련 있고 독자가 알아야 할 사건을 우선합니다.
2. 기업 전략, 실적, 투자, 제휴, 정책, 규제, 기술 도입, 산업 변화는 우선합니다.
3. 후보가 충분하면 서로 다른 사건으로 {candidate_limit}개까지 고릅니다.
4. 홍보성 기사, 단순 행사, 사진기사, 키워드만 맞는 기사는 제외합니다.
5. index는 후보 목록에 있는 번호만 사용하고 중복하지 마세요.

importance_score:
5=영향 큰 핵심 뉴스, 4=주요 변화, 3=일반 참고 뉴스, 2=낮은 영향, 1=거의 제외 대상.
모든 항목을 기계적으로 3점으로 주지 말고 차등화하세요.

출력은 JSON 객체 하나만:
{{"selected":[{{"index":1,"importance_score":5}}]}}

후보:
{candidate_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 편집 데스크입니다. 후보 index 중에서만 선택하고, "
                        "중복 사건을 피하며, 중요도 점수를 1~5로 차등 부여합니다. "
                        "응답은 JSON 객체 하나만 출력합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.1),
            **openai_reasoning_effort_kwargs(MODEL),
            **openai_json_response_format_kwargs(),
            **_openai_token_limit_kwargs(MODEL, 4096 if is_gpt5_model(MODEL) else 1800)
        )

        content = response.choices[0].message.content.strip()
        usage_info = record_openai_usage(
            logger,
            "뉴스 1차 선별",
            MODEL,
            response.usage,
        )
        tokens_used = usage_info["total_tokens"]
        add_selection_tokens("selection_tokens", tokens_used)

        logger.debug(f"🧾 뉴스 1차 선별 토큰 사용량: {tokens_used}")

        try:
            result = _ensure_json_object(_extract_json(content))
        except json.JSONDecodeError:
            logger.error("❌ OpenAI 1차 선별 응답 JSON 파싱 실패")
            logger.error("응답 미리보기: %s", content[:300])
            return _fallback_select(candidate_pool, limit)

        selected_items = _ensure_json_list(result.get("selected"))

        if not selected_items:
            logger.warning("⚠️ OpenAI가 선택한 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        selected_news = []
        used_indexes = set()

        invalid_item_count = 0
        for item in selected_items:
            if not isinstance(item, dict):
                invalid_item_count += 1
                continue

            try:
                selected_index = item.get("index")
                idx = int(selected_index) - 1
            except Exception:
                invalid_item_count += 1
                continue

            if idx < 0 or idx >= len(candidate_pool):
                invalid_item_count += 1
                continue

            if idx in used_indexes:
                invalid_item_count += 1
                continue

            news = candidate_pool[idx]

            news = _prepare_selected_news(
                news=news,
                importance_score=item.get("importance_score", 3)
            )

            selected_news.append(news)
            used_indexes.add(idx)

            if len(selected_news) >= candidate_limit:
                break

        if invalid_item_count:
            logger.debug("기사 선별 응답 무시 항목: %s개", invalid_item_count)

        if not selected_news:
            logger.warning("⚠️ 유효하게 선별된 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        logger.info(f"✅ OpenAI 1차 뉴스 선별 완료: {len(selected_news)}개 후보 선택")

        # 2차 중복 제거:
        # 선택된 후보를 LLM이 같은 사건 기준으로 그룹화하여 중복 제거한다.
        deduped_selected_news = _deduplicate_by_llm_event_group(
            selected_news,
            limit=limit
        )

        if not deduped_selected_news:
            logger.warning("⚠️ 사건 중복 제거 후 남은 뉴스가 없습니다. fallback을 사용합니다.")
            return _fallback_select(candidate_pool, limit)

        # 중복 제거 후 limit보다 부족하면 후보군에서 추가 보충
        if len(deduped_selected_news) < limit:
            deduped_selected_news = _supplement_after_dedup(
                selected_news=deduped_selected_news,
                candidate_pool=candidate_pool,
                used_indexes=used_indexes,
                limit=limit
            )

        final_news = deduped_selected_news[:limit]

        logger.info(f"✅ 최종 뉴스 선별 완료: 1차 {len(selected_news)}개 → 최종 {len(final_news)}개")

        return final_news

    except Exception as e:
        logger.error(f"❌ OpenAI 뉴스 선별 실패: {e}")
        logger.warning(f"⚠️ 실패 시 URL 중복 제거 후 후보 뉴스 앞에서 {limit}개 사용")
        return _fallback_select(news_list, limit)


# ====================================
# 최종 AI 선별 결과 중복 제거 유틸
# ====================================
_FINAL_DEDUP_STOPWORDS = {
    "기자", "뉴스", "단독", "종합", "속보", "오늘", "내일", "오전", "오후",
    "관련", "통해", "대해", "대한", "위해", "이번", "지난", "올해", "내년",
    "밝혔다", "전했다", "설명했다", "말했다", "따르면", "제공", "진행",
    "발표", "공개", "추진", "운영", "지원", "확대", "강화", "개최",
    "서비스", "사업", "기업", "업계", "시장", "정부", "기관", "서울",
}


def _normalize_final_dedup_text(value: Any) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    text = text.replace("美", "미국").replace("韓", "한국").replace("中", "중국")
    text = text.replace("日", "일본").replace("李", "이").replace("金", "김")
    text = re.sub(r"\[[^\]]*\]|【[^】]*】", " ", text)
    text = re.sub(r"[\u201c\u201d\u2018\u2019\"'`~!@#$%^&*()_+=\[\]{};:,.<>/?\\|《》〈〉·ㆍ-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_final_dedup_title(value: Any) -> str:
    text = _normalize_final_dedup_text(value)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text.strip()


def _extract_final_number_tokens(value: Any) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", _safe_text(value)))


def _extract_final_dedup_tokens(news: Dict[str, Any], max_tokens: int = 90) -> List[str]:
    title = _safe_text(news.get("title"))
    description = _safe_text(news.get("description") or news.get("summary") or news.get("content"))
    text = _normalize_final_dedup_text(f"{title} {description}")
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)

    tokens = []
    seen = set()
    for token in raw_tokens:
        token = _normalize_final_token(token)
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


def _final_token_overlap(tokens_a: List[str], tokens_b: List[str]):
    set_a = set(tokens_a or [])
    set_b = set(tokens_b or [])
    if not set_a or not set_b:
        return 0.0, 0
    common = set_a & set_b
    denominator = max(1, min(len(set_a), len(set_b)))
    return len(common) / denominator, len(common)


def _stable_final_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _make_final_simhash(tokens: List[str]) -> str:
    if not tokens:
        return ""
    vector = [0] * 64
    for token in tokens:
        value = _stable_final_token_hash(token)
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


def _final_simhash_distance(a: str, b: str):
    if not a or not b:
        return None
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None




def _normalize_final_token(token: str) -> str:
    """
    외부 형태소 분석기 없이 조사/어미 차이만 가볍게 줄인다.
    특정 주제 단어가 아니라 한국어 문장 공통 형태를 다루는 규칙이다.
    """
    token = _safe_text(token).lower().strip()
    if not token:
        return ""

    # 흔한 한자 약칭을 앞단에서 이미 바꿨지만, 토큰 단위에서도 한 번 더 방어한다.
    token = token.replace("美", "미국").replace("韓", "한국").replace("中", "중국")
    token = token.replace("日", "일본").replace("李", "이").replace("金", "김")

    # 한국어 조사/어미 일부 제거: 발언/발언에, 코스피/코스피는, 김용범/김용범의 등을 맞춘다.
    suffixes = [
        "으로부터", "로부터", "에서는", "에게서", "까지", "부터", "처럼", "보다",
        "으로", "라고", "하고", "에서", "에게", "에도", "에는", "만큼",
        "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로",
    ]
    for suffix in suffixes:
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break

    return token.strip()

def _extract_final_title_tokens(title: Any) -> List[str]:
    normalized = _normalize_final_dedup_text(title)
    tokens = []
    seen = set()
    for token in re.findall(r"[가-힣a-zA-Z0-9]{2,}", normalized):
        token = _normalize_final_token(token)
        if not token or token in _FINAL_DEDUP_STOPWORDS or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _extract_final_anchor_tokens(tokens: List[str]) -> List[str]:
    """
    최종 선별 후 중복 제거용 핵심 토큰.

    특정 브리핑 주제 단어를 코드에 박지 않고, 길이와 형태만으로 정보량이 큰 토큰을 고른다.
    - 영문 약어/혼합 토큰: KDI, AI, EMR 같은 기관·기술명 후보
    - 3자 이상 한글/영문 토큰: 사건을 구분할 가능성이 높은 단어
    """
    anchors = []
    for token in tokens or []:
        token = _safe_text(token).lower()
        if not token or token in _FINAL_DEDUP_STOPWORDS:
            continue
        has_alpha = bool(re.search(r"[a-zA-Z]", token))
        has_korean = bool(re.search(r"[가-힣]", token))
        if has_alpha or len(token) >= 3 or (has_korean and len(token) >= 3):
            anchors.append(token)
    return anchors


def _build_final_dedup_payload(news: Dict[str, Any]) -> Dict[str, Any]:
    title = _safe_text(news.get("title"))
    description = _safe_text(news.get("description") or news.get("summary") or news.get("content"))
    normalized_title = _normalize_final_dedup_title(title)
    normalized_text = _normalize_final_dedup_text(f"{title} {description}")
    compact_text = re.sub(r"\s+", "", normalized_text)
    tokens = _extract_final_dedup_tokens(news)
    title_tokens = _extract_final_title_tokens(title)
    anchor_tokens = _extract_final_anchor_tokens(title_tokens or tokens)
    return {
        "title": title,
        "normalized_title": normalized_title,
        "normalized_text": compact_text,
        "tokens": tokens,
        "title_tokens": title_tokens,
        "anchor_tokens": anchor_tokens,
        "simhash": _make_final_simhash(tokens),
    }


def _has_shared_final_anchor(cand_payload: Dict[str, Any], kept_payload: Dict[str, Any]) -> bool:
    cand_anchors = set(cand_payload.get("anchor_tokens") or [])
    kept_anchors = set(kept_payload.get("anchor_tokens") or [])
    if not cand_anchors or not kept_anchors:
        return False
    return bool(cand_anchors & kept_anchors)


def _is_final_duplicate_news(candidate: Dict[str, Any], kept: Dict[str, Any]):
    cand_payload = candidate.get("_final_dedup_payload") or _build_final_dedup_payload(candidate)
    kept_payload = kept.get("_final_dedup_payload") or _build_final_dedup_payload(kept)

    cand_title = cand_payload.get("normalized_title") or ""
    kept_title = kept_payload.get("normalized_title") or ""
    cand_numbers = _extract_final_number_tokens(cand_payload.get("title") or cand_title)
    kept_numbers = _extract_final_number_tokens(kept_payload.get("title") or kept_title)
    number_conflict = bool(cand_numbers and kept_numbers and cand_numbers != kept_numbers)

    if cand_title and kept_title and cand_title == kept_title:
        return True, "title_exact", 1.0

    # 한쪽 제목이 다른 쪽 제목을 대부분 포함하면 같은 사건으로 본다.
    # 예: 제목 뒤에 '(종합)', 부제, 수치 설명이 붙은 변형 기사.
    if cand_title and kept_title and min(len(cand_title), len(kept_title)) >= 10 and not number_conflict:
        shorter, longer = sorted([cand_title, kept_title], key=len)
        if shorter in longer:
            return True, "title_contains", len(shorter) / max(1, len(longer))

    title_overlap, title_common_count = _final_token_overlap(
        cand_payload.get("title_tokens", []),
        kept_payload.get("title_tokens", []),
    )

    title_similarity = 0.0
    if cand_title and kept_title:
        title_similarity = SequenceMatcher(None, cand_title, kept_title).ratio()
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

    overlap, common_count = _final_token_overlap(
        cand_payload.get("tokens", []),
        kept_payload.get("tokens", []),
    )
    shared_anchor = _has_shared_final_anchor(cand_payload, kept_payload)

    cand_text = cand_payload.get("normalized_text") or ""
    kept_text = kept_payload.get("normalized_text") or ""
    text_similarity = 0.0
    if cand_text and kept_text:
        text_similarity = SequenceMatcher(None, cand_text, kept_text).ratio()
        if text_similarity >= 0.92 and (common_count >= 5 or shared_anchor) and not number_conflict:
            return True, "text_similarity", text_similarity
        if text_similarity >= 0.72 and common_count >= 4 and shared_anchor and not number_conflict:
            return True, f"text_similarity_anchor_common_{common_count}", text_similarity

    # 최종 후보는 이미 AI가 고른 10개 안쪽이므로, 여기서는 중복 제거를 조금 더 적극적으로 적용한다.
    # 단, 단순히 흔한 단어만 겹쳐서 지워지는 것을 막기 위해 공통 토큰 수와 anchor 공유를 함께 본다.
    if overlap >= 0.48 and common_count >= 4 and shared_anchor and not number_conflict:
        return True, f"token_overlap_common_{common_count}", overlap

    distance = _final_simhash_distance(cand_payload.get("simhash"), kept_payload.get("simhash"))
    if distance is not None and distance <= 8 and common_count >= 4 and shared_anchor and not number_conflict:
        return True, f"simhash_distance_{distance}", 1.0 - (distance / 64)

    return False, "", max(title_similarity, text_similarity, overlap, title_overlap)


def _find_final_duplicate_info(
    candidate: Dict[str, Any],
    kept_news: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidate["_final_dedup_payload"] = (
        candidate.get("_final_dedup_payload")
        or _build_final_dedup_payload(candidate)
    )

    for kept in kept_news:
        is_duplicate, method, score = _is_final_duplicate_news(candidate, kept)
        if is_duplicate:
            return {
                "method": method,
                "score": score,
                "kept_title": kept.get("title", ""),
            }

    return None


def _deduplicate_final_selected_news(news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OpenAI가 고른 최종 후보 안에서만 코드 규칙으로 중복을 제거한다.

    의도:
    - AI가 10개를 골랐더라도 같은 사건이 섞이면 제거한다.
    - 제거 수는 LAST_SELECTION_STATS["final_duplicate_excluded_count"]에 저장해
      이메일 대시보드의 "AI 중복제외"에 표시한다.
    """
    kept_news = []
    excluded_count = 0

    for news in news_list or []:
        candidate = dict(news)
        candidate["_final_dedup_payload"] = _build_final_dedup_payload(candidate)

        duplicate_info = _find_final_duplicate_info(candidate, kept_news)

        if duplicate_info:
            excluded_count += 1
            continue

        candidate.pop("_final_dedup_payload", None)
        kept_news.append(candidate)

    LAST_SELECTION_STATS["selected_before_final_dedup_count"] = len(news_list or [])
    LAST_SELECTION_STATS["final_duplicate_excluded_count"] = excluded_count
    LAST_SELECTION_STATS["selected_after_final_dedup_count"] = len(kept_news)

    logger.info(
        f"🧹 AI 선별 후 최종 중복 제거 완료: "
        f"{len(news_list or [])}개 → {len(kept_news)}개 "
        f"(중복 제외 {excluded_count}개)"
    )

    return kept_news


def _supplement_final_news_after_dedup(
    selected_news: List[Dict[str, Any]],
    candidate_news: List[Dict[str, Any]],
    limit: int,
    used_group_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    최종 중복 제거 후 limit보다 적으면 남은 후보에서 중복이 아닌 뉴스만 보충한다.
    중복 방지가 우선이므로, 후보가 있어도 같은 사건이면 채우지 않는다.
    """
    final_news = [dict(news) for news in (selected_news or [])]
    if len(final_news) >= limit:
        return final_news[:limit]

    used_group_ids = set(used_group_ids or set())
    for news in final_news:
        group_id = _safe_text(news.get("group_id"))
        if group_id:
            used_group_ids.add(group_id)

    added_count = 0
    duplicate_skip_count = 0

    for news in candidate_news or []:
        if len(final_news) >= limit:
            break

        group_id = _safe_text(news.get("group_id"))
        if group_id and group_id in used_group_ids:
            continue

        candidate = dict(news)
        _prepare_selected_news(
            candidate,
            importance_score=candidate.get("importance_score", 3),
        )
        candidate["_final_dedup_payload"] = _build_final_dedup_payload(candidate)
        duplicate_info = _find_final_duplicate_info(candidate, final_news)

        if duplicate_info:
            duplicate_skip_count += 1
            continue

        candidate.pop("_final_dedup_payload", None)
        final_news.append(candidate)
        added_count += 1
        if group_id:
            used_group_ids.add(group_id)

    if duplicate_skip_count:
        LAST_SELECTION_STATS["final_duplicate_excluded_count"] = (
            int(LAST_SELECTION_STATS.get("final_duplicate_excluded_count", 0))
            + duplicate_skip_count
        )

    LAST_SELECTION_STATS["selected_after_final_dedup_count"] = len(final_news)

    logger.info(
        f"➕ 최종 뉴스 보충 완료: "
        f"{len(selected_news or [])}개 → {len(final_news)}개 / "
        f"추가 {added_count}개 / 보충 중복 제외 {duplicate_skip_count}개"
    )

    return final_news[:limit]


def _group_sort_key(group: Dict[str, Any]):
    rep = group.get("representative") or {}
    return (
        float(group.get("priority_score") or 0),
        int(group.get("source_count") or 0),
        int(group.get("article_count") or 0),
        _safe_text(rep.get("published_at_kst") or rep.get("published_at")),
    )


def _shortlist_groups_for_ai(
    group_list: List[Dict[str, Any]],
    final_limit: int,
    candidate_group_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    로컬 점수로 넓게 정렬하되, AI에는 설정된 수만 넘긴다.
    단순 상위 N개만 자르면 특정 키워드에 쏠릴 수 있어 키워드별 대표와 최신 그룹을 섞는다.
    """
    sorted_groups = sorted(group_list or [], key=_group_sort_key, reverse=True)
    if not sorted_groups:
        return []

    target = int(candidate_group_limit or DEFAULT_SELECTOR_CANDIDATE_GROUP_LIMIT)
    target = max(int(final_limit or 1), target)
    target = min(len(sorted_groups), target)

    if len(sorted_groups) <= target:
        return sorted_groups

    selected: List[Dict[str, Any]] = []
    seen_group_ids = set()

    def add_group(group: Dict[str, Any]) -> bool:
        if len(selected) >= target:
            return False
        group_id = _safe_text(group.get("group_id"))
        if not group_id or group_id in seen_group_ids:
            return False
        seen_group_ids.add(group_id)
        selected.append(group)
        return True

    priority_seed_count = min(len(sorted_groups), max(final_limit * 3, target // 2))
    for group in sorted_groups[:priority_seed_count]:
        add_group(group)

    keyword_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for group in sorted_groups:
        keywords = group.get("keywords") or ["__unknown__"]
        for keyword in keywords:
            keyword = _safe_text(keyword) or "__unknown__"
            keyword_buckets.setdefault(keyword, []).append(group)

    per_keyword_quota = max(2, final_limit // 2)
    for keyword in sorted(keyword_buckets):
        added_for_keyword = 0
        for group in keyword_buckets[keyword]:
            if add_group(group):
                added_for_keyword += 1
            if len(selected) >= target or added_for_keyword >= per_keyword_quota:
                break
        if len(selected) >= target:
            break

    recent_groups = sorted(
        sorted_groups,
        key=lambda group: _safe_text((group.get("representative") or {}).get("published_at_kst")),
        reverse=True,
    )
    for group in recent_groups[:max(final_limit * 2, 10)]:
        add_group(group)

    for group in sorted_groups:
        if not add_group(group) and len(selected) >= target:
            break

    return selected[:target]


def _build_group_candidate_text(group_list: List[Dict[str, Any]]) -> str:
    """
    Python 그룹화 결과를 OpenAI가 읽기 쉬운 짧은 후보 목록으로 변환한다.
    기사 원문 전체가 아니라 그룹 대표 정보와 그룹 통계만 전달해 토큰을 줄인다.
    """
    lines = []
    for idx, group in enumerate(group_list or [], 1):
        rep = group.get("representative") or {}
        group_id = _safe_text(group.get("group_id") or f"G{idx:03d}")
        title = _clip_text(rep.get("title"), GROUP_CANDIDATE_TITLE_CHARS)
        description = ""
        if idx <= SELECTOR_DETAILED_CANDIDATE_COUNT:
            description = _clip_text(rep.get("description"), GROUP_CANDIDATE_DESCRIPTION_CHARS)
        source = _clip_text(rep.get("source"), 30)
        sources = ", ".join(group.get("sources") or [])[:GROUP_CANDIDATE_SOURCES_CHARS]
        keywords = ", ".join(group.get("keywords") or [])[:GROUP_CANDIDATE_KEYWORDS_CHARS]
        article_count = int(group.get("article_count") or 1)
        source_count = int(group.get("source_count") or 1)
        priority_score = _safe_text(group.get("priority_score"))
        quality_flags = ",".join(group.get("quality_flags") or []) or "-"
        description_part = f" | desc={description}" if description else ""

        lines.append(
            f"[{idx}] id={group_id} | n={article_count} src={source_count} score={priority_score} "
            f"flags={quality_flags} | press={sources or source} | kw={keywords} | title={title}"
            f"{description_part}"
        )
    return "\n\n".join(lines)


def _estimate_importance_score_from_group(group: Dict[str, Any]) -> int:
    """
    AI 점수가 없는 fallback/보충 후보의 중요도를 로컬 그룹 신호로 추정한다.
    중복 보충 과정에서 모든 뉴스가 3점으로 표시되는 것을 막기 위한 안전망이다.
    """
    try:
        priority_score = float(group.get("priority_score") or 0)
    except Exception:
        priority_score = 0.0

    article_count = int(group.get("article_count") or 1)
    source_count = int(group.get("source_count") or 1)
    flags = set(group.get("quality_flags") or [])

    if "low_representative_score" in flags or "photo_like_representative" in flags:
        return 2

    if priority_score >= 24 or (source_count >= 5 and article_count >= 8):
        return 5

    if source_count >= 3 or article_count >= 4 or priority_score >= 13:
        return 4

    if priority_score < 4:
        return 2

    return 3


def _estimate_importance_score_from_news(news: Dict[str, Any]) -> int:
    """
    그룹 dict가 아닌 대표 기사 dict만 있을 때의 fallback 중요도 추정.
    """
    if news.get("importance_score") not in (None, ""):
        return _safe_int(news.get("importance_score"), default=3)

    try:
        priority_score = float(news.get("group_priority_score") or 0)
    except Exception:
        priority_score = 0.0

    article_count = int(news.get("group_article_count") or 1)
    source_count = int(news.get("group_source_count") or 1)
    flags = set(news.get("group_quality_flags") or [])

    return _estimate_importance_score_from_group({
        "priority_score": priority_score,
        "article_count": article_count,
        "source_count": source_count,
        "quality_flags": list(flags),
    })


def _representative_news_from_group(group: Dict[str, Any]) -> Dict[str, Any]:
    rep = dict(group.get("representative") or {})
    articles = group.get("articles") or []

    # 그룹화 결과의 대표 기사에는 published_at_kst만 있는 경우가 있다.
    # 이후 요약/메일 단계는 published_at을 우선 사용하므로 여기서 호환 필드를 보강한다.
    if not rep.get("published_at") and rep.get("published_at_kst"):
        rep["published_at"] = rep.get("published_at_kst")
    if not rep.get("published_at_kst") and rep.get("published_at"):
        rep["published_at_kst"] = rep.get("published_at")

    rep["group_id"] = group.get("group_id")
    rep["group_article_count"] = group.get("article_count", 1)
    rep["group_source_count"] = group.get("source_count", 1)
    rep["group_sources"] = group.get("sources", [])
    rep["group_keywords"] = group.get("keywords", [])
    rep["group_quality_flags"] = group.get("quality_flags", [])
    rep["group_priority_score"] = group.get("priority_score", 0)
    rep["group_article_titles"] = [
        _safe_text(article.get("title"))
        for article in articles[:12]
        if _safe_text(article.get("title"))
    ]
    rep["group_article_urls"] = [
        _safe_text(article.get("url"))
        for article in articles[:12]
        if _safe_text(article.get("url"))
    ]
    rep["local_importance_score"] = _estimate_importance_score_from_group(group)
    if rep.get("importance_score") in (None, ""):
        rep["importance_score"] = rep["local_importance_score"]
    rep["content"] = rep.get("description", "")
    return rep


def _fallback_select_groups(
    group_list: List[Dict[str, Any]],
    fallback_news_list: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """
    그룹 단위 OpenAI 선별 실패 시 로컬 우선순위 순서대로 대표 기사 사용.
    """
    selected_news = []

    if group_list:
        sorted_groups = sorted(
            group_list,
            key=_group_sort_key,
            reverse=True,
        )
        candidate_news = []
        for group in sorted_groups:
            news = _representative_news_from_group(group)
            _prepare_selected_news(
                news,
                importance_score=_estimate_importance_score_from_group(group),
            )
            candidate_news.append(news)

        selected_news = _deduplicate_final_selected_news(candidate_news[:limit])
        return _supplement_final_news_after_dedup(
            selected_news=selected_news,
            candidate_news=candidate_news,
            limit=limit,
        )

    candidate_news = []
    for news in fallback_news_list or []:
        news = dict(news)
        _prepare_selected_news(
            news,
            importance_score=_estimate_importance_score_from_news(news),
        )
        candidate_news.append(news)

    selected_news = _deduplicate_final_selected_news(candidate_news[:limit])
    return _supplement_final_news_after_dedup(
        selected_news=selected_news,
        candidate_news=candidate_news,
        limit=limit,
    )


def select_important_news_groups(
    group_list: List[Dict[str, Any]],
    fallback_news_list: List[Dict[str, Any]],
    topic_name: str,
    topic_description: str,
    limit: int = 10,
    candidate_group_limit: Optional[int] = None
) -> List[Dict]:
    """
    Python 규칙 기반으로 묶인 사건 그룹 중 OpenAI가 중요한 그룹만 선택한다.

    기존 기사 단위 선별과 달리:
    - 입력은 기사 전체가 아니라 사건 그룹 대표 정보다.
    - 같은 사건 중복 제거는 이미 Python 그룹화 단계에서 수행한다.
    - 출력은 group_id 기준으로 받는다.
    """
    reset_selection_stats()

    if not group_list and not fallback_news_list:
        logger.warning("선택할 뉴스 그룹 후보가 없습니다.")
        return []

    logger.info(
        f"🧠 [{topic_name}] 그룹 선별 시작: 후보 {len(group_list or [])}개 / 목표 {limit}개"
    )

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 그룹 fallback 선별을 사용합니다.")
        return _fallback_select_groups(group_list, fallback_news_list, limit)

    prepared_groups = _shortlist_groups_for_ai(
        group_list=group_list or [],
        final_limit=limit,
        candidate_group_limit=candidate_group_limit,
    )

    if not prepared_groups:
        logger.warning("⚠️ OpenAI에 전달할 그룹이 없어 fallback 선별을 사용합니다.")
        return _fallback_select_groups(group_list, fallback_news_list, limit)

    group_text = _build_group_candidate_text(prepared_groups)
    selection_limit = min(len(prepared_groups), max(limit, 1))
    completion_limit = SELECTOR_MAX_COMPLETION_TOKENS if is_gpt5_model(SELECTOR_MODEL) else min(900, SELECTOR_MAX_COMPLETION_TOKENS)

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
1. 주제와 직접 관련 있고 독자가 알아야 할 사건을 우선합니다.
2. src, n, score가 높거나 정책/규제/실적/투자/계약/시장 변화가 있으면 우선합니다.
3. 후보가 충분하면 서로 다른 사건으로 {selection_limit}개를 채웁니다. 관련성이 낮거나 홍보/사진/단순 행사뿐이면 덜 뽑아도 됩니다.
4. photo_like_representative, low_representative_score, overgroup_risk_token_time_span 플래그는 원칙적으로 제외합니다.
5. 같은 group_id를 두 번 쓰지 말고, 후보에 없는 group_id를 만들지 마세요.

importance_score:
- 5: 주제 핵심이며 영향이 큰 정책/규제/실적/투자/대형계약/시장 변화
- 4: 주요 기업/기관의 전략, 서비스, 기술, 제휴, 수급 변화
- 3: 주제 관련 일반 뉴스
- 2: 관련성은 있으나 영향/구체성이 낮음
- 1: 거의 제외 대상이지만 참고용으로만 선택
모든 항목을 기계적으로 3점으로 주지 말고 후보 신호에 따라 차등화하세요.

출력은 JSON 객체 하나만:
{{"selected":[{{"group_id":"G001","importance_score":5}}]}}

후보:
{group_text}
"""

    try:
        response = client.chat.completions.create(
            model=SELECTOR_MODEL,
            messages=[
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

        content = response.choices[0].message.content.strip()
        usage_info = record_openai_usage(
            logger,
            "그룹 단위 뉴스 선별",
            SELECTOR_MODEL,
            response.usage,
        )
        tokens_used = usage_info["total_tokens"]
        add_selection_tokens("selection_tokens", tokens_used)
        logger.debug(f"🧾 그룹 단위 뉴스 선별 토큰 사용량: {tokens_used}")

        try:
            result = _ensure_json_object(_extract_json(content))
        except Exception:
            logger.error("❌ OpenAI 그룹 선별 응답 JSON 파싱 실패")
            logger.error("응답 미리보기: %s", content[:300])
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        selected_items = _ensure_json_list(result.get("selected"))
        if not selected_items:
            logger.warning("⚠️ OpenAI가 선택한 그룹이 없습니다. fallback을 사용합니다.")
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        group_by_id = {str(group.get("group_id")): group for group in prepared_groups}
        selected_news = []
        used_group_ids = set()

        invalid_item_count = 0
        for item in selected_items:
            if not isinstance(item, dict):
                invalid_item_count += 1
                continue

            group_id = _safe_text(item.get("group_id"))
            if not group_id or group_id in used_group_ids:
                invalid_item_count += 1
                continue

            group = group_by_id.get(group_id)
            if not group:
                invalid_item_count += 1
                continue

            news = _representative_news_from_group(group)
            _prepare_selected_news(
                news=news,
                importance_score=item.get("importance_score", 3)
            )
            selected_news.append(news)
            used_group_ids.add(group_id)

            if len(selected_news) >= limit:
                break

        if invalid_item_count:
            logger.debug("그룹 선별 응답 무시 항목: %s개", invalid_item_count)

        if not selected_news:
            logger.warning("⚠️ 유효하게 선택된 그룹이 없습니다. fallback을 사용합니다.")
            return _fallback_select_groups(group_list, fallback_news_list, limit)

        before_final_dedup_count = len(selected_news)
        selected_news = _deduplicate_final_selected_news(selected_news)

        if len(selected_news) < limit:
            supplement_candidates = []

            for group in prepared_groups:
                news = _representative_news_from_group(group)
                _prepare_selected_news(
                    news,
                    importance_score=_estimate_importance_score_from_group(group),
                )
                supplement_candidates.append(news)

            for news in fallback_news_list or []:
                news = dict(news)
                _prepare_selected_news(
                    news,
                    importance_score=_estimate_importance_score_from_news(news),
                )
                supplement_candidates.append(news)

            selected_news = _supplement_final_news_after_dedup(
                selected_news=selected_news,
                candidate_news=supplement_candidates,
                limit=limit,
                used_group_ids=used_group_ids,
            )

        logger.info(
            f"✅ 그룹 단위 뉴스 선별 완료: "
            f"AI 선택 {before_final_dedup_count}개 → 최종 {len(selected_news)}개 "
            f"(중복 제외 {LAST_SELECTION_STATS.get('final_duplicate_excluded_count', 0)}개)"
        )
        return selected_news

    except Exception as e:
        logger.error(f"❌ OpenAI 그룹 단위 뉴스 선별 실패: {e}")
        return _fallback_select_groups(group_list, fallback_news_list, limit)
