"""
OpenAI API를 사용한 뉴스 요약 모듈

개선 사항:
- OpenAI 클라이언트 초기화 방어 로직 추가
- 기사별 개별 호출 대신 기본적으로 배치 요약을 사용해 호출 수와 토큰 낭비를 줄임
- 배치 요약 실패 시 기존 단건 요약 방식으로 fallback
- 요약 토큰 사용량을 main.py에서 집계할 수 있도록 결과에 tokens_used를 포함
"""
import os
import json
import logging
import time
from typing import List, Dict, Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from dotenv import load_dotenv
from openai_usage import (
    create_chat_completion as create_openai_chat_completion,
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

# 모델 설정
MODEL = os.getenv("SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))

# 기본값: 배치 요약 사용. 필요 시 SUMMARY_BATCH_MODE=false 로 단건 방식 사용 가능.
SUMMARY_BATCH_MODE = os.getenv("SUMMARY_BATCH_MODE", "true").lower() not in {"0", "false", "no"}

# GPT-5 계열은 max_completion_tokens 안에 숨은 reasoning token도 포함된다.
# 기본 한도는 동적으로 더 작게 잡고, 빈 응답/JSON 실패 때만 더 크게 재시도한다.
# 첫 시도에서 reasoning token이 출력 한도를 모두 써버리면 같은 요청을 재시도하게 된다.
# 기본 한도를 넉넉히 둬서 2~3문장 요약에서도 불필요한 2회 호출을 줄인다.
# max_completion_tokens는 상한값이라 실제 비용은 사용된 토큰 기준으로만 발생한다.
SUMMARY_BATCH_COMPLETION_LIMIT = int(os.getenv("SUMMARY_BATCH_MAX_COMPLETION_TOKENS", "8000"))
SUMMARY_SINGLE_COMPLETION_LIMIT = int(os.getenv("SUMMARY_SINGLE_MAX_COMPLETION_TOKENS", "1600"))
SUMMARY_INPUT_CONTENT_LIMIT = int(os.getenv("SUMMARY_INPUT_CONTENT_CHARS", "900"))


def _message_content(response: Any) -> str:
    try:
        return _safe_text(response.choices[0].message.content)
    except Exception:
        return ""


def _finish_reason(response: Any) -> str:
    try:
        return _safe_text(response.choices[0].finish_reason)
    except Exception:
        return ""


def _summary_reasoning_effort_kwargs() -> Dict[str, str]:
    """
    요약 호출에서 GPT-5 계열의 숨은 reasoning token 소모를 줄이기 위한 설정.

    기본값은 minimal이다.
    - SUMMARY_REASONING_EFFORT=none 으로 두면 파라미터를 보내지 않는다.
    - OPENAI_REASONING_EFFORT보다 SUMMARY_REASONING_EFFORT를 우선한다.
    - 구버전 SDK에서 reasoning_effort를 직접 지원하지 않으면 호출 wrapper가 extra_body로 옮긴다.
    """
    if not is_gpt5_model(MODEL):
        return {}

    effort = str(
        os.getenv("SUMMARY_REASONING_EFFORT", os.getenv("OPENAI_REASONING_EFFORT", "minimal"))
        or ""
    ).strip().lower()

    if effort in {"", "none", "default", "off", "false", "0"}:
        return {}

    return {"reasoning_effort": effort}


def _create_chat_completion(**kwargs):
    """
    Chat Completions 호출 wrapper.
    OpenAI SDK 버전에 따라 신규 body 필드를 extra_body로 옮겨 호환 호출한다.
    """
    return create_openai_chat_completion(client, logger, **kwargs)


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


def _extract_json(content: str) -> Dict:
    """
    OpenAI 응답에서 JSON 파싱.
    원칙적으로 JSON만 오게 하지만 코드블록이나 앞뒤 설명이 섞이는 경우를 대비한다.
    """
    content = _safe_text(content)

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def _ensure_json_object(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {}


def _ensure_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _build_fallback_summary(article: Dict, max_length: int = 220) -> str:
    content = _safe_text(article.get("content") or article.get("description"))
    return content[:max_length] + "..." if len(content) > max_length else content


def _clip_text(value: Any, limit: int) -> str:
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _summary_min_length(max_length: int) -> int:
    max_length = max(int(max_length or 220), 80)
    return min(max(int(max_length * 0.6), 120), max_length)


def _batch_completion_limits(article_count: int, max_length: int) -> List[int]:
    article_count = max(int(article_count or 1), 1)
    max_length = max(int(max_length or 180), 80)
    per_article_budget = max(300, min(max_length + 260, 760))
    first_limit = max(1600, 700 + article_count * per_article_budget)

    if is_gpt5_model(MODEL):
        # GPT-5 계열은 reasoning token도 max_completion_tokens 안에 포함된다.
        # 첫 시도 한도가 너무 작으면 본문 없이 length로 끝나 같은 요청을 한 번 더 보내게 된다.
        # 상한만 넉넉히 잡고, reasoning_effort는 minimal로 낮춰 실제 사용 토큰을 줄인다.
        first_limit = max(first_limit, min(SUMMARY_BATCH_COMPLETION_LIMIT, 5200))
        first_limit = min(first_limit, SUMMARY_BATCH_COMPLETION_LIMIT)
        retry_limit = max(first_limit + 1200, 600 + article_count * 600)
        retry_limit = min(retry_limit, max(SUMMARY_BATCH_COMPLETION_LIMIT + 1600, first_limit))
        return [first_limit, retry_limit] if retry_limit > first_limit else [first_limit]

    return [min(first_limit, SUMMARY_BATCH_COMPLETION_LIMIT)]


def _build_summary_result(article: Dict, summary: str, tokens_used: int = 0, error: str = None) -> Dict:
    title = _safe_text(article.get("title", "제목 없음"))
    url = _safe_text(article.get("url", "#"))
    keyword = _safe_text(article.get("keyword", ""))
    description = _safe_text(article.get("description"))
    content = _safe_text(article.get("content"))
    # 수집/그룹화 단계에 따라 published_at 또는 published_at_kst 중 하나만 있을 수 있다.
    # 메일에서 발생시간이 사라지지 않도록 둘 다 보존한다.
    published_at = _safe_text(article.get("published_at") or article.get("published_at_kst") or "")
    published_at_kst = _safe_text(article.get("published_at_kst") or article.get("published_at") or "")
    importance_score = _safe_int(article.get("importance_score", 3))
    source = (
        _safe_text(article.get("source"))
        or _safe_text(article.get("press"))
        or _safe_text(article.get("publisher"))
        or _safe_text(article.get("media"))
        or "언론사 미상"
    )

    result = {
        "title": title,
        "summary": _safe_text(summary),
        "url": url,
        "keyword": keyword,
        "description": description,
        "content": content or description,
        "group_id": article.get("group_id"),
        "group_article_count": article.get("group_article_count"),
        "group_source_count": article.get("group_source_count"),
        "group_keywords": article.get("group_keywords", []),
        "group_quality_flags": article.get("group_quality_flags", []),
        "group_priority_score": article.get("group_priority_score"),
        "group_sources": article.get("group_sources", []),
        "group_article_titles": article.get("group_article_titles", []),
        "group_article_urls": article.get("group_article_urls", []),
        "group_article_sources": article.get("group_article_sources", []),
        "published_at": published_at,
        "published_at_kst": published_at_kst,
        "importance_score": importance_score,
        "source": source,
        "tokens_used": int(tokens_used or 0),
    }

    if error:
        result["error"] = error

    return result


def summarize_article(article: Dict, max_length: int = 220) -> Dict:
    """
    단일 기사 요약.
    배치 요약 실패 시 fallback으로도 사용한다.
    """
    title = _safe_text(article.get("title", "제목 없음"))
    content = _safe_text(article.get("content") or article.get("description"))

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 원문 설명을 fallback 요약으로 사용합니다.")
        return _build_summary_result(
            article=article,
            summary=_build_fallback_summary(article, max_length=max_length),
            tokens_used=0,
            error="OPENAI_API_KEY 없음 또는 OpenAI 클라이언트 초기화 실패"
        )

    try:
        min_length = _summary_min_length(max_length)
        prompt = f"""
뉴스를 {min_length}~{max_length}자, 2~3문장으로 요약하세요.

규칙:
1. 제목/내용에 있는 사실만 사용합니다.
2. 기업명, 서비스명, 수치, 일정은 원문에 있을 때만 포함합니다.
3. 추측, 전망, 평가를 새로 만들지 않습니다.
4. 첫 문장에는 핵심 사건/발표를, 이어지는 문장에는 배경·수치·영향·다음 일정 중 원문에 있는 정보를 담습니다.
5. 내용이 부족하면 억지로 늘리지 말고 제목을 바탕으로 확인 가능한 사실만 씁니다.

제목: {title}
내용: {content}

요약:
"""

        logger.debug(f"📝 요약 중: {title[:30]}...")

        response = _create_chat_completion(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 기사를 사실 중심으로 요약합니다. "
                        "핵심 사건과 확인 가능한 배경을 함께 담고, "
                        "원문에 없는 사실을 추가하지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.2),
            **_summary_reasoning_effort_kwargs(),
            **openai_token_limit_kwargs(MODEL, SUMMARY_SINGLE_COMPLETION_LIMIT if is_gpt5_model(MODEL) else 300)
        )

        summary = _message_content(response)
        usage_info = record_openai_usage(
            logger,
            "단건 뉴스 요약",
            MODEL,
            response.usage,
        )
        tokens_used = usage_info["total_tokens"]

        if not summary:
            raise ValueError(
                f"단건 요약 응답 본문이 비어 있습니다. finish_reason={_finish_reason(response)} "
                f"reasoning_tokens={usage_info.get('reasoning_tokens', 0)}"
            )

        logger.debug(f"✅ 단건 요약 완료: {len(summary)}자 / 토큰 {tokens_used}")

        return _build_summary_result(
            article=article,
            summary=summary,
            tokens_used=tokens_used,
        )

    except Exception as e:
        logger.error(f"❌ 요약 실패: {e}")
        return _build_summary_result(
            article=article,
            summary=_build_fallback_summary(article, max_length=max_length),
            tokens_used=0,
            error=str(e)
        )


def summarize_batch_with_llm(articles: List[Dict], max_length: int = 220) -> List[Dict]:
    """
    여러 기사를 한 번의 OpenAI 호출로 요약한다.
    기사별 호출보다 호출 수가 줄고, 전체 실행 시간이 짧아진다.
    """
    if not articles:
        return []

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 배치 요약을 사용할 수 없습니다.")
        return [
            _build_summary_result(
                article=article,
                summary=_build_fallback_summary(article, max_length=max_length),
                tokens_used=0,
                error="OPENAI_API_KEY 없음 또는 OpenAI 클라이언트 초기화 실패"
            )
            for article in articles
        ]

    news_blocks = []
    for idx, article in enumerate(articles, 1):
        title = _clip_text(article.get("title", "제목 없음"), 120)
        content = _clip_text(article.get("content") or article.get("description"), SUMMARY_INPUT_CONTENT_LIMIT)
        source = _safe_text(article.get("source")) or "언론사 미상"
        published_at = _safe_text(article.get("published_at"))

        news_blocks.append(
            f"""
[{idx}]
언론사: {source}
발행일: {published_at}
제목: {title}
내용: {content}
""".strip()
        )

    min_length = _summary_min_length(max_length)
    prompt = f"""
아래 뉴스들을 각각 {min_length}~{max_length}자, 2~3문장으로 요약하세요.

요약 규칙:
1. 제목/내용에 있는 사실만 사용합니다.
2. 기업명, 서비스명, 수치, 일정은 원문에 있을 때만 포함합니다.
3. 추측, 전망, 평가를 새로 만들지 않습니다.
4. 첫 문장에는 핵심 사건/발표를, 이어지는 문장에는 배경·수치·영향·다음 일정 중 원문에 있는 정보를 담습니다.
5. 내용이 부족하면 억지로 늘리지 말고 제목을 바탕으로 확인 가능한 사실만 씁니다.
6. 모든 index를 정확히 한 번씩 포함하고 summary는 빈 문자열로 두지 않습니다.

출력은 JSON 객체 하나만:
{{"summaries":[{{"index":1,"summary":"요약문"}}]}}

뉴스:
{chr(10).join(news_blocks)}
"""

    last_error = None
    parsed = None
    tokens_used = 0

    # GPT-5 계열은 max_completion_tokens 안에 reasoning token이 포함된다.
    # 첫 시도에서 본문이 비거나 JSON 파싱이 실패하면 더 큰 한도로 배치 1회만 재시도한다.
    completion_limits = _batch_completion_limits(len(articles), max_length)

    for attempt_no, completion_limit in enumerate(completion_limits, 1):
        response = _create_chat_completion(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 기사를 사실 중심으로 요약하는 편집자입니다. "
                        "핵심 사건과 확인 가능한 배경을 함께 담고, "
                        "원문에 없는 사실을 추가하지 않고, 반드시 JSON만 출력합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.2),
            **_summary_reasoning_effort_kwargs(),
            **openai_json_response_format_kwargs(),
            **openai_token_limit_kwargs(MODEL, completion_limit)
        )

        content = _message_content(response)
        usage_info = record_openai_usage(
            logger,
            f"배치 뉴스 요약 시도 {attempt_no}",
            MODEL,
            response.usage,
        )
        tokens_used += usage_info["total_tokens"]

        if not content:
            last_error = ValueError(
                f"배치 요약 응답 본문이 비어 있습니다. "
                f"attempt={attempt_no}, finish_reason={_finish_reason(response)}, "
                f"reasoning_tokens={usage_info.get('reasoning_tokens', 0)}, "
                f"completion_limit={completion_limit}"
            )
            logger.warning("⚠️ %s", last_error)
            continue

        try:
            parsed = _ensure_json_object(_extract_json(content))
            break
        except Exception as e:
            last_error = e
            logger.warning(
                "⚠️ 배치 요약 JSON 파싱 실패: attempt=%s | finish_reason=%s | content_preview=%s",
                attempt_no,
                _finish_reason(response),
                content[:300],
            )

    if parsed is None:
        raise last_error or ValueError("배치 요약 JSON 파싱 실패")

    by_index = {}
    for item in _ensure_json_list(parsed.get("summaries")):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        summary_text = _safe_text(item.get("summary"))
        if summary_text:
            by_index[idx] = summary_text

    if not by_index:
        raise ValueError("배치 요약 응답에 summaries가 없습니다.")

    # 배치 호출 1회의 토큰을 각 기사 결과에 나눠서 싣는다.
    per_article_tokens = int(tokens_used / max(len(articles), 1))

    results = []
    for idx, article in enumerate(articles, 1):
        summary = by_index.get(idx) or _build_fallback_summary(article, max_length=max_length)
        results.append(
            _build_summary_result(
                article=article,
                summary=summary,
                tokens_used=per_article_tokens,
                error=None if idx in by_index else "배치 요약에서 해당 index 누락, fallback 사용"
            )
        )

    # 나눗셈 반올림으로 빠진 토큰은 첫 기사에 더해 총합이 맞게 한다.
    distributed = per_article_tokens * len(results)
    remainder = int(tokens_used or 0) - distributed
    if results and remainder > 0:
        results[0]["tokens_used"] += remainder

    logger.debug(f"✅ 배치 요약 완료: {len(results)}개 / 토큰 {tokens_used}")

    return results


def summarize_batch(articles: List[Dict], delay: float = 1.0, max_length: int = 220) -> List[Dict]:
    """
    여러 기사 일괄 요약.

    기본은 배치 요약이며, 실패 시 기존 단건 요약으로 fallback한다.
    """
    logger.info(
        "🤖 뉴스 요약 시작: %s개 / max_length=%s / batch=%s",
        len(articles),
        max_length,
        "on" if SUMMARY_BATCH_MODE else "off",
    )

    if not articles:
        return []

    if SUMMARY_BATCH_MODE:
        try:
            summaries = summarize_batch_with_llm(articles, max_length=max_length)
            total_tokens = sum(summary.get("tokens_used", 0) for summary in summaries)
            logger.info(f"✅ 뉴스 요약 완료: {len(summaries)}개 / 토큰 {total_tokens:,}")
            return summaries
        except Exception as e:
            logger.error(f"❌ 배치 요약 실패, 단건 요약으로 전환: {e}")

    summaries = []
    total_tokens = 0

    for i, article in enumerate(articles, 1):
        logger.debug(f"요약 진행: {i}/{len(articles)}")

        summary = summarize_article(article, max_length=max_length)
        summaries.append(summary)

        total_tokens += summary.get("tokens_used", 0)

        if i < len(articles):
            time.sleep(delay)

    logger.info(f"✅ 뉴스 요약 완료: {len(summaries)}개 / 토큰 {total_tokens:,}")

    return summaries
