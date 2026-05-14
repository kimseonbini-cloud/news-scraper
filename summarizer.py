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
# 너무 작게 잡으면 completion_tokens는 소모되지만 message.content가 빈 값으로 올 수 있다.
SUMMARY_BATCH_COMPLETION_LIMIT = int(os.getenv("SUMMARY_BATCH_MAX_COMPLETION_TOKENS", "8192"))
SUMMARY_SINGLE_COMPLETION_LIMIT = int(os.getenv("SUMMARY_SINGLE_MAX_COMPLETION_TOKENS", "1600"))


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


def _build_fallback_summary(article: Dict, max_length: int = 220) -> str:
    content = _safe_text(article.get("content") or article.get("description"))
    return content[:max_length] + "..." if len(content) > max_length else content


def _build_summary_result(article: Dict, summary: str, tokens_used: int = 0, error: str = None) -> Dict:
    title = _safe_text(article.get("title", "제목 없음"))
    url = _safe_text(article.get("url", "#"))
    keyword = _safe_text(article.get("keyword", ""))
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
        prompt = f"""
다음 뉴스 기사를 {max_length}자 이내로 요약해주세요.

[요약 기준]
1. 핵심 내용만 간결하게 정리하세요.
2. 기사 제목과 내용에 없는 사실은 추가하지 마세요.
3. 기업명, 서비스명, 수치, 일정은 원문에 있는 경우에만 포함하세요.
4. 추측성 표현은 쓰지 마세요.
5. 문장은 자연스러운 한국어로 작성하세요.
6. 요약은 1~2문장으로 작성하세요.

제목: {title}
내용: {content}

요약:
"""

        logger.info(f"📝 요약 중: {title[:30]}...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 기사를 간결하고 정확하게 요약합니다. "
                        "원문에 없는 사실을 추가하지 않습니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.2),
            **openai_reasoning_effort_kwargs(MODEL),
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

        logger.info(f"✅ 요약 완료: {len(summary)}자 (토큰: {tokens_used})")

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
        title = _safe_text(article.get("title", "제목 없음"))
        content = _safe_text(article.get("content") or article.get("description"))
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

    prompt = f"""
아래 뉴스 기사들을 각각 {max_length}자 이내로 요약하세요.

[요약 기준]
1. 각 기사는 1~2문장으로 요약하세요.
2. 기사 제목과 내용에 없는 사실은 추가하지 마세요.
3. 기업명, 서비스명, 수치, 일정은 원문에 있는 경우에만 포함하세요.
4. 추측성 표현은 쓰지 마세요.
5. 문장은 자연스러운 한국어로 작성하세요.
6. 모든 index를 빠짐없이 포함하세요.

[출력 규칙]
반드시 JSON만 출력하세요.
설명 문장, 마크다운, 코드블록은 쓰지 마세요.

[출력 형식]
{{
  "summaries": [
    {{"index": 1, "summary": "요약문"}}
  ]
}}

[뉴스 목록]
{chr(10).join(news_blocks)}
"""

    last_error = None
    parsed = None
    tokens_used = 0

    # GPT-5 계열은 max_completion_tokens 안에 reasoning token이 포함된다.
    # 첫 시도에서 본문이 비거나 JSON 파싱이 실패하면 더 큰 한도로 배치 1회만 재시도한다.
    completion_limits = [
        SUMMARY_BATCH_COMPLETION_LIMIT if is_gpt5_model(MODEL) else max(700, len(articles) * 260),
    ]
    if is_gpt5_model(MODEL):
        completion_limits.append(max(SUMMARY_BATCH_COMPLETION_LIMIT * 2, len(articles) * 1200))

    for attempt_no, completion_limit in enumerate(completion_limits, 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "뉴스 기사를 정확하게 요약하는 편집자입니다. "
                        "원문에 없는 사실을 추가하지 않고, 반드시 JSON만 출력합니다."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **openai_temperature_kwargs(MODEL, 0.2),
            **openai_reasoning_effort_kwargs(MODEL),
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
            parsed = _extract_json(content)
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
    for item in parsed.get("summaries", []):
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        by_index[idx] = _safe_text(item.get("summary"))

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

    logger.info(f"✅ 배치 요약 완료: {len(results)}개 기사 (총 토큰: {tokens_used})")

    return results


def summarize_batch(articles: List[Dict], delay: float = 1.0) -> List[Dict]:
    """
    여러 기사 일괄 요약.

    기본은 배치 요약이며, 실패 시 기존 단건 요약으로 fallback한다.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🤖 {len(articles)}개 기사 요약 시작")
    logger.info(f"{'=' * 60}\n")

    if not articles:
        return []

    if SUMMARY_BATCH_MODE:
        try:
            summaries = summarize_batch_with_llm(articles)
            total_tokens = sum(summary.get("tokens_used", 0) for summary in summaries)
            logger.info(f"총 토큰: {total_tokens:,}")
            logger.info("예상 비용은 OpenAI 응답 usage 기준 모델별 로그를 확인하세요.")
            return summaries
        except Exception as e:
            logger.error(f"❌ 배치 요약 실패, 단건 요약으로 전환: {e}")

    summaries = []
    total_tokens = 0

    for i, article in enumerate(articles, 1):
        logger.info(f"진행: {i}/{len(articles)}")

        summary = summarize_article(article)
        summaries.append(summary)

        total_tokens += summary.get("tokens_used", 0)

        if i < len(articles):
            time.sleep(delay)

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ 요약 완료")
    logger.info(f"총 토큰: {total_tokens:,}")
    logger.info("예상 비용은 OpenAI 응답 usage 기준 모델별 로그를 확인하세요.")
    logger.info(f"{'=' * 60}\n")

    return summaries
