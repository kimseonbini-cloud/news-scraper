# =============================================================================
# [파일 설명]
# - 수행 기능: 선별된 뉴스를 OpenAI로 요약하고, 실패 시 안전한 fallback 요약을 만듭니다.
# - 프로세스: 기사 입력 정리 -> 배치 요약 요청 -> JSON 응답 파싱 -> 길이/품질 검증 -> 표준 요약 dict 생성
# - 호출하는 곳: main.py
# - 주요 파라미터/입력: 선별 뉴스 목록, 요약 길이, OpenAI 모델/토큰 설정
# - 리턴값/출력: 메일/히스토리/관련보도 페이지에서 사용할 요약 dict 목록을 반환합니다.
# =============================================================================

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
    OpenAI = None  # OpenAI클래스

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
logger = logging.getLogger(__name__)  # 모듈로거

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")                                      # OpenAI인증키

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)  # OpenAI클라이언트
else:
    client = None  # OpenAI클라이언트

# 모델 설정
MODEL = os.getenv("SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano"))      # 뉴스요약모델명

# 기본값: 배치 요약 사용. 필요 시 SUMMARY_BATCH_MODE=false 로 단건 방식 사용 가능.
SUMMARY_BATCH_MODE = os.getenv("SUMMARY_BATCH_MODE", "true").lower() not in {"0", "false", "no"}  # 배치요약사용여부

# GPT-5 계열은 max_completion_tokens 안에 숨은 reasoning token도 포함된다.
# 기본 한도는 동적으로 더 작게 잡고, 빈 응답/JSON 실패 때만 더 크게 재시도한다.
# 첫 시도에서 reasoning token이 출력 한도를 모두 써버리면 같은 요청을 재시도하게 된다.
# 기본 한도를 넉넉히 둬서 2~3문장 요약에서도 불필요한 2회 호출을 줄인다.
# max_completion_tokens는 상한값이라 실제 비용은 사용된 토큰 기준으로만 발생한다.
SUMMARY_BATCH_COMPLETION_LIMIT = int(os.getenv("SUMMARY_BATCH_MAX_COMPLETION_TOKENS", "8000"))   # 배치요약응답토큰상한
SUMMARY_SINGLE_COMPLETION_LIMIT = int(os.getenv("SUMMARY_SINGLE_MAX_COMPLETION_TOKENS", "1600"))  # 단건요약응답토큰상한
SUMMARY_INPUT_CONTENT_LIMIT = int(os.getenv("SUMMARY_INPUT_CONTENT_CHARS", "900"))                # 기사별입력본문문자상한


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: response: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _message_content(response: Any) -> str:
    try:
        return _safe_text(response.choices[0].message.content)
    except Exception:
        return ""


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: response: Any
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _finish_reason(response: Any) -> str:
    try:
        return _safe_text(response.choices[0].finish_reason)
    except Exception:
        return ""


# [코드 이해 주석]
# - 역할: 요약 호출에서 GPT-5 계열의 숨은 reasoning token 소모를 줄이기 위한 설정.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: 없음
# - 리턴값: Dict[str, str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
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

    effort = str(  # effort
        os.getenv("SUMMARY_REASONING_EFFORT", os.getenv("OPENAI_REASONING_EFFORT", "minimal"))
        or ""
    ).strip().lower()

    if effort in {"", "none", "default", "off", "false", "0"}:
        return {}

    return {"reasoning_effort": effort}


# [코드 이해 주석]
# - 역할: Chat Completions 호출 wrapper.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: **kwargs: Any
# - 리턴값: 명시 타입은 없지만 처리 결과 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _create_chat_completion(**kwargs):
    """
    Chat Completions 호출 wrapper.
    OpenAI SDK 버전에 따라 신규 body 필드를 extra_body로 옮겨 호환 호출한다.
    """
    return create_openai_chat_completion(client, logger, **kwargs)


# [코드 이해 주석]
# - 역할: None 방지용 문자열 변환.
# - 호출하는 곳: summarizer._build_fallback_summary, summarizer._build_summary_result, summarizer._clip_text,
# summarizer._extract_json, summarizer._finish_reason, summarizer._message_content 외 2곳
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
# - 호출하는 곳: summarizer._build_summary_result
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
# - 역할: OpenAI 응답에서 JSON 파싱.
# - 호출하는 곳: summarizer.summarize_batch_with_llm
# - 파라미터: content: str
# - 리턴값: Dict 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _extract_json(content: str) -> Dict:
    """
    OpenAI 응답에서 JSON 파싱.
    원칙적으로 JSON만 오게 하지만 코드블록이나 앞뒤 설명이 섞이는 경우를 대비한다.
    """
    content = _safe_text(content)  # 본문

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()  # 본문

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")  # 시작값
        end = content.rfind("}")  # 종료값
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_batch_with_llm
# - 파라미터: result: Any
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _ensure_json_object(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {}


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_batch_with_llm
# - 파라미터: value: Any
# - 리턴값: List[Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _ensure_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 내부에서 사용할 출력 구조를 만드는 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: article: Dict, max_length: int = 220
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 내부용 문자열/dict 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def _build_fallback_summary(article: Dict, max_length: int = 220) -> str:
    content = _safe_text(article.get("content") or article.get("description"))  # 본문
    return content[:max_length] + "..." if len(content) > max_length else content


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_batch_with_llm
# - 파라미터: value: Any, limit: int
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _clip_text(value: Any, limit: int) -> str:
    text = _safe_text(value)  # 텍스트
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: max_length: int
# - 리턴값: int 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _summary_min_length(max_length: int) -> int:
    max_length = max(int(max_length or 220), 80)  # 최대length
    return min(max(int(max_length * 0.6), 120), max_length)


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_batch_with_llm
# - 파라미터: article_count: int, max_length: int
# - 리턴값: List[int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _batch_completion_limits(article_count: int, max_length: int) -> List[int]:
    article_count = max(int(article_count or 1), 1)  # 기사건수
    max_length = max(int(max_length or 180), 80)  # 최대length
    per_article_budget = max(300, min(max_length + 260, 760))  # 기사별기사예산
    first_limit = max(1600, 700 + article_count * per_article_budget)  # 1차상한상한

    if is_gpt5_model(MODEL):
        # GPT-5 계열은 reasoning token도 max_completion_tokens 안에 포함된다.
        # 첫 시도 한도가 너무 작으면 본문 없이 length로 끝나 같은 요청을 한 번 더 보내게 된다.
        # 상한만 넉넉히 잡고, reasoning_effort는 minimal로 낮춰 실제 사용 토큰을 줄인다.
        first_limit = max(first_limit, min(SUMMARY_BATCH_COMPLETION_LIMIT, 5200))  # 1차상한상한
        first_limit = min(first_limit, SUMMARY_BATCH_COMPLETION_LIMIT)  # 1차상한상한
        retry_limit = max(first_limit + 1200, 600 + article_count * 600)  # 재시도상한
        retry_limit = min(retry_limit, max(SUMMARY_BATCH_COMPLETION_LIMIT + 1600, first_limit))  # 재시도상한
        return [first_limit, retry_limit] if retry_limit > first_limit else [first_limit]

    return [min(first_limit, SUMMARY_BATCH_COMPLETION_LIMIT)]


# [코드 이해 주석]
# - 역할: 입력 데이터를 조합해 내부에서 사용할 출력 구조를 만드는 보조 함수입니다.
# - 호출하는 곳: summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: article: Dict, summary: str, tokens_used: int = 0, error: str = None
# - 리턴값: Dict 타입 값을 반환합니다.
# - 프로세스 흐름: 필요한 입력값을 안전하게 정리합니다 -> 내부용 문자열/dict 구조를 조립합니다 -> 완성된 결과를 반환합니다.
def _build_summary_result(article: Dict, summary: str, tokens_used: int = 0, error: str = None) -> Dict:
    # 이 함수의 결과 dict는 email_sender.py, issue_history.py, related_pages.py가 공통으로 읽는 표준 뉴스 요약 구조다.
    # 그래서 요약문뿐 아니라 그룹 메타와 관련보도 배열도 article에서 그대로 보존한다.
    title = _safe_text(article.get("title", "제목 없음"))           # 뉴스제목
    url = _safe_text(article.get("url", "#"))                       # 뉴스URL
    keyword = _safe_text(article.get("keyword", ""))                # 수집키워드
    description = _safe_text(article.get("description"))            # 뉴스설명
    content = _safe_text(article.get("content"))                    # 요약입력본문
    # 수집/그룹화 단계에 따라 published_at 또는 published_at_kst 중 하나만 있을 수 있다.
    # 메일에서 발생시간이 사라지지 않도록 둘 다 보존한다.
    published_at = _safe_text(article.get("published_at") or article.get("published_at_kst") or "")      # 표시용발행시각
    published_at_kst = _safe_text(article.get("published_at_kst") or article.get("published_at") or "")  # KST발행시각
    importance_score = _safe_int(article.get("importance_score", 3))                                      # 중요도점수
    source = (  # 출처
        _safe_text(article.get("source"))
        or _safe_text(article.get("press"))
        or _safe_text(article.get("publisher"))
        or _safe_text(article.get("media"))
        or "언론사 미상"
    )

    result = {  # 결과
        "title": title,                                             # 뉴스제목
        "summary": _safe_text(summary),                             # 최종요약문
        "url": url,                                                 # 뉴스URL
        "keyword": keyword,                                         # 수집키워드
        "description": description,                                 # 뉴스설명
        "content": content or description,                          # 히스토리비교용본문
        "group_id": article.get("group_id"),                        # 사건그룹ID
        "group_article_count": article.get("group_article_count"),  # 관련보도기사수
        "group_source_count": article.get("group_source_count"),    # 관련보도언론사수
        "group_keywords": article.get("group_keywords", []),        # 관련보도키워드목록
        "group_quality_flags": article.get("group_quality_flags", []),  # 그룹품질플래그
        "group_priority_score": article.get("group_priority_score"),    # 그룹로컬우선순위점수
        "group_sources": article.get("group_sources", []),          # 관련보도언론사목록
        "group_article_titles": article.get("group_article_titles", []),    # 관련보도제목목록
        "group_article_urls": article.get("group_article_urls", []),        # 관련보도URL목록
        "group_article_sources": article.get("group_article_sources", []),  # 관련보도언론사목록
        "published_at": published_at,                               # 표시용발행시각
        "published_at_kst": published_at_kst,                       # KST발행시각
        "importance_score": importance_score,                       # 중요도점수
        "source": source,                                           # 언론사명
        "tokens_used": int(tokens_used or 0),                       # 요약토큰수
    }

    if error:
        result["error"] = error  # 처리값

    return result


# [코드 이해 주석]
# - 역할: 단일 기사 요약.
# - 호출하는 곳: summarizer.summarize_batch
# - 파라미터: article: Dict, max_length: int = 220
# - 리턴값: Dict 타입 값을 반환합니다.
# - 프로세스 흐름: 기사 입력을 정리합니다 -> AI 요약 또는 fallback을 실행합니다 -> 표준 요약 dict를 반환합니다.
def summarize_article(article: Dict, max_length: int = 220) -> Dict:
    """
    단일 기사 요약.
    배치 요약 실패 시 fallback으로도 사용한다.
    """
    title = _safe_text(article.get("title", "제목 없음"))  # 제목
    content = _safe_text(article.get("content") or article.get("description"))  # 본문

    if client is None:
        logger.warning("⚠️ OpenAI 클라이언트가 없어 원문 설명을 fallback 요약으로 사용합니다.")
        return _build_summary_result(
            article=article,  # 기사
            summary=_build_fallback_summary(article, max_length=max_length),  # 요약
            tokens_used=0,  # 토큰수used
            error="OPENAI_API_KEY 없음 또는 OpenAI 클라이언트 초기화 실패"
        )

    try:
        min_length = _summary_min_length(max_length)  # 최소length
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

        response = _create_chat_completion(  # 응답
            model=MODEL,  # 모델
            messages=[  # messages
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

        summary = _message_content(response)  # 요약
        usage_info = record_openai_usage(  # usageinfo
            logger,
            "단건 뉴스 요약",
            MODEL,
            response.usage,
        )
        tokens_used = usage_info["total_tokens"]  # 토큰수used

        if not summary:
            raise ValueError(
                f"단건 요약 응답 본문이 비어 있습니다. finish_reason={_finish_reason(response)} "
                f"reasoning_tokens={usage_info.get('reasoning_tokens', 0)}"
            )

        logger.debug(f"✅ 단건 요약 완료: {len(summary)}자 / 토큰 {tokens_used}")

        return _build_summary_result(
            article=article,  # 기사
            summary=summary,  # 요약
            tokens_used=tokens_used,  # 토큰수used
        )

    except Exception as e:  # 예외객체
        logger.error(f"❌ 요약 실패: {e}")
        return _build_summary_result(
            article=article,  # 기사
            summary=_build_fallback_summary(article, max_length=max_length),  # 요약
            tokens_used=0,  # 토큰수used
            error=str(e)  # 오류
        )


# [코드 이해 주석]
# - 역할: 여러 기사를 한 번의 OpenAI 호출로 요약한다.
# - 호출하는 곳: summarizer.summarize_batch
# - 파라미터: articles: List[Dict], max_length: int = 220
# - 리턴값: List[Dict] 타입 값을 반환합니다.
# - 프로세스 흐름: 기사 입력을 정리합니다 -> AI 요약 또는 fallback을 실행합니다 -> 표준 요약 dict를 반환합니다.
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
                article=article,  # 기사
                summary=_build_fallback_summary(article, max_length=max_length),  # 요약
                tokens_used=0,  # 토큰수used
                error="OPENAI_API_KEY 없음 또는 OpenAI 클라이언트 초기화 실패"
            )
            for article in articles  # 기사
        ]

    # 1) 선별된 기사 dict를 OpenAI 프롬프트용 블록으로 바꾼다.
    #    입력 article은 main/news_selector가 넘긴 대표 기사이며, title/content/source/published_at만 요약 판단에 사용한다.
    #    content는 main.py에서 description으로 맞춰두었기 때문에, 원문 전문이 없어도 설명 기반 요약이 가능하다.
    news_blocks = []                                                # AI입력뉴스블록목록
    for idx, article in enumerate(articles, 1):  # 순번,기사
        title = _clip_text(article.get("title", "제목 없음"), 120)  # AI입력제목
        content = _clip_text(article.get("content") or article.get("description"), SUMMARY_INPUT_CONTENT_LIMIT)  # AI입력본문
        source = _safe_text(article.get("source")) or "언론사 미상"  # 언론사명
        published_at = _safe_text(article.get("published_at"))     # 발행시각

        news_blocks.append(
            f"""
[{idx}]
언론사: {source}
발행일: {published_at}
제목: {title}
내용: {content}
""".strip()
        )

    # 2) 요약 길이 범위를 계산하고, 모든 index가 반드시 돌아오도록 JSON 출력 규칙을 강하게 둔다.
    #    index를 쓰는 이유는 OpenAI 응답 순서가 흔들려도 원래 기사와 요약을 정확히 다시 매칭하기 위해서다.
    min_length = _summary_min_length(max_length)                   # 요약최소길이
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

    last_error = None                                              # 마지막요약실패예외
    parsed = None                                                  # 파싱된AI응답JSON
    tokens_used = 0                                                # 배치요약총토큰수

    # 3) GPT-5 계열은 max_completion_tokens 안에 reasoning token이 포함된다.
    #    첫 시도에서 본문이 비거나 JSON 파싱이 실패하면 더 큰 한도로 배치 1회만 재시도한다.
    completion_limits = _batch_completion_limits(len(articles), max_length)  # 시도별응답토큰상한목록

    # 4) 배치 요약은 여러 기사를 한 번에 처리하므로, 실패 가능성이 있는 JSON 파싱을 시도별로 분리한다.
    #    parsed가 생기면 루프를 끝내고, 끝까지 실패하면 summarize_batch()가 단건 요약 fallback으로 전환한다.
    for attempt_no, completion_limit in enumerate(completion_limits, 1):  # 시도번호,응답상한
        response = _create_chat_completion(                        # 배치요약AI응답
            model=MODEL,  # 모델
            messages=[  # messages
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

        content = _message_content(response)                       # AI응답본문
        usage_info = record_openai_usage(                          # 요약사용량통계
            logger,
            f"배치 뉴스 요약 시도 {attempt_no}",
            MODEL,
            response.usage,
        )
        tokens_used += usage_info["total_tokens"]  # 처리값

        if not content:
            last_error = ValueError(  # 마지막오류
                f"배치 요약 응답 본문이 비어 있습니다. "
                f"attempt={attempt_no}, finish_reason={_finish_reason(response)}, "
                f"reasoning_tokens={usage_info.get('reasoning_tokens', 0)}, "
                f"completion_limit={completion_limit}"
            )
            logger.warning("⚠️ %s", last_error)
            continue

        try:
            parsed = _ensure_json_object(_extract_json(content))  # parsed
            break
        except Exception as e:  # 예외객체
            last_error = e  # 마지막오류
            logger.warning(
                "⚠️ 배치 요약 JSON 파싱 실패: attempt=%s | finish_reason=%s | content_preview=%s",
                attempt_no,
                _finish_reason(response),
                content[:300],
            )

    if parsed is None:
        raise last_error or ValueError("배치 요약 JSON 파싱 실패")

    # 5) 응답 JSON을 index -> summary dict로 바꾼다.
    #    누락된 index가 있으면 아래 결과 조립 단계에서 해당 기사만 fallback 요약을 사용한다.
    by_index = {}                                                  # 기사index별요약문
    for item in _ensure_json_list(parsed.get("summaries")):  # 항목
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))  # 순번
        except Exception:
            continue
        summary_text = _safe_text(item.get("summary"))             # AI응답요약문
        if summary_text:
            by_index[idx] = summary_text  # 처리값

    if not by_index:
        raise ValueError("배치 요약 응답에 summaries가 없습니다.")

    # 6) 배치 호출 1회의 토큰을 각 기사 결과에 나눠서 싣는다.
    #    메일 대시보드는 뉴스별 tokens_used를 합산하므로, 총합이 실제 사용량과 맞아야 한다.
    per_article_tokens = int(tokens_used / max(len(articles), 1))  # 기사별배분토큰수

    # 7) 최종 summaries는 입력 articles와 같은 순서를 유지한다.
    #    이 순서가 main.py의 selected_news 순서이며, 그대로 메일 카드 순서가 된다.
    results = []                                                   # 최종요약결과목록
    for idx, article in enumerate(articles, 1):  # 순번,기사
        summary = by_index.get(idx) or _build_fallback_summary(article, max_length=max_length)  # 최종기사요약문
        results.append(
            _build_summary_result(
                article=article,  # 기사
                summary=summary,  # 요약
                tokens_used=per_article_tokens,  # 토큰수used
                error=None if idx in by_index else "배치 요약에서 해당 index 누락, fallback 사용"
            )
        )

    # 나눗셈 반올림으로 빠진 토큰은 첫 기사에 더해 총합이 맞게 한다.
    distributed = per_article_tokens * len(results)                # 기사별배분된토큰합계
    remainder = int(tokens_used or 0) - distributed                # 나눗셈후남은토큰수
    if results and remainder > 0:
        results[0]["tokens_used"] += remainder  # 처리값

    logger.debug(f"✅ 배치 요약 완료: {len(results)}개 / 토큰 {tokens_used}")

    return results


# [코드 이해 주석]
# - 역할: 여러 기사 일괄 요약.
# - 호출하는 곳: main.collect_select_and_summarize
# - 파라미터: articles: List[Dict], delay: float = 1.0, max_length: int = 220
# - 리턴값: List[Dict] 타입 값을 반환합니다.
# - 프로세스 흐름: 기사 입력을 정리합니다 -> AI 요약 또는 fallback을 실행합니다 -> 표준 요약 dict를 반환합니다.
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

    # 1) 기본 경로는 배치 요약이다.
    #    기사별 호출보다 빠르고 토큰 관리가 쉬워서, 정상 상황에서는 이 경로가 메일 전체 요약을 만든다.
    if SUMMARY_BATCH_MODE:
        try:
            summaries = summarize_batch_with_llm(articles, max_length=max_length)  # 배치요약결과목록
            total_tokens = sum(summary.get("tokens_used", 0) for summary in summaries)  # 배치요약총토큰수
            logger.info(f"✅ 뉴스 요약 완료: {len(summaries)}개 / 토큰 {total_tokens:,}")
            return summaries
        except Exception as e:  # 예외객체
            logger.error(f"❌ 배치 요약 실패, 단건 요약으로 전환: {e}")

    # 2) 배치 요약이 꺼져 있거나 실패하면 단건 요약으로 전환한다.
    #    한 기사 요약 실패가 전체 배치를 망가뜨리지 않도록, summarize_article() 내부에서도 fallback 요약을 만든다.
    summaries = []  # 요약목록
    total_tokens = 0  # 전체토큰수

    for i, article in enumerate(articles, 1):  # i,기사
        logger.debug(f"요약 진행: {i}/{len(articles)}")

        summary = summarize_article(article, max_length=max_length)  # 요약
        summaries.append(summary)

        total_tokens += summary.get("tokens_used", 0)  # 처리값

        if i < len(articles):
            time.sleep(delay)

    logger.info(f"✅ 뉴스 요약 완료: {len(summaries)}개 / 토큰 {total_tokens:,}")

    return summaries
