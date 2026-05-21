# =============================================================================
# [파일 설명]
# - 수행 기능: OpenAI SDK 호출 파라미터 호환성, 토큰 사용량 추출, 비용 계산, 누적 사용량 로그를 담당합니다.
# - 프로세스: 모델별 옵션 정규화 -> SDK 호출 파라미터 보정 -> 사용량 추출 -> 비용 계산 -> 누적/로그 출력
# - 호출하는 곳: email_sender.py, main.py, news_selector.py, summarizer.py
# - 주요 파라미터/입력: OpenAI 모델명, SDK 응답 usage, 가격표, completion 생성 파라미터
# - 리턴값/출력: 호환 kwargs, usage/cost dict, 누적 사용량 요약 dict를 반환합니다.
# =============================================================================

"""
OpenAI 사용량/비용 로깅 유틸리티

목적:
- OpenAI 응답 usage에서 입력/출력/캐시 입력/총 토큰을 분리한다.
- 모델별 100만 토큰당 단가를 적용해 예상 비용을 계산한다.
- 실행 전체의 모델별 사용량과 비용을 누적해 main.py에서 요약 로그로 표시한다.

주의:
- 실제 청구 금액은 OpenAI 대시보드가 최종 기준이다.
- 여기의 비용은 response.usage 기준 추정값이다.
- 단가는 환경변수로 언제든 덮어쓸 수 있다.
"""

from __future__ import annotations

import logging
import inspect
import os
import re
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)  # 모듈로거

# 기본 단가: USD / 1M tokens
# 필요하면 환경변수로 덮어쓴다.
DEFAULT_MODEL_PRICES: Dict[str, Dict[str, float]] = {                 # 기본모델별백만토큰단가표
    # GPT-4o / GPT-4.1 계열
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 0.60,
    },
    "gpt-4o-mini-2024-07-18": {
        "input_per_1m": 0.15,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 0.60,
    },
    "gpt-4.1-nano": {
        "input_per_1m": 0.10,
        "cached_input_per_1m": 0.025,
        "output_per_1m": 0.40,
    },
    "gpt-4.1-mini": {
        "input_per_1m": 0.40,
        "cached_input_per_1m": 0.10,
        "output_per_1m": 1.60,
    },
    "gpt-4.1": {
        "input_per_1m": 2.00,
        "cached_input_per_1m": 0.50,
        "output_per_1m": 8.00,
    },

    # GPT-5 계열
    "gpt-5-nano": {
        "input_per_1m": 0.05,
        "cached_input_per_1m": 0.005,
        "output_per_1m": 0.40,
    },
    "gpt-5-mini": {
        "input_per_1m": 0.25,
        "cached_input_per_1m": 0.025,
        "output_per_1m": 2.00,
    },
    "gpt-5": {
        "input_per_1m": 1.25,
        "cached_input_per_1m": 0.125,
        "output_per_1m": 10.00,
    },
    "gpt-5-chat-latest": {
        "input_per_1m": 1.25,
        "cached_input_per_1m": 0.125,
        "output_per_1m": 10.00,
    },

    # GPT-5.4 계열
    "gpt-5.4-nano": {
        "input_per_1m": 0.20,
        "cached_input_per_1m": 0.02,
        "output_per_1m": 1.25,
    },
    "gpt-5.4-mini": {
        "input_per_1m": 0.75,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 4.50,
    },
    "gpt-5.4": {
        "input_per_1m": 2.50,
        "cached_input_per_1m": 0.25,
        "output_per_1m": 15.00,
    },

    # GPT-5.5 계열
    "gpt-5.5": {
        "input_per_1m": 5.00,
        "cached_input_per_1m": 0.50,
        "output_per_1m": 30.00,
    },
}

# 모델별 누적 사용량
USAGE_TOTALS_BY_MODEL: Dict[str, Dict[str, float]] = {}  # float

CHAT_COMPLETION_EXTRA_BODY_COMPAT_PARAMS = {  # extra_body호환파라미터목록
    "max_completion_tokens",
    "reasoning_effort",
    "verbosity",
}


# [코드 이해 주석]
# - 역할: 환경변수 키에 사용할 수 있도록 모델명을 정규화한다.
# - 호출하는 곳: openai_usage.get_model_prices
# - 파라미터: model: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 빈 값과 자료형을 보정합니다 -> 비교용 불필요 요소를 제거합니다 -> 표준화된 값을 반환합니다.
def normalize_model_env_key(model: str) -> str:
    """
    환경변수 키에 사용할 수 있도록 모델명을 정규화한다.

    예:
    - gpt-4o-mini -> GPT_4O_MINI
    - gpt-5.4-mini -> GPT_5_4_MINI
    """
    value = str(model or "unknown").upper()  # 값
    value = re.sub(r"[^A-Z0-9]+", "_", value)  # 값
    value = re.sub(r"_+", "_", value).strip("_")  # 값
    return value or "UNKNOWN"


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage.get_model_prices
# - 파라미터: value: Any, default: float
# - 리턴값: float 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage.extract_usage
# - 파라미터: obj: Any, name: str, default: Any = 0
# - 리턴값: Any 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _get_attr_or_key(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# [코드 이해 주석]
# - 역할: 날짜 스냅샷 모델명을 기본 모델명으로 변환한다.
# - 호출하는 곳: openai_usage.get_model_prices, openai_usage.is_gpt5_model, openai_usage.openai_reasoning_effort_kwargs,
# openai_usage.openai_temperature_kwargs, openai_usage.openai_token_limit_kwargs
# - 파라미터: model: str
# - 리턴값: str 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _base_model_name(model: str) -> str:
    """
    날짜 스냅샷 모델명을 기본 모델명으로 변환한다.

    예:
    - gpt-4o-mini-2024-07-18 -> gpt-4o-mini
    - gpt-5-nano-2026-03-17 -> gpt-5-nano
    - gpt-5_4-mini-2026-03-17 -> gpt-5.4-mini
    """
    model = str(model or "").strip()  # 모델
    model = model.replace("gpt-5_4", "gpt-5.4")  # 모델
    model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)  # 모델
    return model


# [코드 이해 주석]
# - 역할: 모델별 출력 토큰 제한 파라미터를 반환한다.
# - 호출하는 곳: email_sender.build_section_insights, news_selector.select_important_news_groups,
# summarizer.summarize_article, summarizer.summarize_batch_with_llm
# - 파라미터: model: str, limit: int
# - 리턴값: Dict[str, int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def openai_token_limit_kwargs(model: str, limit: int) -> Dict[str, int]:
    """
    모델별 출력 토큰 제한 파라미터를 반환한다.

    - GPT-5 계열은 Chat Completions에서 max_tokens 대신 max_completion_tokens를 요구한다.
    - GPT-4o/GPT-4.1 계열은 max_tokens를 사용한다.
    """
    model_name = _base_model_name(model).lower()  # 모델이름

    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": int(limit)}

    return {"max_tokens": int(limit)}




# [코드 이해 주석]
# - 역할: 모델별 temperature 파라미터를 반환한다.
# - 호출하는 곳: email_sender.build_section_insights, news_selector._deduplicate_by_llm_event_group,
# news_selector.select_important_news, news_selector.select_important_news_groups, summarizer.summarize_article,
# summarizer.summarize_batch_with_llm
# - 파라미터: model: str, temperature: float
# - 리턴값: Dict[str, float] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def openai_temperature_kwargs(model: str, temperature: float) -> Dict[str, float]:
    """
    모델별 temperature 파라미터를 반환한다.

    GPT-5 계열 Chat Completions 일부 모델은 temperature를 기본값 1만 허용한다.
    이 경우 temperature 파라미터를 아예 보내지 않아 400 오류를 방지한다.
    GPT-4o/GPT-4.1 계열은 기존처럼 지정값을 사용한다.
    """
    model_name = _base_model_name(model).lower()  # 모델이름

    if model_name.startswith("gpt-5"):
        return {}

    return {"temperature": float(temperature)}




# [코드 이해 주석]
# - 역할: GPT-5 계열 여부를 반환한다.
# - 호출하는 곳: email_sender._email_insight_reasoning_effort_kwargs, email_sender.build_section_insights,
# news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups, openai_usage.openai_reasoning_effort_kwargs 외 3곳
# - 파라미터: model: str
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 정규화합니다 -> 조건식을 평가합니다 -> True/False를 반환합니다.
def is_gpt5_model(model: str) -> bool:
    """GPT-5 계열 여부를 반환한다."""
    return _base_model_name(model).lower().startswith("gpt-5")


# [코드 이해 주석]
# - 역할: GPT-5 계열 Chat Completions의 reasoning effort를 낮춰 숨은 reasoning token 소모를 줄인다.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups
# - 파라미터: model: str
# - 리턴값: Dict[str, str] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def openai_reasoning_effort_kwargs(model: str) -> Dict[str, str]:
    """
    GPT-5 계열 Chat Completions의 reasoning effort를 낮춰 숨은 reasoning token 소모를 줄인다.

    - 기본값: none
    - 변경: OPENAI_REASONING_EFFORT=minimal 또는 low/medium/high
    - 빈 값, none, default이면 파라미터를 보내지 않는다.
    - openai==1.12.0 환경에서는 reasoning_effort 미지원 오류가 날 수 있어 기본 전송하지 않는다.
    """
    if not is_gpt5_model(model):
        return {}

    model_name = _base_model_name(model).lower()  # 모델이름
    effort = str(os.getenv("OPENAI_REASONING_EFFORT", "none") or "").strip().lower()  # effort
    if effort in {"", "none", "default"}:
        return {}

    # gpt-5.4 계열은 minimal을 지원하지 않는다. 비용 절감 의도는 유지하되
    # API가 받는 가장 낮은 값인 none으로 보낸다.
    if effort == "minimal" and model_name.startswith(("gpt-5.4", "gpt-5.5")):
        effort = "none"  # effort

    return {"reasoning_effort": effort}


# [코드 이해 주석]
# - 역할: JSON만 받아야 하는 호출에서 JSON mode를 켠다.
# - 호출하는 곳: news_selector._deduplicate_by_llm_event_group, news_selector.select_important_news,
# news_selector.select_important_news_groups, summarizer.summarize_batch_with_llm
# - 파라미터: 없음
# - 리턴값: Dict[str, Dict[str, str]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def openai_json_response_format_kwargs() -> Dict[str, Dict[str, str]]:
    """JSON만 받아야 하는 호출에서 JSON mode를 켠다."""
    return {"response_format": {"type": "json_object"}}


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage._prepare_chat_completion_kwargs_for_sdk
# - 파라미터: client: Any
# - 리턴값: Optional[Dict[str, inspect.Parameter]] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _chat_completion_create_params(client: Any) -> Optional[Dict[str, inspect.Parameter]]:
    try:
        return dict(inspect.signature(client.chat.completions.create).parameters)
    except Exception:
        return None


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage._prepare_chat_completion_kwargs_for_sdk
# - 파라미터: params: Optional[Dict[str, inspect.Parameter]]
# - 리턴값: bool 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _create_accepts_kwargs(params: Optional[Dict[str, inspect.Parameter]]) -> bool:
    if not params:
        return False
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage._prepare_chat_completion_kwargs_for_sdk,
# openai_usage._retry_chat_completion_after_unexpected_kwarg
# - 파라미터: kwargs: Dict[str, Any], extra_values: Dict[str, Any]
# - 리턴값: None 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _merge_extra_body(kwargs: Dict[str, Any], extra_values: Dict[str, Any]) -> None:
    if not extra_values:
        return

    current_extra_body = kwargs.get("extra_body")  # 현재extra_body
    if current_extra_body is None:
        extra_body = {}  # 추가본문
    else:
        try:
            extra_body = dict(current_extra_body)  # 추가본문
        except Exception:
            extra_body = {}  # 추가본문

    extra_body.update(extra_values)
    kwargs["extra_body"] = extra_body  # 처리값


# [코드 이해 주석]
# - 역할: 구버전 OpenAI SDK가 아직 정식 인자로 모르는 Chat Completions 필드를.
# - 호출하는 곳: openai_usage.create_chat_completion
# - 파라미터: client: Any, kwargs: Dict[str, Any]
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _prepare_chat_completion_kwargs_for_sdk(
    client: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    구버전 OpenAI SDK가 아직 정식 인자로 모르는 Chat Completions 필드를
    extra_body로 옮긴다.

    openai==1.12.x처럼 max_completion_tokens/reasoning_effort를 시그니처에서
    받지 않는 SDK도 extra_body는 요청 JSON에 병합하므로 GPT-5 계열 호출을
    유지할 수 있다.
    """
    # 1) 현재 설치된 OpenAI SDK의 create() 시그니처를 확인한다.
    #    requirements.txt의 openai==1.12.x처럼 오래된 SDK는 max_completion_tokens/reasoning_effort를 직접 인자로 받지 못한다.
    params = _chat_completion_create_params(client)  # 파라미터
    if not params or _create_accepts_kwargs(params):
        return kwargs

    # 2) SDK가 모르는 최신 body 필드를 moved_values로 빼둔다.
    #    이 값들은 요청에서 빠지면 모델 동작이 달라질 수 있으므로 버리지 않고 extra_body로 옮긴다.
    moved_values = {}  # 이동된파라미터값
    for name in CHAT_COMPLETION_EXTRA_BODY_COMPAT_PARAMS:  # 이름
        if name in kwargs and name not in params:
            moved_values[name] = kwargs.pop(name)  # 처리값

    # 3) extra_body를 지원하는 SDK라면 최신 필드를 요청 JSON body에 병합한다.
    #    이렇게 하면 Python 함수 시그니처는 통과하면서 서버에는 원래 의도한 필드가 전달된다.
    if moved_values and "extra_body" in params:
        _merge_extra_body(kwargs, moved_values)
        return kwargs

    # 아주 오래된 SDK가 extra_body도 지원하지 않으면 최후의 호환 처리만 한다.
    # GPT-5 모델에서는 서버가 max_tokens를 거절할 수 있으므로, 가능한 환경에서는
    # extra_body 지원 SDK를 쓰는 것이 가장 안전하다.
    if "max_completion_tokens" in moved_values and "max_tokens" in params:
        kwargs["max_tokens"] = moved_values["max_completion_tokens"]  # 처리값

    return kwargs


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage.create_chat_completion
# - 파라미터: create_fn: Any, kwargs: Dict[str, Any], unexpected_name: str, log: Optional[logging.Logger]
# - 리턴값: Any 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _retry_chat_completion_after_unexpected_kwarg(
    create_fn: Any,
    kwargs: Dict[str, Any],
    unexpected_name: str,
    log: Optional[logging.Logger],
) -> Any:
    retry_kwargs = dict(kwargs)  # retrykwargs

    if unexpected_name in retry_kwargs:
        value = retry_kwargs.pop(unexpected_name)  # 값
        if "extra_body" in _chat_completion_create_params_from_fn(create_fn):
            _merge_extra_body(retry_kwargs, {unexpected_name: value})
            if log is not None:
                log.warning(
                    "⚠️ 현재 OpenAI SDK가 %s 인자를 직접 지원하지 않아 extra_body로 재시도합니다.",
                    unexpected_name,
                )
            return create_fn(**retry_kwargs)

        if unexpected_name == "max_completion_tokens":
            retry_kwargs["max_tokens"] = value  # 처리값
            if log is not None:
                log.warning(
                    "⚠️ 현재 OpenAI SDK가 max_completion_tokens를 지원하지 않아 max_tokens로 재시도합니다."
                )
            return create_fn(**retry_kwargs)

    if unexpected_name == "extra_body":
        retry_kwargs.pop("extra_body", None)
        if log is not None:
            log.warning("⚠️ 현재 OpenAI SDK가 extra_body를 지원하지 않아 해당 옵션 없이 재시도합니다.")
        return create_fn(**retry_kwargs)

    raise TypeError(f"unexpected keyword argument {unexpected_name}")


# [코드 이해 주석]
# - 역할: 모듈의 처리 흐름을 나누어 읽기 쉽게 만든 보조 함수입니다.
# - 호출하는 곳: openai_usage._retry_chat_completion_after_unexpected_kwarg
# - 파라미터: create_fn: Any
# - 리턴값: Dict[str, inspect.Parameter] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def _chat_completion_create_params_from_fn(create_fn: Any) -> Dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(create_fn).parameters)
    except Exception:
        return {}


# [코드 이해 주석]
# - 역할: Chat Completions 호출 호환 wrapper.
# - 호출하는 곳: email_sender._create_chat_completion_for_email_insight, news_selector._deduplicate_by_llm_event_group,
# news_selector.select_important_news, news_selector.select_important_news_groups, summarizer._create_chat_completion
# - 파라미터: client: Any, log: Optional[logging.Logger] = None, **kwargs: Any
# - 리턴값: Any 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def create_chat_completion(client: Any, log: Optional[logging.Logger] = None, **kwargs: Any) -> Any:
    """
    Chat Completions 호출 호환 wrapper.

    최신 SDK에서는 인자를 그대로 보내고, 구버전 SDK에서는 GPT-5 계열에 필요한
    max_completion_tokens/reasoning_effort 같은 신규 body 필드를 extra_body로 옮겨
    TypeError 없이 호출한다.
    """
    if client is None:
        raise ValueError("OpenAI 클라이언트를 초기화할 수 없습니다.")

    # 1) 호출 직전에 SDK 호환 파라미터로 변환한다.
    #    각 모듈(news_selector/summarizer/email_sender)은 모델 기준 kwargs만 만들고,
    #    SDK 버전 차이는 이 wrapper 한 곳에서 흡수한다.
    create_fn = client.chat.completions.create  # createfn
    prepared_kwargs = _prepare_chat_completion_kwargs_for_sdk(client, dict(kwargs))  # 호환보정키워드인자

    try:
        # 2) 정상 SDK에서는 변환된 kwargs로 바로 호출한다.
        return create_fn(**prepared_kwargs)
    except TypeError as e:  # 예외객체
        # 3) 시그니처 검사로 잡지 못한 unexpected keyword가 있으면 한 번 더 호환 재시도한다.
        #    운영 중 SDK 버전이 바뀌어도 뉴스 선별/요약 전체가 중단되지 않게 하는 방어막이다.
        message = str(e)  # 메시지
        match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", message)  # match
        if match:
            return _retry_chat_completion_after_unexpected_kwarg(
                create_fn=create_fn,  # createfn
                kwargs=prepared_kwargs,  # 키워드인자
                unexpected_name=match.group(1),  # unexpected이름
                log=log,  # log
            )
        raise


# [코드 이해 주석]
# - 역할: 모델별 단가를 반환한다.
# - 호출하는 곳: openai_usage.calculate_cost
# - 파라미터: model: str
# - 리턴값: Dict[str, float] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_model_prices(model: str) -> Dict[str, float]:
    """
    모델별 단가를 반환한다.

    환경변수 우선순위:
    1. OPENAI_PRICE_{MODEL_KEY}_INPUT_PER_1M
       OPENAI_PRICE_{MODEL_KEY}_CACHED_INPUT_PER_1M
       OPENAI_PRICE_{MODEL_KEY}_OUTPUT_PER_1M
    2. OPENAI_PRICE_{BASE_MODEL_KEY}_INPUT_PER_1M ...
    3. DEFAULT_MODEL_PRICES
    4. 알 수 없는 모델이면 OPENAI_DEFAULT_* 또는 0
    """
    model = str(model or "unknown").strip() or "unknown"              # 요청모델명
    base_model = _base_model_name(model)                              # 날짜버전제거기본모델명

    # 단가는 환경변수 → 기본 모델명 → 내장 기본값 순으로 찾는다.
    # 운영 중 모델명이 바뀌어도 가격 env만 추가하면 비용 로그를 바로 보정할 수 있게 하기 위한 구조다.
    default = (                                                       # 기본단가정보
        DEFAULT_MODEL_PRICES.get(model)
        or DEFAULT_MODEL_PRICES.get(base_model)
        or {
            "input_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_INPUT_PRICE_PER_1M"), 0.0),
            "cached_input_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_CACHED_INPUT_PRICE_PER_1M"), 0.0),
            "output_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_OUTPUT_PRICE_PER_1M"), 0.0),
        }
    )

    model_key = normalize_model_env_key(model)                        # 환경변수용모델키
    base_key = normalize_model_env_key(base_model)                    # 환경변수용기본모델키

    input_price = _to_float(                                          # 입력토큰단가
        os.getenv(f"OPENAI_PRICE_{model_key}_INPUT_PER_1M"),
        _to_float(os.getenv(f"OPENAI_PRICE_{base_key}_INPUT_PER_1M"), default["input_per_1m"]),
    )
    cached_input_price = _to_float(                                   # 캐시입력토큰단가
        os.getenv(f"OPENAI_PRICE_{model_key}_CACHED_INPUT_PER_1M"),
        _to_float(
            os.getenv(f"OPENAI_PRICE_{base_key}_CACHED_INPUT_PER_1M"),
            default.get("cached_input_per_1m", input_price),
        ),
    )
    output_price = _to_float(                                         # 출력토큰단가
        os.getenv(f"OPENAI_PRICE_{model_key}_OUTPUT_PER_1M"),
        _to_float(os.getenv(f"OPENAI_PRICE_{base_key}_OUTPUT_PER_1M"), default["output_per_1m"]),
    )

    return {
        "input_per_1m": input_price,                                  # 입력토큰백만개당단가
        "cached_input_per_1m": cached_input_price,                    # 캐시입력토큰백만개당단가
        "output_per_1m": output_price,                                # 출력토큰백만개당단가
    }


# [코드 이해 주석]
# - 역할: OpenAI response.usage에서 입력/출력/총 토큰을 안전하게 추출한다.
# - 호출하는 곳: openai_usage.record_openai_usage
# - 파라미터: usage: Any
# - 리턴값: Dict[str, int] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 텍스트/객체를 검사합니다 -> 필요한 부분만 골라냅니다 -> 중복/빈 값을 정리해 반환합니다.
def extract_usage(usage: Any) -> Dict[str, int]:
    """
    OpenAI response.usage에서 입력/출력/총 토큰을 안전하게 추출한다.
    """
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = int(_get_attr_or_key(usage, "prompt_tokens", 0) or 0)      # 입력토큰수
    completion_tokens = int(_get_attr_or_key(usage, "completion_tokens", 0) or 0)  # 출력토큰수
    total_tokens = int(_get_attr_or_key(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)  # 전체토큰수

    prompt_details = _get_attr_or_key(usage, "prompt_tokens_details", None)   # 입력토큰세부정보
    cached_prompt_tokens = int(_get_attr_or_key(prompt_details, "cached_tokens", 0) or 0)  # 캐시입력토큰수

    # 일부 SDK/응답에서는 total만 있고 prompt/completion이 비어 있을 수 있다.
    if total_tokens and not prompt_tokens and not completion_tokens:
        prompt_tokens = total_tokens  # prompt토큰수

    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens  # 전체토큰수

    cached_prompt_tokens = max(0, min(cached_prompt_tokens, prompt_tokens))  # 캐시된prompt토큰수

    completion_details = _get_attr_or_key(usage, "completion_tokens_details", None)  # 출력토큰세부정보
    reasoning_tokens = int(_get_attr_or_key(completion_details, "reasoning_tokens", 0) or 0)  # 추론토큰수

    return {
        "prompt_tokens": prompt_tokens,                              # 입력토큰수
        "completion_tokens": completion_tokens,                      # 출력토큰수
        "cached_prompt_tokens": cached_prompt_tokens,                # 캐시입력토큰수
        "reasoning_tokens": reasoning_tokens,                        # 추론토큰수
        "total_tokens": total_tokens,                                # 전체토큰수
    }


# [코드 이해 주석]
# - 역할: 입력 숫자와 가격표를 바탕으로 비용이나 점수를 계산합니다.
# - 호출하는 곳: openai_usage.record_openai_usage
# - 파라미터: model: str, usage_info: Dict[str, int]
# - 리턴값: Dict[str, float] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def calculate_cost(model: str, usage_info: Dict[str, int]) -> Dict[str, float]:
    prices = get_model_prices(model)                                 # 모델별단가정보

    prompt_tokens = int(usage_info.get("prompt_tokens", 0) or 0)      # 입력토큰수
    completion_tokens = int(usage_info.get("completion_tokens", 0) or 0)  # 출력토큰수
    cached_prompt_tokens = int(usage_info.get("cached_prompt_tokens", 0) or 0)  # 캐시입력토큰수
    non_cached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)     # 일반입력토큰수

    input_cost = non_cached_prompt_tokens / 1_000_000 * prices["input_per_1m"]              # 일반입력예상비용
    cached_input_cost = cached_prompt_tokens / 1_000_000 * prices["cached_input_per_1m"]     # 캐시입력예상비용
    output_cost = completion_tokens / 1_000_000 * prices["output_per_1m"]                    # 출력예상비용
    total_cost = input_cost + cached_input_cost + output_cost                                # 총예상비용

    return {
        "input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "input_price_per_1m": prices["input_per_1m"],
        "cached_input_price_per_1m": prices["cached_input_per_1m"],
        "output_price_per_1m": prices["output_per_1m"],
    }


# [코드 이해 주석]
# - 역할: 누적 통계나 상태 값을 초기 상태로 되돌립니다.
# - 호출하는 곳: main.main
# - 파라미터: 없음
# - 리턴값: None 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def reset_openai_usage_totals() -> None:
    USAGE_TOTALS_BY_MODEL.clear()


# [코드 이해 주석]
# - 역할: 누적 통계나 그룹에 새 값을 더합니다.
# - 호출하는 곳: openai_usage.record_openai_usage
# - 파라미터: model: str, usage_info: Dict[str, int], cost_info: Dict[str, float]
# - 리턴값: None 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def add_openai_usage(model: str, usage_info: Dict[str, int], cost_info: Dict[str, float]) -> None:
    model = str(model or "unknown").strip() or "unknown"            # 누적대상모델명
    if model not in USAGE_TOTALS_BY_MODEL:
        # 모델별 누적 dict는 main.py 마지막 log_openai_usage_summary()에서 그대로 읽는다.
        # 새 모델이 처음 등장하면 모든 누적 항목을 0으로 만들어 이후 += 연산이 단순해지게 한다.
        USAGE_TOTALS_BY_MODEL[model] = {  # 처리값
            "requests": 0,                                         # 모델호출횟수
            "prompt_tokens": 0,                                    # 누적입력토큰수
            "completion_tokens": 0,                                # 누적출력토큰수
            "cached_prompt_tokens": 0,                             # 누적캐시입력토큰수
            "reasoning_tokens": 0,                                 # 누적추론토큰수
            "total_tokens": 0,                                     # 누적전체토큰수
            "input_cost_usd": 0.0,                                 # 누적일반입력비용
            "cached_input_cost_usd": 0.0,                          # 누적캐시입력비용
            "output_cost_usd": 0.0,                                # 누적출력비용
            "total_cost_usd": 0.0,                                 # 누적총비용
        }

    item = USAGE_TOTALS_BY_MODEL[model]                            # 모델별누적사용량
    item["requests"] += 1  # 처리값
    item["prompt_tokens"] += int(usage_info.get("prompt_tokens", 0) or 0)  # 처리값
    item["completion_tokens"] += int(usage_info.get("completion_tokens", 0) or 0)  # 처리값
    item["cached_prompt_tokens"] += int(usage_info.get("cached_prompt_tokens", 0) or 0)  # 처리값
    item["reasoning_tokens"] += int(usage_info.get("reasoning_tokens", 0) or 0)  # 처리값
    item["total_tokens"] += int(usage_info.get("total_tokens", 0) or 0)  # 처리값
    item["input_cost_usd"] += float(cost_info.get("input_cost_usd", 0.0) or 0.0)  # 처리값
    item["cached_input_cost_usd"] += float(cost_info.get("cached_input_cost_usd", 0.0) or 0.0)  # 처리값
    item["output_cost_usd"] += float(cost_info.get("output_cost_usd", 0.0) or 0.0)  # 처리값
    item["total_cost_usd"] += float(cost_info.get("total_cost_usd", 0.0) or 0.0)  # 처리값


# [코드 이해 주석]
# - 역할: 현재 상태, 설정, 입력 dict에서 필요한 값을 조회합니다.
# - 호출하는 곳: openai_usage.log_openai_usage_summary
# - 파라미터: 없음
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력 dict 또는 전역 상태를 확인합니다 -> 기본값을 보정합니다 -> 호출자가 바로 쓸 값을 반환합니다.
def get_openai_usage_totals() -> Dict[str, Any]:
    by_model = deepcopy(USAGE_TOTALS_BY_MODEL)  # 모델별사용량
    grand_total = {  # 전체합계사용량
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "cached_input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }

    for item in by_model.values():  # 항목
        for key in grand_total:  # 키
            grand_total[key] += item.get(key, 0)  # 처리값

    return {
        "by_model": by_model,
        "grand_total": grand_total,
    }


# [코드 이해 주석]
# - 역할: 사용량 추출 → 비용 계산 → 누적 → 로그 출력을 한 번에 수행한다.
# - 호출하는 곳: email_sender.build_section_insights, news_selector._deduplicate_by_llm_event_group,
# news_selector.select_important_news, news_selector.select_important_news_groups, summarizer.summarize_article,
# summarizer.summarize_batch_with_llm
# - 파라미터: log: Optional[logging.Logger], label: str, model: str, usage: Any
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def record_openai_usage(
    log: Optional[logging.Logger],
    label: str,
    model: str,
    usage: Any,
) -> Dict[str, Any]:
    """
    사용량 추출 → 비용 계산 → 누적 → 로그 출력을 한 번에 수행한다.
    """
    # 각 호출부가 usage를 따로 해석하지 않도록, 토큰 추출/비용 계산/전역 누적을 이 함수에서 한 번에 처리한다.
    # 반환 dict는 호출부가 selection_tokens, summary_tokens 같은 섹션별 통계에 다시 반영할 때 사용한다.
    usage_info = extract_usage(usage)                              # 추출된토큰사용량
    cost_info = calculate_cost(model, usage_info)                  # 계산된예상비용
    add_openai_usage(model, usage_info, cost_info)

    if log is None:
        log = logger  # log

    log.info(
        "🧾 %s: model=%s / tokens input=%s, output=%s, reasoning=%s, total=%s / cost=$%.6f",
        label,
        model,
        f"{usage_info['prompt_tokens']:,}",
        f"{usage_info['completion_tokens']:,}",
        f"{usage_info.get('reasoning_tokens', 0):,}",
        f"{usage_info['total_tokens']:,}",
        cost_info["total_cost_usd"],
    )

    return {
        **usage_info,
        **cost_info,
        "model": model,                                            # 사용모델명
    }


# [코드 이해 주석]
# - 역할: 운영자가 확인할 수 있도록 처리 결과를 로그로 출력합니다.
# - 호출하는 곳: main.main
# - 파라미터: log: Optional[logging.Logger] = None
# - 리턴값: Dict[str, Any] 타입 값을 반환합니다.
# - 프로세스 흐름: 입력값을 확인합니다 -> 핵심 처리 로직을 수행합니다 -> 결과를 반환하거나 필요한 부수 효과를 남깁니다.
def log_openai_usage_summary(log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    if log is None:
        log = logger  # log

    totals = get_openai_usage_totals()  # totals
    by_model = totals["by_model"]  # 모델별사용량
    grand_total = totals["grand_total"]  # 전체합계사용량

    if not by_model:
        log.info("🧾 OpenAI 사용량: 기록된 호출 없음")
        return totals

    for model, item in sorted(by_model.items()):  # 모델,항목
        log.info(
            "💰 OpenAI 모델별: model=%s / requests=%s / tokens input=%s, output=%s, reasoning=%s, total=%s / cost=$%.6f",
            model,
            int(item.get("requests", 0)),
            f"{int(item.get('prompt_tokens', 0)):,}",
            f"{int(item.get('completion_tokens', 0)):,}",
            f"{int(item.get('reasoning_tokens', 0)):,}",
            f"{int(item.get('total_tokens', 0)):,}",
            float(item.get("total_cost_usd", 0.0)),
        )

    log.info(
        "💰 OpenAI 전체: requests=%s / tokens input=%s, output=%s, reasoning=%s, total=%s / cost=$%.6f",
        int(grand_total.get("requests", 0)),
        f"{int(grand_total.get('prompt_tokens', 0)):,}",
        f"{int(grand_total.get('completion_tokens', 0)):,}",
        f"{int(grand_total.get('reasoning_tokens', 0)):,}",
        f"{int(grand_total.get('total_tokens', 0)):,}",
        float(grand_total.get("total_cost_usd", 0.0)),
    )

    return totals
