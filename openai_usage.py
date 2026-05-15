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
import os
import re
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 기본 단가: USD / 1M tokens
# 필요하면 환경변수로 덮어쓴다.
DEFAULT_MODEL_PRICES: Dict[str, Dict[str, float]] = {
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
USAGE_TOTALS_BY_MODEL: Dict[str, Dict[str, float]] = {}


def normalize_model_env_key(model: str) -> str:
    """
    환경변수 키에 사용할 수 있도록 모델명을 정규화한다.

    예:
    - gpt-4o-mini -> GPT_4O_MINI
    - gpt-5.4-mini -> GPT_5_4_MINI
    """
    value = str(model or "unknown").upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "UNKNOWN"


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _get_attr_or_key(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _base_model_name(model: str) -> str:
    """
    날짜 스냅샷 모델명을 기본 모델명으로 변환한다.

    예:
    - gpt-4o-mini-2024-07-18 -> gpt-4o-mini
    - gpt-5-nano-2026-03-17 -> gpt-5-nano
    - gpt-5_4-mini-2026-03-17 -> gpt-5.4-mini
    """
    model = str(model or "").strip()
    model = model.replace("gpt-5_4", "gpt-5.4")
    model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    return model


def openai_token_limit_kwargs(model: str, limit: int) -> Dict[str, int]:
    """
    모델별 출력 토큰 제한 파라미터를 반환한다.

    - GPT-5 계열은 Chat Completions에서 max_tokens 대신 max_completion_tokens를 요구한다.
    - GPT-4o/GPT-4.1 계열은 max_tokens를 사용한다.
    """
    model_name = _base_model_name(model).lower()

    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": int(limit)}

    return {"max_tokens": int(limit)}




def openai_temperature_kwargs(model: str, temperature: float) -> Dict[str, float]:
    """
    모델별 temperature 파라미터를 반환한다.

    GPT-5 계열 Chat Completions 일부 모델은 temperature를 기본값 1만 허용한다.
    이 경우 temperature 파라미터를 아예 보내지 않아 400 오류를 방지한다.
    GPT-4o/GPT-4.1 계열은 기존처럼 지정값을 사용한다.
    """
    model_name = _base_model_name(model).lower()

    if model_name.startswith("gpt-5"):
        return {}

    return {"temperature": float(temperature)}




def is_gpt5_model(model: str) -> bool:
    """GPT-5 계열 여부를 반환한다."""
    return _base_model_name(model).lower().startswith("gpt-5")


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

    model_name = _base_model_name(model).lower()
    effort = str(os.getenv("OPENAI_REASONING_EFFORT", "none") or "").strip().lower()
    if effort in {"", "none", "default"}:
        return {}

    # gpt-5.4 계열은 minimal을 지원하지 않는다. 비용 절감 의도는 유지하되
    # API가 받는 가장 낮은 값인 none으로 보낸다.
    if effort == "minimal" and model_name.startswith(("gpt-5.4", "gpt-5.5")):
        effort = "none"

    return {"reasoning_effort": effort}


def openai_json_response_format_kwargs() -> Dict[str, Dict[str, str]]:
    """JSON만 받아야 하는 호출에서 JSON mode를 켠다."""
    return {"response_format": {"type": "json_object"}}


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
    model = str(model or "unknown").strip() or "unknown"
    base_model = _base_model_name(model)

    default = (
        DEFAULT_MODEL_PRICES.get(model)
        or DEFAULT_MODEL_PRICES.get(base_model)
        or {
            "input_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_INPUT_PRICE_PER_1M"), 0.0),
            "cached_input_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_CACHED_INPUT_PRICE_PER_1M"), 0.0),
            "output_per_1m": _to_float(os.getenv("OPENAI_DEFAULT_OUTPUT_PRICE_PER_1M"), 0.0),
        }
    )

    model_key = normalize_model_env_key(model)
    base_key = normalize_model_env_key(base_model)

    input_price = _to_float(
        os.getenv(f"OPENAI_PRICE_{model_key}_INPUT_PER_1M"),
        _to_float(os.getenv(f"OPENAI_PRICE_{base_key}_INPUT_PER_1M"), default["input_per_1m"]),
    )
    cached_input_price = _to_float(
        os.getenv(f"OPENAI_PRICE_{model_key}_CACHED_INPUT_PER_1M"),
        _to_float(
            os.getenv(f"OPENAI_PRICE_{base_key}_CACHED_INPUT_PER_1M"),
            default.get("cached_input_per_1m", input_price),
        ),
    )
    output_price = _to_float(
        os.getenv(f"OPENAI_PRICE_{model_key}_OUTPUT_PER_1M"),
        _to_float(os.getenv(f"OPENAI_PRICE_{base_key}_OUTPUT_PER_1M"), default["output_per_1m"]),
    )

    return {
        "input_per_1m": input_price,
        "cached_input_per_1m": cached_input_price,
        "output_per_1m": output_price,
    }


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

    prompt_tokens = int(_get_attr_or_key(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get_attr_or_key(usage, "completion_tokens", 0) or 0)
    total_tokens = int(_get_attr_or_key(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    prompt_details = _get_attr_or_key(usage, "prompt_tokens_details", None)
    cached_prompt_tokens = int(_get_attr_or_key(prompt_details, "cached_tokens", 0) or 0)

    # 일부 SDK/응답에서는 total만 있고 prompt/completion이 비어 있을 수 있다.
    if total_tokens and not prompt_tokens and not completion_tokens:
        prompt_tokens = total_tokens

    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    cached_prompt_tokens = max(0, min(cached_prompt_tokens, prompt_tokens))

    completion_details = _get_attr_or_key(usage, "completion_tokens_details", None)
    reasoning_tokens = int(_get_attr_or_key(completion_details, "reasoning_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def calculate_cost(model: str, usage_info: Dict[str, int]) -> Dict[str, float]:
    prices = get_model_prices(model)

    prompt_tokens = int(usage_info.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage_info.get("completion_tokens", 0) or 0)
    cached_prompt_tokens = int(usage_info.get("cached_prompt_tokens", 0) or 0)
    non_cached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)

    input_cost = non_cached_prompt_tokens / 1_000_000 * prices["input_per_1m"]
    cached_input_cost = cached_prompt_tokens / 1_000_000 * prices["cached_input_per_1m"]
    output_cost = completion_tokens / 1_000_000 * prices["output_per_1m"]
    total_cost = input_cost + cached_input_cost + output_cost

    return {
        "input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "input_price_per_1m": prices["input_per_1m"],
        "cached_input_price_per_1m": prices["cached_input_per_1m"],
        "output_price_per_1m": prices["output_per_1m"],
    }


def reset_openai_usage_totals() -> None:
    USAGE_TOTALS_BY_MODEL.clear()


def add_openai_usage(model: str, usage_info: Dict[str, int], cost_info: Dict[str, float]) -> None:
    model = str(model or "unknown").strip() or "unknown"
    if model not in USAGE_TOTALS_BY_MODEL:
        USAGE_TOTALS_BY_MODEL[model] = {
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

    item = USAGE_TOTALS_BY_MODEL[model]
    item["requests"] += 1
    item["prompt_tokens"] += int(usage_info.get("prompt_tokens", 0) or 0)
    item["completion_tokens"] += int(usage_info.get("completion_tokens", 0) or 0)
    item["cached_prompt_tokens"] += int(usage_info.get("cached_prompt_tokens", 0) or 0)
    item["reasoning_tokens"] += int(usage_info.get("reasoning_tokens", 0) or 0)
    item["total_tokens"] += int(usage_info.get("total_tokens", 0) or 0)
    item["input_cost_usd"] += float(cost_info.get("input_cost_usd", 0.0) or 0.0)
    item["cached_input_cost_usd"] += float(cost_info.get("cached_input_cost_usd", 0.0) or 0.0)
    item["output_cost_usd"] += float(cost_info.get("output_cost_usd", 0.0) or 0.0)
    item["total_cost_usd"] += float(cost_info.get("total_cost_usd", 0.0) or 0.0)


def get_openai_usage_totals() -> Dict[str, Any]:
    by_model = deepcopy(USAGE_TOTALS_BY_MODEL)
    grand_total = {
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

    for item in by_model.values():
        for key in grand_total:
            grand_total[key] += item.get(key, 0)

    return {
        "by_model": by_model,
        "grand_total": grand_total,
    }


def record_openai_usage(
    log: Optional[logging.Logger],
    label: str,
    model: str,
    usage: Any,
) -> Dict[str, Any]:
    """
    사용량 추출 → 비용 계산 → 누적 → 로그 출력을 한 번에 수행한다.
    """
    usage_info = extract_usage(usage)
    cost_info = calculate_cost(model, usage_info)
    add_openai_usage(model, usage_info, cost_info)

    if log is None:
        log = logger

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
        "model": model,
    }


def log_openai_usage_summary(log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    if log is None:
        log = logger

    totals = get_openai_usage_totals()
    by_model = totals["by_model"]
    grand_total = totals["grand_total"]

    if not by_model:
        log.info("🧾 OpenAI 사용량: 기록된 호출 없음")
        return totals

    for model, item in sorted(by_model.items()):
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
