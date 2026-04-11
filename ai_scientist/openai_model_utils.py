import re
from typing import Any, Dict, Optional, Tuple


GPT5_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
GPT5_CHAT_MODEL_PATTERN = re.compile(
    r"^gpt-5(?:\.\d+)?(?:-codex)?(?:-mini|-nano|-max)?(?:-\d{4}-\d{2}-\d{2})?$"
)


def split_reasoning_model_alias(model: str) -> Tuple[str, Optional[str]]:
    base, sep, suffix = model.rpartition("-")
    if sep and suffix in GPT5_REASONING_EFFORTS and GPT5_CHAT_MODEL_PATTERN.match(base):
        return base, suffix
    return model, None


def normalize_openai_model_name(model: str) -> str:
    return split_reasoning_model_alias(model)[0]


def is_supported_gpt5_chat_model(model: str) -> bool:
    return GPT5_CHAT_MODEL_PATTERN.match(normalize_openai_model_name(model)) is not None


def resolve_reasoning_effort(
    model: str, reasoning_effort: Optional[str] = None
) -> Optional[str]:
    _, alias_effort = split_reasoning_model_alias(model)
    return alias_effort or reasoning_effort


def build_openai_chat_completion_kwargs(
    model: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Any = None,
    seed: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    api_model = normalize_openai_model_name(model)
    kwargs: Dict[str, Any] = {"model": api_model}

    if is_supported_gpt5_chat_model(api_model):
        effort = resolve_reasoning_effort(model, reasoning_effort)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if effort in (None, "none") and temperature is not None:
            kwargs["temperature"] = temperature
    elif api_model.startswith(("o1", "o3")):
        kwargs["reasoning_effort"] = reasoning_effort or "high"
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

    if n is not None:
        kwargs["n"] = n
    if stop is not None:
        kwargs["stop"] = stop
    if seed is not None:
        kwargs["seed"] = seed

    return kwargs
