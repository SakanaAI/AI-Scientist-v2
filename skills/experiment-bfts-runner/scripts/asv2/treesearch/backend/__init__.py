from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def _normalize_model(model: str) -> tuple[str, str]:
    """
    Normalize model identifier and choose backend.
    Returns (backend, model_name).
    backend in {"anthropic", "openai"}.
    """
    if model.startswith("bedrock/") and "claude" in model:
        return "anthropic", model.split("/", 1)[1]
    if model.startswith("vertex_ai/") and "claude" in model:
        return "anthropic", model.split("/", 1)[1]
    if model.startswith("claude-"):
        return "anthropic", model
    return "openai", model


def get_ai_client(model: str, **model_kwargs):
    backend, model_name = _normalize_model(model)
    if backend == "anthropic":
        return backend_anthropic.get_ai_client(model=model_name, **model_kwargs)
    return backend_openai.get_ai_client(model=model_name, **model_kwargs)


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    backend, model_name = _normalize_model(model)

    model_kwargs = model_kwargs | {
        "model": model_name,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    if model_name.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    query_func = backend_anthropic.query if backend == "anthropic" else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
