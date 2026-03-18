import json
import logging
import os
import time
import threading

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")

# Global rate limiter for Copilot API
_rate_lock = threading.Lock()
_last_request_time = 0.0
_COPILOT_MIN_INTERVAL = 3.0  # minimum seconds between requests


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.PermissionDeniedError,  # Some APIs return 403 on quota exhaustion
)

def _is_copilot_mode() -> bool:
    """Check if we're using GitHub Copilot API."""
    return bool(os.environ.get("COPILOT_OAUTH_TOKEN"))


def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if _is_copilot_mode():
        from ai_scientist.copilot_auth import get_copilot_jwt, copilot_default_headers, COPILOT_API
        jwt = get_copilot_jwt()
        client = openai.OpenAI(
            api_key=jwt,
            base_url=COPILOT_API,
            default_headers=copilot_default_headers(),
            max_retries=max_retries,
        )
    elif model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    # Claude via OpenAI-compatible proxy requires at least one user message.
    # The proxy extracts system role into Anthropic's system parameter,
    # leaving no messages if only system was provided.
    if messages and not any(m.get("role") == "user" for m in messages):
        messages.append({"role": "user", "content": "Please proceed with the task described above."})

    # Claude requires max_tokens to be set explicitly
    model_name = filtered_kwargs.get("model", "")
    if "claude" in model_name and "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "", 1)

    t0 = time.time()

    # Copilot API rate limiting: space out requests
    if _is_copilot_mode():
        global _last_request_time
        with _rate_lock:
            elapsed = time.time() - _last_request_time
            if elapsed < _COPILOT_MIN_INTERVAL:
                wait = _COPILOT_MIN_INTERVAL - elapsed
                logger.debug(f"Rate limit sleep {wait:.1f}s")
                time.sleep(wait)
            _last_request_time = time.time()

    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # Copilot Claude may split content and tool_calls into separate choices.
    # Find the right choice depending on whether we expect a function call.
    choice = completion.choices[0]
    if func_spec is not None and not getattr(choice.message, 'tool_calls', None):
        # Look for a choice that has tool_calls
        for c in completion.choices:
            if getattr(c.message, 'tool_calls', None):
                choice = c
                break

    if func_spec is None:
        # For plain text, prefer the choice with content
        if not choice.message.content:
            for c in completion.choices:
                if c.message.content:
                    choice = c
                    break
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
