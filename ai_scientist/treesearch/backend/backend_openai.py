import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from ai_scientist.openai_client_utils import (
    OpenAICompatibleServerError,
    create_openai_client,
)
from ai_scientist.openai_model_utils import normalize_openai_model_name
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    OpenAICompatibleServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    else:
        client = create_openai_client(max_retries=max_retries)
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
    if not messages:
        messages = [{"role": "user", "content": "Please continue."}]
    elif not any(message["role"] == "user" for message in messages):
        # Some OpenAI-compatible gateways reject system-only chat requests.
        messages = [{"role": "user", "content": messages[0]["content"]}]

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
        filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")
    else:
        filtered_kwargs["model"] = normalize_openai_model_name(filtered_kwargs["model"])

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
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
