import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)


def get_ai_client(model: str, max_retries=2):
    # Select the Anthropic client by model prefix (mirrors llm.py:create_client()):
    #   bedrock/...   -> AWS Bedrock
    #   vertex_ai/... -> GCP Vertex
    #   else          -> direct Anthropic API (uses ANTHROPIC_API_KEY)
    if model and model.startswith("bedrock/"):
        return anthropic.AnthropicBedrock(max_retries=max_retries)
    if model and model.startswith("vertex_ai/"):
        return anthropic.AnthropicVertex(max_retries=max_retries)
    return anthropic.Anthropic(max_retries=max_retries)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        # force the model to call our function
        filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    # get_ai_client picked the client from the prefix; strip it for the actual call.
    model_name = filtered_kwargs.get("model", "")
    for _prefix in ("bedrock/", "vertex_ai/"):
        if model_name.startswith(_prefix):
            filtered_kwargs["model"] = model_name[len(_prefix) :]
            break

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    print(filtered_kwargs)

    if func_spec is not None:
        tool_blocks = [b for b in message.content if b.type == "tool_use"]
        assert (
            tool_blocks and tool_blocks[0].name == func_spec.name
        ), f"expected tool_use '{func_spec.name}', got: {message.content}"
        output: OutputType = tool_blocks[0].input
    elif "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
