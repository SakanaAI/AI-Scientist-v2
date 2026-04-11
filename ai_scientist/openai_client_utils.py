import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import openai

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None


class OpenAICompatibleServerError(RuntimeError):
    """Transient gateway/server failure from an OpenAI-compatible endpoint."""


DEFAULT_OPENAI_TIMEOUT_SECONDS = 180.0


def _first_nonempty_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_flag(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _load_codex_auth_config() -> Dict[str, Any]:
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return {}
    try:
        return json.loads(auth_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _load_codex_base_url() -> Optional[str]:
    if tomllib is None:
        return None

    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return None

    try:
        with config_path.open("rb") as f:
            config = tomllib.load(f)
    except OSError:
        return None

    provider_name = config.get("model_provider")
    providers = config.get("model_providers", {})
    if not provider_name or not isinstance(providers, dict):
        return None

    provider_cfg = providers.get(provider_name)
    if not isinstance(provider_cfg, dict):
        return None

    base_url = provider_cfg.get("base_url")
    if isinstance(base_url, str) and base_url:
        return base_url.rstrip("/")
    return None


def get_openai_api_key() -> Optional[str]:
    return _first_nonempty_env(
        "AI_SCIENTIST_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ) or _load_codex_auth_config().get("OPENAI_API_KEY")


def get_openai_base_url() -> Optional[str]:
    base_url = _first_nonempty_env(
        "AI_SCIENTIST_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
    )
    if base_url:
        return base_url.rstrip("/")
    return _load_codex_base_url()


def get_openai_timeout_seconds() -> float:
    timeout = _env_float("AI_SCIENTIST_OPENAI_TIMEOUT_SECONDS")
    if timeout is None:
        timeout = _env_float("OPENAI_TIMEOUT_SECONDS")
    return timeout if timeout is not None else DEFAULT_OPENAI_TIMEOUT_SECONDS


def is_rightcode_codex_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    normalized = base_url.rstrip("/").lower()
    return "right.codes" in normalized and "/codex" in normalized


def should_use_responses_compat(base_url: Optional[str] = None) -> bool:
    forced = _env_flag("AI_SCIENTIST_OPENAI_USE_RESPONSES")
    if forced is not None:
        return forced
    return is_rightcode_codex_base_url(base_url)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {
                "text",
                "input_text",
                "output_text",
            }:
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def _convert_chat_content_item(item: Any, role: str) -> dict[str, Any]:
    text_type = "output_text" if role == "assistant" else "input_text"

    if isinstance(item, str):
        return {"type": text_type, "text": item}

    if not isinstance(item, dict):
        return {"type": text_type, "text": str(item)}

    item_type = item.get("type")
    if item_type in {"text", "input_text", "output_text"}:
        return {"type": text_type, "text": item.get("text", "")}

    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            detail = image_url.get("detail")
        else:
            url = image_url
            detail = None

        payload: dict[str, Any] = {"type": "input_image", "image_url": url}
        if detail:
            payload["detail"] = detail
        return payload

    if item_type == "input_image":
        return item

    return item


def _convert_chat_content(content: Any, role: str) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [_convert_chat_content_item(item, role) for item in content]
    return [_convert_chat_content_item(content, role)]


def _split_chat_messages_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[Optional[str], list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    input_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            text = _message_text(content).strip()
            if text:
                instructions_parts.append(text)
            continue

        input_messages.append(
            {
                "type": "message",
                "role": role,
                "content": _convert_chat_content(content, role),
            }
        )

    if not input_messages:
        input_messages = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Please continue."}],
            }
        ]

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_messages


def _flatten_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return {"type": "function", **tool["function"]}
    return tool


def _normalize_tool_choice(
    tool_choice: Any, tools: list[dict[str, Any]] | None
) -> Any:
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return "required" if tools else "auto"
    return tool_choice


def _parse_sse_events(raw: str) -> list[tuple[str, dict[str, Any] | str]]:
    events: list[tuple[str, dict[str, Any] | str]] = []
    event_name: Optional[str] = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is None:
            return
        payload = "\n".join(data_lines).strip()
        try:
            parsed: dict[str, Any] | str = json.loads(payload)
        except json.JSONDecodeError:
            parsed = payload
        events.append((event_name, parsed))
        event_name = None
        data_lines = []

    for line in raw.splitlines():
        if line.startswith("event: "):
            flush()
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
        elif not line.strip():
            flush()

    flush()
    return events


def _summarize_failed_response_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)

    response_id = payload.get("id")
    status = payload.get("status")
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    error_code = error.get("code")
    message = error.get("message")
    request_id = None
    if isinstance(message, str):
        match = re.search(
            r"request ID\s+([0-9A-Za-z-]{8,})",
            message,
            flags=re.IGNORECASE,
        )
        if match is not None:
            request_id = match.group(1)
        message = re.sub(r"\s+", " ", message).strip()
        if len(message) > 240:
            message = message[:237] + "..."

    summary = {
        "response_id": response_id,
        "status": status,
        "error_code": error_code,
        "request_id": request_id,
        "message": message,
    }
    return json.dumps(summary, ensure_ascii=True)


def _reconstruct_response_payload_from_sse(
    events: list[tuple[str, dict[str, Any] | str]],
) -> Optional[dict[str, Any]]:
    latest_response: Optional[dict[str, Any]] = None
    output_items_by_index: dict[int, dict[str, Any]] = {}
    function_arguments_by_item_id: dict[str, str] = {}
    content_parts_by_item_id: dict[str, dict[int, dict[str, Any]]] = {}
    saw_output_item_done = False

    for event_name, payload in events:
        if not isinstance(payload, dict):
            continue

        response = payload.get("response")
        if isinstance(response, dict):
            latest_response = dict(response)

        output_index = payload.get("output_index")
        item = payload.get("item")
        if isinstance(output_index, int) and isinstance(item, dict):
            output_items_by_index[output_index] = dict(item)
            if event_name == "response.output_item.done":
                saw_output_item_done = True

        if event_name in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            item_id = payload.get("item_id")
            if not isinstance(item_id, str):
                continue

            if event_name.endswith(".done") and isinstance(payload.get("arguments"), str):
                function_arguments_by_item_id[item_id] = payload["arguments"]
            elif event_name.endswith(".delta") and isinstance(payload.get("delta"), str):
                function_arguments_by_item_id[item_id] = (
                    function_arguments_by_item_id.get(item_id, "") + payload["delta"]
                )

        if event_name == "response.content_part.done":
            item_id = payload.get("item_id")
            content_index = payload.get("content_index")
            part = payload.get("part")
            if (
                isinstance(item_id, str)
                and isinstance(content_index, int)
                and isinstance(part, dict)
            ):
                content_parts_by_item_id.setdefault(item_id, {})[content_index] = dict(part)

        if event_name == "response.output_text.done":
            item_id = payload.get("item_id")
            content_index = payload.get("content_index")
            text = payload.get("text")
            if (
                isinstance(item_id, str)
                and isinstance(content_index, int)
                and isinstance(text, str)
            ):
                content_parts_by_item_id.setdefault(item_id, {})[content_index] = {
                    "type": "output_text",
                    "text": text,
                }

    has_meaningful_output = bool(
        output_items_by_index or function_arguments_by_item_id or content_parts_by_item_id
    )
    if latest_response is None and not has_meaningful_output:
        return None

    reconstructed_response = dict(latest_response or {})
    reconstructed_output: list[dict[str, Any]] = []

    for output_index in sorted(output_items_by_index):
        item = dict(output_items_by_index[output_index])
        item_id = item.get("id")
        if isinstance(item_id, str):
            arguments = function_arguments_by_item_id.get(item_id)
            if arguments and not item.get("arguments"):
                item["arguments"] = arguments

            content_parts = content_parts_by_item_id.get(item_id)
            if content_parts and not item.get("content"):
                item["content"] = [
                    content_parts[index] for index in sorted(content_parts)
                ]

        reconstructed_output.append(item)

    if reconstructed_output:
        reconstructed_response["output"] = reconstructed_output
        if saw_output_item_done and not reconstructed_response.get("status"):
            reconstructed_response["status"] = "completed"

    if not reconstructed_response.get("output") and not has_meaningful_output:
        return None

    return reconstructed_response


def _extract_response_payload(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, str):
        completed_response = None
        failed_response = None
        events = _parse_sse_events(raw_response)
        event_names: list[str] = []

        for event_name, payload in events:
            event_names.append(event_name)
            if event_name == "response.completed" and isinstance(payload, dict):
                completed_response = payload.get("response")
            elif event_name == "response.failed" and isinstance(payload, dict):
                failed_response = payload.get("response", payload)

        if completed_response is not None:
            return completed_response
        if failed_response is not None:
            error = failed_response.get("error") if isinstance(failed_response, dict) else None
            error_code = error.get("code") if isinstance(error, dict) else None
            failed_summary = _summarize_failed_response_payload(failed_response)
            if error_code == "server_error":
                raise OpenAICompatibleServerError(
                    f"Responses API returned a failed response: {failed_summary}"
                )
            raise RuntimeError(
                f"Responses API returned a failed response: {failed_summary}"
            )

        reconstructed_response = _reconstruct_response_payload_from_sse(events)
        if reconstructed_response is not None:
            return reconstructed_response

        raise RuntimeError(
            "Could not find response.completed in Responses API payload. "
            f"Observed events: {event_names or ['<none>']}"
        )

    if hasattr(raw_response, "model_dump"):
        payload = raw_response.model_dump()
        if isinstance(payload, dict):
            return payload

    if isinstance(raw_response, dict):
        return raw_response

    raise TypeError(f"Unsupported Responses API payload type: {type(raw_response)}")


def _extract_output_text(output: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "".join(texts)


def _extract_tool_calls(output: list[dict[str, Any]]) -> list[SimpleNamespace]:
    tool_calls: list[SimpleNamespace] = []
    for item in output:
        if item.get("type") != "function_call":
            continue
        tool_calls.append(
            SimpleNamespace(
                id=item.get("call_id"),
                type="function",
                function=SimpleNamespace(
                    name=item.get("name"),
                    arguments=item.get("arguments", "{}"),
                ),
            )
        )
    return tool_calls


def _usage_namespace(response_payload: dict[str, Any]) -> SimpleNamespace:
    usage = response_payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return SimpleNamespace(
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=input_details.get("cached_tokens", 0)
        ),
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=output_details.get("reasoning_tokens", 0)
        ),
    )


def _response_to_chat_completion(
    response_payload: dict[str, Any], fallback_model: str
) -> SimpleNamespace:
    output = response_payload.get("output") or []
    tool_calls = _extract_tool_calls(output)
    text = _extract_output_text(output)

    return SimpleNamespace(
        id=response_payload.get("id"),
        created=response_payload.get("created_at"),
        model=response_payload.get("model", fallback_model),
        system_fingerprint=response_payload.get("system_fingerprint"),
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls" if tool_calls else "stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls or None,
                ),
            )
        ],
        usage=_usage_namespace(response_payload),
    )


def _merge_chat_completions(completions: list[SimpleNamespace]) -> SimpleNamespace:
    if len(completions) == 1:
        return completions[0]

    total_prompt_tokens = sum(c.usage.prompt_tokens for c in completions)
    total_completion_tokens = sum(c.usage.completion_tokens for c in completions)
    total_tokens = sum(c.usage.total_tokens for c in completions)
    total_reasoning_tokens = sum(
        c.usage.completion_tokens_details.reasoning_tokens for c in completions
    )
    total_cached_tokens = sum(
        c.usage.prompt_tokens_details.cached_tokens for c in completions
    )

    choices = []
    for index, completion in enumerate(completions):
        choice = completion.choices[0]
        choices.append(
            SimpleNamespace(
                index=index,
                finish_reason=choice.finish_reason,
                message=choice.message,
            )
        )

    return SimpleNamespace(
        id=completions[0].id,
        created=completions[-1].created,
        model=completions[0].model,
        system_fingerprint=completions[0].system_fingerprint,
        choices=choices,
        usage=SimpleNamespace(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=total_cached_tokens),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=total_reasoning_tokens
            ),
        ),
    )


class ResponsesCompatOpenAIClient:
    def __init__(self, client: openai.OpenAI):
        self._client = client
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )
        self.responses = client.responses

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create_chat_completion(self, **kwargs: Any) -> SimpleNamespace:
        messages = kwargs.pop("messages")
        model = kwargs.pop("model")
        n = kwargs.pop("n", 1)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        max_tokens = kwargs.pop("max_tokens", None)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        # Unsupported chat-only parameters in the RightCode Responses shim.
        kwargs.pop("stream", None)
        kwargs.pop("stop", None)

        instructions, response_input = _split_chat_messages_for_responses(messages)

        response_kwargs: dict[str, Any] = {
            "model": model,
            "input": response_input,
        }
        if instructions:
            response_kwargs["instructions"] = instructions
        if max_completion_tokens is not None:
            response_kwargs["max_output_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            response_kwargs["max_output_tokens"] = max_tokens
        if reasoning_effort is not None:
            response_kwargs["reasoning"] = {"effort": reasoning_effort}
        if tools is not None:
            flattened_tools = [_flatten_tool(tool) for tool in tools]
            response_kwargs["tools"] = flattened_tools
            normalized_tool_choice = _normalize_tool_choice(tool_choice, flattened_tools)
            if normalized_tool_choice is not None:
                response_kwargs["tool_choice"] = normalized_tool_choice
        elif tool_choice is not None:
            response_kwargs["tool_choice"] = _normalize_tool_choice(tool_choice, None)

        for key in ("temperature", "top_p"):
            value = kwargs.pop(key, None)
            if value is not None:
                response_kwargs[key] = value

        # RightCode's Responses endpoint currently rejects seed.
        kwargs.pop("seed", None)

        if kwargs:
            response_kwargs.update(kwargs)

        completions = []
        for _ in range(max(1, int(n or 1))):
            raw_response = self._client.responses.create(**response_kwargs)
            payload = _extract_response_payload(raw_response)
            completions.append(_response_to_chat_completion(payload, fallback_model=model))

        return _merge_chat_completions(completions)


def create_openai_client(*, max_retries: Optional[int] = None) -> Any:
    kwargs: Dict[str, Any] = {}
    api_key = get_openai_api_key()
    base_url = get_openai_base_url()

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["timeout"] = get_openai_timeout_seconds()
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    client = openai.OpenAI(**kwargs)
    if should_use_responses_compat(base_url):
        return ResponsesCompatOpenAIClient(client)
    return client


def describe_openai_endpoint() -> str:
    base_url = get_openai_base_url()
    if base_url:
        return f"OpenAI-compatible API at {base_url}"
    return "OpenAI API"
