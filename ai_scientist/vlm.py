import base64
from typing import Any
import re
import json
import os
from PIL import Image

from ai_scientist.agent_backend import AgentBackend

MAX_NUM_TOKENS = 4096

# v3: Backend mode. Set AI_SCIENTIST_BACKEND=legacy to use old SDK clients.
_BACKEND_MODE = os.environ.get("AI_SCIENTIST_BACKEND", "agent")

AVAILABLE_VLMS = None  # Set below based on backend mode

if _BACKEND_MODE == "legacy":
    import backoff
    import openai
    from ai_scientist.utils.token_tracker import track_token_usage

    AVAILABLE_VLMS = [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "o3-mini",
        "ollama/llama4:16x17b",
        "ollama/mistral-small3.2:24b",
        "ollama/qwen2.5vl:32b",
        "ollama/z-uo/qwen2.5vl_tools:32b",
    ]
else:
    AVAILABLE_VLMS = None


# Module-level agent backend (lazily initialized, v3 only)
_agent_backend: AgentBackend | None = None


def _get_agent_backend() -> AgentBackend:
    global _agent_backend
    if _agent_backend is None:
        _agent_backend = AgentBackend()
    return _agent_backend


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def create_client(model: str) -> tuple[Any, str]:
    """Create client for vision-language model.

    In v3 (default): returns (AgentBackend, model_str). No API keys needed.
    In legacy mode: returns (OpenAI_client, model_str) as before.
    """
    if _BACKEND_MODE == "legacy":
        return _create_client_legacy(model)

    print(f"Using agent backend (v3 mode) for VLM model '{model}'.")
    return _get_agent_backend(), model


def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from vision-language model.

    In v3: routes through AgentBackend with image file paths.
    """
    if _BACKEND_MODE == "legacy":
        return _get_response_from_vlm_legacy(
            msg, image_paths, client, model, system_message,
            print_debug=print_debug, msg_history=msg_history,
            temperature=temperature, max_images=max_images,
        )

    if msg_history is None:
        msg_history = []

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Limit images
    images_to_send = image_paths[:max_images]

    # Build user message with image references
    backend: AgentBackend = client
    content = backend.query(
        system_message=system_message,
        user_message=msg,
        images=images_to_send,
    )

    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history):
            print(f'{j}, {m["role"]}: {m["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return content, new_msg_history


def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from VLM for the same input.

    In v3: N sequential agent invocations with image support.
    """
    if _BACKEND_MODE == "legacy":
        return _get_batch_responses_from_vlm_legacy(
            msg, image_paths, client, model, system_message,
            print_debug=print_debug, msg_history=msg_history,
            temperature=temperature, n_responses=n_responses,
            max_images=max_images,
        )

    contents = []
    histories = []
    for _ in range(n_responses):
        c, hist = get_response_from_vlm(
            msg, image_paths, client, model, system_message,
            print_debug=False, msg_history=msg_history,
            temperature=temperature, max_images=max_images,
        )
        contents.append(c)
        histories.append(hist)

    if print_debug and histories:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, m in enumerate(histories[0]):
            print(f'{j}, {m["role"]}: {m["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, histories


def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a raw LLM call (text-only, used by VLM pipeline).

    In v3: routes through AgentBackend.
    """
    if _BACKEND_MODE == "legacy":
        return _make_llm_call_legacy(client, model, temperature, system_message, prompt)

    user_parts = []
    for entry in prompt:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if isinstance(content, list):
            # Extract text from multimodal content blocks
            text_parts = [
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts)
        user_parts.append(f"[{role}]: {content}")
    user_message = "\n".join(user_parts)

    backend: AgentBackend = client
    return backend.query(
        system_message=system_message,
        user_message=user_message,
    )


def make_vlm_call(client, model, temperature, system_message, prompt):
    """Make a VLM call with images.

    In v3: routes through AgentBackend with image support.
    """
    if _BACKEND_MODE == "legacy":
        return _make_vlm_call_legacy(client, model, temperature, system_message, prompt)

    # Extract text and image paths from multimodal prompt
    text_parts = []
    image_paths = []
    for entry in prompt:
        content = entry.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        # Image is base64 encoded, just note it
                        image_paths.append("[base64 image]")
        elif isinstance(content, str):
            text_parts.append(content)

    user_message = "\n".join(text_parts)

    backend: AgentBackend = client
    return backend.query(
        system_message=system_message,
        user_message=user_message,
        images=image_paths if image_paths else None,
    )


def prepare_vlm_prompt(msg, image_paths, max_images):
    pass


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON from LLM output. Unchanged between v2 and v3."""
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    return None


# ---- Legacy implementations (only loaded when AI_SCIENTIST_BACKEND=legacy) ----

def _create_client_legacy(model: str) -> tuple[Any, str]:
    if model in [
        "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18", "o3-mini",
    ]:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("ollama/"):
        print(f"Using Ollama API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url="http://localhost:11434/v1"
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")


def _make_llm_call_legacy(client, model, temperature, system_message, prompt):
    @track_token_usage
    def _inner():
        if model.startswith("ollama/"):
            return client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[{"role": "system", "content": system_message}, *prompt],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None, seed=0,
            )
        elif "gpt" in model:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message}, *prompt],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None, seed=0,
            )
        elif "o1" in model or "o3" in model:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": system_message}, *prompt],
                temperature=1, n=1, seed=0,
            )
        else:
            raise ValueError(f"Model {model} not supported.")
    return _inner()


def _make_vlm_call_legacy(client, model, temperature, system_message, prompt):
    @track_token_usage
    def _inner():
        if model.startswith("ollama/"):
            return client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[{"role": "system", "content": system_message}, *prompt],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS,
            )
        elif "gpt" in model:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message}, *prompt],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS,
            )
        else:
            raise ValueError(f"Model {model} not supported.")
    return _inner()


def _get_response_from_vlm_legacy(
    msg, image_paths, client, model, system_message,
    print_debug=False, msg_history=None, temperature=0.7, max_images=25,
):
    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
    def _inner():
        if msg_history is None:
            _msg_history = []
        else:
            _msg_history = msg_history

        if model in AVAILABLE_VLMS:
            _image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
            content = [{"type": "text", "text": msg}]
            for image_path in _image_paths[:max_images]:
                base64_image = encode_image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                })
            new_msg_history = _msg_history + [{"role": "user", "content": content}]
            response = _make_vlm_call_legacy(client, model, temperature, system_message, new_msg_history)
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        else:
            raise ValueError(f"Model {model} not supported.")

        if print_debug:
            print()
            print("*" * 20 + " VLM START " + "*" * 20)
            for j, m in enumerate(new_msg_history):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " VLM END " + "*" * 21)
            print()

        return content, new_msg_history
    return _inner()


def _get_batch_responses_from_vlm_legacy(
    msg, image_paths, client, model, system_message,
    print_debug=False, msg_history=None, temperature=0.7,
    n_responses=1, max_images=200,
):
    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
    def _inner():
        if msg_history is None:
            _msg_history = []
        else:
            _msg_history = msg_history

        if model in AVAILABLE_VLMS:
            _image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
            content = [{"type": "text", "text": msg}]
            for image_path in _image_paths[:max_images]:
                base64_image = encode_image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                })
            new_msg_history = _msg_history + [{"role": "user", "content": content}]

            if model.startswith("ollama/"):
                response = client.chat.completions.create(
                    model=model.replace("ollama/", ""),
                    messages=[{"role": "system", "content": system_message}, *new_msg_history],
                    temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, seed=0,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_message}, *new_msg_history],
                    temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, seed=0,
                )
            contents = [r.message.content for r in response.choices]
            new_msg_histories = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in contents
            ]
        else:
            raise ValueError(f"Model {model} not supported.")

        if print_debug:
            print()
            print("*" * 20 + " VLM START " + "*" * 20)
            for j, m in enumerate(new_msg_histories[0]):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(contents[0])
            print("*" * 21 + " VLM END " + "*" * 21)
            print()

        return contents, new_msg_histories
    return _inner()
