import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any

import backoff
import openai
import os
from PIL import Image
from ai_scientist.utils.token_tracker import track_token_usage
import logging
logger = logging.getLogger(__name__)


MAX_NUM_TOKENS = 4096

# DashScope OpenAI-compatible API (Qwen VL). Override with env QWEN_BASE_URL if needed
# (e.g. international: https://dashscope-intl.aliyuncs.com/compatible-mode/v1).
DEFAULT_QWEN_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Default local VLM when Ollama responds at OLLAMA_HOST (see resolve_vlm_model).
DEFAULT_OLLAMA_VLM_MODEL = "ollama/qwen3-vl:32b"
# Default DashScope VL model id (no ``qwen/`` prefix). Override with env QWEN_VLM_MODEL.
# See: https://www.alibabacloud.com/help/en/model-studio/developer-reference/qwen-vl-compatible-with-openai
DEFAULT_QWEN_VLM_DASHSCOPE_MODEL = "qwen3-vl-plus"

AVAILABLE_VLMS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "o3-mini",

    # Ollama models

    # llama4
    "ollama/llama4:16x17b",

    # mistral
    "ollama/mistral-small3.2:24b",

    # qwen
    "ollama/qwen2.5vl:32b",
    "ollama/qwen3-vl:32b",

    "ollama/z-uo/qwen2.5vl_tools:32b",
]


def is_supported_vlm_model(model: str) -> bool:
    """True for built-in IDs or DashScope models ``qwen/<dashscope-model-name>``."""
    return model in AVAILABLE_VLMS or model.startswith("qwen/")


def _ollama_http_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host:
        host = "127.0.0.1:11434"
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


def is_ollama_server_reachable(timeout: float = 2.0) -> bool:
    """True if an Ollama daemon responds (GET /api/tags). Uses OLLAMA_HOST like the Ollama CLI."""
    url = f"{_ollama_http_base()}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(code) < 300
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def resolve_vlm_model(model: str) -> str:
    """Pick a concrete VLM id.

    If ``model`` is ``auto`` (case-insensitive): use DashScope Qwen-VL
    ``qwen/<QWEN_VLM_MODEL>`` (requires ``QWEN_API_KEY``). Ollama is not used for
    ``auto``; pass ``ollama/<tag>`` explicitly for local VLMs.
    Any other value is returned unchanged.
    """
    raw = (model or "").strip()
    if raw.lower() != "auto":
        return raw

    dash = os.environ.get("QWEN_VLM_MODEL", DEFAULT_QWEN_VLM_DASHSCOPE_MODEL).strip()
    if dash.startswith("qwen/"):
        resolved = dash
    else:
        resolved = f"qwen/{dash}"
    logger.info("VLM auto: using DashScope model %s", resolved)
    if "QWEN_API_KEY" not in os.environ:
        raise ValueError(
            "VLM auto mode requires QWEN_API_KEY (DashScope Qwen-VL). "
            "Optional: QWEN_VLM_MODEL (default qwen3-vl-plus), QWEN_BASE_URL. "
            "For Ollama, set agent.vlm_feedback.model to ollama/<tag> explicitly."
        )
    return resolved


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    elif model.startswith("qwen/"):
        return client.chat.completions.create(
            model=model.replace("qwen/", "", 1),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def prepare_vlm_prompt(msg, image_paths, max_images):
    pass


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
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
    """Get response from vision-language model."""
    if msg_history is None:
        msg_history = []

    if is_supported_vlm_model(model):
        # Convert single image path to list for consistent handling
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Create content list starting with the text message
        content = [{"type": "text", "text": msg}]

        # Add each image to the content list
        for image_path in image_paths[:max_images]:
            base64_image = encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )
        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        response = make_vlm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )

        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        logger.info()
        logger.info("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            logger.info(f'{j}, {msg["role"]}: {msg["content"]}')
        logger.info(content)
        logger.info("*" * 21 + " VLM END " + "*" * 21)
        logger.info()

    return content, new_msg_history


def create_client(model: str) -> tuple[Any, str]:
    """Create client for vision-language model."""
    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "o3-mini",
    ]:
        logger.info(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("ollama/"):
        logger.info(f"Using Ollama API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url="http://localhost:11434/v1"
        ), model
    elif model.startswith("qwen/"):
        logger.info(f"Using Qwen (DashScope) API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ["QWEN_API_KEY"],
            base_url=os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_OPENAI_BASE_URL),
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
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
    """Get multiple responses from vision-language model for the same input.

    Args:
        msg: Text message to send
        image_paths: Path(s) to image file(s)
        client: OpenAI client instance
        model: Name of model to use
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate

    Returns:
        Tuple of (list of response strings, list of message histories)
    """
    if msg_history is None:
        msg_history = []

    if is_supported_vlm_model(model):
        # Convert single image path to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Create content list with text and images
        content = [{"type": "text", "text": msg}]
        for image_path in image_paths[:max_images]:
            base64_image = encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )

        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        if model.startswith("ollama/"):
            api_model = model.replace("ollama/", "", 1)
        elif model.startswith("qwen/"):
            api_model = model.replace("qwen/", "", 1)
        else:
            api_model = model

        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            seed=0,
        )

        # Extract content from all responses
        contents = [r.message.content for r in response.choices]
        new_msg_histories = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in contents
        ]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        # Just print the first response
        logger.info()
        logger.info("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            logger.info(f'{j}, {msg["role"]}: {msg["content"]}')
        logger.info(contents[0])
        logger.info("*" * 21 + " VLM END " + "*" * 21)
        logger.info()

    return contents, new_msg_histories
