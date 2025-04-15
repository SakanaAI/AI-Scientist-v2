import base64
from typing import Any
import re
import json
import backoff
import openai
from PIL import Image
from ai_scientist.utils.token_tracker import track_token_usage
import os
from dotenv import load_dotenv

load_dotenv()
MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 4096))

AVAILABLE_VLMS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "o3-mini",
]


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
    if "gpt" in model:
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
    if "gpt" in model:
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

    if model in AVAILABLE_VLMS or "gemini" in model:
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
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f"{j}, {msg['role']}: {msg['content']}")
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

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
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "gemini" in model:
        print(f"Using Gemini API with model {model}.")
        return openai.OpenAI(
            max_retries=0,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "o3-mini",
    ] or "gemini" in model:
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

        # Get multiple responses
        response = client.chat.completions.create(
            model=model,
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
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f"{j}, {msg['role']}: {msg['content']}")
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, new_msg_histories
