import json
import os
import re
from typing import Any

from ai_scientist.agent_backend import AgentBackend

MAX_NUM_TOKENS = 4096

# v3: Backend mode. Set AI_SCIENTIST_BACKEND=legacy to use old SDK clients.
_BACKEND_MODE = os.environ.get("AI_SCIENTIST_BACKEND", "agent")

# In v3, AVAILABLE_LLMS accepts any string (no model restriction).
# Kept as a list so argparse choices still works for consumers.
AVAILABLE_LLMS = None  # Set below based on backend mode

if _BACKEND_MODE == "legacy":
    import anthropic
    import backoff
    import openai
    from ai_scientist.utils.token_tracker import track_token_usage

    AVAILABLE_LLMS = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "o1",
        "o1-2024-12-17",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-mini",
        "o3-mini-2025-01-31",
        "deepseek-coder-v2-0724",
        "deepcoder-14b",
        "llama3.1-405b",
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        "vertex_ai/claude-3-opus@20240229",
        "vertex_ai/claude-3-5-sonnet@20240620",
        "vertex_ai/claude-3-5-sonnet@20241022",
        "vertex_ai/claude-3-sonnet@20240229",
        "vertex_ai/claude-3-haiku@20240307",
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-03-25",
        "ollama/gpt-oss:20b",
        "ollama/gpt-oss:120b",
        "ollama/qwen3:8b",
        "ollama/qwen3:32b",
        "ollama/qwen3:235b",
        "ollama/qwen2.5vl:8b",
        "ollama/qwen2.5vl:32b",
        "ollama/qwen3-coder:70b",
        "ollama/qwen3-coder:480b",
        "ollama/deepseek-r1:8b",
        "ollama/deepseek-r1:32b",
        "ollama/deepseek-r1:70b",
        "ollama/deepseek-r1:671b",
    ]
else:
    # v3: No model restriction -- agent backend accepts any model string.
    # None means argparse won't restrict choices.
    AVAILABLE_LLMS = None


# Module-level agent backend (lazily initialized, v3 only)
_agent_backend: AgentBackend | None = None


def _get_agent_backend() -> AgentBackend:
    """Get or create the module-level AgentBackend for v3 mode."""
    global _agent_backend
    if _agent_backend is None:
        _agent_backend = AgentBackend()
    return _agent_backend


def create_client(model) -> tuple[Any, str]:
    """Create an LLM client.

    In v3 (default): returns (AgentBackend, model_str). No API keys needed.
    In legacy mode: returns (SDK_client, model_str) as before.
    """
    if _BACKEND_MODE == "legacy":
        return _create_client_legacy(model)

    print(f"Using agent backend (v3 mode) for model '{model}'.")
    return _get_agent_backend(), model


def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Get a single response from the LLM.

    In v3: routes through AgentBackend. Client param is an AgentBackend instance.
    Returns (content_str, msg_history) as before.
    """
    if _BACKEND_MODE == "legacy":
        return _get_response_from_llm_legacy(
            prompt, client, model, system_message,
            print_debug=print_debug, msg_history=msg_history,
            temperature=temperature,
        )

    msg = prompt
    if msg_history is None:
        msg_history = []

    # Build user message from history + new prompt
    history_context = ""
    if msg_history:
        for entry in msg_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if isinstance(content, list):
                # Handle Anthropic-style content blocks
                content = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            history_context += f"\n[{role}]: {content}\n"

    user_message = msg
    if history_context:
        user_message = f"Previous conversation:\n{history_context}\n\nCurrent message:\n{msg}"

    backend: AgentBackend = client
    content = backend.query(
        system_message=system_message,
        user_message=user_message,
    )

    # Build new message history in the standard format
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history):
            print(f'{j}, {m["role"]}: {m["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get N responses from a single message, used for ensembling.

    In v3: N sequential agent invocations.
    """
    if _BACKEND_MODE == "legacy":
        return _get_batch_responses_from_llm_legacy(
            prompt, client, model, system_message,
            print_debug=print_debug, msg_history=msg_history,
            temperature=temperature, n_responses=n_responses,
        )

    content_list = []
    history_list = []
    for _ in range(n_responses):
        c, hist = get_response_from_llm(
            prompt, client, model, system_message,
            print_debug=False, msg_history=msg_history,
            temperature=temperature,
        )
        content_list.append(c)
        history_list.append(hist)

    if print_debug and history_list:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, m in enumerate(history_list[0]):
            print(f'{j}, {m["role"]}: {m["content"]}')
        print(content_list)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content_list, history_list


def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a raw LLM call.

    In v3: routes through AgentBackend, returns the content string directly.
    Note: Legacy mode returns the raw SDK response object.
    """
    if _BACKEND_MODE == "legacy":
        return _make_llm_call_legacy(client, model, temperature, system_message, prompt)

    # Build message from prompt list
    user_parts = []
    for entry in prompt:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        user_parts.append(f"[{role}]: {content}")
    user_message = "\n".join(user_parts)

    backend: AgentBackend = client
    return backend.query(
        system_message=system_message,
        user_message=user_message,
    )


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON from LLM output. Unchanged between v2 and v3."""
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


# ---- Legacy implementations (only loaded when AI_SCIENTIST_BACKEND=legacy) ----

def _create_client_legacy(model) -> tuple[Any, str]:
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif model.startswith("ollama/"):
        print(f"Using Ollama with model {model}.")
        return openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url="http://localhost:11434/v1",
        ), model
    elif "gpt" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    elif model == "deepcoder-14b":
        print(f"Using HuggingFace API with {model}.")
        if "HUGGINGFACE_API_KEY" not in os.environ:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return (
            openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ),
            model,
        )
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "meta-llama/llama-3.1-405b-instruct",
        )
    elif 'gemini' in model:
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            model,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def _get_response_from_llm_legacy(
    prompt, client, model, system_message,
    print_debug=False, msg_history=None, temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            anthropic.RateLimitError,
        ),
    )
    def _inner():
        msg = prompt
        if msg_history is None:
            _msg_history = []
        else:
            _msg_history = msg_history

        if "claude" in model:
            new_msg_history = _msg_history + [
                {"role": "user", "content": [{"type": "text", "text": msg}]}
            ]
            response = client.messages.create(
                model=model, max_tokens=MAX_NUM_TOKENS,
                temperature=temperature, system=system_message,
                messages=new_msg_history,
            )
            content = response.content[0].text
            new_msg_history = new_msg_history + [
                {"role": "assistant", "content": [{"type": "text", "text": content}]}
            ]
        elif model.startswith("ollama/"):
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif "gpt" in model:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = _make_llm_call_legacy(client, model, temperature, system_message, new_msg_history)
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif "o1" in model or "o3" in model:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = _make_llm_call_legacy(client, model, temperature, system_message, new_msg_history)
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif model == "deepseek-coder-v2-0724":
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif model == "deepcoder-14b":
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            try:
                response = client.chat.completions.create(
                    model="agentica-org/DeepCoder-14B-Preview",
                    messages=[{"role": "system", "content": system_message}, *new_msg_history],
                    temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None,
                )
                content = response.choices[0].message.content
            except Exception:
                import requests
                headers = {
                    "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "inputs": {
                        "system": system_message,
                        "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                    },
                    "parameters": {
                        "temperature": temperature, "max_new_tokens": MAX_NUM_TOKENS,
                        "return_full_text": False
                    }
                }
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                    headers=headers, json=payload
                )
                if resp.status_code == 200:
                    content = resp.json()["generated_text"]
                else:
                    raise ValueError(f"Error from HuggingFace API: {resp.text}")
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        elif 'gemini' in model:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        else:
            raise ValueError(f"Model {model} not supported.")

        if print_debug:
            print()
            print("*" * 20 + " LLM START " + "*" * 20)
            for j, m in enumerate(new_msg_history):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21)
            print()

        return content, new_msg_history

    return _inner()


def _get_batch_responses_from_llm_legacy(
    prompt, client, model, system_message,
    print_debug=False, msg_history=None, temperature=0.7, n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            anthropic.RateLimitError,
        ),
    )
    @track_token_usage
    def _inner():
        msg = prompt
        if msg_history is None:
            _msg_history = []
        else:
            _msg_history = msg_history

        if model.startswith("ollama/"):
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, stop=None,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        elif "gpt" in model:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, stop=None, seed=0,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        elif model == "deepseek-coder-v2-0724":
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, stop=None,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        elif model == "llama-3-1-405b-instruct":
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, stop=None,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        elif 'gemini' in model:
            new_msg_history = _msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message}, *new_msg_history],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=n_responses, stop=None,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                new_msg_history + [{"role": "assistant", "content": c}] for c in content
            ]
        else:
            content, new_msg_history = [], []
            for _ in range(n_responses):
                c, hist = _get_response_from_llm_legacy(
                    msg, client, model, system_message,
                    print_debug=False, msg_history=None, temperature=temperature,
                )
                content.append(c)
                new_msg_history.append(hist)

        if print_debug:
            print()
            print("*" * 20 + " LLM START " + "*" * 20)
            for j, m in enumerate(new_msg_history[0]):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21)
            print()

        return content, new_msg_history

    return _inner()


def _make_llm_call_legacy(client, model, temperature, system_message, prompt):
    @track_token_usage
    def _inner():
        if model.startswith("ollama/"):
            return client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[{"role": "system", "content": system_message}, *prompt],
                temperature=temperature, max_tokens=MAX_NUM_TOKENS, n=1, stop=None,
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
