# guardian_angel/llm.py

# Modified based on ai_scientist/llm.py from AI-Scientist-v2
# Original work © 2024 SakanaAI Contributors
# Licensed under the Apache License, Version 2.0
# Modifications by Sho Watanabe (2025):
# - Adapted for Guardian Security Agent architecture
# - Added async request handling and enhanced caching
# - Removed speculative reasoning API hooks

# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0


import os
from typing import Any
import numpy as np

import backoff
import openai
import anthropic

from ai_steward.utils.token_tracker import track_token_usage

MAX_NUM_TOKENS = 32768
LOCAL_NUM_PREDICT = 8196
LOCAL_NUM_KEEP = 1024
LOCAL_NUM_CTX = 131072

AVAILABLE_LLMS = [
    # Anthropic
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI
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
    # DeepSeek
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama(OpenRouter)
    "llama3.1-405b",
    # Bedrock Anthropic
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Vertex AI Anthropic
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # Google (OpenAI-compatible proxy)
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    # Local (Ollama等)
    "gpt-oss:20b",
    "gpt-oss:120b",
    "kun432/cl-nagoya-ruri-large",
]

def is_local_model(model: str) -> bool:
    if os.environ.get("LOCAL_LLM_URL"):
        return True
    if model.startswith("gpt-oss:"):
        return True
    if "/" in model and not any(k in model for k in ("gpt-", "claude", "gemini", "bedrock", "vertex_ai", "o1", "o3")):
        return True
    return False

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError))
def get_embedding(text: str, client: Any, model: str) -> np.ndarray:
    text = text.replace("\n", " ")
    res = client.embeddings.create(model=model, input=[text])
    return np.array(res.data[0].embedding, dtype=np.float32)

def _messages_to_tracker_text(system_message: str, messages: list[dict[str, str]]) -> str:
    parts = [f"[SYSTEM]\n{system_message}", "[MESSAGES]"]
    for m in messages:
        parts.append(f"{m.get('role')}: {m.get('content')}")
    return "\n".join(parts)

@track_token_usage
def _remote_chat_call(
    client,
    model: str,
    temperature: float,
    system_message: str,
    prompt_messages: list[dict[str, str]],
    prompt: str = "",
    system_message_for_tracker: str = "",
):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, *prompt_messages],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
        seed=0,
    )

@track_token_usage
def _local_chat_call(
    client,
    model: str,
    temperature: float,
    system_message: str,
    prompt_messages: list[dict[str, str]],
    num_ctx: int,
    num_keep: int,
    prompt: str = "",
    system_message_for_tracker: str = "",
):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_message}, *prompt_messages],
        temperature=temperature,
        max_tokens=min(MAX_NUM_TOKENS, num_ctx // 2),
        n=1,
        seed=0,
        extra_body={"options": {"num_ctx": num_ctx, "num_predict": LOCAL_NUM_PREDICT, "num_keep": num_keep, "temperature": temperature}},
    )

def _family(model: str) -> str:
    if model.startswith("claude") or model.startswith("bedrock/") or model.startswith("vertex_ai/"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3")):
        return "openai"
    if model in ("deepseek-coder-v2-0724", "deepcoder-14b"):
        return "deepseek"
    if model in ("llama3.1-405b", "meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"):
        return "openrouter"
    if model.startswith("gemini-"):
        return "gemini"
    return "local"

def create_client(model) -> tuple[Any, str]:
    if os.environ.get("LOCAL_LLM_URL"):
        base_url = os.environ["LOCAL_LLM_URL"]
        api_key = os.environ.get("LOCAL_LLM_API_KEY", "ollama")
        print(f"Using LOCAL LLM endpoint: {base_url} (model={model})")
        return openai.OpenAI(api_key=api_key, base_url=base_url), model

    fam = _family(model)

    if fam == "anthropic":
        if model.startswith("bedrock/"):
            m = model.split("/")[-1]
            print(f"Using Bedrock Anthropic: {m}")
            return anthropic.AnthropicBedrock(), m
        if model.startswith("vertex_ai/"):
            m = model.split("/")[-1]
            print(f"Using Vertex Anthropic: {m}")
            return anthropic.AnthropicVertex(), m
        print(f"Using Anthropic: {model}")
        return anthropic.Anthropic(), model

    if fam == "openai":
        print(f"Using OpenAI: {model}")
        return openai.OpenAI(), model

    if fam == "deepseek":
        if model == "deepseek-coder-v2-0724":
            print("Using DeepSeek API")
            return openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"), "deepseek-coder"
        if model == "deepcoder-14b":
            print("Using HuggingFace Inference (DeepCoder-14B)")
            if "HUGGINGFACE_API_KEY" not in os.environ:
                raise ValueError("HUGGINGFACE_API_KEY not set")
            return openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ), model

    if fam == "openrouter":
        print("Using OpenRouter for Llama 3.1 405B")
        return openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"), "meta-llama/llama-3.1-405b-instruct"

    if fam == "gemini":
        print(f"Using Gemini via OpenAI-compatible proxy: {model}")
        return openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/"), model

    print(f"Using default OpenAI client for local-like model: {model}")
    return openai.OpenAI(api_key=os.environ.get("LOCAL_LLM_API_KEY", "ollama"), base_url=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:11434/v1")), model

def _as_chat_history(msg_history: list[dict[str, str]] | None, user_msg: str) -> list[dict[str, str]]:
    return (msg_history or []) + [{"role": "user", "content": user_msg}]

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, anthropic.RateLimitError))
def get_response_from_llm(
    prompt: str,
    client: Any,
    model: str,
    system_message: str,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> tuple[str, list[dict[str, Any]]]:
    fam = _family(model)
    new_msg_history = _as_chat_history(msg_history, prompt)

    if fam == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=[{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in new_msg_history],
        )
        content = resp.content[0].text
        new_msg_history.append({"role": "assistant", "content": content})
        return content, new_msg_history

    tracker_prompt = _messages_to_tracker_text(system_message, new_msg_history)

    if is_local_model(model) or fam == "local":
        resp = _local_chat_call(
            client,
            model,
            temperature,
            system_message,
            new_msg_history,
            num_ctx=LOCAL_NUM_CTX,
            num_keep=LOCAL_NUM_KEEP,
            prompt=tracker_prompt,
            system_message_for_tracker=system_message,
        )
        content = resp.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
        return content, new_msg_history

    if fam in ("openai", "deepseek", "openrouter", "gemini"):
        resp = _remote_chat_call(
            client,
            model,
            temperature,
            system_message,
            new_msg_history,
            prompt=tracker_prompt,
            system_message_for_tracker=system_message,
        )
        content = resp.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
        return content, new_msg_history

    raise ValueError(f"Unsupported model family for: {model}")
