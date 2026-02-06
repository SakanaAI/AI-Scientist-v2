#!/usr/bin/env python3
# Lightweight LLM adapter with offline-by-default safeguards.
from __future__ import annotations

import json
import os
import urllib.request

ONLINE_ENV_VAR = "ASV2_ONLINE"


def _require_online() -> None:
    if os.getenv(ONLINE_ENV_VAR) != "1":
        raise RuntimeError(
            "Offline mode: set ASV2_ONLINE=1 or pass --online to allow network calls."
        )


def _ollama_chat(prompt: str, system: str, model: str, temperature: float) -> str:
    _require_online()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model": model.replace("ollama/", ""),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


def chat(
    prompt: str,
    system: str,
    model: str,
    temperature: float = 0.4,
    max_tokens: int = 1200,
) -> str:
    _require_online()
    if model.startswith("ollama/"):
        return _ollama_chat(prompt, system, model, temperature)
    if model.startswith("bedrock/"):
        raise RuntimeError("Bedrock adapter not implemented in this lightweight skill.")
    if model.startswith("vertex_ai/"):
        raise RuntimeError("Vertex AI adapter not implemented in this lightweight skill.")
    if model.startswith("gemini-"):
        raise RuntimeError("Gemini adapter not implemented in this lightweight skill.")

    # Default: OpenAI-compatible
    try:
        import openai
    except Exception as e:
        raise RuntimeError(
            "OpenAI adapter requires the openai package. Install with --with openai."
        ) from e

    base_url = os.getenv("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
