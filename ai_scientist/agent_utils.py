"""
Agent helper utilities for AI-Scientist v3.

Convenience wrappers and shared utilities used by both the pipeline stage
calls (llm.py/vlm.py replacement) and tree search backend (backend_agent.py).
"""

import json
import logging
import re
from typing import Any

from ai_scientist.agent_backend import (
    AgentBackend,
    AgentError,
    AgentOutputError,
    _extract_json_from_text,
)

logger = logging.getLogger(__name__)

# Module-level default backend instance (lazily initialized)
_default_backend: AgentBackend | None = None


def get_default_backend(**kwargs: Any) -> AgentBackend:
    """Get or create the module-level default AgentBackend instance.

    Args:
        **kwargs: Passed to AgentBackend() on first creation.

    Returns:
        The shared AgentBackend instance.
    """
    global _default_backend
    if _default_backend is None:
        _default_backend = AgentBackend(**kwargs)
    return _default_backend


def reset_default_backend() -> None:
    """Reset the default backend (useful for testing)."""
    global _default_backend
    _default_backend = None


def invoke_agent(
    prompt: str,
    system_message: str | None = None,
    json_schema: dict | None = None,
    images: list[str] | None = None,
    timeout: int = 300,
    model: str | None = None,
    backend: AgentBackend | None = None,
) -> str | dict:
    """Convenience wrapper for a single agent invocation.

    This is the simplest way to call the agent. For repeated calls or when
    you need invocation tracking, use AgentBackend directly.

    Args:
        prompt: The user message / prompt.
        system_message: Optional system prompt.
        json_schema: Optional JSON schema for structured output.
        images: Optional list of image file paths.
        timeout: Timeout in seconds (default 300).
        model: Model override (default uses backend's model).
        backend: AgentBackend instance (uses default if None).

    Returns:
        str or dict depending on whether json_schema was provided.
    """
    if backend is None:
        backend = get_default_backend()

    return backend.query_with_retry(
        system_message=system_message,
        user_message=prompt,
        json_schema=json_schema,
        images=images,
        timeout=timeout,
        model=model,
    )


def parse_agent_json(response: str, schema: dict | None = None) -> dict:
    """Extract and validate JSON from agent text output.

    Handles common patterns: JSON in code blocks, raw JSON objects, and
    JSON mixed with explanatory text.

    Args:
        response: Raw text response from the agent.
        schema: Optional JSON schema for validation (currently logs warnings
            on mismatch but does not raise).

    Returns:
        Parsed dict.

    Raises:
        AgentOutputError: If no valid JSON can be extracted.
    """
    # Try direct parse
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extraction from text
    extracted = _extract_json_from_text(response)
    if extracted is not None:
        return extracted

    raise AgentOutputError(
        f"Could not extract JSON from response: {response[:500]}..."
    )


def estimate_tokens(text: str) -> int:
    """Rough token count estimate from text length.

    Uses the ~4 chars per token heuristic. This is intentionally simple --
    we don't want to depend on tiktoken or any tokenizer library in v3.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def format_json_schema_prompt(schema: dict) -> str:
    """Format a JSON schema as a prompt instruction for the agent.

    Used when converting FunctionSpec schemas into prompt-based instructions
    instead of API-level tool_choice parameters.

    Args:
        schema: JSON schema dict.

    Returns:
        Formatted instruction string.
    """
    schema_str = json.dumps(schema, indent=2)
    return (
        "Respond with a JSON object matching this schema. "
        "Output ONLY valid JSON, no other text.\n\n"
        f"Schema:\n```json\n{schema_str}\n```"
    )


def build_function_call_prompt(
    name: str,
    description: str,
    json_schema: dict,
    user_message: str,
) -> str:
    """Build a prompt that mimics function calling via structured output.

    Replaces the FunctionSpec -> tool_choice API pattern with a prompt-based
    approach where the agent is instructed to return JSON matching the schema.

    Args:
        name: Function name (for context).
        description: Function description.
        json_schema: Expected output schema.
        user_message: The actual user request.

    Returns:
        Combined prompt string.
    """
    schema_str = json.dumps(json_schema, indent=2)
    return (
        f"{user_message}\n\n"
        f"---\n"
        f"You must respond by calling the function '{name}'.\n"
        f"Description: {description}\n"
        f"Output ONLY a valid JSON object matching this schema:\n"
        f"```json\n{schema_str}\n```\n"
        f"Do not include any text outside the JSON object."
    )
