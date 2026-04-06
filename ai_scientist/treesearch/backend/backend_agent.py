"""
Agent-based backend for AI-Scientist v3 tree search.

Replaces backend_openai.py and backend_anthropic.py with a unified agent
backend that routes all queries through AgentBackend (claude CLI subprocess).
"""

import json
import logging
import time

from .utils import FunctionSpec, OutputType

from ai_scientist.agent_backend import AgentBackend, AgentError
from ai_scientist.agent_utils import (
    build_function_call_prompt,
    estimate_tokens,
)

logger = logging.getLogger("ai-scientist")

# Module-level backend instance (lazily initialized)
_backend: AgentBackend | None = None


def get_ai_client(model: str = "agent", **model_kwargs) -> AgentBackend:
    """Get or create the module-level AgentBackend instance.

    Matches the signature of backend_openai.get_ai_client / backend_anthropic.get_ai_client
    but returns an AgentBackend instead of an SDK client.

    Args:
        model: Model identifier. Ignored for routing (always uses agent).
        **model_kwargs: Additional kwargs (timeout, verbose, etc.) passed to AgentBackend.

    Returns:
        AgentBackend instance.
    """
    global _backend
    if _backend is None:
        backend_kwargs = {}
        if "timeout" in model_kwargs:
            backend_kwargs["timeout"] = model_kwargs["timeout"]
        if "verbose" in model_kwargs:
            backend_kwargs["verbose"] = model_kwargs["verbose"]
        if "max_retries" in model_kwargs:
            backend_kwargs["max_retries"] = model_kwargs["max_retries"]
        _backend = AgentBackend(**backend_kwargs)
    return _backend


def reset_client() -> None:
    """Reset the module-level backend (useful for testing)."""
    global _backend
    _backend = None


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Execute a query through the agent backend.

    Matches the signature of backend_openai.query and backend_anthropic.query:
        (output, req_time, in_tokens, out_tokens, info)

    Args:
        system_message: System prompt (already compiled to markdown string).
        user_message: User prompt (already compiled to markdown string).
        func_spec: Optional FunctionSpec for structured JSON output.
            The schema is converted to prompt instructions instead of
            API-level tool_choice parameters.
        **model_kwargs: Model configuration. Recognized keys:
            - model (str): Model identifier (passed to AgentBackend).
            - temperature (float): Ignored (agent handles its own sampling).
            - max_tokens (int): Ignored (agent handles its own limits).
            - timeout (int): Override per-query timeout.

    Returns:
        Tuple of (output, req_time, in_tokens, out_tokens, info):
            - output: str if no func_spec, dict if func_spec provided
            - req_time: wall-clock seconds
            - in_tokens: estimated input token count
            - out_tokens: estimated output token count
            - info: dict with metadata
    """
    backend = get_ai_client(model_kwargs.get("model", "agent"))

    # Build the effective user message, incorporating FunctionSpec if present
    effective_user_message = user_message
    json_schema = None

    if func_spec is not None:
        json_schema = func_spec.json_schema
        # Augment the user message with function-calling instructions
        effective_user_message = build_function_call_prompt(
            name=func_spec.name,
            description=func_spec.description,
            json_schema=func_spec.json_schema,
            user_message=user_message or "",
        )

    # Extract per-query overrides from model_kwargs
    timeout = model_kwargs.get("timeout", None)
    model = model_kwargs.get("model", None)

    # Estimate input tokens before the call
    input_text = (system_message or "") + (effective_user_message or "")
    in_tokens = estimate_tokens(input_text)

    t0 = time.time()
    try:
        result = backend.query(
            system_message=system_message,
            user_message=effective_user_message,
            json_schema=json_schema,
            timeout=timeout,
            model=model,
        )
        req_time = time.time() - t0
    except AgentError as e:
        req_time = time.time() - t0
        logger.error("Agent query failed after %.1fs: %s", req_time, e)
        raise

    # Determine output and estimate output tokens
    if func_spec is not None:
        # result should be a dict from json_schema parsing
        if isinstance(result, dict):
            output = result
            out_tokens = estimate_tokens(json.dumps(result))
        else:
            # Fallback: try to parse the string result as JSON
            try:
                output = json.loads(result)
                out_tokens = estimate_tokens(result)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Expected JSON output for func_spec '%s', got string. "
                    "Returning raw string.",
                    func_spec.name,
                )
                output = result
                out_tokens = estimate_tokens(str(result))
    else:
        output = result
        out_tokens = estimate_tokens(str(result))

    info = {
        "backend": "agent",
        "wall_clock_seconds": req_time,
        "estimated_tokens": True,
    }

    return output, req_time, in_tokens, out_tokens, info
