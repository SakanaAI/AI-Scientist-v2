import os
import time
from functools import wraps
from typing import Dict, Optional, List
try:
    import tiktoken
except ImportError:
    tiktoken = None  # Not needed in v3 agent mode
from collections import defaultdict
import asyncio
from datetime import datetime
import logging

_BACKEND_MODE = os.environ.get("AI_SCIENTIST_BACKEND", "agent")


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.

        In v3 agent mode, token counts are estimated from text length (~4 chars/token).
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        # v3 agent-specific tracking
        self.agent_stats = defaultdict(
            lambda: {"invocations": 0, "wall_clock_seconds": 0.0}
        )

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,
                "cached": 1.25 / 1000000,
                "completion": 10 / 1000000,
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,
                "cached": 1.25 / 1000000,
                "completion": 10 / 1000000,
            },
            "gpt-4o-2024-05-13": {
                "prompt": 5.0 / 1000000,
                "completion": 15 / 1000000,
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,
                "cached": 0.075 / 1000000,
                "completion": 0.6 / 1000000,
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,
                "cached": 7.5 / 1000000,
                "completion": 60 / 1000000,
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,
                "cached": 7.5 / 1000000,
                "completion": 60 / 1000000,
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,
                "cached": 0.55 / 1000000,
                "completion": 4.4 / 1000000,
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_agent_invocation(
        self,
        model: str,
        wall_clock_seconds: float,
        prompt_text: str = "",
        response_text: str = "",
        system_message: str = "",
    ):
        """Record an agent invocation with estimated token counts.

        Used in v3 mode where actual token counts aren't available.
        Estimates tokens at ~4 chars per token.
        """
        est_prompt = max(1, len(prompt_text + system_message) // 4)
        est_completion = max(1, len(response_text) // 4)

        self.token_counts[model]["prompt"] += est_prompt
        self.token_counts[model]["completion"] += est_completion
        self.agent_stats[model]["invocations"] += 1
        self.agent_stats[model]["wall_clock_seconds"] += wall_clock_seconds

        self.interactions[model].append(
            {
                "system_message": system_message[:200] if system_message else "",
                "prompt": prompt_text[:200] if prompt_text else "",
                "response": response_text[:200] if response_text else "",
                "timestamp": datetime.now(),
                "wall_clock_seconds": wall_clock_seconds,
                "estimated_tokens": True,
            }
        )

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts, interactions, and agent stats."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        self.agent_stats = defaultdict(
            lambda: {"invocations": 0, "wall_clock_seconds": 0.0}
        )

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of token usage and costs for all models.

        In v3 mode, also includes agent-specific stats (invocations, wall-clock time).
        """
        summary = {}
        for model, tokens in self.token_counts.items():
            entry = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
            if model in self.agent_stats:
                entry["agent_stats"] = self.agent_stats[model].copy()
            summary[model] = entry
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    """Decorator to track token usage from LLM calls.

    In v3 agent mode: gracefully handles non-SDK results (strings, tuples, dicts).
    In legacy mode: extracts token counts from SDK response objects as before.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt") or kwargs.get("system_message") or ""
        system_message = kwargs.get("system_message") or ""

        t0 = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - t0

        _extract_and_track(result, prompt, system_message, elapsed)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt") or kwargs.get("system_message") or ""
        system_message = kwargs.get("system_message") or ""

        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0

        _extract_and_track(result, prompt, system_message, elapsed)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def _extract_and_track(result, prompt, system_message, elapsed):
    """Extract token info from result and record it.

    Handles both SDK response objects (legacy) and plain values (v3).
    """
    # Try SDK response object format (legacy mode)
    if hasattr(result, "model") and hasattr(result, "usage"):
        model = result.model
        timestamp = getattr(result, "created", datetime.now())
        usage = result.usage

        if usage and getattr(usage, "completion_tokens_details", None) is not None:
            token_tracker.add_tokens(
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.completion_tokens_details.reasoning_tokens,
                (
                    usage.prompt_tokens_details.cached_tokens
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details
                    else 0
                ),
            )
            response_text = ""
            if hasattr(result, "choices") and result.choices:
                response_text = result.choices[0].message.content or ""
            token_tracker.add_interaction(
                model, system_message, prompt, response_text, timestamp,
            )
        return

    # v3 agent mode: result is a string, tuple, dict, or other non-SDK object
    model = "agent"
    response_text = ""
    if isinstance(result, str):
        response_text = result
    elif isinstance(result, tuple) and len(result) >= 1:
        response_text = str(result[0]) if result[0] else ""
    elif isinstance(result, dict):
        response_text = str(result)

    prompt_text = str(prompt) if prompt else ""
    sys_text = str(system_message) if system_message else ""

    token_tracker.add_agent_invocation(
        model=model,
        wall_clock_seconds=elapsed,
        prompt_text=prompt_text,
        response_text=response_text,
        system_message=sys_text,
    )
