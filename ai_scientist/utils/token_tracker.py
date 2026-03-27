from functools import wraps
from typing import Dict, Optional, List
import tiktoken
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

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
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")

        result = await func(*args, **kwargs)

        # Handle cases where the LLM response lacks usage information
        # (e.g., some providers/models don't return token usage)
        if not hasattr(result, "usage") or result.usage is None:
            logging.warning(
                "track_token_usage: LLM response has no usage information. "
                "Token tracking will be skipped for this call."
            )
            return result

        model = result.model
        timestamp = result.created

        completion_details = getattr(result.usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0)
        else:
            reasoning_tokens = 0

        prompt_details = getattr(result.usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

        token_tracker.add_tokens(
            model,
            result.usage.prompt_tokens or 0,
            result.usage.completion_tokens or 0,
            reasoning_tokens or 0,
            cached_tokens or 0,
        )
        # Add interaction details
        token_tracker.add_interaction(
            model,
            system_message,
            prompt,
            result.choices[0].message.content if result.choices else "",
            timestamp,
        )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")

        result = func(*args, **kwargs)

        # Handle cases where the LLM response lacks usage information
        # (e.g., some providers/models don't return token usage)
        if not hasattr(result, "usage") or result.usage is None:
            logging.warning(
                "track_token_usage: LLM response has no usage information. "
                "Token tracking will be skipped for this call."
            )
            return result

        model = result.model
        timestamp = result.created

        completion_details = getattr(result.usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0)
        else:
            reasoning_tokens = 0

        prompt_details = getattr(result.usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

        token_tracker.add_tokens(
            model,
            result.usage.prompt_tokens or 0,
            result.usage.completion_tokens or 0,
            reasoning_tokens or 0,
            cached_tokens or 0,
        )
        # Add interaction details
        token_tracker.add_interaction(
            model,
            system_message,
            prompt,
            result.choices[0].message.content if result.choices else "",
            timestamp,
        )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
