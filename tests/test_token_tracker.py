"""Tests for ai_scientist/utils/token_tracker.py (v3 agent mode)."""

import time
import pytest

from ai_scientist.utils.token_tracker import (
    TokenTracker,
    token_tracker,
    track_token_usage,
)


@pytest.fixture
def tracker():
    t = TokenTracker()
    yield t


class TestTokenTracker:
    def test_initial_state(self, tracker):
        summary = tracker.get_summary()
        assert summary == {}

    def test_add_tokens(self, tracker):
        tracker.add_tokens("agent", prompt_tokens=100, completion_tokens=50)
        summary = tracker.get_summary()
        assert "agent" in summary
        assert summary["agent"]["tokens"]["prompt"] == 100
        assert summary["agent"]["tokens"]["completion"] == 50

    def test_add_agent_invocation(self, tracker):
        tracker.add_agent_invocation(
            model="agent",
            wall_clock_seconds=2.5,
            prompt_text="Hello world",
            response_text="Hi there, how can I help?",
            system_message="You are helpful.",
        )
        summary = tracker.get_summary()
        assert "agent" in summary
        assert summary["agent"]["tokens"]["prompt"] > 0
        assert summary["agent"]["tokens"]["completion"] > 0
        assert summary["agent"]["agent_stats"]["invocations"] == 1
        assert summary["agent"]["agent_stats"]["wall_clock_seconds"] == 2.5

    def test_multiple_invocations_accumulate(self, tracker):
        tracker.add_agent_invocation("agent", 1.0, "prompt1", "response1")
        tracker.add_agent_invocation("agent", 2.0, "prompt2", "response2")
        summary = tracker.get_summary()
        assert summary["agent"]["agent_stats"]["invocations"] == 2
        assert summary["agent"]["agent_stats"]["wall_clock_seconds"] == 3.0

    def test_get_interactions(self, tracker):
        tracker.add_agent_invocation("agent", 1.0, "p", "r", "sys")
        interactions = tracker.get_interactions()
        assert "agent" in interactions
        assert len(interactions["agent"]) == 1
        assert interactions["agent"][0]["estimated_tokens"] is True

    def test_get_interactions_filtered(self, tracker):
        tracker.add_agent_invocation("agent", 1.0, "p1", "r1")
        tracker.add_agent_invocation("other", 2.0, "p2", "r2")
        filtered = tracker.get_interactions(model="agent")
        assert "agent" in filtered
        assert "other" not in filtered

    def test_reset(self, tracker):
        tracker.add_agent_invocation("agent", 1.0, "p", "r")
        tracker.reset()
        assert tracker.get_summary() == {}
        assert tracker.get_interactions() == {}

    def test_calculate_cost_unknown_model(self, tracker):
        tracker.add_tokens("agent", 1000, 500)
        cost = tracker.calculate_cost("agent")
        assert cost == 0.0  # No price info for "agent"

    def test_calculate_cost_known_model(self, tracker):
        tracker.add_tokens("gpt-4o-2024-08-06", 1000, 500)
        cost = tracker.calculate_cost("gpt-4o-2024-08-06")
        assert cost > 0


class TestTrackTokenUsageDecorator:
    def test_decorator_with_string_result(self):
        @track_token_usage
        def my_func(prompt=None, system_message=None):
            return "Hello from agent"

        token_tracker.reset()
        result = my_func(prompt="Test prompt", system_message="sys")
        assert result == "Hello from agent"

        summary = token_tracker.get_summary()
        assert "agent" in summary
        assert summary["agent"]["tokens"]["completion"] > 0

    def test_decorator_with_tuple_result(self):
        @track_token_usage
        def my_func(prompt=None, system_message=None):
            return ("response text", [{"role": "user", "content": "test"}])

        token_tracker.reset()
        result = my_func(prompt="Test", system_message="sys")
        assert isinstance(result, tuple)
        assert result[0] == "response text"

    def test_decorator_with_dict_result(self):
        @track_token_usage
        def my_func(prompt=None, system_message=None):
            return {"is_bug": False, "summary": "All good"}

        token_tracker.reset()
        result = my_func(prompt="Test", system_message="sys")
        assert isinstance(result, dict)

        summary = token_tracker.get_summary()
        assert "agent" in summary

    def test_decorator_preserves_function_name(self):
        @track_token_usage
        def my_special_func(prompt=None, system_message=None):
            return "ok"

        assert my_special_func.__name__ == "my_special_func"

    def test_decorator_tracks_wall_clock_time(self):
        @track_token_usage
        def slow_func(prompt=None, system_message=None):
            time.sleep(0.05)
            return "done"

        token_tracker.reset()
        slow_func(prompt="test", system_message="sys")

        summary = token_tracker.get_summary()
        assert summary["agent"]["agent_stats"]["wall_clock_seconds"] >= 0.04


class TestGlobalTracker:
    def test_global_instance_exists(self):
        assert token_tracker is not None
        assert isinstance(token_tracker, TokenTracker)
