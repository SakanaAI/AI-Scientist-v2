"""Tests for ai_scientist.llm (v3 agent mode)."""

import json
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.llm import (
    create_client,
    extract_json_between_markers,
    get_batch_responses_from_llm,
    get_response_from_llm,
    make_llm_call,
)


class TestCreateClient:
    def test_returns_agent_backend(self):
        client, model = create_client("gpt-4o")
        assert isinstance(client, AgentBackend)
        assert model == "gpt-4o"

    def test_any_model_string_works(self):
        client, model = create_client("claude-3-5-sonnet-20241022")
        assert isinstance(client, AgentBackend)
        assert model == "claude-3-5-sonnet-20241022"

    def test_returns_same_backend_instance(self):
        c1, _ = create_client("model-a")
        c2, _ = create_client("model-b")
        assert c1 is c2  # Same module-level singleton


class TestGetResponseFromLlm:
    @patch.object(AgentBackend, "query", return_value="The answer is 42.")
    def test_basic_response(self, mock_query):
        client, model = create_client("agent")
        content, history = get_response_from_llm(
            "What is the meaning of life?",
            client, model,
            system_message="You are a philosopher.",
        )
        assert content == "The answer is 42."
        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "The answer is 42."

    @patch.object(AgentBackend, "query", return_value="Response with history.")
    def test_with_msg_history(self, mock_query):
        client, model = create_client("agent")
        existing_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        content, history = get_response_from_llm(
            "Follow up question",
            client, model,
            system_message="You are helpful.",
            msg_history=existing_history,
        )
        assert content == "Response with history."
        assert len(history) == 4  # 2 existing + 2 new
        # Verify the agent received history context
        call_kwargs = mock_query.call_args.kwargs
        assert "Hello" in call_kwargs["user_message"]
        assert "Follow up question" in call_kwargs["user_message"]

    @patch.object(AgentBackend, "query", return_value="test")
    def test_passes_system_message(self, mock_query):
        client, model = create_client("agent")
        get_response_from_llm(
            "prompt", client, model,
            system_message="Be concise.",
        )
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["system_message"] == "Be concise."

    @patch.object(AgentBackend, "query", return_value="test")
    def test_none_msg_history(self, mock_query):
        client, model = create_client("agent")
        content, history = get_response_from_llm(
            "test prompt", client, model,
            system_message="sys",
            msg_history=None,
        )
        assert len(history) == 2

    @patch.object(AgentBackend, "query", return_value="test")
    def test_anthropic_style_history(self, mock_query):
        """Handle Anthropic-style content blocks in history."""
        client, model = create_client("agent")
        history = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        content, new_history = get_response_from_llm(
            "Next", client, model,
            system_message="sys",
            msg_history=history,
        )
        # Should have processed the history without error
        assert content == "test"
        call_kwargs = mock_query.call_args.kwargs
        assert "Hello" in call_kwargs["user_message"]


class TestGetBatchResponsesFromLlm:
    @patch.object(AgentBackend, "query", return_value="response")
    def test_single_response(self, mock_query):
        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_llm(
            "prompt", client, model,
            system_message="sys",
            n_responses=1,
        )
        assert len(contents) == 1
        assert len(histories) == 1
        assert contents[0] == "response"

    @patch.object(AgentBackend, "query", side_effect=["r1", "r2", "r3"])
    def test_multiple_responses(self, mock_query):
        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_llm(
            "prompt", client, model,
            system_message="sys",
            n_responses=3,
        )
        assert len(contents) == 3
        assert contents == ["r1", "r2", "r3"]
        assert len(histories) == 3
        assert mock_query.call_count == 3


class TestMakeLlmCall:
    @patch.object(AgentBackend, "query", return_value="raw response")
    def test_basic_call(self, mock_query):
        client, model = create_client("agent")
        prompt = [
            {"role": "user", "content": "Hello"},
        ]
        result = make_llm_call(client, model, 0.7, "system msg", prompt)
        assert result == "raw response"
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["system_message"] == "system msg"
        assert "Hello" in call_kwargs["user_message"]


class TestExtractJsonBetweenMarkers:
    def test_json_code_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json_between_markers(text)
        assert result == {"key": "value"}

    def test_raw_json(self):
        text = 'The result is {"answer": 42} as shown.'
        result = extract_json_between_markers(text)
        assert result == {"answer": 42}

    def test_no_json(self):
        result = extract_json_between_markers("No JSON here")
        assert result is None

    def test_malformed_json_with_control_chars(self):
        text = '```json\n{"key": "val\x00ue"}\n```'
        result = extract_json_between_markers(text)
        assert result == {"key": "value"}
