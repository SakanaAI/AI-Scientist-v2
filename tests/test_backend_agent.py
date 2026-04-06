"""Tests for ai_scientist.treesearch.backend.backend_agent."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

# In v3 mode (default), __init__.py only imports backend_agent (no SDK deps),
# so we can import directly.
from ai_scientist.treesearch.backend.backend_agent import (
    get_ai_client,
    query,
    reset_client,
)
from ai_scientist.treesearch.backend.utils import FunctionSpec
from ai_scientist.agent_backend import AgentBackend, AgentError, AgentTimeoutError


@pytest.fixture(autouse=True)
def _reset():
    """Reset the module-level backend before each test."""
    reset_client()
    yield
    reset_client()


# --- get_ai_client ---


class TestGetAiClient:
    def test_returns_agent_backend(self):
        client = get_ai_client("agent")
        assert isinstance(client, AgentBackend)

    def test_returns_same_instance(self):
        c1 = get_ai_client("agent")
        c2 = get_ai_client("agent")
        assert c1 is c2

    def test_accepts_timeout_kwarg(self):
        client = get_ai_client("agent", timeout=600)
        assert client.timeout == 600

    def test_accepts_verbose_kwarg(self):
        client = get_ai_client("agent", verbose=True)
        assert client.verbose is True

    def test_reset_clears_instance(self):
        c1 = get_ai_client("agent")
        reset_client()
        c2 = get_ai_client("agent")
        assert c1 is not c2


# --- query without func_spec ---


class TestQueryPlainText:
    @patch.object(AgentBackend, "query", return_value="Hello, world!")
    def test_returns_string(self, mock_query):
        output, req_time, in_tok, out_tok, info = query(
            system_message="You are helpful.",
            user_message="Say hello.",
            model="agent",
        )
        assert output == "Hello, world!"
        assert isinstance(req_time, float)
        assert req_time >= 0
        assert in_tok > 0
        assert out_tok > 0
        assert info["backend"] == "agent"

    @patch.object(AgentBackend, "query", return_value="test response")
    def test_passes_system_and_user_messages(self, mock_query):
        query(
            system_message="sys",
            user_message="usr",
            model="agent",
        )
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args
        assert call_kwargs.kwargs["system_message"] == "sys"
        assert call_kwargs.kwargs["user_message"] == "usr"

    @patch.object(AgentBackend, "query", return_value="ok")
    def test_none_messages(self, mock_query):
        output, _, _, _, _ = query(
            system_message=None,
            user_message=None,
            model="agent",
        )
        assert output == "ok"

    @patch.object(AgentBackend, "query", return_value="response")
    def test_estimated_tokens_flag(self, mock_query):
        _, _, _, _, info = query(
            system_message="test",
            user_message="test",
            model="agent",
        )
        assert info["estimated_tokens"] is True


# --- query with func_spec ---


class TestQueryWithFuncSpec:
    @pytest.fixture
    def sample_spec(self):
        return FunctionSpec(
            name="test_function",
            description="A test function.",
            json_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["result", "score"],
            },
        )

    @patch.object(
        AgentBackend,
        "query",
        return_value={"result": "success", "score": 0.95},
    )
    def test_returns_dict(self, mock_query, sample_spec):
        output, req_time, in_tok, out_tok, info = query(
            system_message="You are a scorer.",
            user_message="Score this.",
            func_spec=sample_spec,
            model="agent",
        )
        assert isinstance(output, dict)
        assert output["result"] == "success"
        assert output["score"] == 0.95

    @patch.object(AgentBackend, "query")
    def test_passes_json_schema(self, mock_query, sample_spec):
        mock_query.return_value = {"result": "x", "score": 1}
        query(
            system_message="sys",
            user_message="usr",
            func_spec=sample_spec,
            model="agent",
        )
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["json_schema"] == sample_spec.json_schema

    @patch.object(AgentBackend, "query")
    def test_augments_user_message_with_func_spec(self, mock_query, sample_spec):
        mock_query.return_value = {"result": "x", "score": 1}
        query(
            system_message="sys",
            user_message="Do something.",
            func_spec=sample_spec,
            model="agent",
        )
        call_kwargs = mock_query.call_args.kwargs
        user_msg = call_kwargs["user_message"]
        # Should contain function calling instructions
        assert "test_function" in user_msg
        assert "Do something." in user_msg
        assert "JSON" in user_msg

    @patch.object(
        AgentBackend,
        "query",
        return_value='{"result": "parsed", "score": 0.5}',
    )
    def test_string_result_parsed_as_json(self, mock_query, sample_spec):
        """If backend returns a string instead of dict, query() tries JSON parse."""
        output, _, _, _, _ = query(
            system_message="sys",
            user_message="usr",
            func_spec=sample_spec,
            model="agent",
        )
        assert isinstance(output, dict)
        assert output["result"] == "parsed"

    @patch.object(
        AgentBackend,
        "query",
        return_value="not json at all",
    )
    def test_unparseable_string_returned_raw(self, mock_query, sample_spec):
        """If string result can't be parsed as JSON, return raw string with warning."""
        output, _, _, _, _ = query(
            system_message="sys",
            user_message="usr",
            func_spec=sample_spec,
            model="agent",
        )
        assert output == "not json at all"


# --- error handling ---


class TestQueryErrors:
    @patch.object(AgentBackend, "query", side_effect=AgentError("fail"))
    def test_agent_error_propagates(self, mock_query):
        with pytest.raises(AgentError, match="fail"):
            query(
                system_message="sys",
                user_message="usr",
                model="agent",
            )

    @patch.object(AgentBackend, "query", side_effect=AgentTimeoutError("timeout"))
    def test_timeout_error_propagates(self, mock_query):
        with pytest.raises(AgentTimeoutError):
            query(
                system_message="sys",
                user_message="usr",
                model="agent",
            )


# --- token estimation ---


class TestTokenEstimation:
    @patch.object(AgentBackend, "query", return_value="short")
    def test_in_tokens_estimated_from_input(self, mock_query):
        _, _, in_tok, _, _ = query(
            system_message="a" * 100,
            user_message="b" * 100,
            model="agent",
        )
        # ~200 chars / 4 = ~50 tokens
        assert in_tok >= 40
        assert in_tok <= 60

    @patch.object(AgentBackend, "query", return_value="x" * 400)
    def test_out_tokens_estimated_from_output(self, mock_query):
        _, _, _, out_tok, _ = query(
            system_message="sys",
            user_message="usr",
            model="agent",
        )
        # 400 chars / 4 = 100 tokens
        assert out_tok == 100
