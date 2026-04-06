"""Tests for ai_scientist.agent_backend and ai_scientist.agent_utils."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ai_scientist.agent_backend import (
    AgentBackend,
    AgentError,
    AgentInvocationResult,
    AgentOutputError,
    AgentTimeoutError,
    _extract_json_from_text,
)
from ai_scientist.agent_utils import (
    build_function_call_prompt,
    estimate_tokens,
    format_json_schema_prompt,
    invoke_agent,
    parse_agent_json,
    reset_default_backend,
)


# --- AgentBackend initialization ---


class TestAgentBackendInit:
    def test_default_init(self):
        backend = AgentBackend()
        assert backend.model == "sonnet"
        assert backend.timeout == 300
        assert backend.max_retries == 2
        assert backend.verbose is False
        assert backend.allowed_tools == []
        assert backend._invocation_log == []

    def test_custom_init(self):
        backend = AgentBackend(
            model="opus",
            timeout=600,
            max_retries=3,
            verbose=True,
            allowed_tools=["Bash", "Read"],
        )
        assert backend.model == "opus"
        assert backend.timeout == 600
        assert backend.max_retries == 3
        assert backend.verbose is True
        assert backend.allowed_tools == ["Bash", "Read"]


# --- Prompt building ---


class TestPromptBuilding:
    def test_build_prompt_user_only(self):
        backend = AgentBackend()
        prompt = backend._build_prompt(None, "Hello agent", None)
        assert prompt == "Hello agent"

    def test_build_prompt_with_images(self):
        backend = AgentBackend()
        prompt = backend._build_prompt(
            None, "Describe this", ["/tmp/img1.png", "/tmp/img2.jpg"]
        )
        assert "Describe this" in prompt
        assert "/tmp/img1.png" in prompt
        assert "/tmp/img2.jpg" in prompt

    def test_build_prompt_empty(self):
        backend = AgentBackend()
        prompt = backend._build_prompt(None, None, None)
        assert prompt == ""

    def test_build_prompt_system_not_in_prompt(self):
        """System message is passed via --system-prompt flag, not in prompt text."""
        backend = AgentBackend()
        prompt = backend._build_prompt("You are a scientist", "Do research", None)
        assert "You are a scientist" not in prompt
        assert "Do research" in prompt


# --- Command building ---


class TestCommandBuilding:
    def test_basic_command(self):
        backend = AgentBackend()
        cmd = backend._build_command(
            prompt="test prompt",
            system_message=None,
            json_schema=None,
            model="sonnet",
        )
        assert "claude" in cmd
        assert "-p" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "test prompt" in cmd
        # Should use text output by default
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "text"

    def test_command_with_system_message(self):
        backend = AgentBackend()
        cmd = backend._build_command(
            prompt="test",
            system_message="You are helpful",
            json_schema=None,
            model="sonnet",
        )
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "You are helpful"

    def test_command_with_json_schema(self):
        backend = AgentBackend()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cmd = backend._build_command(
            prompt="test",
            system_message=None,
            json_schema=schema,
            model="sonnet",
        )
        assert "--json-schema" in cmd
        # Output format should be json when schema is provided
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"

    def test_command_with_allowed_tools(self):
        backend = AgentBackend(allowed_tools=["Bash", "Read"])
        cmd = backend._build_command(
            prompt="test",
            system_message=None,
            json_schema=None,
            model="sonnet",
        )
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Bash,Read"

    def test_command_with_extra_args(self):
        backend = AgentBackend(extra_cli_args=["--bare"])
        cmd = backend._build_command(
            prompt="test",
            system_message=None,
            json_schema=None,
            model="sonnet",
        )
        assert "--bare" in cmd


# --- CLI execution (mocked) ---


class TestCLIExecution:
    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_successful_execution(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Hello from agent", stderr=""
        )
        backend = AgentBackend()
        result = backend._execute_cli(["claude", "-p", "test"], timeout=60)
        assert result == "Hello from agent"

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        backend = AgentBackend()
        with pytest.raises(AgentTimeoutError, match="timed out"):
            backend._execute_cli(["claude", "-p", "test"], timeout=60)

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error occurred"
        )
        backend = AgentBackend()
        with pytest.raises(AgentError, match="error occurred"):
            backend._execute_cli(["claude", "-p", "test"], timeout=60)


# --- Full query (mocked) ---


class TestQuery:
    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_query_text_response(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="The hypothesis is X", stderr=""
        )
        backend = AgentBackend()
        result = backend.query(
            system_message="You are a scientist",
            user_message="Generate a hypothesis",
        )
        assert result == "The hypothesis is X"
        assert len(backend._invocation_log) == 1
        assert backend._invocation_log[0].success is True

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_query_json_response(self, mock_run):
        json_output = json.dumps({
            "result": json.dumps({"name": "test", "value": 42})
        })
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json_output, stderr=""
        )
        backend = AgentBackend()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }
        result = backend.query(user_message="Give me data", json_schema=schema)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_query_json_direct_dict(self, mock_run):
        """Test when claude returns JSON directly (not wrapped in result envelope)."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"name": "test"}),
            stderr="",
        )
        backend = AgentBackend()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = backend.query(user_message="test", json_schema=schema)
        assert result == {"name": "test"}

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_query_tracks_failure(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        backend = AgentBackend()
        with pytest.raises(AgentTimeoutError):
            backend.query(user_message="test")
        assert len(backend._invocation_log) == 1
        assert backend._invocation_log[0].success is False
        assert backend._invocation_log[0].error == "timeout"

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_query_model_override(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        backend = AgentBackend(model="sonnet")
        backend.query(user_message="test", model="opus")
        # Check that opus was used in the command
        call_args = mock_run.call_args[0][0]
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "opus"


# --- Stats ---


class TestStats:
    def test_empty_stats(self):
        backend = AgentBackend()
        stats = backend.get_stats()
        assert stats["total_invocations"] == 0
        assert stats["successful"] == 0

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_stats_after_queries(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="response", stderr=""
        )
        backend = AgentBackend()
        backend.query(user_message="test1")
        backend.query(user_message="test2")
        stats = backend.get_stats()
        assert stats["total_invocations"] == 2
        assert stats["successful"] == 2
        assert stats["total_output_chars"] > 0


# --- JSON parsing ---


class TestJSONParsing:
    def test_parse_json_direct(self):
        backend = AgentBackend()
        result = backend._parse_json_output(
            '{"key": "value"}', {"type": "object"}
        )
        assert result == {"key": "value"}

    def test_parse_json_from_code_block(self):
        backend = AgentBackend()
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = backend._parse_json_output(text, {"type": "object"})
        assert result == {"key": "value"}

    def test_parse_json_from_result_envelope(self):
        backend = AgentBackend()
        text = json.dumps({"result": '{"key": "value"}'})
        result = backend._parse_json_output(text, {"type": "object"})
        assert result == {"key": "value"}

    def test_parse_json_from_result_dict(self):
        backend = AgentBackend()
        text = json.dumps({"result": {"key": "value"}})
        result = backend._parse_json_output(text, {"type": "object"})
        assert result == {"key": "value"}

    def test_parse_json_failure(self):
        backend = AgentBackend()
        with pytest.raises(AgentOutputError, match="Could not parse"):
            backend._parse_json_output("not json at all", {"type": "object"})


class TestExtractJsonFromText:
    def test_code_block_json(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json_from_text(text) == {"a": 1}

    def test_plain_code_block(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json_from_text(text) == {"a": 1}

    def test_raw_json_in_text(self):
        text = 'The result is {"a": 1} as expected.'
        assert _extract_json_from_text(text) == {"a": 1}

    def test_no_json(self):
        assert _extract_json_from_text("no json here") is None

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}}'
        result = _extract_json_from_text(text)
        assert result == {"outer": {"inner": 42}}


# --- agent_utils ---


class TestInvokeAgent:
    @patch("ai_scientist.agent_utils.get_default_backend")
    def test_invoke_agent_uses_default_backend(self, mock_get):
        mock_backend = MagicMock()
        mock_backend.query_with_retry.return_value = "response"
        mock_get.return_value = mock_backend

        reset_default_backend()
        result = invoke_agent("test prompt", system_message="sys")
        assert result == "response"
        mock_backend.query_with_retry.assert_called_once()

    def test_invoke_agent_with_explicit_backend(self):
        mock_backend = MagicMock(spec=AgentBackend)
        mock_backend.query_with_retry.return_value = "ok"
        result = invoke_agent("test", backend=mock_backend)
        assert result == "ok"


class TestParseAgentJson:
    def test_direct_json(self):
        assert parse_agent_json('{"a": 1}') == {"a": 1}

    def test_json_in_text(self):
        assert parse_agent_json('result: {"a": 1}') == {"a": 1}

    def test_invalid_json(self):
        with pytest.raises(AgentOutputError):
            parse_agent_json("not json")


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") >= 1

    def test_longer_text(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100  # 400 chars / 4


class TestFormatJsonSchemaPrompt:
    def test_includes_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = format_json_schema_prompt(schema)
        assert "Schema:" in result
        assert '"type": "object"' in result
        assert "ONLY valid JSON" in result


class TestBuildFunctionCallPrompt:
    def test_includes_all_parts(self):
        result = build_function_call_prompt(
            name="evaluate",
            description="Evaluate the result",
            json_schema={"type": "object"},
            user_message="Check this experiment",
        )
        assert "Check this experiment" in result
        assert "evaluate" in result
        assert "Evaluate the result" in result
        assert '"type": "object"' in result


# --- Retry logic ---


class TestRetry:
    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_retry_on_agent_error(self, mock_run):
        # Fail twice, then succeed
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="transient"
            ),
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="transient"
            ),
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout="success", stderr=""
            ),
        ]
        backend = AgentBackend(max_retries=2)
        result = backend.query_with_retry(user_message="test")
        assert result == "success"
        assert len(backend._invocation_log) == 3

    @patch("ai_scientist.agent_backend.subprocess.run")
    def test_no_retry_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        backend = AgentBackend(max_retries=2)
        with pytest.raises(AgentTimeoutError):
            backend.query_with_retry(user_message="test")
        # Should only have been called once (no retry on timeout)
        assert len(backend._invocation_log) == 1
