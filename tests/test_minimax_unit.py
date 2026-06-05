"""Unit tests for MiniMax provider integration."""

import os
import re
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for ai_scientist/llm.py helpers
# ---------------------------------------------------------------------------

class TestIsMiniMaxModel:
    """Tests for _is_minimax_model helper."""

    def test_minimax_m3(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("MiniMax-M3") is True

    def test_minimax_m27(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("MiniMax-M2.7") is True

    def test_minimax_m27_highspeed(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("MiniMax-M2.7-highspeed") is True

    def test_openai_model(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("gpt-4o") is False

    def test_claude_model(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("claude-3-5-sonnet-20241022") is False

    def test_ollama_model(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("ollama/qwen3:8b") is False

    def test_gemini_model(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("gemini-2.0-flash") is False

    def test_empty_string(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("") is False

    def test_partial_match(self):
        from ai_scientist.llm import _is_minimax_model
        assert _is_minimax_model("minimax-M2.7") is False  # case-sensitive


class TestClampTemperatureMiniMax:
    """Tests for _clamp_temperature_minimax helper."""

    def test_normal_temperature(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(0.7) == 0.7

    def test_zero_temperature_clamped(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(0.0) == 0.01

    def test_negative_temperature_clamped(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(-1.0) == 0.01

    def test_high_temperature_clamped(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(2.0) == 1.0

    def test_boundary_one(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(1.0) == 1.0

    def test_small_positive(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(0.01) == 0.01

    def test_mid_range(self):
        from ai_scientist.llm import _clamp_temperature_minimax
        assert _clamp_temperature_minimax(0.5) == 0.5


class TestStripThinkTags:
    """Tests for _strip_think_tags helper."""

    def test_no_think_tags(self):
        from ai_scientist.llm import _strip_think_tags
        assert _strip_think_tags("Hello world") == "Hello world"

    def test_with_think_tags(self):
        from ai_scientist.llm import _strip_think_tags
        text = "<think>Let me reason about this.</think>The answer is 42."
        assert _strip_think_tags(text) == "The answer is 42."

    def test_multiline_think_tags(self):
        from ai_scientist.llm import _strip_think_tags
        text = "<think>\nStep 1: analyze\nStep 2: conclude\n</think>\nFinal answer."
        assert _strip_think_tags(text) == "Final answer."

    def test_empty_string(self):
        from ai_scientist.llm import _strip_think_tags
        assert _strip_think_tags("") == ""

    def test_none_passthrough(self):
        from ai_scientist.llm import _strip_think_tags
        assert _strip_think_tags(None) is None

    def test_think_tag_in_middle(self):
        from ai_scientist.llm import _strip_think_tags
        text = "Prefix <think>reasoning</think> suffix"
        result = _strip_think_tags(text)
        assert "reasoning" not in result
        assert "Prefix" in result
        assert "suffix" in result


# ---------------------------------------------------------------------------
# Tests for AVAILABLE_LLMS
# ---------------------------------------------------------------------------

class TestAvailableLLMs:
    """Tests that MiniMax models are registered in AVAILABLE_LLMS."""

    def test_minimax_m3_in_list(self):
        from ai_scientist.llm import AVAILABLE_LLMS
        assert "MiniMax-M3" in AVAILABLE_LLMS

    def test_minimax_m27_in_list(self):
        from ai_scientist.llm import AVAILABLE_LLMS
        assert "MiniMax-M2.7" in AVAILABLE_LLMS

    def test_minimax_m27_highspeed_in_list(self):
        from ai_scientist.llm import AVAILABLE_LLMS
        assert "MiniMax-M2.7-highspeed" in AVAILABLE_LLMS

    def test_minimax_m3_listed_first(self):
        from ai_scientist.llm import AVAILABLE_LLMS
        minimax_models = [m for m in AVAILABLE_LLMS if m.startswith("MiniMax-")]
        assert minimax_models[0] == "MiniMax-M3"


# ---------------------------------------------------------------------------
# Tests for create_client (llm.py)
# ---------------------------------------------------------------------------

class TestCreateClient:
    """Tests for create_client with MiniMax models."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_creates_openai_client_for_m3(self):
        from ai_scientist.llm import create_client
        client, model = create_client("MiniMax-M3")
        assert model == "MiniMax-M3"
        assert client.base_url.host == "api.minimax.io"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_creates_openai_client_for_minimax(self):
        from ai_scientist.llm import create_client
        client, model = create_client("MiniMax-M2.7")
        assert model == "MiniMax-M2.7"
        # Client should be an OpenAI instance configured for MiniMax
        assert client.base_url.host == "api.minimax.io"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_creates_client_for_highspeed(self):
        from ai_scientist.llm import create_client
        client, model = create_client("MiniMax-M2.7-highspeed")
        assert model == "MiniMax-M2.7-highspeed"
        assert client.base_url.host == "api.minimax.io"

    def test_missing_api_key_raises(self):
        from ai_scientist.llm import create_client
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(KeyError):
                create_client("MiniMax-M2.7")


# ---------------------------------------------------------------------------
# Tests for get_response_from_llm with MiniMax (mocked)
# ---------------------------------------------------------------------------

class TestGetResponseFromLLMMiniMax:
    """Tests for get_response_from_llm with MiniMax model (mocked API)."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_response_path(self):
        from ai_scientist.llm import get_response_from_llm

        mock_choice = MagicMock()
        mock_choice.message.content = "Test response from MiniMax"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        content, history = get_response_from_llm(
            prompt="Hello",
            client=mock_client,
            model="MiniMax-M2.7",
            system_message="You are helpful.",
            temperature=0.7,
        )

        assert content == "Test response from MiniMax"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "MiniMax-M2.7"
        assert call_kwargs.kwargs["temperature"] == 0.7

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_strips_think_tags(self):
        from ai_scientist.llm import get_response_from_llm

        mock_choice = MagicMock()
        mock_choice.message.content = "<think>reasoning here</think>Clean answer"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        content, _ = get_response_from_llm(
            prompt="Hello",
            client=mock_client,
            model="MiniMax-M2.7",
            system_message="You are helpful.",
            temperature=0.5,
        )

        assert content == "Clean answer"
        assert "<think>" not in content

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_temperature_clamping_zero(self):
        from ai_scientist.llm import get_response_from_llm

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        get_response_from_llm(
            prompt="Hello",
            client=mock_client,
            model="MiniMax-M2.7",
            system_message="system",
            temperature=0.0,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        # Temperature 0.0 should be clamped to 0.01
        assert call_kwargs.kwargs["temperature"] == 0.01


# ---------------------------------------------------------------------------
# Tests for make_llm_call with MiniMax (mocked)
# ---------------------------------------------------------------------------

class TestMakeLLMCallMiniMax:
    """Tests for make_llm_call with MiniMax model."""

    def test_minimax_call_format(self):
        from ai_scientist.llm import make_llm_call

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        make_llm_call(
            client=mock_client,
            model="MiniMax-M2.7-highspeed",
            temperature=0.5,
            system_message="system",
            prompt=[{"role": "user", "content": "test"}],
        )

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "MiniMax-M2.7-highspeed"
        assert call_kwargs.kwargs["temperature"] == 0.5

    def test_minimax_temperature_clamped_high(self):
        from ai_scientist.llm import make_llm_call

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        make_llm_call(
            client=mock_client,
            model="MiniMax-M2.7",
            temperature=1.5,
            system_message="system",
            prompt=[{"role": "user", "content": "test"}],
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Tests for treesearch backend __init__.py
# ---------------------------------------------------------------------------

class TestTreesearchBackendMiniMax:
    """Tests for treesearch backend MiniMax routing."""

    def test_is_minimax_model(self):
        try:
            from ai_scientist.treesearch.backend import _is_minimax_model
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        assert _is_minimax_model("MiniMax-M2.7") is True
        assert _is_minimax_model("gpt-4o") is False

    def test_clamp_temperature(self):
        try:
            from ai_scientist.treesearch.backend import _clamp_temperature_minimax
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        assert _clamp_temperature_minimax(0.0) == 0.01
        assert _clamp_temperature_minimax(0.5) == 0.5
        assert _clamp_temperature_minimax(2.0) == 1.0

    def test_strip_think_tags(self):
        try:
            from ai_scientist.treesearch.backend import _strip_think_tags
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        assert _strip_think_tags("<think>x</think>result") == "result"
        assert _strip_think_tags("no tags") == "no tags"


# ---------------------------------------------------------------------------
# Tests for treesearch backend_openai.py client creation
# ---------------------------------------------------------------------------

class TestBackendOpenAIMiniMax:
    """Tests for backend_openai.get_ai_client with MiniMax models."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_client_base_url(self):
        try:
            from ai_scientist.treesearch.backend.backend_openai import get_ai_client
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        client = get_ai_client("MiniMax-M2.7")
        assert "minimax" in str(client.base_url).lower()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_highspeed_client(self):
        try:
            from ai_scientist.treesearch.backend.backend_openai import get_ai_client
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        client = get_ai_client("MiniMax-M2.7-highspeed")
        assert "minimax" in str(client.base_url).lower()

    def test_openai_client_default(self):
        try:
            from ai_scientist.treesearch.backend.backend_openai import get_ai_client
        except (ImportError, AttributeError):
            pytest.skip("treesearch backend not importable (missing anthropic[bedrock])")
        client = get_ai_client("gpt-4o")
        assert "minimax" not in str(client.base_url).lower()
