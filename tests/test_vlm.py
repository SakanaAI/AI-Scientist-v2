"""Tests for ai_scientist.vlm (v3 agent mode)."""

import json
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.vlm import (
    create_client,
    extract_json_between_markers,
    get_batch_responses_from_vlm,
    get_response_from_vlm,
    make_llm_call,
    make_vlm_call,
)


class TestCreateClient:
    def test_returns_agent_backend(self):
        client, model = create_client("gpt-4o")
        assert isinstance(client, AgentBackend)
        assert model == "gpt-4o"

    def test_any_model_string(self):
        client, model = create_client("ollama/qwen2.5vl:32b")
        assert isinstance(client, AgentBackend)


class TestGetResponseFromVlm:
    @patch.object(AgentBackend, "query", return_value="The image shows a loss curve.")
    def test_single_image(self, mock_query):
        client, model = create_client("agent")
        content, history = get_response_from_vlm(
            "Describe this plot.",
            "/tmp/plot.png",
            client, model,
            system_message="You analyze plots.",
        )
        assert content == "The image shows a loss curve."
        assert len(history) == 2
        # Verify images were passed to the backend
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["images"] == ["/tmp/plot.png"]

    @patch.object(AgentBackend, "query", return_value="Multiple plots analyzed.")
    def test_multiple_images(self, mock_query):
        client, model = create_client("agent")
        images = ["/tmp/plot1.png", "/tmp/plot2.png", "/tmp/plot3.png"]
        content, history = get_response_from_vlm(
            "Compare these plots.",
            images, client, model,
            system_message="sys",
        )
        assert content == "Multiple plots analyzed."
        call_kwargs = mock_query.call_args.kwargs
        assert len(call_kwargs["images"]) == 3

    @patch.object(AgentBackend, "query", return_value="Limited analysis.")
    def test_max_images_limit(self, mock_query):
        client, model = create_client("agent")
        images = [f"/tmp/img{i}.png" for i in range(30)]
        content, _ = get_response_from_vlm(
            "Analyze.", images, client, model,
            system_message="sys", max_images=5,
        )
        call_kwargs = mock_query.call_args.kwargs
        assert len(call_kwargs["images"]) == 5

    @patch.object(AgentBackend, "query", return_value="response")
    def test_passes_system_message(self, mock_query):
        client, model = create_client("agent")
        get_response_from_vlm(
            "msg", "/tmp/img.png", client, model,
            system_message="You are a vision expert.",
        )
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["system_message"] == "You are a vision expert."

    @patch.object(AgentBackend, "query", return_value="resp")
    def test_with_msg_history(self, mock_query):
        client, model = create_client("agent")
        history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "prev resp"},
        ]
        content, new_history = get_response_from_vlm(
            "follow up", "/tmp/img.png", client, model,
            system_message="sys", msg_history=history,
        )
        assert len(new_history) == 4


class TestGetBatchResponsesFromVlm:
    @patch.object(AgentBackend, "query", side_effect=["r1", "r2", "r3"])
    def test_multiple_responses(self, mock_query):
        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_vlm(
            "Describe.", "/tmp/img.png", client, model,
            system_message="sys", n_responses=3,
        )
        assert len(contents) == 3
        assert contents == ["r1", "r2", "r3"]
        assert len(histories) == 3
        assert mock_query.call_count == 3

    @patch.object(AgentBackend, "query", return_value="single")
    def test_single_response(self, mock_query):
        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_vlm(
            "Describe.", "/tmp/img.png", client, model,
            system_message="sys", n_responses=1,
        )
        assert len(contents) == 1


class TestMakeLlmCall:
    @patch.object(AgentBackend, "query", return_value="text response")
    def test_basic_call(self, mock_query):
        client, model = create_client("agent")
        result = make_llm_call(
            client, model, 0.7, "system msg",
            [{"role": "user", "content": "Hello"}],
        )
        assert result == "text response"

    @patch.object(AgentBackend, "query", return_value="multimodal response")
    def test_multimodal_content_blocks(self, mock_query):
        client, model = create_client("agent")
        prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            ]},
        ]
        result = make_llm_call(client, model, 0.7, "sys", prompt)
        assert result == "multimodal response"
        # Verify text was extracted from content blocks
        call_kwargs = mock_query.call_args.kwargs
        assert "Describe this." in call_kwargs["user_message"]


class TestMakeVlmCall:
    @patch.object(AgentBackend, "query", return_value="vlm response")
    def test_basic_vlm_call(self, mock_query):
        client, model = create_client("agent")
        prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in the image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            ]},
        ]
        result = make_vlm_call(client, model, 0.7, "sys", prompt)
        assert result == "vlm response"


class TestExtractJsonBetweenMarkers:
    def test_json_code_block(self):
        text = 'Text\n```json\n{"key": "val"}\n```\nMore'
        assert extract_json_between_markers(text) == {"key": "val"}

    def test_raw_json(self):
        assert extract_json_between_markers('{"a": 1}') == {"a": 1}

    def test_no_json(self):
        assert extract_json_between_markers("no json") is None
