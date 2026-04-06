"""
Integration test: Verify backend_agent produces valid outputs for tree search consumers.

Simulates the key query patterns used by parallel_agent.py, agent_manager.py,
and journal.py through the backend/__init__.py router (in agent mode).
"""

import json
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.treesearch.backend.utils import FunctionSpec, compile_prompt_to_md

# Import backend_agent and __init__ carefully to avoid SDK deps
import ai_scientist.treesearch.backend.backend_agent as backend_agent_mod


@pytest.fixture(autouse=True)
def _reset():
    backend_agent_mod.reset_client()
    yield
    backend_agent_mod.reset_client()


# ---- FunctionSpec instances matching what real consumers use ----

review_func_spec = FunctionSpec(
    name="submit_review",
    description="Submit a review of the execution output",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": ["is_bug", "summary"],
    },
)

stage_completion_eval_spec = FunctionSpec(
    name="evaluate_stage_completion",
    description="Evaluate if the current stage is complete",
    json_schema={
        "type": "object",
        "properties": {
            "is_complete": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "missing_criteria": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["is_complete", "reasoning", "missing_criteria"],
    },
)

node_selection_spec = FunctionSpec(
    name="select_best_implementation",
    description="Select the best implementation from candidates",
    json_schema={
        "type": "object",
        "properties": {
            "selected_id": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["selected_id", "reasoning"],
    },
)

metric_parse_spec = FunctionSpec(
    name="parse_metrics",
    description="Parse metrics from execution output",
    json_schema={
        "type": "object",
        "properties": {
            "valid_metrics_received": {"type": "boolean"},
            "metric_names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "lower_is_better": {"type": "boolean"},
                        "description": {"type": "string"},
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dataset_name": {"type": "string"},
                                    "final_value": {"type": "number"},
                                    "best_value": {"type": "number"},
                                },
                                "required": ["dataset_name", "final_value", "best_value"],
                            },
                        },
                    },
                    "required": ["data", "metric_name", "lower_is_better", "description"],
                },
            },
        },
        "required": ["valid_metrics_received", "metric_names"],
    },
)


class TestPlanAndCodeQuery:
    """Simulate parallel_agent.plan_and_code_query() pattern: plain text query."""

    @patch.object(
        AgentBackend,
        "query",
        return_value=(
            "## Plan\nI will implement a simple baseline model.\n\n"
            "```python\nimport torch\nmodel = torch.nn.Linear(10, 1)\n```"
        ),
    )
    def test_plain_text_query_returns_string(self, mock_query):
        """The plan_and_code_query pattern: system_message dict, no func_spec."""
        prompt = {
            "Introduction": "You are an AI researcher.",
            "Task": "Implement a baseline model.",
        }
        compiled = compile_prompt_to_md(prompt)

        output, req_time, in_tok, out_tok, info = backend_agent_mod.query(
            system_message=compiled,
            user_message=None,
            model="agent",
            temperature=0.7,
        )

        assert isinstance(output, str)
        assert "python" in output
        assert req_time >= 0
        assert info["backend"] == "agent"

    @patch.object(
        AgentBackend,
        "query",
        return_value="Here is the implementation but no code block.",
    )
    def test_plain_text_no_code_block(self, mock_query):
        """If agent returns no code block, output is still a valid string."""
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Write code.",
            user_message=None,
            model="agent",
        )
        assert isinstance(output, str)
        assert "implementation" in output


class TestParseExecResult:
    """Simulate parallel_agent.parse_exec_result() pattern: FunctionSpec query."""

    @patch.object(
        AgentBackend,
        "query",
        return_value={"is_bug": False, "summary": ""},
    )
    def test_successful_execution(self, mock_query):
        prompt = {
            "Introduction": "Analyze the execution output.",
            "Execution output": "Epoch 10: loss=0.05, accuracy=0.98",
        }
        compiled = compile_prompt_to_md(prompt)

        output, _, _, _, _ = backend_agent_mod.query(
            system_message=compiled,
            user_message=None,
            func_spec=review_func_spec,
            model="agent",
            temperature=0.0,
        )

        assert isinstance(output, dict)
        assert output["is_bug"] is False
        assert isinstance(output["summary"], str)

    @patch.object(
        AgentBackend,
        "query",
        return_value={
            "is_bug": True,
            "summary": "IndexError: list index out of range at line 42",
        },
    )
    def test_buggy_execution(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Analyze output.",
            user_message=None,
            func_spec=review_func_spec,
            model="agent",
        )

        assert output["is_bug"] is True
        assert "IndexError" in output["summary"]


class TestStageCompletionEval:
    """Simulate agent_manager evaluate_stage_completion() pattern."""

    @patch.object(
        AgentBackend,
        "query",
        return_value={
            "is_complete": True,
            "reasoning": "All goals met",
            "missing_criteria": [],
        },
    )
    def test_stage_complete(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Evaluate stage completion.",
            user_message=None,
            func_spec=stage_completion_eval_spec,
            model="agent",
            temperature=0.0,
        )

        assert output["is_complete"] is True
        assert isinstance(output["missing_criteria"], list)

    @patch.object(
        AgentBackend,
        "query",
        return_value={
            "is_complete": False,
            "reasoning": "Ablation study incomplete",
            "missing_criteria": ["Run on dataset B", "Compare with baseline"],
        },
    )
    def test_stage_incomplete(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Evaluate stage.",
            user_message=None,
            func_spec=stage_completion_eval_spec,
            model="agent",
        )

        assert output["is_complete"] is False
        assert len(output["missing_criteria"]) == 2


class TestNodeSelection:
    """Simulate journal.py select_best_implementation() pattern."""

    @patch.object(
        AgentBackend,
        "query",
        return_value={
            "selected_id": "node_7",
            "reasoning": "Best accuracy with reasonable training time",
        },
    )
    def test_node_selection(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Select the best implementation.",
            user_message=None,
            func_spec=node_selection_spec,
            model="agent",
        )

        assert output["selected_id"] == "node_7"
        assert isinstance(output["reasoning"], str)


class TestMetricParsing:
    """Simulate parallel_agent metric parsing pattern (complex nested schema)."""

    @patch.object(
        AgentBackend,
        "query",
        return_value={
            "valid_metrics_received": True,
            "metric_names": [
                {
                    "metric_name": "test accuracy",
                    "lower_is_better": False,
                    "description": "Classification accuracy on test set",
                    "data": [
                        {"dataset_name": "CIFAR-10", "final_value": 0.923, "best_value": 0.931},
                        {"dataset_name": "CIFAR-100", "final_value": 0.712, "best_value": 0.725},
                    ],
                },
                {
                    "metric_name": "training loss",
                    "lower_is_better": True,
                    "description": "Cross-entropy loss during training",
                    "data": [
                        {"dataset_name": "CIFAR-10", "final_value": 0.05, "best_value": 0.03},
                    ],
                },
            ],
        },
    )
    def test_complex_metric_parsing(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Parse the metrics from execution output.",
            user_message=None,
            func_spec=metric_parse_spec,
            model="agent",
        )

        assert output["valid_metrics_received"] is True
        assert len(output["metric_names"]) == 2
        assert output["metric_names"][0]["metric_name"] == "test accuracy"
        assert len(output["metric_names"][0]["data"]) == 2
        assert output["metric_names"][1]["lower_is_better"] is True


class TestCompilePromptToMd:
    """Verify compile_prompt_to_md works with dict prompts (used by all consumers)."""

    def test_dict_prompt(self):
        prompt = {
            "Introduction": "You are an AI researcher.",
            "Task": "Implement a model.",
            "Constraints": ["Use PyTorch", "Keep it simple"],
        }
        compiled = compile_prompt_to_md(prompt)
        assert isinstance(compiled, str)
        assert "Introduction" in compiled
        assert "AI researcher" in compiled
        assert "PyTorch" in compiled

    def test_string_prompt(self):
        result = compile_prompt_to_md("Just a string")
        assert isinstance(result, str)
        assert "Just a string" in result

    def test_none_prompt(self):
        result = compile_prompt_to_md(None)
        assert result == ""


class TestQueryRouterIntegration:
    """Test that backend_agent.query() works with compiled prompt inputs (as __init__.py would call it)."""

    @patch.object(AgentBackend, "query", return_value="agent response")
    def test_routes_dict_prompt_through_agent(self, mock_query):
        """Dict system_message gets compiled to markdown, then passed to agent."""
        compiled = compile_prompt_to_md({"Task": "Do something"})
        output, _, _, _, _ = backend_agent_mod.query(
            system_message=compiled,
            user_message=None,
            model="agent",
            temperature=0.5,
        )

        assert output == "agent response"

    @patch.object(
        AgentBackend,
        "query",
        return_value={"is_bug": False, "summary": "All good"},
    )
    def test_routes_func_spec_through_agent(self, mock_query):
        output, _, _, _, _ = backend_agent_mod.query(
            system_message="Review output.",
            user_message=None,
            func_spec=review_func_spec,
            model="agent",
        )

        assert isinstance(output, dict)
        assert output["is_bug"] is False
