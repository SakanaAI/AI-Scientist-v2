"""
Tests verifying that FunctionSpec schemas work correctly as agent prompt templates.

Phase 1 verification: ensures all real FunctionSpec schemas from agent_manager.py,
parallel_agent.py, and journal.py produce valid prompts and that matching JSON
responses parse correctly through backend_agent.query().
"""

import json
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.agent_utils import build_function_call_prompt
from ai_scientist.treesearch.backend.utils import FunctionSpec

# Import backend_agent directly to avoid SDK-dependent __init__.py
import ai_scientist.treesearch.backend.backend_agent as backend_agent


@pytest.fixture(autouse=True)
def _reset():
    backend_agent.reset_client()
    yield
    backend_agent.reset_client()


# ---- Define all real FunctionSpec instances from the codebase ----

STAGE_CONFIG_SPEC = FunctionSpec(
    name="generate_stage_config",
    description="Generate configuration for the next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Brief, descriptive name for the stage"},
            "description": {"type": "string", "description": "Detailed description of the stage's purpose"},
            "goals": {"type": "array", "items": {"type": "string"}, "description": "List of specific, measurable goals"},
            "max_iterations": {"type": "integer", "description": "Maximum number of iterations"},
        },
        "required": ["name", "description", "goals", "max_iterations"],
    },
)

STAGE_PROGRESS_EVAL_SPEC = FunctionSpec(
    name="evaluate_stage_progression",
    description="Evaluate readiness to progress to next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "ready_for_next_stage": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "suggested_focus": {"type": "string"},
        },
        "required": ["ready_for_next_stage", "reasoning", "recommendations"],
    },
)

STAGE_COMPLETION_EVAL_SPEC = FunctionSpec(
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

REVIEW_FUNC_SPEC = FunctionSpec(
    name="submit_review",
    description="Submit a review of the execution output",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {"type": "boolean", "description": "true if execution failed or has a bug"},
            "summary": {"type": "string", "description": "bug summary and proposed fix, or empty"},
        },
        "required": ["is_bug", "summary"],
    },
)

VLM_FEEDBACK_SPEC = FunctionSpec(
    name="analyze_experiment_plots",
    description="Analyze experiment plots and provide feedback",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"analysis": {"type": "string"}},
                    "required": ["analysis"],
                },
            },
            "valid_plots_received": {"type": "boolean"},
            "vlm_feedback_summary": {"type": "string"},
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
    },
)

METRIC_PARSE_SPEC = FunctionSpec(
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

PLOT_SELECTION_SPEC = FunctionSpec(
    name="select_plots",
    description="Select the most relevant plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
        },
        "required": ["selected_plots"],
    },
)

NODE_SELECTION_SPEC = FunctionSpec(
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

SUBSTAGE_GOAL_SPEC = FunctionSpec(
    name="generate_substage_goals",
    description="Generate goals for the next sub-stage",
    json_schema={
        "type": "object",
        "properties": {
            "goals": {"type": "string", "description": "Detailed goals for the next sub-stage"},
            "sub_stage_name": {"type": "string", "description": "Name of the next sub-stage"},
        },
        "required": ["goals", "sub_stage_name"],
    },
)

# All specs with matching sample responses
ALL_SPECS_AND_RESPONSES = [
    (
        STAGE_CONFIG_SPEC,
        {
            "name": "Baseline Implementation",
            "description": "Implement the baseline model",
            "goals": ["Implement baseline", "Verify training loop"],
            "max_iterations": 5,
        },
    ),
    (
        STAGE_PROGRESS_EVAL_SPEC,
        {
            "ready_for_next_stage": True,
            "reasoning": "Baseline metrics are stable",
            "recommendations": ["Try larger batch size"],
            "suggested_focus": "hyperparameter tuning",
        },
    ),
    (
        STAGE_COMPLETION_EVAL_SPEC,
        {
            "is_complete": False,
            "reasoning": "Not all ablations complete",
            "missing_criteria": ["Run ablation on dataset B"],
        },
    ),
    (
        REVIEW_FUNC_SPEC,
        {"is_bug": True, "summary": "IndexError in data loader at line 42"},
    ),
    (
        VLM_FEEDBACK_SPEC,
        {
            "plot_analyses": [{"analysis": "Loss curve shows convergence after epoch 20"}],
            "valid_plots_received": True,
            "vlm_feedback_summary": "Training converged successfully",
        },
    ),
    (
        METRIC_PARSE_SPEC,
        {
            "valid_metrics_received": True,
            "metric_names": [
                {
                    "metric_name": "test accuracy",
                    "lower_is_better": False,
                    "description": "Classification accuracy on test set",
                    "data": [
                        {"dataset_name": "CIFAR-10", "final_value": 0.923, "best_value": 0.931},
                    ],
                }
            ],
        },
    ),
    (
        PLOT_SELECTION_SPEC,
        {"selected_plots": ["/tmp/plots/loss.png", "/tmp/plots/accuracy.png"]},
    ),
    (
        NODE_SELECTION_SPEC,
        {"selected_id": "node_42", "reasoning": "Best validation accuracy"},
    ),
    (
        SUBSTAGE_GOAL_SPEC,
        {"goals": "Implement attention mechanism", "sub_stage_name": "attention_integration"},
    ),
]


class TestPromptGeneration:
    """Verify each FunctionSpec produces a valid prompt string."""

    @pytest.mark.parametrize(
        "spec,_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_build_prompt_includes_schema(self, spec, _response):
        prompt = build_function_call_prompt(
            name=spec.name,
            description=spec.description,
            json_schema=spec.json_schema,
            user_message="Test input.",
        )
        assert spec.name in prompt
        assert "JSON" in prompt
        # All required fields should appear in the prompt
        for field in spec.json_schema.get("required", []):
            assert field in prompt

    @pytest.mark.parametrize(
        "spec,_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_prompt_contains_description(self, spec, _response):
        prompt = build_function_call_prompt(
            name=spec.name,
            description=spec.description,
            json_schema=spec.json_schema,
            user_message="Evaluate this.",
        )
        assert spec.description in prompt


class TestBackendAgentQueryWithFuncSpec:
    """Verify backend_agent.query() correctly handles each FunctionSpec."""

    @pytest.mark.parametrize(
        "spec,mock_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_dict_response_passthrough(self, spec, mock_response):
        """When AgentBackend returns a dict, it passes through correctly."""
        with patch.object(AgentBackend, "query", return_value=mock_response):
            output, req_time, in_tok, out_tok, info = backend_agent.query(
                system_message="You are an AI scientist.",
                user_message="Evaluate the experiment.",
                func_spec=spec,
                model="agent",
            )
            assert isinstance(output, dict)
            for key in spec.json_schema.get("required", []):
                assert key in output

    @pytest.mark.parametrize(
        "spec,mock_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_json_string_response_parsed(self, spec, mock_response):
        """When AgentBackend returns a JSON string, it gets parsed to dict."""
        json_str = json.dumps(mock_response)
        with patch.object(AgentBackend, "query", return_value=json_str):
            output, _, _, _, _ = backend_agent.query(
                system_message="sys",
                user_message="usr",
                func_spec=spec,
                model="agent",
            )
            assert isinstance(output, dict)
            for key in spec.json_schema.get("required", []):
                assert key in output


class TestSchemaValidity:
    """Verify all schemas are valid JSON Schema drafts."""

    @pytest.mark.parametrize(
        "spec,_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_schema_is_valid(self, spec, _response):
        """FunctionSpec __post_init__ validates the schema -- if we got here, it's valid."""
        import jsonschema
        jsonschema.Draft7Validator.check_schema(spec.json_schema)

    @pytest.mark.parametrize(
        "spec,mock_response",
        ALL_SPECS_AND_RESPONSES,
        ids=[s.name for s, _ in ALL_SPECS_AND_RESPONSES],
    )
    def test_sample_response_validates_against_schema(self, spec, mock_response):
        """Verify our sample responses actually match the schemas."""
        import jsonschema
        jsonschema.validate(mock_response, spec.json_schema)
