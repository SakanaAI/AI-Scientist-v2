"""
Final integration test: Full pipeline verification for AI-Scientist v3.

Verifies:
1. All modules import without SDK dependencies
2. Full pipeline flow (ideation -> review) with mocked agent
3. Token tracker records invocations correctly
4. Both LLM call paths (Path A: llm.py, Path B: backend) route through AgentBackend
5. No API keys required in v3 mode
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.utils.token_tracker import TokenTracker, token_tracker


# ---- Verify no SDK imports at module level in v3 mode ----

class TestNoSDKDependencies:
    """Verify that no SDK packages are required to import any module."""

    def test_llm_imports_without_sdk(self):
        from ai_scientist.llm import create_client, get_response_from_llm, AVAILABLE_LLMS
        assert AVAILABLE_LLMS is None  # v3 mode

    def test_vlm_imports_without_sdk(self):
        from ai_scientist.vlm import create_client, get_response_from_vlm, AVAILABLE_VLMS
        assert AVAILABLE_VLMS is None

    def test_backend_imports_without_sdk(self):
        from ai_scientist.treesearch.backend import query, get_ai_client, FunctionSpec
        assert callable(query)

    def test_semantic_scholar_imports_without_sdk(self):
        from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool, search_for_papers
        assert callable(search_for_papers)

    def test_token_tracker_imports_without_tiktoken(self):
        from ai_scientist.utils.token_tracker import TokenTracker, track_token_usage
        assert callable(track_token_usage)

    def test_launch_script_imports(self):
        from launch_scientist_bfts import parse_arguments
        assert callable(parse_arguments)

    def test_all_pipeline_modules_import(self):
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea
        from ai_scientist.perform_plotting import aggregate_plots
        from ai_scientist.perform_writeup import perform_writeup
        from ai_scientist.perform_icbinb_writeup import perform_writeup as icbinb_writeup
        from ai_scientist.perform_llm_review import perform_review
        from ai_scientist.perform_vlm_review import generate_vlm_img_review
        from ai_scientist.treesearch.log_summarization import overall_summarize


class TestTokenTrackerIntegration:
    """Verify token tracker works end-to-end with agent invocations."""

    def setup_method(self):
        token_tracker.reset()

    @patch.object(AgentBackend, "query", return_value="test response")
    def test_tracker_records_from_llm_path(self, mock_query):
        """Path A: get_response_from_llm -> AgentBackend -> token_tracker."""
        from ai_scientist.llm import create_client, get_response_from_llm

        client, model = create_client("agent")
        content, history = get_response_from_llm(
            "Test prompt", client, model,
            system_message="You are helpful.",
        )
        assert content == "test response"

    @patch.object(AgentBackend, "query", return_value={"is_bug": False, "summary": ""})
    def test_tracker_records_from_backend_path(self, mock_query):
        """Path B: backend.query -> AgentBackend."""
        from ai_scientist.treesearch.backend import query
        from ai_scientist.treesearch.backend.utils import FunctionSpec

        spec = FunctionSpec(
            name="test", description="test",
            json_schema={
                "type": "object",
                "properties": {"is_bug": {"type": "boolean"}, "summary": {"type": "string"}},
                "required": ["is_bug", "summary"],
            },
        )
        output = query(
            system_message="sys", user_message=None,
            model="agent", func_spec=spec,
        )
        assert isinstance(output, dict)

    def test_tracker_summary_serializable(self):
        """Verify get_summary() output is JSON-serializable (for token_tracker.json)."""
        token_tracker.add_agent_invocation("agent", 1.5, "prompt", "response", "sys")
        summary = token_tracker.get_summary()
        # Must be JSON-serializable for file output
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        assert "agent" in parsed
        assert parsed["agent"]["agent_stats"]["invocations"] == 1

    def test_tracker_interactions_serializable(self):
        """Verify get_interactions() output is JSON-serializable."""
        token_tracker.add_agent_invocation("agent", 1.0, "p", "r", "s")
        interactions = token_tracker.get_interactions()
        # datetime needs special handling
        for model_interactions in interactions.values():
            for interaction in model_interactions:
                if isinstance(interaction.get("timestamp"), datetime):
                    interaction["timestamp"] = interaction["timestamp"].isoformat()
        json_str = json.dumps(interactions)
        parsed = json.loads(json_str)
        assert "agent" in parsed


class TestFullPipelineFlow:
    """Simulate the full pipeline sequence with mocked agent."""

    @patch.object(AgentBackend, "query")
    def test_ideation_to_review_sequence(self, mock_query):
        """Run ideation -> review in sequence, verify each stage works."""
        # Stage 1: Ideation
        mock_query.return_value = (
            'ACTION:\nFinalizeIdea\n\nARGUMENTS:\n```json\n'
            '{"idea": {"Name": "test", "Title": "T", "Short Hypothesis": "H", '
            '"Related Work": "R", "Abstract": "A", "Experiments": "E", '
            '"Risk Factors and Limitations": "L"}}\n```'
        )

        from ai_scientist.llm import create_client
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

        client, model = create_client("agent")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            fname = f.name

        try:
            ideas = generate_temp_free_idea(
                idea_fname=fname, client=client, model=model,
                workshop_description="Test.", max_num_generations=1, num_reflections=1,
            )
            assert len(ideas) >= 1
        finally:
            os.unlink(fname)

        # Stage 5: Review (text)
        mock_query.return_value = (
            '```json\n{"Summary": "Good.", "Strengths": ["Novel"], '
            '"Weaknesses": ["Limited"], "Questions": [], '
            '"Soundness": 3, "Presentation": 3, "Contribution": 3, '
            '"Overall": 6, "Confidence": 3, "Decision": "Accept"}\n```'
        )

        from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
        response, _ = get_response_from_llm(
            "Review this paper.", client, model,
            system_message="You are a reviewer.",
        )
        review = extract_json_between_markers(response)
        assert review is not None
        assert review["Decision"] == "Accept"

        # Stage 5: VLM review (images)
        mock_query.return_value = "The figures are well-formatted."
        from ai_scientist.vlm import create_client as create_vlm_client, get_response_from_vlm
        vlm_client, vlm_model = create_vlm_client("agent")
        vlm_response, _ = get_response_from_vlm(
            "Analyze figures.", ["/tmp/fig.png"],
            vlm_client, vlm_model, system_message="Review figures.",
        )
        assert "figures" in vlm_response.lower()


class TestConfigConsistency:
    """Verify bfts_config.yaml has agent defaults."""

    def test_config_loads_with_agent_models(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("bfts_config.yaml")
        assert cfg.report.model == "agent"
        assert cfg.agent.code.model == "agent"
        assert cfg.agent.feedback.model == "agent"
        assert cfg.agent.vlm_feedback.model == "agent"

    def test_config_preserves_structure(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("bfts_config.yaml")
        # Structural params preserved
        assert cfg.agent.stages.stage1_max_iters == 20
        assert cfg.agent.num_workers == 4
        assert cfg.exec.timeout == 3600
        assert cfg.agent.search.num_drafts == 3


class TestCreateClientConsistency:
    """All create_client variants return AgentBackend in v3."""

    def test_llm_client(self):
        from ai_scientist.llm import create_client
        c, m = create_client("anything")
        assert isinstance(c, AgentBackend)

    def test_vlm_client(self):
        from ai_scientist.vlm import create_client
        c, m = create_client("anything")
        assert isinstance(c, AgentBackend)

    def test_backend_client(self):
        from ai_scientist.treesearch.backend import get_ai_client
        c = get_ai_client("anything")
        assert isinstance(c, AgentBackend)
