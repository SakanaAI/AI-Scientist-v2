"""
Integration test: End-to-end pipeline verification for v3 agent backend.

Verifies all pipeline stages can be called in sequence through the agent backend:
  ideation -> experiments (tree search) -> plotting -> writeup -> review

All agent calls are mocked — this tests the wiring, not the agent itself.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from ai_scientist.agent_backend import AgentBackend


# --- Sample responses for each pipeline stage ---

IDEATION_RESPONSE = (
    'ACTION:\nFinalizeIdea\n\nARGUMENTS:\n```json\n'
    '{"idea": {"Name": "test_idea", "Title": "Test Idea", '
    '"Short Hypothesis": "Testing the pipeline.", '
    '"Related Work": "None.", "Abstract": "A test.", '
    '"Experiments": "Run tests.", '
    '"Risk Factors and Limitations": "None."}}\n```'
)

WRITEUP_RESPONSE = (
    "Here is a LaTeX section:\n\n"
    "\\section{Introduction}\nThis paper presents our findings.\n"
)

REVIEW_RESPONSE = (
    '```json\n{"Summary": "Good paper.", "Strengths": ["Novel"], '
    '"Weaknesses": ["Limited"], "Questions": [], '
    '"Soundness": 3, "Presentation": 3, "Contribution": 3, '
    '"Overall": 6, "Confidence": 3, "Decision": "Accept"}\n```'
)

PLOTTING_RESPONSE = (
    "```python\nimport os\nos.makedirs('figures', exist_ok=True)\n"
    "print('Plots generated')\n```"
)

SUMMARY_RESPONSE = (
    'THOUGHT:\nAnalysis complete.\n\nJSON:\n```json\n'
    '{"Experiment_description": "Baseline test", '
    '"Significance": "Validates pipeline", '
    '"Description": "Ran baseline model", '
    '"List_of_included_plots": [], '
    '"Key_numerical_results": []}\n```'
)


class TestPipelineImportChain:
    """Verify all pipeline modules import cleanly in v3 mode."""

    def test_ideation_imports(self):
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea
        assert callable(generate_temp_free_idea)

    def test_plotting_imports(self):
        from ai_scientist.perform_plotting import aggregate_plots
        assert callable(aggregate_plots)

    def test_writeup_imports(self):
        from ai_scientist.perform_writeup import perform_writeup
        assert callable(perform_writeup)

    def test_icbinb_writeup_imports(self):
        from ai_scientist.perform_icbinb_writeup import perform_writeup
        assert callable(perform_writeup)

    def test_llm_review_imports(self):
        from ai_scientist.perform_llm_review import perform_review
        assert callable(perform_review)

    def test_vlm_review_imports(self):
        from ai_scientist.perform_vlm_review import generate_vlm_img_review
        assert callable(generate_vlm_img_review)

    def test_log_summarization_imports(self):
        from ai_scientist.treesearch.log_summarization import overall_summarize
        assert callable(overall_summarize)

    def test_backend_imports(self):
        from ai_scientist.treesearch.backend import query, get_ai_client, FunctionSpec
        assert callable(query)
        assert callable(get_ai_client)


class TestPipelineStageSequence:
    """Test that pipeline stages can be called in sequence with mocked agent."""

    @patch.object(AgentBackend, "query")
    def test_ideation_stage(self, mock_query):
        """Stage 1: Ideation generates an idea JSON."""
        mock_query.return_value = IDEATION_RESPONSE

        from ai_scientist.llm import create_client
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

        client, model = create_client("agent")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            fname = f.name

        try:
            ideas = generate_temp_free_idea(
                idea_fname=fname, client=client, model=model,
                workshop_description="Test workshop.",
                max_num_generations=1, num_reflections=1,
            )
            assert len(ideas) >= 1
            assert ideas[0]["Name"] == "test_idea"
        finally:
            os.unlink(fname)

    @patch.object(AgentBackend, "query")
    def test_review_stage(self, mock_query):
        """Stage 5: LLM review produces scores."""
        mock_query.return_value = REVIEW_RESPONSE

        from ai_scientist.llm import create_client, get_response_from_llm, extract_json_between_markers

        client, model = create_client("agent")
        response, history = get_response_from_llm(
            prompt="Review this paper: ...",
            client=client, model=model,
            system_message="You are a reviewer.",
        )
        review_json = extract_json_between_markers(response)
        assert review_json is not None
        assert "Overall" in review_json
        assert review_json["Decision"] == "Accept"

    @patch.object(AgentBackend, "query")
    def test_batch_review_ensembling(self, mock_query):
        """Review ensembling: N responses aggregated."""
        mock_query.side_effect = [REVIEW_RESPONSE] * 3

        from ai_scientist.llm import create_client, get_batch_responses_from_llm, extract_json_between_markers

        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_llm(
            prompt="Review this paper.",
            client=client, model=model,
            system_message="You are a reviewer.",
            n_responses=3,
        )
        assert len(contents) == 3
        for content in contents:
            review = extract_json_between_markers(content)
            assert review is not None
            assert "Overall" in review

    @patch.object(AgentBackend, "query")
    def test_vlm_review_stage(self, mock_query):
        """VLM review: image + text input."""
        mock_query.return_value = "The figures look well-formatted and clear."

        from ai_scientist.vlm import create_client, get_response_from_vlm

        client, model = create_client("agent")
        content, history = get_response_from_vlm(
            msg="Analyze these figures.",
            image_paths=["/tmp/fig1.png", "/tmp/fig2.png"],
            client=client, model=model,
            system_message="You review figures.",
        )
        assert "figures" in content.lower()
        # Verify images were passed
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["images"] == ["/tmp/fig1.png", "/tmp/fig2.png"]

    @patch.object(AgentBackend, "query")
    def test_tree_search_backend_query(self, mock_query):
        """Tree search: backend query with FunctionSpec."""
        mock_query.return_value = {"is_bug": False, "summary": "All good"}

        from ai_scientist.treesearch.backend import query
        from ai_scientist.treesearch.backend.utils import FunctionSpec

        spec = FunctionSpec(
            name="submit_review",
            description="Review execution output",
            json_schema={
                "type": "object",
                "properties": {
                    "is_bug": {"type": "boolean"},
                    "summary": {"type": "string"},
                },
                "required": ["is_bug", "summary"],
            },
        )

        output = query(
            system_message={"Task": "Review output"},
            user_message=None,
            model="agent",
            func_spec=spec,
        )
        assert isinstance(output, dict)
        assert output["is_bug"] is False

    @patch.object(AgentBackend, "query")
    def test_log_summarization_dual_path(self, mock_query):
        """Log summarization: uses both Path A (llm.py) and Path B (backend)."""
        mock_query.return_value = SUMMARY_RESPONSE

        from ai_scientist.treesearch.log_summarization import get_stage_summary
        from ai_scientist.treesearch.journal import Journal, Node
        from ai_scientist.llm import create_client

        journal = Journal()
        node = Node(plan="Test baseline", code="print('test')")
        node.analysis = "Results look good"
        journal.nodes.append(node)

        client, model = create_client("agent")
        summary = get_stage_summary(journal, "baseline", model, client)
        assert summary is not None
        assert "Experiment_description" in summary


class TestCreateClientConsistency:
    """Verify create_client returns AgentBackend across all modules."""

    def test_llm_create_client(self):
        from ai_scientist.llm import create_client
        client, model = create_client("any-model")
        assert isinstance(client, AgentBackend)

    def test_vlm_create_client(self):
        from ai_scientist.vlm import create_client
        client, model = create_client("any-model")
        assert isinstance(client, AgentBackend)

    def test_backend_get_ai_client(self):
        from ai_scientist.treesearch.backend import get_ai_client
        client = get_ai_client("any-model")
        assert isinstance(client, AgentBackend)
