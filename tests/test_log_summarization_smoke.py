"""Smoke test for treesearch/log_summarization.py — bridges both LLM call paths."""

from unittest.mock import patch
import pytest

from ai_scientist.agent_backend import AgentBackend


class TestLogSummarizationImports:
    def test_module_imports(self):
        from ai_scientist.treesearch.log_summarization import (
            get_stage_summary,
            annotate_history,
            update_summary,
            overall_summarize,
            get_nodes_infos,
            get_summarizer_prompt,
        )
        assert callable(get_stage_summary)
        assert callable(annotate_history)
        assert callable(update_summary)

    def test_uses_both_call_paths(self):
        """Verify it imports from both llm.py (Path A) and backend (Path B)."""
        from ai_scientist.treesearch.log_summarization import (
            get_response_from_llm,
            extract_json_between_markers,
            get_ai_client,
        )
        assert callable(get_response_from_llm)
        assert callable(extract_json_between_markers)
        assert callable(get_ai_client)


class TestLogSummarizationFlow:
    @patch.object(AgentBackend, "query")
    def test_get_ai_client_returns_agent_backend(self, mock_query):
        from ai_scientist.treesearch.log_summarization import get_ai_client
        client = get_ai_client("agent")
        assert isinstance(client, AgentBackend)

    @patch.object(
        AgentBackend,
        "query",
        return_value=(
            'THOUGHT:\nAnalysis complete.\n\n'
            'JSON:\n```json\n'
            '{"Experiment_description": "Test", '
            '"Significance": "Important", '
            '"Description": "Method details", '
            '"List_of_included_plots": [], '
            '"Key_numerical_results": []}\n```'
        ),
    )
    def test_get_stage_summary_through_agent(self, mock_query):
        """Verify get_stage_summary calls get_response_from_llm which routes to agent."""
        from ai_scientist.treesearch.log_summarization import get_stage_summary
        from ai_scientist.treesearch.journal import Journal, Node

        # Create a minimal journal with one node
        journal = Journal()
        node = Node(plan="Test plan", code="print('hello')")
        node.analysis = "Test analysis"
        journal.nodes.append(node)

        from ai_scientist.llm import create_client
        client, model = create_client("agent")

        result = get_stage_summary(journal, "test_stage", model, client)
        assert result is not None
        assert "Experiment_description" in result
        assert mock_query.call_count >= 1

    @patch.object(
        AgentBackend,
        "query",
        return_value=(
            'THOUGHT:\nDone.\n\n'
            'JSON:\n```json\n'
            '{"overall_plan": "Implement baseline and test"}\n```'
        ),
    )
    def test_annotate_history_through_agent(self, mock_query):
        """Verify annotate_history uses get_ai_client (Path B) and get_response_from_llm (Path A)."""
        from ai_scientist.treesearch.log_summarization import annotate_history
        from ai_scientist.treesearch.journal import Journal, Node
        from omegaconf import OmegaConf

        journal = Journal()
        parent = Node(plan="Parent plan", code="print('parent')")
        parent.overall_plan = "Original plan"
        child = Node(plan="Child plan", code="print('child')")
        child.parent = parent
        parent.children.add(child)
        journal.nodes.extend([parent, child])

        cfg = OmegaConf.create({"agent": {"summary": {"model": "agent"}}})
        annotate_history(journal, cfg=cfg)

        assert parent.overall_plan == "Parent plan"  # Root node keeps its plan
        assert child.overall_plan == "Implement baseline and test"  # Annotated from LLM
