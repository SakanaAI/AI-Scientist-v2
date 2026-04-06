"""Tests for ai_scientist/tools/semantic_scholar.py (v3 agent mode)."""

from unittest.mock import patch
import pytest

from ai_scientist.agent_backend import AgentBackend
from ai_scientist.tools.semantic_scholar import (
    SemanticScholarSearchTool,
    search_for_papers,
)


MOCK_PAPERS_RESPONSE = {
    "papers": [
        {
            "title": "Attention Is All You Need",
            "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
            "venue": "NeurIPS",
            "year": 2017,
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            "citationCount": 100000,
            "citationStyles": {"bibtex": "@article{vaswani2017attention}"},
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": [{"name": "Jacob Devlin"}],
            "venue": "NAACL",
            "year": 2019,
            "abstract": "We introduce BERT.",
            "citationCount": 50000,
        },
    ]
}


class TestSemanticScholarSearchTool:
    @patch.object(AgentBackend, "query", return_value=MOCK_PAPERS_RESPONSE)
    def test_use_tool_returns_formatted_string(self, mock_query):
        tool = SemanticScholarSearchTool()
        result = tool.use_tool(query="transformer attention")
        assert "Attention Is All You Need" in result
        assert "Ashish Vaswani" in result
        assert "NeurIPS" in result

    @patch.object(AgentBackend, "query", return_value=MOCK_PAPERS_RESPONSE)
    def test_search_returns_sorted_papers(self, mock_query):
        tool = SemanticScholarSearchTool()
        papers = tool.search_for_papers("transformers")
        assert len(papers) == 2
        # Should be sorted by citation count (descending)
        assert papers[0]["citationCount"] >= papers[1]["citationCount"]

    @patch.object(AgentBackend, "query", return_value={"papers": []})
    def test_empty_results(self, mock_query):
        tool = SemanticScholarSearchTool()
        result = tool.use_tool(query="xyznonexistent")
        assert result == "No papers found."

    def test_empty_query_returns_none(self):
        tool = SemanticScholarSearchTool()
        papers = tool.search_for_papers("")
        assert papers is None


class TestSearchForPapers:
    @patch.object(AgentBackend, "query", return_value=MOCK_PAPERS_RESPONSE)
    def test_returns_list_of_dicts(self, mock_query):
        papers = search_for_papers("deep learning")
        assert isinstance(papers, list)
        assert len(papers) == 2
        assert papers[0]["title"] == "Attention Is All You Need"

    @patch.object(AgentBackend, "query", return_value={"papers": []})
    def test_empty_results_returns_none(self, mock_query):
        result = search_for_papers("xyznonexistent")
        assert result is None

    def test_empty_query_returns_none(self):
        result = search_for_papers("")
        assert result is None

    @patch.object(AgentBackend, "query", side_effect=Exception("agent error"))
    def test_agent_error_returns_none(self, mock_query):
        result = search_for_papers("test query")
        assert result is None


class TestToolInterface:
    def test_tool_name_and_description(self):
        tool = SemanticScholarSearchTool()
        assert tool.name == "SearchSemanticScholar"
        assert "search" in tool.description.lower()

    def test_tool_parameters(self):
        tool = SemanticScholarSearchTool()
        assert len(tool.parameters) == 1
        assert tool.parameters[0]["name"] == "query"

    @patch.object(AgentBackend, "query", return_value=MOCK_PAPERS_RESPONSE)
    def test_format_papers(self, mock_query):
        tool = SemanticScholarSearchTool()
        papers = tool.search_for_papers("test")
        formatted = tool.format_papers(papers)
        assert "1:" in formatted
        assert "2:" in formatted
        assert "Number of citations:" in formatted
        assert "Abstract:" in formatted
