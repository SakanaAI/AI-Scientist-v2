"""Smoke test for perform_plotting.py with v3 agent backend."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend


class TestPlottingImports:
    def test_module_imports(self):
        from ai_scientist.perform_plotting import (
            aggregate_plots,
            build_aggregator_prompt,
            extract_code_snippet,
            run_aggregator_script,
            AGGREGATOR_SYSTEM_MSG,
        )
        assert callable(aggregate_plots)
        assert callable(build_aggregator_prompt)
        assert "researcher" in AGGREGATOR_SYSTEM_MSG

    def test_extract_code_snippet(self):
        from ai_scientist.perform_plotting import extract_code_snippet
        text = "Here is the code:\n```python\nimport matplotlib\nprint('hello')\n```\nDone."
        code = extract_code_snippet(text)
        assert "import matplotlib" in code
        assert "print('hello')" in code

    def test_extract_code_no_block(self):
        from ai_scientist.perform_plotting import extract_code_snippet
        text = "No code block here, just text."
        result = extract_code_snippet(text)
        assert result == text.strip()


class TestPlottingFlow:
    @patch.object(AgentBackend, "query")
    def test_aggregate_plots_calls_agent(self, mock_query):
        """Verify aggregate_plots creates client and calls LLM through agent backend."""
        # First call: initial aggregator script generation
        # Second call: reflection (says "I am done")
        mock_query.side_effect = [
            "```python\nimport os\nos.makedirs('figures', exist_ok=True)\nprint('done')\n```",
            "I am done",
        ]

        from ai_scientist.perform_plotting import aggregate_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal required files
            os.makedirs(os.path.join(tmpdir, "figures"), exist_ok=True)
            # Create research_idea.md
            with open(os.path.join(tmpdir, "research_idea.md"), "w") as f:
                f.write("# Test Idea\nThis is a test research idea.")
            # Create minimal experiment summaries
            for name in ["baseline", "research", "ablation"]:
                summary = {
                    "experiment_name": name,
                    "metrics": {"accuracy": 0.95},
                    "exp_results_npy_files": [],
                }
                with open(os.path.join(tmpdir, f"{name}_summary.json"), "w") as f:
                    json.dump(summary, f)

            aggregate_plots(base_folder=tmpdir, model="agent", n_reflections=1)

            # Agent should have been called at least once
            assert mock_query.call_count >= 1
