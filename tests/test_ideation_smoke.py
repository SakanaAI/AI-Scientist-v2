"""
Smoke test: Verify perform_ideation_temp_free.py imports and core flow work
through the refactored llm.py agent backend.

This does NOT run actual ideation (that requires a real agent). It verifies:
1. All imports resolve without SDK dependencies
2. create_client returns an AgentBackend
3. The generate_temp_free_idea function can be called (with mocked agent)
4. AVAILABLE_LLMS is importable (None in v3 mode)
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from ai_scientist.agent_backend import AgentBackend


class TestIdeationImports:
    """Verify that perform_ideation_temp_free.py imports work in v3 mode."""

    def test_import_llm_functions(self):
        from ai_scientist.llm import (
            AVAILABLE_LLMS,
            create_client,
            get_response_from_llm,
        )
        assert create_client is not None
        assert get_response_from_llm is not None

    def test_available_llms_is_none_in_v3(self):
        from ai_scientist.llm import AVAILABLE_LLMS
        # In v3 mode, AVAILABLE_LLMS is None (no model restriction)
        assert AVAILABLE_LLMS is None

    def test_available_vlms_is_none_in_v3(self):
        from ai_scientist.vlm import AVAILABLE_VLMS
        assert AVAILABLE_VLMS is None

    def test_ideation_module_imports(self):
        """The ideation module should import without errors."""
        from ai_scientist.perform_ideation_temp_free import (
            generate_temp_free_idea,
            system_prompt,
            idea_generation_prompt,
        )
        assert callable(generate_temp_free_idea)
        assert isinstance(system_prompt, str)
        assert "researcher" in system_prompt


class TestIdeationCreateClient:
    def test_create_client_for_ideation(self):
        from ai_scientist.llm import create_client
        client, model = create_client("gpt-4o")
        assert isinstance(client, AgentBackend)
        assert model == "gpt-4o"

    def test_create_client_agent_default(self):
        from ai_scientist.llm import create_client
        client, model = create_client("agent")
        assert isinstance(client, AgentBackend)
        assert model == "agent"


class TestIdeationArgparse:
    """Verify the __main__ argparse block works in v3 mode."""

    def test_available_llms_none_allows_any_model(self):
        """When AVAILABLE_LLMS is None, argparse choices=None accepts any string."""
        import argparse
        from ai_scientist.llm import AVAILABLE_LLMS

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="agent", choices=AVAILABLE_LLMS)
        # Should not raise with any model string
        args = parser.parse_args(["--model", "my-custom-model"])
        assert args.model == "my-custom-model"

    def test_default_model_is_agent(self):
        """In v3 mode, default model should be 'agent'."""
        import argparse
        from ai_scientist.llm import AVAILABLE_LLMS

        default = "agent" if AVAILABLE_LLMS is None else "gpt-4o-2024-05-13"
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default=default, choices=AVAILABLE_LLMS)
        args = parser.parse_args([])
        assert args.model == "agent"


class TestIdeationFlow:
    """Simulate the core ideation loop with mocked agent responses."""

    @patch.object(AgentBackend, "query")
    def test_single_idea_generation(self, mock_query):
        """Simulate one idea generation round with a FinalizeIdea response."""
        # Mock agent returning a FinalizeIdea action
        mock_query.return_value = (
            'ACTION:\nFinalizeIdea\n\n'
            'ARGUMENTS:\n```json\n'
            '{"idea": {'
            '"Name": "test_idea", '
            '"Title": "A Test Idea", '
            '"Short Hypothesis": "Testing works.", '
            '"Related Work": "None relevant.", '
            '"Abstract": "This is a test abstract for smoke testing.", '
            '"Experiments": "Run pytest.", '
            '"Risk Factors and Limitations": "None."'
            '}}\n```'
        )

        from ai_scientist.llm import create_client
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

        client, model = create_client("agent")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            idea_fname = f.name

        try:
            ideas = generate_temp_free_idea(
                idea_fname=idea_fname,
                client=client,
                model=model,
                workshop_description="Test workshop about ML.",
                max_num_generations=1,
                num_reflections=1,
            )
            # Should have generated at least one idea
            assert len(ideas) >= 1
            # ideas are already dicts (json.loads was done in generate_temp_free_idea)
            assert ideas[0]["Name"] == "test_idea"
        finally:
            os.unlink(idea_fname)

    @patch.object(AgentBackend, "query")
    def test_get_response_preserves_history(self, mock_query):
        """Verify message history accumulates correctly across rounds."""
        mock_query.side_effect = [
            "ACTION:\nSearchSemanticScholar\n\nARGUMENTS:\n{\"query\": \"test\"}",
            (
                'ACTION:\nFinalizeIdea\n\n'
                'ARGUMENTS:\n```json\n'
                '{"idea": {"Name": "test", "Title": "T", '
                '"Short Hypothesis": "H", "Related Work": "R", '
                '"Abstract": "A", "Experiments": "E", '
                '"Risk Factors and Limitations": "L"}}\n```'
            ),
        ]

        from ai_scientist.llm import create_client, get_response_from_llm

        client, model = create_client("agent")

        # First call
        content1, history1 = get_response_from_llm(
            "Generate idea", client, model,
            system_message="You are a researcher.",
        )
        assert len(history1) == 2

        # Second call with history
        content2, history2 = get_response_from_llm(
            "Refine idea", client, model,
            system_message="You are a researcher.",
            msg_history=history1,
        )
        assert len(history2) == 4  # 2 previous + 2 new
