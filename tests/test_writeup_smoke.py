"""Smoke test for perform_writeup.py and perform_icbinb_writeup.py with v3 agent backend."""

import pytest

from ai_scientist.agent_backend import AgentBackend


class TestWriteupImports:
    def test_perform_writeup_imports(self):
        from ai_scientist.perform_writeup import (
            perform_writeup,
            compile_latex,
        )
        assert callable(perform_writeup)
        assert callable(compile_latex)

    def test_perform_icbinb_writeup_imports(self):
        from ai_scientist.perform_icbinb_writeup import (
            perform_writeup,
            load_idea_text,
            load_exp_summaries,
            filter_experiment_summaries,
        )
        assert callable(perform_writeup)
        assert callable(load_idea_text)

    def test_llm_imports_from_writeup(self):
        """Both writeup modules import from llm.py — verify they get v3 functions."""
        from ai_scientist.llm import (
            AVAILABLE_LLMS,
            create_client,
            get_response_from_llm,
            extract_json_between_markers,
        )
        # v3 mode: AVAILABLE_LLMS is None
        assert AVAILABLE_LLMS is None
        client, model = create_client("agent")
        assert isinstance(client, AgentBackend)

    def test_vlm_imports_from_writeup(self):
        """Both writeup modules import create_client from vlm.py."""
        from ai_scientist.vlm import create_client as create_vlm_client
        client, model = create_vlm_client("agent")
        assert isinstance(client, AgentBackend)


class TestWriteupArgparse:
    def test_writeup_defaults_v3(self):
        """Verify argparse defaults are 'agent' in v3 mode."""
        import argparse
        from ai_scientist.llm import AVAILABLE_LLMS

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model", type=str,
            default="agent" if AVAILABLE_LLMS is None else "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            choices=AVAILABLE_LLMS,
        )
        parser.add_argument(
            "--big-model", type=str,
            default="agent" if AVAILABLE_LLMS is None else "o1-2024-12-17",
            choices=AVAILABLE_LLMS,
        )
        args = parser.parse_args([])
        assert args.model == "agent"
        assert args.big_model == "agent"
