"""Smoke test for perform_llm_review.py and perform_vlm_review.py with v3 agent backend."""

from unittest.mock import patch
import pytest

from ai_scientist.agent_backend import AgentBackend


class TestReviewImports:
    def test_llm_review_imports(self):
        from ai_scientist.perform_llm_review import (
            perform_review,
            get_meta_review,
            load_paper,
            reviewer_system_prompt_neg,
        )
        assert callable(perform_review)
        assert callable(get_meta_review)
        assert callable(load_paper)
        assert "reviewing" in reviewer_system_prompt_neg.lower()

    def test_vlm_review_imports(self):
        from ai_scientist.perform_vlm_review import (
            generate_vlm_img_review,
            perform_imgs_cap_ref_review,
            perform_imgs_cap_ref_review_selection,
            detect_duplicate_figures,
        )
        assert callable(generate_vlm_img_review)
        assert callable(perform_imgs_cap_ref_review)

    def test_llm_review_uses_v3_functions(self):
        """Verify llm_review imports functions that are v3-compatible."""
        from ai_scientist.llm import get_response_from_llm, get_batch_responses_from_llm
        assert callable(get_response_from_llm)
        assert callable(get_batch_responses_from_llm)

    def test_vlm_review_uses_v3_functions(self):
        """Verify vlm_review imports functions that are v3-compatible."""
        from ai_scientist.vlm import get_response_from_vlm, get_batch_responses_from_vlm
        assert callable(get_response_from_vlm)
        assert callable(get_batch_responses_from_vlm)


class TestReviewBatchPattern:
    """Verify the N-response ensembling pattern works through the agent backend."""

    @patch.object(AgentBackend, "query", side_effect=["review 1", "review 2", "review 3"])
    def test_batch_responses_for_ensembling(self, mock_query):
        from ai_scientist.llm import create_client, get_batch_responses_from_llm

        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_llm(
            prompt="Review this paper.",
            client=client,
            model=model,
            system_message="You are a reviewer.",
            n_responses=3,
        )
        assert len(contents) == 3
        assert mock_query.call_count == 3

    @patch.object(AgentBackend, "query", side_effect=["vlm review 1", "vlm review 2"])
    def test_vlm_batch_responses(self, mock_query):
        from ai_scientist.vlm import create_client, get_batch_responses_from_vlm

        client, model = create_client("agent")
        contents, histories = get_batch_responses_from_vlm(
            msg="Analyze these figures.",
            image_paths=["/tmp/page1.png"],
            client=client,
            model=model,
            system_message="You review figures.",
            n_responses=2,
        )
        assert len(contents) == 2
        assert mock_query.call_count == 2
