import sys
import types
import unittest
from unittest.mock import patch


def _install_test_stubs():
    if "ai_scientist.treesearch.journal" not in sys.modules:
        journal_module = types.ModuleType("ai_scientist.treesearch.journal")
        journal_module.Node = object
        journal_module.Journal = object
        sys.modules["ai_scientist.treesearch.journal"] = journal_module

    if "ai_scientist.llm" not in sys.modules:
        llm_module = types.ModuleType("ai_scientist.llm")
        llm_module.get_response_from_llm = lambda *args, **kwargs: ("", [])
        llm_module.extract_json_between_markers = lambda *args, **kwargs: {}
        sys.modules["ai_scientist.llm"] = llm_module

    if "ai_scientist.treesearch.backend" not in sys.modules:
        backend_module = types.ModuleType("ai_scientist.treesearch.backend")
        backend_module.get_ai_client = lambda *args, **kwargs: object()
        sys.modules["ai_scientist.treesearch.backend"] = backend_module

    if "dataclasses_json" not in sys.modules:
        dataclasses_json = types.ModuleType("dataclasses_json")
        dataclasses_json.DataClassJsonMixin = object
        sys.modules["dataclasses_json"] = dataclasses_json

    if "tiktoken" not in sys.modules:
        sys.modules["tiktoken"] = types.ModuleType("tiktoken")

    if "backoff" not in sys.modules:
        backoff = types.ModuleType("backoff")
        backoff.expo = lambda *args, **kwargs: None

        def passthrough_decorator(*args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

        backoff.on_exception = passthrough_decorator
        backoff.on_predicate = passthrough_decorator
        sys.modules["backoff"] = backoff

    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")

        class _OpenAIError(Exception):
            pass

        class _OpenAIClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        openai.RateLimitError = _OpenAIError
        openai.APITimeoutError = _OpenAIError
        openai.InternalServerError = _OpenAIError
        openai.APIConnectionError = _OpenAIError
        openai.OpenAI = _OpenAIClient
        sys.modules["openai"] = openai

    if "anthropic" not in sys.modules:
        anthropic = types.ModuleType("anthropic")

        class _AnthropicError(Exception):
            pass

        class _AnthropicClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        anthropic.RateLimitError = _AnthropicError
        anthropic.Anthropic = _AnthropicClient
        anthropic.AnthropicBedrock = _AnthropicClient
        anthropic.AnthropicVertex = _AnthropicClient
        sys.modules["anthropic"] = anthropic

    if "rich" not in sys.modules:
        rich = types.ModuleType("rich")
        rich.print = print
        sys.modules["rich"] = rich

    if "funcy" not in sys.modules:
        funcy = types.ModuleType("funcy")
        funcy.notnone = lambda value: value is not None
        funcy.once = lambda func: func
        funcy.select_values = (
            lambda predicate, mapping: {
                key: value for key, value in mapping.items() if predicate(value)
            }
        )
        sys.modules["funcy"] = funcy


_install_test_stubs()

from ai_scientist.treesearch import log_summarization


class FakeNode:
    def __init__(
        self,
        name,
        *,
        is_seed_node=False,
        is_seed_agg_node=False,
        is_leaf=True,
        ablation_name=None,
        children=None,
    ):
        self.name = name
        self.is_seed_node = is_seed_node
        self.is_seed_agg_node = is_seed_agg_node
        self.is_leaf = is_leaf
        self.ablation_name = ablation_name
        self.children = children or []


class FakeJournal:
    def __init__(self, name, *, best_node=None, good_nodes=None):
        self.name = name
        self._best_node = best_node
        self.good_nodes = good_nodes or []

    def get_best_node(self, cfg=None):
        return self._best_node


class FakeAgentConfig:
    def __init__(self):
        self.summary = {"model": "fake-summary-model"}

    def get(self, key, default=None):
        if key == "summary":
            return self.summary
        return default


class LogSummarizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = types.SimpleNamespace(agent=FakeAgentConfig())

    def test_returns_four_slots_when_only_stage_one_exists(self):
        journals = [
            (
                "1_initial_implementation_1_preliminary",
                FakeJournal("draft-only"),
            )
        ]

        with (
            patch.object(log_summarization, "annotate_history"),
            patch.object(
                log_summarization,
                "get_stage_summary",
                side_effect=lambda journal, stage_name, model, client: {
                    "stage_name": stage_name,
                    "journal_name": journal.name,
                },
            ),
            patch.object(log_summarization, "get_ai_client", return_value=object()),
        ):
            draft, baseline, research, ablation = log_summarization.overall_summarize(
                journals, self.cfg
            )

        self.assertEqual(
            draft,
            {
                "stage_name": "1_initial_implementation_1_preliminary",
                "journal_name": "draft-only",
            },
        )
        self.assertEqual(baseline, {})
        self.assertEqual(research, {})
        self.assertEqual(ablation, [])

    def test_uses_latest_substage_for_research_summary(self):
        older_best = FakeNode("older-best")
        newer_best = FakeNode("newer-best")
        seen_stage_names = []
        journals = [
            ("1_initial_implementation_1_preliminary", FakeJournal("draft")),
            (
                "3_creative_research_1_first_attempt",
                FakeJournal("research-older", best_node=older_best),
            ),
            (
                "3_creative_research_2_followup",
                FakeJournal("research-newer", best_node=newer_best),
            ),
        ]

        with (
            patch.object(
                log_summarization,
                "annotate_history",
                side_effect=lambda journal, cfg=None: seen_stage_names.append(
                    journal.name
                ),
            ),
            patch.object(
                log_summarization,
                "get_stage_summary",
                return_value={"stage_name": "draft"},
            ),
            patch.object(log_summarization, "get_ai_client", return_value=object()),
            patch.object(
                log_summarization,
                "get_node_log",
                side_effect=lambda node: {"node_name": node.name},
            ),
        ):
            _, _, research, _ = log_summarization.overall_summarize(journals, self.cfg)

        self.assertEqual(
            research,
            {
                "best node": {"node_name": "newer-best"},
                "best node with different seeds": [],
            },
        )
        self.assertEqual(seen_stage_names.count("research-older"), 0)
        self.assertEqual(seen_stage_names.count("research-newer"), 1)

    def test_maps_stages_by_name_instead_of_input_order(self):
        baseline_best = FakeNode("baseline-best")
        research_best = FakeNode("research-best")
        ablation_leaf = FakeNode(
            "ablation-leaf", is_leaf=True, ablation_name="dropout_ablation"
        )
        journals = [
            (
                "3_creative_research_1_first_attempt",
                FakeJournal("research", best_node=research_best),
            ),
            (
                "1_initial_implementation_1_preliminary",
                FakeJournal("draft"),
            ),
            (
                "4_ablation_studies_1_first_attempt",
                FakeJournal("ablation", good_nodes=[ablation_leaf]),
            ),
            (
                "2_baseline_tuning_1_first_attempt",
                FakeJournal("baseline", best_node=baseline_best),
            ),
        ]

        with (
            patch.object(log_summarization, "annotate_history"),
            patch.object(
                log_summarization,
                "get_stage_summary",
                side_effect=lambda journal, stage_name, model, client: {
                    "draft_from": journal.name
                },
            ),
            patch.object(log_summarization, "get_ai_client", return_value=object()),
            patch.object(
                log_summarization,
                "get_node_log",
                side_effect=lambda node: {"node_name": node.name},
            ),
        ):
            draft, baseline, research, ablation = log_summarization.overall_summarize(
                journals, self.cfg
            )

        self.assertEqual(draft, {"draft_from": "draft"})
        self.assertEqual(
            baseline,
            {
                "best node": {"node_name": "baseline-best"},
                "best node with different seeds": [],
            },
        )
        self.assertEqual(
            research,
            {
                "best node": {"node_name": "research-best"},
                "best node with different seeds": [],
            },
        )
        self.assertEqual(ablation, [{"node_name": "ablation-leaf"}])

    def test_ignores_unknown_stage_names(self):
        baseline_best = FakeNode("baseline-best")
        journals = [
            ("notes_stage_without_numbering", FakeJournal("unknown")),
            (
                "2_baseline_tuning_1_first_attempt",
                FakeJournal("baseline", best_node=baseline_best),
            ),
        ]

        with self.assertLogs("ai_scientist.treesearch.log_summarization", level="WARNING") as logs:
            with (
                patch.object(log_summarization, "annotate_history"),
                patch.object(log_summarization, "get_ai_client", return_value=object()),
                patch.object(
                    log_summarization,
                    "get_node_log",
                    side_effect=lambda node: {"node_name": node.name},
                ),
            ):
                _, baseline, _, _ = log_summarization.overall_summarize(
                    journals, self.cfg
                )

        self.assertEqual(
            baseline,
            {
                "best node": {"node_name": "baseline-best"},
                "best node with different seeds": [],
            },
        )
        self.assertIn("Skipping unrecognized stage name: notes_stage_without_numbering", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
