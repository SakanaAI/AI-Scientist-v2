"""Offline regression tests for multi-seed process-pool failures.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_multi_seed_evaluation.py' -v
"""

import multiprocessing
import os
import unittest
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import Mock

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import ParallelAgent
from ai_scientist.treesearch.utils.metric import MetricValue


def terminate_fixture_worker(*args):
    """Terminate only the dedicated test worker, without running experiment code."""
    os._exit(17)


class MultiSeedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.parent = Node(code="pass", id="parent")
        self.journal = Journal()
        self.journal.append(self.parent)

        # Avoid constructor model calls and GPU/process discovery. The method
        # under test and its Node/Journal integration are the real implementations.
        self.agent = ParallelAgent.__new__(ParallelAgent)
        self.agent.cfg = SimpleNamespace(
            agent=SimpleNamespace(multi_seed_eval=SimpleNamespace(num_seeds=3))
        )
        self.agent.journal = self.journal
        self.agent.gpu_manager = None
        self.agent.task_desc = "offline fixture"
        self.agent.evaluation_metrics = "fixture metric"
        self.agent.stage_name = "3_creative_research_1_fixture"
        self.agent.timeout = 20
        self.agent.executor = Mock(spec=["submit"])

    def successful_future(self, node_id):
        node = Node(
            id=node_id,
            code="pass",
            is_seed_node=True,
            is_buggy=False,
            is_buggy_plots=False,
            metric=MetricValue(1.0, maximize=True),
        )
        data = node.to_dict()
        data["parent_id"] = self.parent.id
        future = Future()
        future.set_result(data)
        return future

    def failed_future(self, error):
        future = Future()
        future.set_exception(error)
        return future

    def assert_seed_nodes(self, expected_ids):
        self.assertEqual([n.id for n in self.journal.nodes], ["parent", *expected_ids])
        self.assertEqual(
            [n.step for n in self.journal.nodes], list(range(len(self.journal)))
        )
        self.assertEqual({n.id for n in self.parent.children}, set(expected_ids))
        for node_id in expected_ids:
            self.assertIs(self.journal.get_node_by_id(node_id).parent, self.parent)

    def test_successful_seeds_are_returned_and_linked(self):
        self.agent.executor.submit.side_effect = [
            self.successful_future(f"seed-{i}") for i in range(3)
        ]

        results = self.agent._run_multi_seed_evaluation(self.parent)

        self.assertEqual([n.id for n in results], ["seed-0", "seed-1", "seed-2"])
        self.assert_seed_nodes(["seed-0", "seed-1", "seed-2"])
        for node in results:
            self.assertIs(node, self.journal.get_node_by_id(node.id))

    def test_ordinary_worker_failure_keeps_existing_behavior(self):
        self.agent.executor.submit.side_effect = [
            self.successful_future("seed-0"),
            self.failed_future(ValueError("fixture task failure")),
            self.successful_future("seed-2"),
        ]

        with self.assertLogs("ai-scientist", level="ERROR") as logs:
            results = self.agent._run_multi_seed_evaluation(self.parent)

        self.assertEqual([n.id for n in results], ["seed-0", "seed-2"])
        self.assertIn("fixture task failure", "\n".join(logs.output))
        self.assert_seed_nodes(["seed-0", "seed-2"])

    def test_broken_pool_before_any_seed_result_is_fatal(self):
        error = BrokenProcessPool("fixture broken pool")
        self.agent.executor.submit.side_effect = [
            self.failed_future(error),
            self.successful_future("seed-1"),
            self.successful_future("seed-2"),
        ]

        with self.assertRaises(BrokenProcessPool) as raised:
            self.agent._run_multi_seed_evaluation(self.parent)

        self.assertIs(raised.exception, error)
        self.assert_seed_nodes([])

    def test_broken_pool_preserves_already_incorporated_seeds_but_aborts(self):
        error = BrokenProcessPool("fixture broken pool after one success")
        self.agent.executor.submit.side_effect = [
            self.successful_future("seed-0"),
            self.failed_future(error),
            self.successful_future("seed-2"),
        ]

        with self.assertRaises(BrokenProcessPool) as raised:
            self.agent._run_multi_seed_evaluation(self.parent)

        self.assertIs(raised.exception, error)
        self.assert_seed_nodes(["seed-0"])

    def test_real_worker_termination_propagates_broken_pool(self):
        # Use an isolated spawn pool; no model calls, GPU work or launcher
        # cleanup. Abrupt termination must not turn into a partial-success return.
        self.agent.cfg.agent.multi_seed_eval.num_seeds = 1
        self.agent._process_node_wrapper = terminate_fixture_worker
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            self.agent.executor = executor

            with self.assertRaises(BrokenProcessPool):
                self.agent._run_multi_seed_evaluation(self.parent)

            with self.assertRaises(BrokenProcessPool):
                executor.submit(terminate_fixture_worker)

        self.assert_seed_nodes([])


if __name__ == "__main__":
    unittest.main()
