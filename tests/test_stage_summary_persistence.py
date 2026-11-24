import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    STAGE_SUMMARY_FILENAMES,
    write_stage_summary,
)


class _FakeAgent:
    def __init__(self, stage_name: str, fail_on_stage: str | None):
        self.stage_name = stage_name
        self.fail_on_stage = fail_on_stage

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self, exec_callback):
        if self.fail_on_stage and self.stage_name.startswith(self.fail_on_stage):
            raise RuntimeError("simulated interruption")


def _make_stage(main_stage_number: int, sequence_number: int) -> Stage:
    return Stage(
        name=f"{main_stage_number}_main_{main_stage_number}_first_attempt",
        description="test stage",
        goals=[],
        max_iterations=1,
        num_drafts=0,
        stage_number=sequence_number,
    )


def _make_manager(stages, fail_on_stage=None):
    manager = object.__new__(AgentManager)
    manager.current_stage = stages[0]
    manager.stages = [stages[0]]
    manager.journals = {stage.name: object() for stage in stages}
    manager.stage_history = []
    manager.completed_stages = []
    manager._save_checkpoint = lambda: None
    manager._create_agent_for_stage = lambda stage: _FakeAgent(
        stage.name, fail_on_stage
    )
    manager._check_stage_completion = lambda stage: (True, "stage complete")
    manager._check_substage_completion = lambda stage, journal: (False, "")

    next_stages = dict(zip((stage.name for stage in stages), stages[1:]))
    manager._create_next_main_stage = lambda stage, journal: next_stages.get(
        stage.name
    )
    return manager


class StageSummaryPersistenceTest(unittest.TestCase):
    def test_writes_each_main_stage_summary_to_its_expected_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(log_dir=Path(tmpdir) / "logs" / "0-run")
            calls = []

            def fake_summarize(journal, stage_name, stage_index, cfg):
                calls.append((journal, stage_name, stage_index, cfg))
                return {"stage_index": stage_index}

            for main_stage_number, filename in STAGE_SUMMARY_FILENAMES.items():
                path = write_stage_summary(
                    cfg=cfg,
                    main_stage_number=main_stage_number,
                    stage_name=f"{main_stage_number}_main_1_first_attempt",
                    journal=f"journal-{main_stage_number}",
                    summarizer=fake_summarize,
                )

                self.assertEqual(path, cfg.log_dir / filename)
                with open(path, encoding="utf-8") as summary_file:
                    self.assertEqual(
                        json.load(summary_file),
                        {"stage_index": main_stage_number - 1},
                    )

            self.assertEqual(
                [call[2] for call in calls],
                list(range(len(STAGE_SUMMARY_FILENAMES))),
            )
            self.assertFalse(list(cfg.log_dir.glob("*.tmp")))

    def test_failed_summarization_keeps_previous_summary_intact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(log_dir=Path(tmpdir) / "logs" / "0-run")
            cfg.log_dir.mkdir(parents=True)
            summary_path = cfg.log_dir / STAGE_SUMMARY_FILENAMES[1]
            summary_path.write_text('{"old": true}\n', encoding="utf-8")

            def fail_summarize(journal, stage_name, stage_index, cfg):
                raise RuntimeError("LLM unavailable")

            with self.assertRaisesRegex(RuntimeError, "LLM unavailable"):
                write_stage_summary(
                    cfg=cfg,
                    main_stage_number=1,
                    stage_name="1_main_1_first_attempt",
                    journal=object(),
                    summarizer=fail_summarize,
                )

            self.assertEqual(
                summary_path.read_text(encoding="utf-8"), '{"old": true}\n'
            )
            self.assertFalse(list(cfg.log_dir.glob("*.tmp")))

    def test_manager_notifies_after_each_main_stage_before_next_stage(self):
        stages = [_make_stage(number, number + 4) for number in range(1, 5)]
        manager = _make_manager(stages)
        completed = []

        manager.run(
            exec_callback=lambda: None,
            main_stage_callback=lambda number, stage_name, journal: completed.append(
                (number, stage_name, journal)
            ),
        )

        self.assertEqual(
            [(number, stage_name) for number, stage_name, _ in completed],
            [(stage.stage_number - 4, stage.name) for stage in stages],
        )

    def test_completed_stage_is_saved_even_when_next_stage_interrupts(self):
        stages = [_make_stage(1, 5), _make_stage(2, 6)]
        manager = _make_manager(stages, fail_on_stage="2_")
        completed = []

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            manager.run(
                exec_callback=lambda: None,
                main_stage_callback=lambda number, stage_name, journal: completed.append(
                    number
                ),
            )

        self.assertEqual(completed, [1])


if __name__ == "__main__":
    unittest.main()
