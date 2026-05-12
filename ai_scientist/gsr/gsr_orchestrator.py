"""GSR orchestrator: the main Generate-Select-Refine loop.

Implements Algorithm 2 from the GSR paper, adapted for AI Scientist v2.
Each "task" is a research idea; within-task evaluation runs a partial
BFTS experiment pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from datetime import datetime
from typing import List, Optional

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file, idea_to_markdown
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)

from .config import GSRConfig
from .idea_mutator import IdeaMutator
from .task_manager import TaskManager, TaskState
from .utility_evaluator import UtilityEvaluator

logger = logging.getLogger(__name__)


class GSROrchestrator:
    """Main GSR loop wrapping the AI Scientist v2 pipeline.

    Pseudocode (Algorithm 2):
        1. Initialize seed tasks from user-provided ideas
        2. For t = 1, ..., T:
           a. SELECT: i_t = argmax UCB(i) over active tasks
           b. EVALUATE: Run partial BFTS on idea i_t
           c. UPDATE: Update confidence intervals
           d. ANCHOR: Find anchor a_t (best well-resolved task)
           e. REFINE: If anchor width <= c_g * epsilon_m^U,
              advance level m and generate new child ideas
        3. Return best idea and its experiment results
    """

    def __init__(
        self,
        gsr_config: GSRConfig,
        bfts_config_path: str,
        base_experiment_dir: str = "experiments",
        code_context: Optional[str] = None,
    ):
        self.gsr_config = gsr_config
        self.bfts_config_path = bfts_config_path
        self.base_experiment_dir = base_experiment_dir
        self.code_context = code_context

        self.task_manager = TaskManager(gsr_config)
        self.mutator = IdeaMutator(gsr_config)
        self.evaluator = UtilityEvaluator(gsr_config)

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.gsr_dir = osp.join(base_experiment_dir, f"gsr_{self.run_id}")
        os.makedirs(self.gsr_dir, exist_ok=True)

        self._save_config()

    def _save_config(self) -> None:
        config_path = osp.join(self.gsr_dir, "gsr_config.json")
        with open(config_path, "w") as f:
            json.dump(self.gsr_config.__dict__, f, indent=2)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, seed_ideas: List[dict]) -> dict:
        """Execute the full GSR loop.

        Args:
            seed_ideas: Initial research ideas (seed tasks at level 0).

        Returns:
            Summary dict with the best idea, its results, and GSR trace.
        """
        logger.info(
            "Starting GSR run with %d seed ideas, budget=%d",
            len(seed_ideas),
            self.gsr_config.budget,
        )

        # Step 0: Initialize seed tasks
        for idea in seed_ideas:
            self.task_manager.add_task(idea, level=0)

        # Main GSR loop (Algorithm 2)
        for t in range(1, self.gsr_config.budget + 1):
            logger.info("=" * 60)
            logger.info(
                "GSR round %d/%d (level=%d, active_tasks=%d)",
                t,
                self.gsr_config.budget,
                self.task_manager.current_level,
                self.task_manager.num_active,
            )

            # (a) SELECT: pick task with highest UCB
            task = self.task_manager.select_task()

            # (b) EVALUATE: run partial BFTS + utility assessment
            utility, metric, exp_dir = self._evaluate_task(task, round_num=t)

            # (c) UPDATE: record result → recomputes CIs
            self.task_manager.record_evaluation(
                task.task_id,
                utility,
                metric=metric,
                stages_completed=task.stages_completed + 1,
                experiment_dir=exp_dir,
            )

            # (d-e) ANCHOR + REFINE: check if we should generate
            anchor = self.task_manager.try_refine()
            if anchor is not None:
                self._refine(anchor)

            # Checkpoint
            self._save_checkpoint(t)

        # Return best
        best = self.task_manager.get_best_task()
        summary = self._build_summary(best)
        self._save_summary(summary)
        logger.info(
            "GSR complete. Best idea: %s (utility=%.4f)",
            best.idea.get("Title", "?") if best else "none",
            best.incumbent if best else 0.0,
        )
        return summary

    # ------------------------------------------------------------------
    # Evaluate a task (within-task BO step)
    # ------------------------------------------------------------------

    def _evaluate_task(
        self, task: TaskState, round_num: int
    ) -> tuple[float, Optional[float], Optional[str]]:
        """Run one evaluation step on a task.

        This runs a BFTS experiment and then uses the committee evaluator
        to produce a utility score.
        """
        logger.info(
            "Evaluating task %s: '%s'", task.task_id, task.idea.get("Title", "?")
        )

        # Set up experiment directory
        idea_dir = osp.join(
            self.gsr_dir,
            f"round_{round_num:03d}_task_{task.task_id}",
        )
        os.makedirs(idea_dir, exist_ok=True)

        # Write idea to files — include code context in JSON so BFTS
        # can read it via load_task_desc (mirrors launch_scientist_bfts.py).
        idea_json_path = osp.join(idea_dir, "idea.json")
        idea_for_bfts = dict(task.idea)
        if self.code_context is not None:
            idea_for_bfts["Code"] = self.code_context
        with open(idea_json_path, "w") as f:
            json.dump(idea_for_bfts, f, indent=4)

        idea_md_path = osp.join(idea_dir, "idea.md")
        idea_to_markdown(idea_for_bfts, idea_md_path, None)

        # Prepare BFTS config pointing to this idea
        run_config_path = edit_bfts_config_file(
            self.bfts_config_path, idea_dir, idea_json_path
        )

        # Run BFTS (partial: respects evaluation_depth via config stages)
        best_metric = None
        try:
            perform_experiments_bfts(run_config_path)
            best_metric = self._extract_best_metric(idea_dir)
        except Exception as e:
            logger.error("BFTS failed for task %s: %s", task.task_id, e)

        # Committee utility evaluation
        utility = self.evaluator.evaluate_absolute(
            task.idea,
            metric=best_metric,
            stages=task.stages_completed + 1,
        )

        logger.info(
            "Task %s evaluation: utility=%.4f, metric=%s",
            task.task_id,
            utility,
            f"{best_metric:.4f}" if best_metric is not None else "N/A",
        )
        return utility, best_metric, idea_dir

    def _extract_best_metric(self, idea_dir: str) -> Optional[float]:
        """Extract the best metric from a BFTS run directory."""
        logs_dir = osp.join(idea_dir, "logs")
        if not osp.exists(logs_dir):
            return None

        best = None
        for root, dirs, files in os.walk(logs_dir):
            for fname in files:
                if fname == "stage_progress.json":
                    try:
                        with open(osp.join(root, fname), "r") as f:
                            progress = json.load(f)
                        metric_str = progress.get("best_metric", "None")
                        if metric_str and metric_str != "None":
                            val = float(metric_str)
                            if best is None or val > best:
                                best = val
                    except (ValueError, json.JSONDecodeError):
                        continue
        return best

    # ------------------------------------------------------------------
    # Refinement (Alg. 2, Lines 8-10)
    # ------------------------------------------------------------------

    def _refine(self, anchor: TaskState) -> None:
        """Generate child ideas and advance refinement level only on success."""
        next_level = self.task_manager.current_level + 1
        logger.info(
            "REFINE: Mutating anchor '%s' at level %d",
            anchor.idea.get("Title", "?"),
            next_level,
        )

        children = self.mutator.generate(
            anchor.idea,
            level=next_level,
            batch_size=self.gsr_config.generation_batch_size,
        )

        if children:
            # Only advance the level if we successfully generated children
            self.task_manager.advance_level()
            self.task_manager.add_tasks_batch(
                children, level=next_level, parent_id=anchor.task_id
            )
            logger.info("Added %d child tasks at level %d", len(children), next_level)

            # Save generated children
            children_path = osp.join(self.gsr_dir, f"mutations_level_{next_level}.json")
            with open(children_path, "w") as f:
                json.dump(children, f, indent=2)
        else:
            logger.warning("No valid mutations generated at level %d", next_level)

    # ------------------------------------------------------------------
    # Checkpointing and summary
    # ------------------------------------------------------------------

    def _save_checkpoint(self, round_num: int) -> None:
        checkpoint = {
            "round": round_num,
            "task_manager_status": self.task_manager.get_status_summary(),
        }
        path = osp.join(self.gsr_dir, "checkpoint.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _build_summary(self, best_task: Optional[TaskState]) -> dict:
        return {
            "run_id": self.run_id,
            "gsr_dir": self.gsr_dir,
            "total_rounds": self.gsr_config.budget,
            "final_level": self.task_manager.current_level,
            "total_tasks_explored": len(self.task_manager.tasks),
            "best_task": (
                {
                    "task_id": best_task.task_id,
                    "idea": best_task.idea,
                    "incumbent_utility": best_task.incumbent,
                    "best_metric": best_task.best_metric,
                    "num_evaluations": best_task.num_evaluations,
                    "refinement_level": best_task.refinement_level,
                    "experiment_dir": best_task.experiment_dir,
                }
                if best_task
                else None
            ),
            "task_manager_status": self.task_manager.get_status_summary(),
        }

    def _save_summary(self, summary: dict) -> None:
        path = osp.join(self.gsr_dir, "gsr_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
