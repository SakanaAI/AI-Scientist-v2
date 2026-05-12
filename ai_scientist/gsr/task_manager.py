"""Task pool and UCB-based task selection for the GSR loop.

Implements the task-UCB policy (Algorithm 2, Lines 4-7) and confidence
interval tracking from the GSR paper. Each "task" wraps a research idea
with evaluation history and confidence bookkeeping.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import GSRConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Tracks the evaluation state of a single research-idea task.

    Corresponds to task *i* in the GSR paper. Maintains the incumbent
    (best observed utility), evaluation count, and confidence interval.
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    idea: dict = field(default_factory=dict)
    refinement_level: int = 0
    parent_task_id: Optional[str] = None

    # Within-task evaluation bookkeeping
    num_evaluations: int = 0
    utility_observations: List[float] = field(default_factory=list)
    incumbent: float = float("-inf")  # best observed utility
    metric_observations: List[float] = field(default_factory=list)

    # BFTS progress
    stages_completed: int = 0
    best_metric: Optional[float] = None
    experiment_dir: Optional[str] = None

    # Confidence interval (updated by TaskManager)
    ucb: float = float("inf")
    lcb: float = float("-inf")
    ci_width: float = float("inf")

    @property
    def mean_utility(self) -> float:
        if not self.utility_observations:
            return 0.0
        return sum(self.utility_observations) / len(self.utility_observations)


class TaskManager:
    """Manages the pool of active tasks and implements UCB task selection.

    This corresponds to the outer loop of Algorithm 2 in the GSR paper:
    maintaining confidence intervals, selecting the next task via task-UCB,
    and identifying the anchor task for refinement.
    """

    def __init__(self, config: GSRConfig):
        self.config = config
        self.tasks: Dict[str, TaskState] = {}
        self.active_task_ids: List[str] = []
        self.eliminated_task_ids: List[str] = []
        self.global_step: int = 0
        self.current_level: int = 0

    def add_task(
        self, idea: dict, level: int = 0, parent_id: Optional[str] = None
    ) -> TaskState:
        task = TaskState(
            idea=idea,
            refinement_level=level,
            parent_task_id=parent_id,
        )
        self.tasks[task.task_id] = task
        self.active_task_ids.append(task.task_id)
        logger.info(
            "Added task %s (level=%d, parent=%s): %s",
            task.task_id,
            level,
            parent_id,
            idea.get("Title", idea.get("Name", "unnamed")),
        )
        return task

    def add_tasks_batch(
        self, ideas: List[dict], level: int = 0, parent_id: Optional[str] = None
    ) -> List[TaskState]:
        return [self.add_task(idea, level, parent_id) for idea in ideas]

    # ------------------------------------------------------------------
    # UCB task selection (Alg. 2, Line 4)
    # ------------------------------------------------------------------

    def select_task(self) -> TaskState:
        """Select the next task to evaluate via task-UCB.

        Returns the task with the highest UCB. Ties are broken in favor
        of the task with fewer evaluations.
        """
        self._update_confidence_intervals()

        best_task = None
        best_ucb = float("-inf")
        for tid in self.active_task_ids:
            t = self.tasks[tid]
            if t.ucb > best_ucb or (
                t.ucb == best_ucb
                and best_task is not None
                and t.num_evaluations < best_task.num_evaluations
            ):
                best_ucb = t.ucb
                best_task = t

        if best_task is None:
            raise RuntimeError("No active tasks to select from.")
        logger.info(
            "Selected task %s (UCB=%.4f, evals=%d)",
            best_task.task_id,
            best_task.ucb,
            best_task.num_evaluations,
        )
        return best_task

    # ------------------------------------------------------------------
    # Anchor selection (Alg. 2, Line 7 / Eq. 3)
    # ------------------------------------------------------------------

    def select_anchor(self) -> Tuple[Optional[TaskState], float]:
        """Select the anchor task for refinement.

        The anchor is the most promising task (highest LCB) whose
        confidence width is below the resolution threshold.

        Returns:
            (anchor_task, anchor_ci_width) or (None, inf) if no task
            qualifies.
        """
        self._update_confidence_intervals()

        epsilon_u_m = self.config.epsilon_u_0 * (2.0**-self.current_level)
        min_width = min(
            (self.tasks[tid].ci_width for tid in self.active_task_ids),
            default=float("inf"),
        )
        threshold = max(self.config.confidence_gate * epsilon_u_m, min_width)

        best_anchor = None
        best_lcb = float("-inf")
        for tid in self.active_task_ids:
            t = self.tasks[tid]
            if t.num_evaluations < self.config.min_evaluations_before_refine:
                continue
            if t.ci_width <= threshold and t.lcb > best_lcb:
                best_lcb = t.lcb
                best_anchor = t

        if best_anchor is not None:
            logger.info(
                "Anchor task %s (LCB=%.4f, width=%.4f, threshold=%.4f)",
                best_anchor.task_id,
                best_anchor.lcb,
                best_anchor.ci_width,
                threshold,
            )
        return best_anchor, min_width if best_anchor is None else best_anchor.ci_width

    # ------------------------------------------------------------------
    # Should-refine gate (Alg. 2, Line 8)
    # ------------------------------------------------------------------

    def try_refine(self) -> Optional["TaskState"]:
        """Check the refinement gate and return the anchor if we should refine.

        Returns the anchor task if refinement should proceed, or None.
        This avoids the redundant double-call to select_anchor().
        """
        if self.current_level >= self.config.max_refinement_levels:
            return None
        anchor, width = self.select_anchor()
        if anchor is None:
            return None
        epsilon_u_m = self.config.epsilon_u_0 * (2.0**-self.current_level)
        if width <= self.config.confidence_gate * epsilon_u_m:
            return anchor
        return None

    def advance_level(self) -> int:
        """Step up m <- m + 1 (Alg. 2, Line 9)."""
        self.current_level += 1
        logger.info("Advanced to refinement level %d", self.current_level)
        return self.current_level

    # ------------------------------------------------------------------
    # Record evaluation results
    # ------------------------------------------------------------------

    def record_evaluation(
        self,
        task_id: str,
        utility: float,
        metric: Optional[float] = None,
        stages_completed: int = 0,
        experiment_dir: Optional[str] = None,
    ) -> None:
        """Record the result of evaluating a task."""
        task = self.tasks[task_id]
        task.num_evaluations += 1
        task.utility_observations.append(utility)
        # Update experiment_dir only when this is the best run so far,
        # so the writeup uses artifacts from the highest-utility evaluation.
        if experiment_dir is not None and utility >= task.incumbent:
            task.experiment_dir = experiment_dir
        task.incumbent = max(task.incumbent, utility)
        if metric is not None:
            task.metric_observations.append(metric)
            if task.best_metric is None or metric > task.best_metric:
                task.best_metric = metric
        task.stages_completed = max(task.stages_completed, stages_completed)
        self.global_step += 1
        self._update_confidence_intervals()

    # ------------------------------------------------------------------
    # Confidence interval computation
    # ------------------------------------------------------------------

    def _update_confidence_intervals(self) -> None:
        """Recompute UCB/LCB for all active tasks.

        UCB_s^{(i)} = mean_utility + L_bar * eps_f_s  (Eq. in §3.1)
        where eps_f_s ~ 1/sqrt(s) is the optimization gap proxy.
        """
        for tid in self.active_task_ids:
            t = self.tasks[tid]
            s = max(t.num_evaluations, 1)
            log_term = math.log(max(math.e * self.global_step, math.e))

            # Utility uncertainty from finite observations
            beta = self.config.noise_std * math.sqrt(2.0 * log_term / s)

            # Optimization gap proxy: shrinks as sqrt(log(s)/s)
            opt_gap = self.config.lipschitz_bound * math.sqrt(log_term / s)

            mean_u = t.mean_utility
            t.ucb = mean_u + beta + opt_gap
            t.lcb = mean_u - beta - opt_gap
            t.ci_width = t.ucb - t.lcb

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_best_task(self) -> Optional[TaskState]:
        """Return the task with the highest incumbent utility."""
        best = None
        for tid in self.active_task_ids:
            t = self.tasks[tid]
            if best is None or t.incumbent > best.incumbent:
                best = t
        return best

    @property
    def num_active(self) -> int:
        return len(self.active_task_ids)

    def get_status_summary(self) -> dict:
        """Return a summary dict for logging / checkpointing.

        Non-finite floats (inf, -inf) are replaced with None so the
        output is valid RFC 8259 JSON for non-Python parsers.
        """

        def _safe(v: float) -> Optional[float]:
            if math.isinf(v) or math.isnan(v):
                return None
            return v

        return {
            "global_step": self.global_step,
            "current_level": self.current_level,
            "num_active_tasks": self.num_active,
            "num_eliminated": len(self.eliminated_task_ids),
            "tasks": {
                tid: {
                    "title": self.tasks[tid].idea.get("Title", ""),
                    "evals": self.tasks[tid].num_evaluations,
                    "incumbent": _safe(self.tasks[tid].incumbent),
                    "ucb": _safe(self.tasks[tid].ucb),
                    "lcb": _safe(self.tasks[tid].lcb),
                    "level": self.tasks[tid].refinement_level,
                }
                for tid in self.active_task_ids
            },
        }
