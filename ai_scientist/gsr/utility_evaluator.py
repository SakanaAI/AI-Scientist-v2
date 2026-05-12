"""Committee-based utility evaluation for GSR.

Implements the LLM committee evaluator from §3.4 of the GSR paper.
A committee of K voters performs pairwise comparisons between the
current task's incumbent and an anchor, producing a utility estimate
with confidence intervals.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.llm import create_client, get_response_from_llm

from .config import GSRConfig

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an experienced ML research reviewer evaluating research ideas and their experimental results.

You will be shown two research ideas (A and B) along with their experimental progress.
Your job is to judge which idea is more promising based on:
1. Scientific merit and novelty of the hypothesis
2. Quality and significance of experimental results so far
3. Feasibility of completing the research within reasonable budget
4. Potential impact if successful

Respond with a JSON object:
{
  "preferred": "A" or "B",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}
"""

EVALUATOR_USER_PROMPT = """## Idea A (anchor)
Title: {anchor_title}
Hypothesis: {anchor_hypothesis}
Best metric so far: {anchor_metric}
Stages completed: {anchor_stages}

## Idea B (candidate)
Title: {candidate_title}
Hypothesis: {candidate_hypothesis}
Best metric so far: {candidate_metric}
Stages completed: {candidate_stages}

Which idea is more promising? Respond with ONLY a JSON object."""

ABSOLUTE_EVALUATOR_PROMPT = """Rate the following research idea on a scale from 0.0 to 1.0 based on:
1. Scientific merit and novelty (0.3 weight)
2. Feasibility within academic lab budget (0.2 weight)
3. Potential impact (0.3 weight)
4. Clarity of hypothesis and experiments (0.2 weight)

Research Idea:
Title: {title}
Hypothesis: {hypothesis}
Abstract: {abstract}
Experiments: {experiments}
Best experimental metric so far: {metric}
Stages completed: {stages}

Respond with ONLY a JSON object:
{{
  "score": 0.0 to 1.0,
  "breakdown": {{
    "novelty": 0.0 to 1.0,
    "feasibility": 0.0 to 1.0,
    "impact": 0.0 to 1.0,
    "clarity": 0.0 to 1.0
  }},
  "reasoning": "brief explanation"
}}
"""


class UtilityEvaluator:
    """Committee-based utility evaluator for cross-idea comparison.

    Uses K committee members (LLM calls with varied prompting) to produce
    pairwise preference votes, then aggregates into a utility estimate.
    """

    def __init__(self, config: GSRConfig):
        self.config = config
        self.client, self.model = create_client(config.model_evaluator)
        self.evaluation_cache: Dict[str, List[float]] = {}

    def evaluate_absolute(
        self, idea: dict, metric: Optional[float] = None, stages: int = 0
    ) -> float:
        """Produce an absolute utility score for a single idea.

        Uses K committee members and averages their scores.
        """
        scores = []
        for k in range(self.config.committee_size):
            try:
                score = self._single_absolute_vote(idea, metric, stages)
                scores.append(score)
            except Exception as e:
                logger.warning("Committee vote %d failed: %s", k, e)

        if not scores:
            logger.warning("All committee votes failed, returning 0.5")
            return 0.5

        mean_score = sum(scores) / len(scores)
        logger.info(
            "Absolute utility for '%s': %.3f (K=%d votes, std=%.3f)",
            idea.get("Title", "?"),
            mean_score,
            len(scores),
            (sum((s - mean_score) ** 2 for s in scores) / max(len(scores) - 1, 1))
            ** 0.5,
        )
        return mean_score

    def evaluate_pairwise(
        self,
        anchor: dict,
        candidate: dict,
        anchor_metric: Optional[float] = None,
        candidate_metric: Optional[float] = None,
        anchor_stages: int = 0,
        candidate_stages: int = 0,
    ) -> float:
        """Produce a pairwise preference score for candidate vs. anchor.

        Returns the fraction of committee members preferring the candidate.
        """
        votes_for_candidate = 0
        total_votes = 0

        for k in range(self.config.committee_size):
            try:
                prefers_candidate = self._single_pairwise_vote(
                    anchor,
                    candidate,
                    anchor_metric,
                    candidate_metric,
                    anchor_stages,
                    candidate_stages,
                )
                if prefers_candidate:
                    votes_for_candidate += 1
                total_votes += 1
            except Exception as e:
                logger.warning("Pairwise vote %d failed: %s", k, e)

        if total_votes == 0:
            return 0.5

        preference = votes_for_candidate / total_votes
        logger.info(
            "Pairwise: '%s' vs '%s' → %.0f%% prefer candidate (K=%d)",
            anchor.get("Title", "?"),
            candidate.get("Title", "?"),
            preference * 100,
            total_votes,
        )
        return preference

    def compute_utility_with_ci(
        self, idea: dict, metric: Optional[float] = None, stages: int = 0
    ) -> Tuple[float, float, float]:
        """Compute utility with confidence interval.

        Returns (mean, lower, upper) where the CI comes from committee
        vote variance (sub-Gaussian concentration).
        """
        scores = []
        for k in range(self.config.committee_size):
            try:
                score = self._single_absolute_vote(idea, metric, stages)
                scores.append(score)
            except Exception as e:
                logger.warning("Committee vote %d failed: %s", k, e)

        if not scores:
            return 0.5, 0.0, 1.0

        mean = sum(scores) / len(scores)
        K = len(scores)
        # Sub-Gaussian CI: sigma / sqrt(K) * sqrt(2 log(1/delta))
        delta = 0.05
        ci_half = self.config.noise_std * math.sqrt(2.0 * math.log(1.0 / delta) / K)
        lower = max(0.0, mean - ci_half)
        upper = min(1.0, mean + ci_half)
        return mean, lower, upper

    # ------------------------------------------------------------------
    # Internal vote methods
    # ------------------------------------------------------------------

    def _single_absolute_vote(
        self, idea: dict, metric: Optional[float], stages: int
    ) -> float:
        prompt = ABSOLUTE_EVALUATOR_PROMPT.format(
            title=idea.get("Title", "Untitled"),
            hypothesis=idea.get("Short Hypothesis", "N/A"),
            abstract=idea.get("Abstract", "N/A"),
            experiments=_format_experiments(idea.get("Experiments", "N/A")),
            metric=f"{metric:.4f}" if metric is not None else "not yet evaluated",
            stages=stages,
        )

        response, _ = get_response_from_llm(
            prompt=prompt,
            client=self.client,
            model=self.model,
            system_message="You are a research evaluation expert. Respond with ONLY valid JSON.",
            temperature=self.config.temperature_evaluator,
        )

        parsed = _parse_json_response(response)
        if parsed and "score" in parsed:
            return float(parsed["score"])
        return 0.5

    def _single_pairwise_vote(
        self,
        anchor: dict,
        candidate: dict,
        anchor_metric: Optional[float],
        candidate_metric: Optional[float],
        anchor_stages: int,
        candidate_stages: int,
    ) -> bool:
        prompt = EVALUATOR_USER_PROMPT.format(
            anchor_title=anchor.get("Title", "Untitled"),
            anchor_hypothesis=anchor.get("Short Hypothesis", "N/A"),
            anchor_metric=(
                f"{anchor_metric:.4f}"
                if anchor_metric is not None
                else "not yet evaluated"
            ),
            anchor_stages=anchor_stages,
            candidate_title=candidate.get("Title", "Untitled"),
            candidate_hypothesis=candidate.get("Short Hypothesis", "N/A"),
            candidate_metric=(
                f"{candidate_metric:.4f}"
                if candidate_metric is not None
                else "not yet evaluated"
            ),
            candidate_stages=candidate_stages,
        )

        response, _ = get_response_from_llm(
            prompt=prompt,
            client=self.client,
            model=self.model,
            system_message=EVALUATOR_SYSTEM_PROMPT,
            temperature=self.config.temperature_evaluator,
        )

        parsed = _parse_json_response(response)
        if parsed and "preferred" in parsed:
            return parsed["preferred"].upper() == "B"
        return False


def _format_experiments(experiments: Any) -> str:
    if isinstance(experiments, list):
        return "\n".join(
            f"- {json.dumps(e) if isinstance(e, dict) else str(e)}" for e in experiments
        )
    return str(experiments)


def _parse_json_response(text: str) -> Optional[dict]:
    import re

    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start : brace_end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
