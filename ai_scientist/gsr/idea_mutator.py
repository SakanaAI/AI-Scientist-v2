"""LLM-powered coarse-to-fine idea mutation for GSR.

Implements the Gen(a_t, m, J) operator from §3.3 of the GSR paper.
Given an anchor idea, generates J mutated child ideas at refinement
level m with decreasing mutation ratio rho_m = rho_0 * 2^{-m}.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ai_scientist.llm import create_client, get_response_from_llm

from .config import GSRConfig

logger = logging.getLogger(__name__)

# Editable fields in an idea JSON, ordered from coarse to fine
IDEA_FIELDS = [
    "Title",
    "Short Hypothesis",
    "Abstract",
    "Experiments",
    "Risk Factors and Limitations",
    "Related Work",
    "Name",
]

COARSE_FIELDS = ["Short Hypothesis", "Abstract", "Experiments"]
FINE_FIELDS = ["Risk Factors and Limitations", "Related Work", "Title", "Name"]

MUTATION_SYSTEM_PROMPT = """You are an experienced AI researcher helping to refine research ideas.
You will be given a parent research idea and asked to create a mutated variant.
The mutation should be creative but grounded — it must remain a feasible ML research idea.

You MUST respond with a valid JSON object containing the mutated idea.
The JSON should have the same fields as the parent idea.

Important constraints:
- Mutation ratio target: {mutation_ratio:.0%} of editable fields should change.
- Fields to consider changing: {fields_to_edit}
- Fields to keep unchanged: {fields_to_keep}
- The mutation should explore a meaningfully different direction while
  preserving the unchanged fields exactly.
- Ensure the Name field is lowercase with underscores, no spaces.
"""

MUTATION_USER_PROMPT = """Parent idea (anchor):
```json
{parent_idea_json}
```

Mutation level: {level} (0=coarse/radical, higher=fine/incremental)
Target mutation ratio: {mutation_ratio:.0%} of fields should change.

Please produce a mutated child idea. Change approximately {num_fields_to_change} of
these fields: {fields_to_edit}.
Keep these fields identical to the parent: {fields_to_keep}.

Respond with ONLY a JSON object containing the complete mutated idea.
"""


class IdeaMutator:
    """Generates mutated research ideas at controlled granularity.

    This is the GSR task generator: Gen(anchor, m, J) → B_m.
    """

    def __init__(self, config: GSRConfig):
        self.config = config
        self.client, self.model = create_client(config.model_mutator)

    def mutation_ratio(self, level: int) -> float:
        """Compute rho_m = rho_0 * 2^{-m}."""
        return self.config.initial_mutation_ratio * (2.0**-level)

    def _select_fields_to_edit(
        self, level: int, attempt: int = 0
    ) -> tuple[list[str], list[str]]:
        """Decide which fields to edit vs. keep at this refinement level.

        At fine levels (>=3), rotates which fields are selected across
        attempts so that different children in a batch mutate different
        fields rather than always targeting the first element.
        """
        rho = self.mutation_ratio(level)

        if level == 0:
            candidate_fields = list(IDEA_FIELDS)
        elif level <= 2:
            candidate_fields = list(COARSE_FIELDS)
        else:
            candidate_fields = list(FINE_FIELDS)

        num_to_change = max(1, round(rho * len(candidate_fields)))

        # Rotate the candidate list by `attempt` so each child in a batch
        # targets different fields when num_to_change < len(candidates).
        if len(candidate_fields) > num_to_change:
            offset = attempt % len(candidate_fields)
            candidate_fields = candidate_fields[offset:] + candidate_fields[:offset]

        fields_to_edit = candidate_fields[:num_to_change]
        fields_to_keep = [f for f in IDEA_FIELDS if f not in fields_to_edit]
        return fields_to_edit, fields_to_keep

    def generate(
        self, anchor_idea: dict, level: int, batch_size: Optional[int] = None
    ) -> List[dict]:
        """Generate a batch of mutated child ideas from the anchor.

        Args:
            anchor_idea: The anchor (parent) idea dict.
            level: Refinement level m.
            batch_size: Number of children J; defaults to config value.

        Returns:
            List of mutated idea dicts.
        """
        J = batch_size if batch_size is not None else self.config.generation_batch_size
        rho = self.mutation_ratio(level)

        logger.info(
            "Generating %d mutations at level %d (rho=%.2f)",
            J,
            level,
            rho,
        )

        children: List[dict] = []
        for j in range(J):
            try:
                fields_to_edit, fields_to_keep = self._select_fields_to_edit(
                    level, attempt=j
                )
                num_to_change = len(fields_to_edit)
                child = self._generate_single(
                    anchor_idea,
                    level,
                    rho,
                    fields_to_edit,
                    fields_to_keep,
                    num_to_change,
                    attempt=j,
                )
                if child is not None:
                    children.append(child)
            except Exception as e:
                logger.warning("Mutation %d/%d failed: %s", j + 1, J, e)

        logger.info("Generated %d/%d valid mutations", len(children), J)
        return children

    def _generate_single(
        self,
        anchor: dict,
        level: int,
        rho: float,
        fields_to_edit: list[str],
        fields_to_keep: list[str],
        num_to_change: int,
        attempt: int = 0,
    ) -> Optional[dict]:
        """Generate one mutated idea."""
        system_msg = MUTATION_SYSTEM_PROMPT.format(
            mutation_ratio=rho,
            fields_to_edit=", ".join(fields_to_edit),
            fields_to_keep=", ".join(fields_to_keep),
        )

        user_msg = MUTATION_USER_PROMPT.format(
            parent_idea_json=json.dumps(anchor, indent=2),
            level=level,
            mutation_ratio=rho,
            num_fields_to_change=num_to_change,
            fields_to_edit=", ".join(fields_to_edit),
            fields_to_keep=", ".join(fields_to_keep),
        )

        response, _ = get_response_from_llm(
            prompt=user_msg,
            client=self.client,
            model=self.model,
            system_message=system_msg,
            temperature=self.config.temperature_mutator,
        )

        child = self._parse_idea_json(response)
        if child is None:
            return None

        child = self._validate_and_fix(child, anchor, fields_to_keep)
        return child

    @staticmethod
    def _parse_idea_json(text: str) -> Optional[dict]:
        """Extract a JSON object from LLM response text."""
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
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse idea JSON: %s", e)
            return None

    @staticmethod
    def _validate_and_fix(child: dict, anchor: dict, fields_to_keep: list[str]) -> dict:
        """Ensure kept fields match the anchor and required fields exist."""
        for f in fields_to_keep:
            if f in anchor:
                child[f] = anchor[f]

        required = {"Name", "Title", "Short Hypothesis", "Abstract", "Experiments"}
        for req in required:
            if req not in child and req in anchor:
                child[req] = anchor[req]

        if "Name" in child and isinstance(child["Name"], str):
            child["Name"] = child["Name"].lower().replace(" ", "_").replace("-", "_")

        return child
