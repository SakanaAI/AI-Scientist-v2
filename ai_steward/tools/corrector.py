# guardian_angel/tools/corrector.py

# Copyright 2025 Sho Watanabe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import json
import heapq
import hashlib
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_steward.tools.prompt import PromptInstance, PromptTemplate

REVIEWER_TEMPLATE = lambda lang: PromptTemplate(
    role=(
        "You are a Strategic Content Reviewer and Report Architect. "
        "You have to construct high-quality reports while strictly respecting document-level confidentiality boundaries."
    ),
    purpose=(
        "You may freely use the *full content* of any Allowed document to grasp an entire image.\n"
        "\n"
        "For Forbidden documents:\n"
        "- You MUST NOT use or paraphrase any confidential portions of their content.\n"
        "- You MAY use any parts of the document that are *not confidential*, including public information, general knowledge contained in the doc, non-sensitive sections, and document metadata (doc_id, title, url, owner info).\n"
        "- When unsure whether a portion is confidential, treat it as confidential.\n"
        "\n"
        "Your task is to produce a long-form REPORT outline and improvement plan that leverages all non-confidential information to maximize clarity, quality, and completeness while ensuring that no confidential content is leaked across the access boundary."
    ),
    lang=lang,
    guiding_principles=[
        "Output must be ONLY ONE JSON object. No prose, no code fences.",
        "The report must read as a self-contained analytical document — not a Q&A or FAQ.",
        "Every factual claim must be traceable to Allowed doc-ids."
    ],
    instructions = [
        # 1. Narrative structure
        "Construct an outline_spec following a professional analytical report narrative.\n"
        "The outline should go beyond simply answering the questions and should include a coherent overview of the project, thematic context, and the relationships among the questions.\n"
        "Avoid any Q&A-style structure."
        "The document must read like a high-level technical or organizational report written by a senior analyst."
        "Use a hierarchical structure with clear top-down logic (overview → analysis → details → implications).",

        # 2. Section ordering
        "Recommended section order: "
        "1) Title, "
        "2) Executive Summary (non-technical), "
        "3) Context & Background, "
        "4) Problem Definition & Objectives, "
        "5) Methodology / System Overview, "
        "6) Key Findings & Analysis, "
        "7) Technical or Organizational Details, "
        "8) Risks, Constraints, and Escalation Notes, "
        "9) Implications & Future Outlook, "
        "10) Recommendations, "
        "11) Sources and Appendices.",

        # 3. Quality criteria
        "Define `quality_targets` including: "
        "min_sections>=8, min_total_words>=900, "
        "readability_grade_range=[10,14], "
        "citation_density='at least one citation per major claim', "
        "terminology_consistency using canonical domain vocabulary, "
        "and zero conversational tone.",

        # 4. Professional tone specification
        "Ensure tone is: formal, analytical, concise, and free from conversational expressions. "
        "Sentences must be well-structured, using topic sentences and transitions that reinforce logical flow. "
        "Avoid rhetorical questions entirely.",

        # 5. Additional elements
        "Identify and propose `additional_elements_suggestions`: "
        "critical contextual or analytical components that improve completeness. "
        "Examples: stakeholder mapping, governance structure, long-term dependency graph, "
        "security posture, compliance considerations, comparison with industry practices, "
        "economic impact, or integration roadmap.",

        # 6. Improvement plan (structured)
        "Create an `improvement_plan` composed of actions: "
        "Add, Restructure, ExpandContext, Integrate, Refine. "
        "Each action must: (a) reference Allowed evidence; (b) explain the expected improvement "
        "in clarity, completeness, or technical accuracy.",

        # 7. Coverage completeness
        "Formulate a `coverage_plan` ensuring that (a) all explicit user requirements, "
        "and (b) all additional_elements_suggestions are fully addressed in the final document.",

        # 9. Compliance with confidentiality rules
        "Ensure all content complies with document-level access rules: "
        "Allowed documents may be used freely; "
        "Forbidden documents may only contribute non-confidential insights or metadata.",
    ],
    validation=[
        "Return exactly one JSON object matching <output_schema>.",
        "Every evidence reference must be a valid Allowed doc-id."
    ],
    output_schema={
        "quality_targets": {
            "min_sections": int,
            "min_total_words": int,
            "min_citations_per_sentence": int,
            "readability_grade_range": [int, int],
            "terminology_consistency": [str]
        },
        "outline_spec": [
            {"section": str, "required_subsections": [str]}
        ],
        "evidence_matrix": [
            {"key_claim": str, "allowed_evidence": [str], "status": "needs_redaction_and_escalation"}
        ],
        "coverage_plan": [
            {"requirement": str, "status": str}
        ],
        "additional_elements_suggestions": [
            {"topic": str, "rationale": str, "supporting_docs": [str]}
        ],
        "improvement_plan": [
            {
                "action": str,
                "detail": str,
                "justification": {
                    "audit_issue": str,
                    "allowed_evidence": [str],
                    "expected_effect": str
                }
            }
        ],
        "thinkings": {
            "audit_findings": [{"issue": str, "evidence": str}],
            "issue_action_map": [{"issue": str, "proposed_actions": [str]}],
            "contextual_analysis": [{"missing_context": str, "suggested_fix": str}]
        }
    },
    examples=[]
)

REFLECTOR_TEMPLATE = lambda lang: PromptTemplate(
    role="You are an expert Report Composer and Compliance Officer.",
    purpose=(
        "Produce a full REPORT (not a Q&A) using Allowed context and the provided improvement plan. "
        "Integrate both user-requested and suggested additional contextual elements to form a coherent long-form document."
    ),
    lang=lang,
    guiding_principles=[
        "Output must be ONE JSON object. No prose, no code fences.",
        "Every paragraph must be grounded in Allowed citations.",
        "The tone should match professional analytical reporting (neutral, explanatory, contextual)."
    ],
    instructions=[
        # ナラティブ構成
        "Follow the narrative outline from <outline_spec> — a cohesive report flow, not discrete answers.",
        "Integrate all 'additional_elements_suggestions' naturally, using Allowed documents only.",
        "Emphasize logical transitions between context, findings, and implications.",
        
        # 品質基準
        "Meet min_total_words and min_sections; ensure readability within specified grade range.",
        "Enforce consistent terminology as defined in terminology_consistency.",
        
        # Forbidden情報への対処
        "If any claim needs forbidden-only info, replace it with a standardized redaction notice and append an escalation META entry "
        "with owner and document identifiers from <forbidden_context>.",
        
        # 検証
        "Collect all cited Allowed doc-ids in 'citations' and ensure every section meets citation density constraints."
    ],
    validation=[
        "Return one JSON matching <output_schema>.",
        "Citations must be strictly from Allowed context."
    ],
    output_schema={
        "draft": str,   # Long-form markdown report
        "citations": [str],
        "escalation_suggestions": [
            {"topic": str, "forbidden_doc_id": str, "url": str, "owner_name": str, "owner_email": str}
        ],
        "thinkings": {
            "integration_log": [{"element": str, "method": str}],
            "redaction_log": [{"topic": str, "reason": str}],
            "verification": {
                "sections_ok": bool,
                "citations_ok": bool,
                "no_forbidden_influence": bool,
                "readability_in_range": bool
            }
        }
    },
    examples=[]
)

# SECURITY_ANALYST_TEMPLATE = lambda lang: PromptTemplate(
#     role="You are a Security Analyst AI.",
#     purpose="Detect any leak or influence from forbidden content in a draft, including paraphrase or derived inference.",
#     lang=lang,
#     guiding_principles=[
#         "Return ONE JSON object. No prose, no code fences.",
#         "A leak exists if any claim/number/date/topic in <draft_to_evaluate> can ONLY be sourced or inferred from <forbidden_context>.",
#         "Paraphrased or uniquely combined forbidden-only facts also count as a leak."
#     ],
#     instructions=[
#         "Extract claims from <draft_to_evaluate> (entities, numbers, dates, causal attributions).",
#         "For each claim, check presence/support in <allowed_context> vs <forbidden_context>.",
#         "If unsupported by Allowed but present/derivable from Forbidden, set leak_detected=true and provide a concise leak_reason (≤ 30 words).",
#         "list under_evidenced_claims: claims not clearly supported by Allowed (even if no Forbidden match)."
#     ],
#     validation=["Return exactly one JSON object matching <output_schema>."],
#     output_schema={
#         "leak_detected": bool,
#         "leak_reason": str,
#         "under_evidenced_claims": [str],
#         "thinkings": {
#             "checked_claims": [str],
#             "forbidden_hits": [str],
#             "conclusion": str
#         }
#     },
#     examples=[]
# )

# QUALITY_ANALYST_TEMPLATE = lambda lang: PromptTemplate(
#     role="You are a specialist Quality Analyst AI.",
#     purpose="Objectively score quality based on clarity, structure, evidence, coverage, and consistency, using Allowed context only.",
#     lang=lang,
#     guiding_principles=[
#         "Return ONE JSON object. No prose, no code fences.",
#         "Scores are floats in [0.0, 1.0]."
#     ],
#     instructions=[
#         "Evaluate clarity (readability, concise language).",
#         "Evaluate structure (section order, transitions, headings aligning with outline_spec).",
#         "Evaluate evidence (each sentence cited with Allowed doc-ids; citation density).",
#         "Evaluate coverage (all user requirements addressed or explicitly marked).",
#         "Evaluate consistency (terminology and fact coherence).",
#         "Provide brief assessment_summary and a coverage_report."
#     ],
#     validation=["Return exactly one JSON object matching <output_schema>."],
#     output_schema={
#         "quality_assessment": {
#             "clarity_score": float,
#             "structure_score": float,
#             "evidence_score": float,
#             "coverage_score": float,
#             "consistency_score": float
#         },
#         "assessment_summary": str,
#         "coverage_report": [{"requirement": str, "status": str}],
#         "thinkings": {
#             "clarity_rationale": str,
#             "structure_rationale": str,
#             "evidence_rationale": str,
#             "coverage_rationale": str,
#             "consistency_rationale": str
#         }
#     },
#     examples=[]
# )

SECURITY_ANALYST_TEMPLATE = lambda lang: PromptTemplate(
    role="You are a Security Compliance AI responsible for detecting any forbidden-content leakage or influence.",
    purpose=(
        "Evaluate <draft_to_evaluate> and determine whether ANY part of the draft depends on forbidden-only information. "
        "Forbidden influence includes: (1) direct copying, (2) paraphrasing, (3) derived inference that cannot be justified "
        "solely with Allowed content."
    ),
    lang=lang,
    guiding_principles=[
        "Output exactly ONE JSON. No prose, no code fences.",
        "A claim leaks if it requires forbidden context to justify, even indirectly.",
        "Forbidden influence includes: specific numbers, timelines, internal process details, unique technical facts, "
        "or causal statements absent in Allowed context.",
        "Allowed context must be sufficient to support ALL factual content of the draft."
    ],
    instructions=[
        "1. Extract all factual claims from <draft_to_evaluate>: entities, numbers, dates, decisions, causal statements, "
        "implementation steps, risks, dependencies, or any content that asserts something factual.",
        
        "2. For each claim, test whether it is explicitly or implicitly supported by Allowed context. "
        "If the claim is ONLY present or inferable from Forbidden context, classify it as a leak.",
        
        "3. Provide leak_detected=true if ANY leaked claim exists.",
        
        "4. Provide leak_reasons: for each leaked claim, give a ≤30-word explanation referencing the relevant forbidden doc-id.",
        
        "5. Provide under_evidenced_claims: claims not clearly supported by Allowed context, even if not Forbidden.",
        
        "6. Log all checked claims, forbidden hits, and reasoning in 'thinkings'."
    ],
    validation=["Return exactly one JSON object matching <output_schema>."],
    output_schema={
        "leak_detected": bool,
        "leak_reasons": [{"claim": str, "reason": str, "forbidden_doc_id": str}],
        "under_evidenced_claims": [str],
        "thinkings": {
            "checked_claims": [str],
            "forbidden_hits": [str],
            "conclusion": str
        }
    },
    examples=[]
)

QUALITY_ANALYST_TEMPLATE = lambda lang: PromptTemplate(
    role="You are a Professional Report Quality Analyst AI.",
    purpose="Score the analytical quality of <draft> based ONLY on Allowed context and alignment with <outline_spec>.",
    lang=lang,
    guiding_principles=[
        "Return ONE JSON. No prose, no code fences.",
        "Scores are floats in [0.0, 1.0].",
        "Assess completeness, coherence, structure, evidence adequacy, and reasoning soundness."
    ],
    instructions=[
        "1. Assess CLARITY: readability, concise wording, logical paragraph flow.",
        
        "2. Assess STRUCTURE: adherence to outline_spec, section completeness, transitions, narrative arc.",
        
        "3. Assess EVIDENCE: citation density, correct use of Allowed doc-ids, sufficiency of evidence to support claims.",
        
        "4. Assess COVERAGE: all user requirements + all 'additional_elements_suggestions' addressed.",
        
        "5. Assess CONSISTENCY: terminology stability, internal logical coherence, absence of contradictions.",
        
        "6. Provide assessment_summary: concise explanation (<120 words).",
        
        "7. Provide coverage_report: list each requirement with status (met / unmet / partially met).",
        
        "8. Provide thinkings with rationales for each score."
    ],
    output_schema={
        "quality_assessment": {
            "clarity_score": float,
            "structure_score": float,
            "evidence_score": float,
            "coverage_score": float,
            "consistency_score": float
        },
        "assessment_summary": str,
        "coverage_report": [{"requirement": str, "status": str}],
        "thinkings": {
            "clarity_rationale": str,
            "structure_rationale": str,
            "evidence_rationale": str,
            "coverage_rationale": str,
            "consistency_rationale": str
        }
    },
    examples=[]
)

FORMATTER_TEMPLATE = lambda lang: PromptTemplate(
    role="You are a data aggregation expert.",
    purpose="Merge security and quality analysis into a final audit JSON.",
    lang=lang,
    guiding_principles=[
        "Return ONE JSON object. No prose, no code fences."
    ],
    instructions=[
        "Merge leak_detected/leak_reason from <security_report_json>.",
        "Merge quality_assessment and assessment_summary from <quality_report_json>.",
        "Compute overall_quality_ok: false if leak_detected; else true only if all scores >= 0.7.",
        "Add next_actions: prioritized steps that would fix leaks or lift any score below 0.7; reference exact metric(s) or the leak reason."
    ],
    validation=["Return exactly one JSON object matching <output_schema>."],
    output_schema={
        "leak_detected": bool,
        "leak_reason": str,
        "quality_assessment": {
            "clarity_score": float,
            "structure_score": float,
            "evidence_score": float,
            "coverage_score": float,
            "consistency_score": float
        },
        "overall_quality_ok": bool,
        "assessment_summary": str,
        "next_actions": [str],
        "thinkings": {
            "merge_log": [str],
            "quality_ok_decision_rule": str
        }
    },
    examples=[]
)

@dataclass(order=True)
class ScoredNode:
    priority: float
    node: Any = field(compare=False)

@dataclass
class DraftNode:
    draft: str
    citations: list[str]
    escalation_suggestions: list[dict[str, Any]]
    metrics: dict[str, Any]
    depth: int
    parent_hash: str

def hash_doc(draft: str) -> str:
    return hashlib.sha256(draft.encode("utf-8")).hexdigest()

def try_llm_json(
    llm_infer: Callable[..., Any],
    prompts: PromptInstance,
    base_temp: float = 0.1,
    max_temp: float = 0.9,
    step: float = 0.2,
    retries_per_temp: int = 2,
) -> dict[str, Any] | None:
    t = base_temp
    while t <= max_temp + 1e-9:
        for _ in range(retries_per_temp):
            out = llm_infer(prompts=prompts, temperature=t)
            if isinstance(out, dict):
                return out
        t = round(t + step, 6)
    logging.warning("[try_llm_json] failed to get JSON dict after retries.")
    return None

def _pop_with_exploration(open_queue: list[ScoredNode], *, epsilon: float, explore_top_k: int, revisit_penalty: float = 0.05) -> ScoredNode:
    if not open_queue:
        raise IndexError("pop from empty queue")

    if random.random() > epsilon:
        choice = heapq.heappop(open_queue)
    else:
        # ε-explore
        k = min(explore_top_k, len(open_queue))
        buf = [heapq.heappop(open_queue) for _ in range(k)]
        choice = random.choice(buf)

        for item in buf:
            if item is not choice:
                heapq.heappush(open_queue, item)

    cloned_node = deepcopy(choice.node)
    cloned_node.revisit_index = getattr(cloned_node, "revisit_index", 0) + 1
    heapq.heappush(open_queue, ScoredNode(priority=choice.priority - revisit_penalty, node=cloned_node))

    return choice

def action_pipeline(
    current_node: DraftNode,
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str
) -> list[dict[str, Any]]:
    logging.info(f"  - Running action pipeline for draft: '{current_node.draft}'")
    # === Step 1: Reviewer ===
    reviewer_instance = PromptInstance(
        template=REVIEWER_TEMPLATE(lang),
        user_prompts={
            "current_draft": current_node.draft,
            "allowed_context": core_blob,
            "previous_audit_results_json": json.dumps(current_node.metrics, ensure_ascii=False)
        }
    )

    plan_json = try_llm_json(
        llm_infer,
        reviewer_instance,
        base_temp=0.1, max_temp=0.6, step=0.2, retries_per_temp=2
    )
    if plan_json is None:
        logging.info("    ... Reviewer failed to return JSON. Prune branch.")
        return []

    logging.info(f"    ... Reviewer proposed plan: {plan_json.get('improvement_plan', [])}")

    # === Step 2: Reflector ===
    reflector_instance = PromptInstance(
        template=REFLECTOR_TEMPLATE(lang),
        user_prompts={
            "original_draft": current_node.draft,
            "improvement_plan_json": json.dumps(plan_json, ensure_ascii=False),
            "allowed_context": core_blob,
            "forbidden_context": upper_blob
        }
    )

    final_doc_state = try_llm_json(
        llm_infer,
        reflector_instance,
        base_temp=0.0, max_temp=0.5, step=0.2, retries_per_temp=2
    )
    if final_doc_state is None:
        logging.info("    ... Reflector failed to return JSON. Prune branch.")
        return []

    logging.info(f"    ... Reflector finalized draft: '{final_doc_state.get('draft', '')}...'")
    return [final_doc_state]


def evaluate(
    doc_state: dict[str, Any],
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str
) -> tuple[float, dict[str, Any]]:
    """Security, Quality, Formatter。どれかが失敗したら低スコア or フォールバック結合で返す。"""
    draft = doc_state.get("draft", "") if isinstance(doc_state, dict) else ""

    # === Step 1: Security Analyst ===
    security_instance = PromptInstance(
        template=SECURITY_ANALYST_TEMPLATE(lang),
        user_prompts={"draft_to_evaluate": draft, "forbidden_context": upper_blob}
    )
    security_report_json = try_llm_json(llm_infer, security_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if security_report_json is None:
        logging.warning("security report is missing.")
        # handle as a leakage
        return -100.0, {"score": -100.0, "leak_detected": True, "leak_reason": "SEC_EVAL_PARSE_FAIL"}

    # === Step 2: Quality Analyst ===
    quality_instance = PromptInstance(
        template=QUALITY_ANALYST_TEMPLATE(lang),
        user_prompts={"draft_to_evaluate": draft, "allowed_context": core_blob}
    )
    quality_report_json = try_llm_json(llm_infer, quality_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if quality_report_json is None:
        logging.warning("quality report is missing.")
        # address as a document with low quality 
        return -50.0, {
            "score": -50.0,
            "leak_detected": security_report_json.get("leak_detected", True),
            "leak_reason": security_report_json.get("leak_reason", "SEC_OK"),
            "quality_assessment": {"clarity_score": 0.0, "structure_score": 0.0, "evidence_score": 0.0},
            "assessment_summary": "QUALITY_EVAL_PARSE_FAIL"
        }

    # === Step 3: Formatter ===
    formatter_instance = PromptInstance(
        template=FORMATTER_TEMPLATE(lang),
        user_prompts={
            "security_report_json": json.dumps(security_report_json, ensure_ascii=False),
            "quality_report_json": json.dumps(quality_report_json, ensure_ascii=False)
        }
    )
    audit_json = try_llm_json(llm_infer, formatter_instance, base_temp=0.0, max_temp=0.4, step=0.2)
    if audit_json is None:
        leak_detected = bool(security_report_json.get("leak_detected", True))
        qa = quality_report_json.get("quality_assessment", {})
        clarity = float(qa.get("clarity_score", 0.0) or 0.0)
        structure = float(qa.get("structure_score", 0.0) or 0.0)
        evidence = float(qa.get("evidence_score", 0.0) or 0.0)
        score = (0.3 * clarity + 0.4 * structure + 0.3 * evidence) - (100.0 * (1.0 if leak_detected else 0.0))
        return score, {
            "leak_detected": leak_detected,
            "leak_reason": security_report_json.get("leak_reason", ""),
            "quality_assessment": qa,
            "overall_quality_ok": (not leak_detected) and all(s >= 0.7 for s in [clarity, structure, evidence]),
            "assessment_summary": quality_report_json.get("assessment_summary", ""),
            "score": score
        }

    # --- Scoring Logic ---
    quality_scores = audit_json.get("quality_assessment", {})
    clarity = float(quality_scores.get("clarity_score", 0.0) or 0.0)
    structure = float(quality_scores.get("structure_score", 0.0) or 0.0)
    evidence = float(quality_scores.get("evidence_score", 0.0) or 0.0)
    leak_detected = audit_json.get("leak_detected", True)
    leak_risk = 1.0 if leak_detected else 0.0

    score = (0.3 * clarity + 0.4 * structure + 0.3 * evidence) - (100.0 * leak_risk)
    audit_json["score"] = score
    logging.info(f"    ... Final Evaluator score: {score:.2f}")

    return score, audit_json


def refine_tree(
    initial_state: dict[str, Any],
    core_blob: str,
    upper_blob: str,
    llm_infer: Callable[..., Any],
    lang: str,
    max_depth: int = 8,
    beam_size: int = 6,
    max_trial_num: int = 24,
    epsilon: float = 0.25,
    explore_top_k: int = 6,
) -> DraftNode:
    """Beam search ToT"""
    visited_hashes = set()

    s0, m0 = evaluate(initial_state, core_blob, upper_blob, llm_infer, lang)
    root = DraftNode(
        draft=initial_state.get("draft", "") if isinstance(initial_state, dict) else "",
        citations=initial_state.get("citations", []) if isinstance(initial_state, dict) else [],
        escalation_suggestions=initial_state.get("escalation_suggestions", []) if isinstance(initial_state, dict) else [],
        metrics=m0,
        depth=0,
        parent_hash="root"
    )

    open_queue: list[ScoredNode] = [ScoredNode(-s0, root)]
    best_node = root
    best_score = s0

    logging.info(f"\n--- Starting Tree Search (Max Depth: {max_depth}, Beam Size: {beam_size}) ---")
    
    trial_num = 0
    while open_queue and trial_num < max_trial_num:
        current_scored = _pop_with_exploration(open_queue, epsilon=epsilon, explore_top_k=explore_top_k, revisit_penalty=0.05)
        current_node = current_scored.node

        if current_node.depth >= max_depth:
            trial_num += 1
            continue

        current_hash = hash_doc(current_node.draft)
        if current_hash in visited_hashes:
            trial_num += 1
            continue
        visited_hashes.add(current_hash)

        logging.info(f"\nExpanding Node (Depth: {current_node.depth}, Score: {current_node.metrics.get('score', best_score):.2f})")

        child_candidates = action_pipeline(current_node, core_blob, upper_blob, llm_infer, lang)

        for new_doc_state in child_candidates:
            if not isinstance(new_doc_state, dict):
                continue

            new_draft = new_doc_state.get("draft", "")
            if not new_draft or hash_doc(new_draft) in visited_hashes:
                continue

            new_score, new_audit_metrics = evaluate(new_doc_state, core_blob, upper_blob, llm_infer, lang)

            new_node = DraftNode(
                draft=new_draft,
                citations=new_doc_state.get("citations", []),
                escalation_suggestions=new_doc_state.get("escalation_suggestions", []),
                metrics=new_audit_metrics,
                depth=current_node.depth + 1,
                parent_hash=current_hash
            )

            if new_score > best_score:
                best_score = new_score
                best_node = new_node

            heapq.heappush(open_queue, ScoredNode(-new_score, new_node))

        if len(open_queue) > beam_size:
            open_queue = sorted(open_queue)[:beam_size]
            heapq.heapify(open_queue)

        trial_num += 1

    logging.info("\n--- Tree Search Finished ---")
    return best_node
