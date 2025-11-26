# guardian_angel/tools/generate_draft.py
#
# Copyright 2025 Sho Watanabe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from ai_steward.tools.prompt import PromptInstance, PromptTemplate

# DRAFT_WRITER_TEMPLATE = lambda lang: PromptTemplate(
#     role="You are an expert Strategic Intelligence and Security-Aware Report Writer.",
#     purpose=(
#         "Generate a comprehensive analytical REPORT (not Q&A) in JSON format. "
#         "Use Allowed context for factual grounding, while understanding—but never reproducing—Forbidden content. "
#         "Forbidden context may inform structure and escalation suggestions (META: doc_id, url, owner_name, owner_email), "
#         "but must not influence specific wording, numbers, or sensitive claims."
#     ),
#     lang=lang,
#     guiding_principles=[
#         "Return exactly ONE JSON object. No prose, no code fences.",
#         "The JSON must include: 'draft', 'citations', 'escalation_suggestions', 'thinkings'.",
#         "The report must read as a cohesive analytical document, not a bullet-point or Q&A list.",
#         "Structure the report as: Title → Executive Summary → Background → Findings → Related Context → Implications → Constraints → References.",
#         "Focus on overall project insight and causal understanding, not just direct answers to the user’s questions.",
#         "Add unasked but contextually relevant background and implications if they improve completeness and coherence.",
#         "If a key point cannot be supported without Forbidden context, replace it with a neutral placeholder ('権限外のため非開示') "
#         "and generate an escalation suggestion with META only.",
#         "Every factual paragraph must cite at least one Allowed document [doc-id].",
#     ],
#     instructions=[
#         # --- Step 1: Question understanding ---
#         "Read <questions> carefully. Group related ones thematically (e.g., technical feasibility, schedule, market scope).",
#         "Record this grouping as a 'thinkings' entry (action='grouping').",

#         # --- Step 2: Evidence collection ---
#         "Extract relevant Allowed evidence for each theme. Record doc-ids and summary facts.",
#         "If Forbidden context includes exclusive details, mark that theme for escalation only.",
#         "Record this as 'thinkings' entry (action='evidence_selection').",

#         # --- Step 3: Outline planning ---
#         "Construct a professional outline with mandatory sections: "
#         "1) # タイトル, 2) ## エグゼクティブサマリー, 3) ## 背景, 4) ## 主な所見, "
#         "5) ## 関連情報・周辺文脈, 6) ## 影響・今後の見通し, 7) ## 制約と未解決事項, 8) ## 参考情報.",
#         "Record this as 'thinkings' entry (action='outline_design').",

#         # --- Step 4: Draft writing ---
#         "Compose the report in neutral and analytical Japanese, suitable for internal strategic review.",
#         "Include the user’s direct question(s) as part of '主な所見', highlighted in context.",
#         "Integrate additional related facts and trends from Allowed context to enhance completeness.",
#         "Each paragraph that states a fact must end with at least one citation [doc-id].",
#         "When forbidden-only content is encountered, insert a neutral note ('詳細は権限外のため非開示') "
#         "and add an escalation entry with that document’s META.",
#         "Record a 'thinkings' entry (action='drafting').",

#         # --- Step 5: Quality validation ---
#         "Verify that: (a) all mandatory sections exist, (b) every factual statement is cited, "
#         "(c) forbidden influence is absent, and (d) escalation suggestions are provided where needed.",
#         "Record a final 'thinkings' entry (action='quality_review').",

#         # --- Step 6: Output assembly ---
#         "Return only the JSON object with keys: draft, citations, escalation_suggestions, thinkings.",
#     ],
#     output_schema={
#         "draft": str,
#         "citations": [str],
#         "escalation_suggestions": [str],
#         "thinkings": [{
#             "step": int, 
#             "action": str, 
#             "input_questions": [str], 
#             "decision": str, 
#             "rationale": str,
#             "sources_used": [str]
#         }]
#     },
#     examples=[
#         {
#             "thinkings": [
#                 {
#                     "step": 1,
#                     "action": "grouping",
#                     "themes": [
#                         {"theme": "AURAティザー計画の時期と地域"},
#                         {"theme": "Cell-Nova量産性および関連リスク"}
#                     ],
#                     "rationale": "マーケティング計画と技術計画で整理することで因果を明確化"
#                 },
#                 {
#                     "step": 2,
#                     "action": "evidence_selection",
#                     "sources_used": ["doc-l2-514"],
#                     "notes": "Allowed内に開始時期は記載。国および量産安定性は上位機密にのみ存在。"
#                 },
#                 {
#                     "step": 3,
#                     "action": "outline_design",
#                     "decision": "全8章体制を採用し、背景と関連情報を拡張して文脈を強化"
#                 },
#                 {
#                     "step": 4,
#                     "action": "drafting",
#                     "rationale": "Forbiddenに触れず、Allowedのみで記述。非開示部分にはエスカレーションMETAを付与。"
#                 },
#                 {
#                     "step": 5,
#                     "action": "quality_review",
#                     "checks": {
#                         "sections_present": True,
#                         "citations_coverage": "100%",
#                         "no_forbidden_influence": True
#                     }
#                 }
#             ],
#             "draft": "........",
#             "citations": ["doc-l?-???"],
#             "escalation_suggestions": [
#                 {
#                     "topic": "...",
#                     "forbidden_doc_id": "...",
#                     "url": "https://...",
#                     "owner_name": "...",
#                     "owner_email": "...@example.com"
#                 }
#             ]
#         }
#     ]
# )

DRAFT_WRITER_TEMPLATE = lambda lang: PromptTemplate(
    role="You are an Analytical Draft Writer for a Security-Aware LLM Agent Pipeline.",
    purpose=(
        "Produce an initial high-level DRAFT and structured reasoning plan. "
        "Use Allowed context only. Forbidden context may contribute metadata only "
        "(doc_id, url, owner_name, owner_email), never semantic content. "
        "You do NOT produce the final report—that is the Reflector's role."
    ),
    lang=lang,
    guiding_principles=[
        "Output exactly ONE JSON object. No prose outside the JSON. No code fences.",
        "This agent produces an INITIAL DRAFT and reasoning structure, not a polished report.",
        "The draft may be partial; clarity > completeness.",
        "Forbidden information must not influence content. Metadata may be used only for escalation markers.",
        "All factual statements must cite Allowed document IDs.",
        "Language of the draft must match {lang}.",
    ],
    instructions=[
        # --- Step 1: Understand the input questions ---
        "Read <questions>. Identify key topics and thematic clusters.",
        "Record a thinking entry (step=1, action='grouping').",

        # --- Step 2: Collect relevant Allowed evidence ---
        "Extract only Allowed evidence relevant to each theme.",
        "If a theme relies on Forbidden-only information, mark it as 'needs_escalation'.",
        "Record a thinking entry (step=2, action='evidence_review').",

        # --- Step 3: Produce initial high-level outline (draft structure) ---
        "Create a coarse outline appropriate for {lang}, such as:",
        "  1) Title",
        "  2) Summary of Key Themes",
        "  3) Context",
        "  4) Initial Observations",
        "  5) Gaps & Needed Escalation",
        "This outline must be flexible because Reviewer will redesign it.",
        "Record a thinking entry (step=3, action='outline_proposal').",

        # --- Step 4: Generate a rough draft ---
        "Write a short high-level draft (not the final report). "
        "Include only Allowed-derived statements with citations.",
        "Where Forbidden-only info would be required, insert a placeholder "
        "({lang == 'ja' and '詳細は権限外のため非開示' or '[REDACTED — insufficient permission]'}).",
        "Record a thinking entry (step=4, action='drafting').",

        # --- Step 5: Final assembly ---
        "Return JSON with keys: draft, citations, escalation_suggestions, thinkings.",
    ],
    output_schema={
        "draft": str,
        "citations": [str],
        "escalation_suggestions": [
            {"topic": str, "forbidden_doc_id": str, "url": str, "owner_name": str, "owner_email": str}
        ],
        "thinkings": [{
            "step": int,
            "action": str,
            "decision": str | None,
            "rationale": str | None,
            "sources_used": list[str] | None
        }]
    },
    examples=[]
)

def draft_writer(
    original_question: str,
    questions: list[str],
    core_blob: str,
    upper_blob: str,
    allow_upper_context: bool,
    llm_infer: Callable,
    lang: str,
    temperature: float = 0.2,
    increment_temperature: float = 0.1,
    max_temperature: float = 0.80,
):
    forbidden_placeholder = (
        # for strict access style
        "(ACCESS DENIED: This context is unavailable at the current access level. "
        "Do not include, paraphrase, or infer its contents in the draft.)"
    )

    user_data = {
        "user_question": original_question,
        "questions": "\n- ".join(questions),
        "allowed_context": core_blob,
        "forbidden_context": upper_blob if allow_upper_context else forbidden_placeholder
    }
    
    prompts = PromptInstance(template=DRAFT_WRITER_TEMPLATE(lang), user_prompts=user_data)

    current_temperature = temperature
    while temperature < max_temperature:
        drafts = llm_infer(prompts, current_temperature)
        if isinstance(drafts, dict):
            break
        current_temperature += increment_temperature
    
    return drafts
