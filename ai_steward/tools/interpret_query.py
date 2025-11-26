# guardian_angel/tools/interpret_query.py
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

INTERPRETER_TEMPLATE = lambda lang: PromptTemplate(
        role=f"You are a Master Query Analyst. Decompose complex user questions into clear, atomic sub-queries that downstream retrieval systems can use directly. Work in {lang}.",
        purpose="Decompose user question into reasonable questions for next tasks as a professional task.",
        lang=lang,
        guiding_principles=[
            "Output EXACTLY ONE artifact: a SINGLE JSON object (no tags, no code fences, no explanations).",
            (
                "REQUIRED top-level keys:\n"
                "questions: array of strings (each atomic and retrieval-ready)\n"
                "thinkings: object with fields (ALL required):\n"
                '  - language: string\n'
                '  - assumptions: [string]\n'
                '  - purposes: [string]\n'
                '  - keypoints: [string]\n'
                '  - methods: {"approach": string, "steps": [string]}\n'
                '  - question_analysis: [{"original": string, "resolved": string}]\n'
                '  - decomposition_plan: {"strategy": string, "checks": [string]}\n'
                '  - validation_notes: [string]\n'
                '  - summary: string'
            ),
            "No extra keys beyond the schema.",
            "Use ONLY the information in the user prompt; do not invent facts.",
            f"Maintain {lang} unless proper nouns or technical terms require otherwise."
        ],
        instructions=[
            "Read <user question>.",
            "Identify coordinated/composite intents and split them into atomic queries.",
            'Resolve pronouns or deictic references (e.g., "it") to explicit entities.',
            "Ensure coverage and deduplicate.",
            "Return JSON following the schema exactly."
        ],
        validation=[
            '"questions" items must be atomic and directly retrievable.',
            '"question_analysis" MUST include pronoun/ambiguity resolutions if present.',
            '"validation_notes" briefly justify the decomposition and coverage.'
        ],
        examples = [
            {
                "thinkings": {
                    "language": "English",
                    "assumptions": ["User wants definition and launch date of Apple M4 chip."],
                    "purposes": ["Enable factual retrieval without redundancy."],
                    "keypoints": ["entity=M4 chip", "definition", "launch date"],
                    "methods": {
                        "approach": "Coreference + clause splitting + dedup",
                        "steps": [
                            "detect entity mentions",
                            'resolve "it" -> Apple M4 chip',
                            "remove near-duplicate asks",
                        ],
                    },
                    "question_analysis": [
                        {"original": "it", "resolved": "Apple M4 chip"},
                        {"original": "What is the M4?", "resolved": "Definition of Apple M4 chip"},
                        {"original": "Explain the M4", "resolved": "Definition of Apple M4 chip (duplicate)"},
                    ],
                    "decomposition_plan": {
                        "strategy": "Deduplicate near-synonymous asks; keep atomic facts separate",
                        "checks": ["coverage", "deduplicate", "no-pronouns"],
                    },
                    "validation_notes": ["Definition duplicates merged; launch date kept separate."],
                    "summary": "Two atomic questions: definition and launch date.",
                },
                "questions": [
                    "What is the Apple M4 chip?",
                    "When was the Apple M4 chip launched?",
                ],
            },
            {
                "thinkings": {
                    "language": "English",
                    "assumptions": ['"Jaguar" can mean animal or car brand.'],
                    "purposes": ["Cover both plausible senses without inventing."],
                    "keypoints": ["sense1=animal", "sense2=car"],
                    "methods": {
                        "approach": "Sense disambiguation by branching",
                        "steps": ["list senses", "spawn atomic queries per sense"],
                    },
                    "question_analysis": [
                        {"original": "Jaguar", "resolved": "Ambiguous: animal vs. Jaguar (car brand)"}
                    ],
                    "decomposition_plan": {
                        "strategy": "Branch per sense",
                        "checks": ["sense-coverage", "no cross-contamination"],
                    },
                    "validation_notes": ["Each atomic query targets a single sense."],
                    "summary": 'Two atomic queries for two senses of "Jaguar".',
                },
                "questions": [
                    "What is the maximum running speed of a jaguar (animal)?",
                    "What is the top speed of the latest Jaguar car model?",
                ],
            },
            {
                "thinkings": {
                    "language": "English",
                    "assumptions": ["Policies change over time; retrieval needs explicit time windows."],
                    "purposes": ["Compare last-year vs this-year rules."],
                    "keypoints": ["country=Japan", "topic=COVID entry rules", "time windows"],
                    "methods": {
                        "approach": "Temporal scoping",
                        "steps": ["identify topic", "split by time period", "add compare query if needed"],
                    },
                    "question_analysis": [
                        {"original": "this year", "resolved": "YYYY (current year)"},
                        {"original": "last year", "resolved": "YYYY-1"},
                    ],
                    "decomposition_plan": {
                        "strategy": "Split by time ranges + optional diff",
                        "checks": ["per-period atomicity", "no leakage across periods"],
                    },
                    "validation_notes": ["Two atomic retrievals plus one comparison keeps sources distinct."],
                    "summary": "Two scoped queries and one optional diff.",
                },
                "questions": [
                    "What were Japan’s COVID-19 entry rules in <YYYY-1>?",
                    "What are Japan’s COVID-19 entry rules in <YYYY>?",
                    "What changed between <YYYY-1> and <YYYY> in Japan’s COVID-19 entry rules?",
                ],
            },
            {
                "thinkings": {
                    "language": "English",
                    "assumptions": ["User needs training recipe details and a direct comparison."],
                    "purposes": ["Retrieve per-model facts; then produce a comparison."],
                    "keypoints": ["model1=YOLOv8", "model2=YOLOv11", "training recipes", "datasets"],
                    "methods": {
                        "approach": "Per-entity retrieval + comparator",
                        "steps": ["split per model", "add comparator ask"],
                    },
                    "question_analysis": [
                        {"original": "training recipes", "resolved": "optimizer, lr schedule, aug, epochs"},
                        {"original": "datasets", "resolved": "officially used/common benchmarks"},
                    ],
                    "decomposition_plan": {
                        "strategy": "Entity-wise split + comparison",
                        "checks": ["coverage of both entities", "no duplication"],
                    },
                    "validation_notes": ["Two atomic retrievals + one comparison query."],
                    "summary": "Entity pages feed the comparator.",
                },
                "questions": [
                    "What are the official training recipes and common datasets for YOLOv8?",
                    "What are the official training recipes and common datasets for YOLOv11?",
                    "How do YOLOv8 and YOLOv11 differ in training recipe and dataset usage?",
                ],
            },
            {
                "thinkings": {
                    "language": "English",
                    "assumptions": ["User wants 2024-only diffusion+RL papers and code links."],
                    "purposes": ["Collect top-K then fetch code URLs."],
                    "keypoints": ["topic=diffusion RL", "year=2024", "K=3", "code links"],
                    "methods": {
                        "approach": "Constrained retrieval + follow-up metadata",
                        "steps": ["filter by year/topic", "rank/select top 3", "resolve repo links"],
                    },
                    "question_analysis": [
                        {
                            "original": "top 3 papers",
                            "resolved": "ranked subset of 2024 diffusion+RL papers",
                        },
                        {
                            "original": "code links",
                            "resolved": "official GitHub/Code repo URLs per paper",
                        },
                    ],
                    "decomposition_plan": {
                        "strategy": "Primary list + per-item metadata fetch",
                        "checks": ["year filter active", "K respected"],
                    },
                    "validation_notes": [
                        "First query yields candidate set; follow-ups fetch code links."
                    ],
                    "summary": "One list query + three code-link lookups.",
                },
                "questions": [
                    "List the top 3 diffusion-model-for-RL papers published in 2024.",
                    "Provide the official code repository for paper #1.",
                    "Provide the official code repository for paper #2.",
                    "Provide the official code repository for paper #3.",
                ],
            },
        ],
        output_schema={
            "questions": [str],
            "thinkings": {
                "language": str,
                "assumptions": [str],
                "purposes": [str],
                "keypoints": [str],
                "methods": {"approach": str, "steps": [str]},
                "question_analysis": [{"original": str, "resolved": str}],
                "decomposition_plan": {"strategy": str, "checks": [str]},
                "validation_notes": [str],
                "summary": str
            }
        }
)

def interpreter(
    question: str,
    lang: str,
    llm_infer: Callable[[PromptInstance, float], str | dict],
    temperature: float = 0.05,
    increment_temperature: float = 0.05,
    max_temperature: float = 0.85,
) -> dict | str:
    interpreter_template = INTERPRETER_TEMPLATE(lang)
    interpreter_prompts = PromptInstance(template=interpreter_template, user_prompts={"user question": question})
    current_temperature = temperature

    while temperature < max_temperature:
        questions = llm_infer(interpreter_prompts, current_temperature)
        if isinstance(questions, dict):
            break
        current_temperature += increment_temperature

    return questions
