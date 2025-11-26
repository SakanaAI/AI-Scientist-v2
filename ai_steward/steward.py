# guardian_security/confidence.py
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

import datetime
import json
import os
import os.path as osp
import sys
from typing import Any, Callable
import logging

from ai_steward.tools.prompt import PromptInstance

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_steward.llm import (
    create_client,
)
from ai_steward.tools.interpret_query import interpreter
from ai_steward.tools.search_db import searcher
from ai_steward.tools.generate_draft import draft_writer
from ai_steward.tools.json_recorrection import exclude_thinkings, get_json_response
from ai_steward.tools.corrector import refine_tree

def run_secure_answer(
    model: str,
    embed_model: str,
    question: str, user_level: str, kb_path: str, out_path: str, lang: str,
    allow_upper_context: bool = True, debug: bool = False, max_retries: int = 10
) -> None:

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    def _save_debug_output(step_name: str, content: Any, file_type: str = "txt"):
        if not debug: return
        try:
            debug_dir = osp.join("outputs", "debug", run_id)
            os.makedirs(debug_dir, exist_ok=True)
            filepath = osp.join(debug_dir, f"{step_name}.{file_type}")
            with open(filepath, "w", encoding="utf-8") as f:
                if isinstance(content, (dict, list)): json.dump(content, f, ensure_ascii=False, indent=2)
                else: f.write(str(content))
            print(f"[DEBUG] Saved step output to: {filepath}")
        except Exception as e:
            print(f"[DEBUG] Failed to save debug output for {step_name}: {e}")

    llm_client, llm_model = create_client(model)
    llm_infer: Callable[[PromptInstance, float], str | dict] = lambda prompts, temperature: get_json_response(prompts=prompts, client=llm_client, model=llm_model, max_retries=max_retries, temperature=temperature)
    embed_client, real_embed_model = create_client(embed_model)

    analyzer_json = interpreter(question, lang, llm_infer)
    logging.debug(f'-----Pick up questions-----\n{exclude_thinkings(analyzer_json)}')

    if not isinstance(analyzer_json, dict):
        raise ValueError(f"Invalid format is given. {type(analyzer_json)}")
    _save_debug_output("01_query_analyzer_parsed", analyzer_json, "json")
    questions = analyzer_json.get("questions")
    if questions is None:
        raise ValueError(f"Questions are not extracted")


    core_blob, upper_blob = searcher(questions, kb_path, embed_client, real_embed_model, user_level)
    logging.debug(f'-----Search done-----')

    drafts = draft_writer(question, questions, core_blob, upper_blob, allow_upper_context, llm_infer, lang)
    logging.debug(f'-----Write up drafts-----\n{exclude_thinkings(drafts)}')

    refined_draft = refine_tree(drafts, core_blob, upper_blob, llm_infer, lang)
    logging.debug(f'-----Refined drafts-----\n{refined_draft.draft}\n{refined_draft.citations}\n{refined_draft.escalation_suggestions}')

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined_draft.draft, f, ensure_ascii=False, indent=2)
