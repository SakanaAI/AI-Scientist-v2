# guardian_angel/tools/search_db.py
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

from typing import Any

from ai_steward.tools.knowledge_database import LocalKnowledgeBaseSearchTool, SecurityLevel, allowed, pack_doc_snippet, parse_search_payload

def searcher(questions: list[str], kb_path: str, embed_client: Any, real_embed_model: Any, user_level: str | SecurityLevel):
    kb_tool = LocalKnowledgeBaseSearchTool(kb_path, embed_client, real_embed_model)
    search_query = " ".join(questions)
    search_payload = kb_tool.use_tool(query=search_query, top_k=12)

    hits = parse_search_payload(search_payload)
    core_docs: list[dict[str, Any]] = [d for d in hits if allowed(d.get("level", "L0"), user_level)]
    upper_docs: list[dict[str, Any]] = [d for d in hits if not allowed(d.get("level", "L0"), user_level)]

    core_blob = "\n".join([pack_doc_snippet(d) for d in core_docs]) or "(none)"
    upper_blob = "\n".join([pack_doc_snippet(d) for d in upper_docs]) or "(none)"

    return core_blob, upper_blob
