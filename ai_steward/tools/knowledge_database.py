# guardian_angel/tools/knowledge_database.py
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

import enum
import json
import logging
from typing import Any
from os import path as osp
import numpy as np

from ai_steward.llm import get_embedding


class LocalKnowledgeBaseSearchTool:
    name = "SearchKnowledgeBase"
    description = (
        "Search local knowledge base using semantic search. Arguments: {\"query\": str, \"top_k\": int}. "
        "Each item includes: id, title, url, level (L0/L1/L2/L3), has_confidentiality (bool), owner_name, owner_email, summary."
    )

    def __init__(self, kb_path: str, embed_client: Any, embed_model: str):
        self.kb_path = kb_path
        self.client = embed_client
        self.model = embed_model
        self.index: list[dict[str, Any]] = []
        
        print(f"Loading knowledge base from: {kb_path}")
        if osp.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                original_index = json.load(f)
            
            print("Generating embeddings for all documents in memory... (This may take a while for a large KB)")
            for i, item in enumerate(original_index):
                content_to_embed = (item.get("title", "") + "\n" + item.get("summary", "")).strip()
                if content_to_embed:
                    try:
                        item["embedding_vec"] = get_embedding(content_to_embed, client=self.client, model=self.model)
                        self.index.append(item)
                    except Exception as e:
                        print(f"[ERROR] Failed to generate embedding for document ID {item.get('id')}: {e}")
                else:
                     print(f"  - Skipped document {i+1}/{len(original_index)} (no content)")
            print("In-memory embedding generation complete.")
        else:
            print(f"[ERROR] Knowledge base file not found: {kb_path}.")
            self.index = []

    def _get_query_embedding(self, query: str) -> np.ndarray:
        return get_embedding(query, client=self.client, model=self.model)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray) or vec1.shape != vec2.shape:
             return 0.0
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

    def use_tool(self, query: str, top_k: int = 12) -> str:
        if not self.index:
            return json.dumps({"query": query, "count": 0, "results": []}, ensure_ascii=False, indent=2)
        query_vec = self._get_query_embedding(query)
        scored_items = []
        for item in self.index:
            if "embedding_vec" in item:
                score = self._cosine_similarity(query_vec, item["embedding_vec"])
                item_copy = item.copy()
                item_copy.pop("embedding_vec", None)
                scored_items.append((score, item_copy))
        scored_items.sort(key=lambda x: x[0], reverse=True)
        results = [item for score, item in scored_items[:top_k]]
        payload = {"query": query, "count": len(results), "results": results}
        return json.dumps(payload, ensure_ascii=False, indent=2)

class SecurityLevel(enum.IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3

def level_of(x: str) -> SecurityLevel:
    try:
        return SecurityLevel[x]
    except KeyError:
        logging.warning(f"Invalid security level {x} is detected.")
        return SecurityLevel.L0
 
def allowed(level: str, user_level: str | SecurityLevel) -> bool: 
    if isinstance(user_level, SecurityLevel):
        return level_of(level) <= user_level
    else:
        return level_of(level) <= level_of(user_level)


def pack_doc_snippet(d: dict[str, Any]) -> str:
    return (
        f"- id: {d.get('id')}\n"
        f"  title: {d.get('title')}\n"
        f"  level: {level_of(d.get('level', 'L0'))}\n"
        f"  has_confidentiality: {bool(d.get('has_confidentiality'))}\n"
        f"  url: {d.get('url')}\n"
        f"  owner: {d.get('owner_name')} <{d.get('owner_email')}>\n"
        f"  summary: {d.get('summary')}\n"
    )

def citations_from_core(docs: list[dict[str, Any]]) -> list[str]:
    return [d.get("id") or d.get("title") or "" for d in docs if d.get("id") or d.get("title")]

def parse_search_payload(payload_str: str) -> list[dict[str, Any]]:
    try:
        p = json.loads(payload_str)
        return p.get("results", []) if isinstance(p, dict) else []
    except Exception:
        return []
