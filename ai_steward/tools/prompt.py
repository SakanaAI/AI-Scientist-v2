# guardian_angel/tools/prompt.py
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

import json
from dataclasses import dataclass, field
from typing import Any
from functools import cached_property

def pytype_to_name(t):
    return {
        str: "string", int: "integer", float: "number",
        bool: "boolean", type(None): "null",
    }.get(t, "any")

def to_serializable_schema(spec):
    if isinstance(spec, dict):
        return {k: to_serializable_schema(v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [to_serializable_schema(spec[0])] if spec else []
    if isinstance(spec, type):
        return pytype_to_name(spec)
    return spec

@dataclass(frozen=True)
class PromptTemplate:
    role: str
    purpose: str
    lang: str
    output_schema: dict[str, Any]
    guiding_principles: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    def _format_section(self, tag: str, content: Any, use_numbering: bool = False) -> str:
        if not content:
            return ""
        if isinstance(content, list):
            items = "\n".join(
                (f"{i}. {item}" for i, item in enumerate(content, 1)) if use_numbering
                else (f"- {item}" for item in content)
            )
            return f"<{tag}>\n{items}\n</{tag}>\n\n"
        if isinstance(content, dict):
            json_str = json.dumps(content, indent=2, ensure_ascii=False)
            return f"<{tag}>\n{json_str}\n</{tag}>\n\n"
        return f"<{tag}>\n{str(content)}\n</{tag}>\n\n"

    @cached_property
    def system_prompt(self) -> str:
        schema_dict = self.schema_str
        
        prompt_parts = [
            self._format_section("role", self.role),
            self._format_section("purpose", self.purpose),
            self._format_section("lang", self.lang),
            self._format_section("guiding_principles", self.guiding_principles),
            self._format_section("instructions", self.instructions, use_numbering=True),
            self._format_section("validation", self.validation),
            self._format_section("examples", self.examples),
            self._format_section("output_schema", schema_dict),
        ]
        return "".join(part for part in prompt_parts if part).strip()
    
    @cached_property
    def schema(self) -> dict:
        return self.output_schema
    @cached_property
    def schema_str(self) -> dict:
        schema_dict: dict = to_serializable_schema(self.output_schema) # type: ignore
        return schema_dict


class PromptInstance:
    def __init__(self, template: PromptTemplate, user_prompts: dict[str, str]):
        self.template = template
        self.user_prompts = user_prompts

    @property
    def system_prompt(self) -> str:
        return self.template.system_prompt
    
    @property
    def schema_str(self) -> dict:
        return self.template.schema_str
    @property
    def schema(self) -> dict:
        return self.template.schema

    @cached_property
    def user_prompt(self) -> str:
        prompt_parts = [
            self.template._format_section(k, v) for k, v in self.user_prompts.items()
        ]
        return "".join(part for part in prompt_parts if part).strip()
    