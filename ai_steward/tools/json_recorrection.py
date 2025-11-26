# guardian_angel/tools/json_recorrection.py
# 
# Modified based on ai_scientist/llm.py from AI-Scientist-v2
# Original work © 2024 SakanaAI Contributors
# Licensed under the Apache License, Version 2.0
# Modifications by Sho Watanabe (2025):
# - Adapted for Guardian Security Agent architecture
# - Added async request handling and enhanced caching
# - Removed speculative reasoning API hooks
#
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import re
import unicodedata
import logging
from typing import Any
from copy import deepcopy

from ai_steward.llm import get_response_from_llm
from ai_steward.tools.prompt import PromptInstance, PromptTemplate


THINKING_KEY = "thinkings"

def exclude_thinkings(response: str | dict) -> str | dict:
    if isinstance(response, dict) and THINKING_KEY in response:
        response = dict(response)
        response.pop(THINKING_KEY, None)
    return response

THINKING_JSON_ERROR_SHAPE = {
    "attempts": 0,
    "root_cause": "",
    "parser_errors": [""],
    "missing_keys": [""],
    "extra_keys_after": 0,
    "smells": [""],
    "selected_fix": "",
    "applied_patches": [{"rule": "", "effect": ""}],
    "notes": [""],
}

def augment_shape_with_diag(shape: dict) -> dict:
    if "thinkings_json_error" in shape:
        return shape
    new_shape = deepcopy(shape)
    new_shape["thinkings_json_error"] = THINKING_JSON_ERROR_SHAPE
    return new_shape

def normalize_text(text: str) -> str:
    """Unicode NFKC"""
    return unicodedata.normalize("NFKC", text)

def _build_norm_key_index(d: dict) -> dict[str, str]:
    return {normalize_text(k): k for k in d.keys()}

def _light_repair(js: str) -> str:
    js = re.sub(r"[\x00-\x1F\x7F]", "", js)
    js = re.sub(r",\s*([}\]])", r"\1", js)
    js = js.replace("'", '"')
    js = re.sub(r'(?m)(^|[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', js)
    return js

def _infer_top_type(shape: Any) -> str:
    if isinstance(shape, dict): return "dict"
    if isinstance(shape, list): return "list"
    return "unknown"

def _count_extra_keys(target: Any, shape: Any) -> int:
    cnt = 0
    if isinstance(target, dict) and isinstance(shape, dict):
        shape_keys_norm = {normalize_text(k) for k in shape.keys()}
        for k, v in target.items():
            nk = normalize_text(k)
            if nk not in shape_keys_norm:
                cnt += 1
            else:
                cnt += _count_extra_keys(v, shape[nk])
    elif isinstance(target, list) and isinstance(shape, list) and len(shape) > 0:
        proto = shape[0]
        for tv in target:
            cnt += _count_extra_keys(tv, proto)
    return cnt

def _strip_bom_ws(s: str) -> str:
    return s.lstrip("\ufeff").strip()

def _collect_code_fence_blocks(text: str) -> list[str]:
    blocks = []
    for m in re.finditer(r"```json\s+([\s\S]*?)```", text, flags=re.IGNORECASE):
        blocks.append(m.group(1))
    for m in re.finditer(r"```\s+([\s\S]*?)```", text):
        blocks.append(m.group(1))
    return blocks

def _collect_xml_tag_blocks(text: str, tag: str = "json") -> list[str]:
    """<json> ... </json> の中身"""
    pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
    return [m.group(1) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]

def _strict_object_candidates(text: str) -> list[str]:
    """
      - { ... }
      - 'json { ... }' / 'json: { ... }' / 'json= { ... }'
      - ```json ... ```
      - <json> ... </json>
    """
    n_open = text.count("{")
    n_close = text.count("}")
    if n_open > n_close:
        text = text + ("}" * (n_open - n_close))
        
    text = _strip_bom_ws(text)
    cands: list[str] = []

    # 1) ```json ... ```
    for block in _collect_code_fence_blocks(text):
        block = _strip_bom_ws(block)
        if block.startswith("{") and block.endswith("}"):
            cands.append(block)

    # 2) <json> ... </json>
    for block in _collect_xml_tag_blocks(text, "json"):
        block = _strip_bom_ws(block)
        if block.startswith("{") and block.endswith("}"):
            cands.append(block)

    # 3) { ... }
    if text.startswith("{") and text.endswith("}"):
        cands.append(text)

    # 4) json[:=]? { ... }
    m = re.match(r"^\s*json\s*[:=]?\s*(\{[\s\S]*\})\s*$", text, flags=re.IGNORECASE)
    if m:
        block = _strip_bom_ws(m.group(1))
        if block.startswith("{") and block.endswith("}"):
            cands.append(block)

    seen = set()
    uniq: list[str] = []
    for s in sorted(cands, key=lambda x: -len(x)):
        key = (s[:512], len(s))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq

def compare_dict(target: Any, shape: Any, path: list[str] | None = None) -> list[str]:
    path = path or []
    if isinstance(shape, dict):
        if not isinstance(target, dict):
            return [f"{'.'.join(path) or '<root>'} should be dict but is {type(target)} in {target}"]
        errors: list[str] = []
        norm_idx = _build_norm_key_index(target)
        for k, vshape in shape.items():
            if "thinkings" in k:
                continue
            nk = normalize_text(k)
            if nk not in norm_idx:
                errors.append(f"missing key: {'.'.join(path+[k])} in {target}")
                continue
            real_k = norm_idx[nk]
            errors += compare_dict(target[real_k], vshape, path+[k])
        return errors
    if isinstance(shape, list):
        if not isinstance(target, list):
            return [f"{'.'.join(path) or '<root>'} should be list but is {type(target)} in {target}"]
        if len(shape) == 0 or len(target) == 0:
            return []
        proto = shape[0]
        errs: list[str] = []
        for i, el in enumerate(target):
            errs += compare_dict(el, proto, path+[f"[{i}]"])
        return errs
    if isinstance(shape, type):
        if not isinstance(target, shape):
            return [f"type mismatch at {'.'.join(path) or '<root>'}: expected {shape}, got {type(target)} in {target}"]
        return []
    return []

def extract_json_from_response(
    llm_output: str, json_shape: dict
) -> tuple[dict | None, list[str] | None]:
    text = (llm_output or "")
    candidates = _strict_object_candidates(text)
    if not candidates:
        return None, None

    expected_top = _infer_top_type(json_shape)
    req_keys = {normalize_text(k) for k in json_shape.keys()} if isinstance(json_shape, dict) else set()

    best: tuple[dict | None, list[str] | None, tuple[int, int, int]] = (
        None, None, (10**9, 10**9, 10**9)
    )

    for raw in candidates:
        for variant_idx, js in enumerate((raw, _light_repair(raw))):
            try:
                parsed = json.loads(js)
            except json.JSONDecodeError:
                continue

            top_pen = 0
            if expected_top == "dict" and not isinstance(parsed, dict):
                top_pen = 1000
            elif expected_top == "list" and not isinstance(parsed, list):
                top_pen = 1000

            missing_required = 0
            if isinstance(parsed, dict) and req_keys:
                pkeys = {normalize_text(k) for k in parsed.keys()}
                missing_required = len(req_keys - pkeys)

            missing = compare_dict(parsed, json_shape) or []
            if top_pen == 0 and missing_required == 0 and len(missing) == 0:
                return parsed, [] # type: ignore

            score = (1000 * missing_required + len(missing) + top_pen, -len(js), variant_idx)
            if score < best[2]:
                best = (parsed, missing if missing else [f"missing_required={missing_required}"], score) # type: ignore

    return best[0], best[1]

def _tighten_template_for_structure(
    base: PromptTemplate,
    require_shape: dict,
    reason: str,
    missing_keys: list[str] | None = None,
) -> PromptTemplate:
    effective_shape = augment_shape_with_diag(require_shape)

    extra_gp = [
        "Output EXACTLY ONE JSON object (no prose, no code fences).",
        "Keys and value types MUST match <output_schema> exactly.",
        "Do NOT invent content; if unknown/forbidden, use empty string or empty array.",
        "Populate 'thinkings_json_error' with concise diagnostics.",
    ]
    extra_instructions = [
        "Produce the JSON directly without any preface or explanation.",
        "Ensure every required key from <output_schema> is present.",
        "For arrays, use [] if no safe content. For strings, use \"\" if content cannot be provided safely.",
    ]
    extra_validation = [
        "The final answer MUST parse as JSON.",
        "All required top-level keys exist; no extra top-level keys.",
        "All value types match exactly.",
        "Include 'thinkings_json_error' with root_cause and attempts incremented.",
    ]
    notes: list[dict[str, str | list[str]]] = [{"retry_reason": reason}]
    if missing_keys:
        extra_instructions.append(f"Add the missing keys exactly as listed: {missing_keys}.")
        extra_validation.append(f"Confirm the following keys now exist: {missing_keys}.")
        notes.append({"missing_keys": missing_keys})

    example_min: dict[str, Any] = {}
    for k, v in effective_shape.items():
        if isinstance(v, list): example_min[k] = []
        elif isinstance(v, dict): example_min[k] = {}
        elif v is str: example_min[k] = ""
        elif v is int: example_min[k] = 0
        elif v is float: example_min[k] = 0.0
        elif v is bool: example_min[k] = False
        else: example_min[k] = None
    if "thinkings_json_error" in effective_shape:
        example_min["thinkings_json_error"] = {
            "attempts": 1,
            "root_cause": reason,
            "parser_errors": [],
            "missing_keys": missing_keys or [],
            "extra_keys_after": 0,
            "smells": [],
            "selected_fix": "schema_enforced_retry",
            "applied_patches": [],
            "notes": ["auto example"],
        }

    return PromptTemplate(
        role=base.role,
        purpose=base.purpose,
        lang=base.lang,
        guiding_principles=[*base.guiding_principles, *extra_gp],
        instructions=[*base.instructions, *extra_instructions],
        validation=[*base.validation, *extra_validation],
        examples=[*base.examples, example_min, *notes],
        output_schema=effective_shape,
    )

def get_json_response(
    prompts: PromptInstance,
    client: Any,
    model: str,
    temperature: float,
    max_retries: int = 10,
) -> dict | str:
    parsed_json: dict | None = None
    missing_keys: list[str] | None = None
    full_response: str = ""
    original_prompts = deepcopy(prompts)
    base_template = prompts.template
    user_prompts = prompts.user_prompts
    for attempt in range(max_retries + 1):
        llm_response, _ = get_response_from_llm(
            prompt=prompts.user_prompt,
            client=client,
            model=model,
            system_message=prompts.system_prompt,
            temperature=temperature,
        )
        full_response = llm_response

        parsed_json, missing_keys = extract_json_from_response(full_response, prompts.schema)

        if parsed_json is not None and missing_keys is not None and len(missing_keys) == 0:
            if attempt > 0 and isinstance(parsed_json, dict) and "thinkings_json_error" not in parsed_json:
                parsed_json = dict(parsed_json)
                parsed_json["thinkings_json_error"] = {
                    "attempts": attempt + 1,
                    "root_cause": "not_provided",
                    "parser_errors": [],
                    "missing_keys": [],
                    "extra_keys_after": 0,
                    "smells": [],
                    "selected_fix": "not_provided",
                    "applied_patches": [],
                    "notes": ["auto-filled"],
                }
            return parsed_json

        logging.warning(f"(Attempt {attempt}/{max_retries}), missing_keys: {missing_keys}")
        logging.debug(full_response)

        if parsed_json is None and missing_keys is None:
            tightened = _tighten_template_for_structure(
                base=base_template,
                require_shape=original_prompts.schema,
                reason="NO_JSON_CANDIDATE_FOUND",
                missing_keys=None,
            )
            prompts = PromptInstance(tightened, user_prompts)
            continue

        if missing_keys:
            tightened = _tighten_template_for_structure(
                base=base_template,
                require_shape=original_prompts.schema,
                reason="MISSING_KEYS",
                missing_keys=missing_keys,
            )
            prompts = PromptInstance(tightened, user_prompts)
            continue

    logging.error("[ERROR] Failed to parse JSON even after schema-enforced retries.")
    return full_response
