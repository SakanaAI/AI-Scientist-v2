#!/usr/bin/env python3
"""
Validate idea JSON files against a lightweight schema.

This is intentionally strict about required keys, but flexible about types
for some fields (e.g., Experiments may be a list or a string).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


RE_NAME = re.compile(r"^[a-z0-9_-]+$")
REQUIRED_KEYS = [
    "Name",
    "Title",
    "Short Hypothesis",
    "Related Work",
    "Abstract",
    "Experiments",
    "Risk Factors and Limitations",
]


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON: {path}: {e}")


def _unwrap(obj):
    if isinstance(obj, dict) and "ideas" in obj and isinstance(obj["ideas"], list):
        return obj["ideas"]
    if isinstance(obj, dict) and "idea" in obj and isinstance(obj["idea"], dict):
        return [obj["idea"]]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def _validate_one(idea: dict, idx: int) -> list[str]:
    errs: list[str] = []
    if not isinstance(idea, dict):
        return [f"idea[{idx}]: not an object"]

    for k in REQUIRED_KEYS:
        if k not in idea:
            errs.append(f"idea[{idx}]: missing key: {k}")

    name = idea.get("Name")
    if not isinstance(name, str) or not name.strip():
        errs.append(f"idea[{idx}]: Name must be a non-empty string")
    else:
        if " " in name:
            errs.append(f"idea[{idx}]: Name must not contain spaces: {name!r}")
        if not RE_NAME.match(name):
            errs.append(
                f"idea[{idx}]: Name should match {RE_NAME.pattern}: {name!r}"
            )

    # Basic type checks (flexible where common)
    for k in ["Title", "Short Hypothesis", "Related Work", "Abstract"]:
        v = idea.get(k)
        if v is not None and not isinstance(v, str):
            errs.append(f"idea[{idx}]: {k} must be a string")

    exps = idea.get("Experiments")
    if exps is not None and not isinstance(exps, (list, str)):
        errs.append(f"idea[{idx}]: Experiments must be a list or a string")
    if isinstance(exps, list):
        for j, e in enumerate(exps):
            if not isinstance(e, (str, dict)):
                errs.append(f"idea[{idx}]: Experiments[{j}] must be str or dict")

    risks = idea.get("Risk Factors and Limitations")
    if risks is not None and not isinstance(risks, (list, str)):
        errs.append(
            f"idea[{idx}]: Risk Factors and Limitations must be a list or a string"
        )
    if isinstance(risks, list):
        for j, r in enumerate(risks):
            if not isinstance(r, str):
                errs.append(
                    f"idea[{idx}]: Risk Factors and Limitations[{j}] must be a string"
                )

    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate idea JSON schema.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file path.")
    args = ap.parse_args(argv)

    path = Path(args.in_path)
    obj = _load(path)
    ideas = _unwrap(obj)

    if not ideas:
        print("[ERROR] No ideas found. Expected list or object with idea fields.", file=sys.stderr)
        return 2

    all_errs: list[str] = []
    for i, idea in enumerate(ideas):
        all_errs.extend(_validate_one(idea, i))

    if all_errs:
        print("[FAIL] Schema validation failed:", file=sys.stderr)
        for e in all_errs:
            print(f"- {e}", file=sys.stderr)
        return 1

    print(f"[OK] {len(ideas)} idea(s) validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

