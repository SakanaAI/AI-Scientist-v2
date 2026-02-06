#!/usr/bin/env python3
"""
Convert idea JSON to a structured Markdown file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON: {path}: {e}")


def _normalize_ideas(obj: Any) -> list[dict]:
    if isinstance(obj, dict) and "ideas" in obj and isinstance(obj["ideas"], list):
        return obj["ideas"]
    if isinstance(obj, dict) and "idea" in obj and isinstance(obj["idea"], dict):
        return [obj["idea"]]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def _render_one(idea: dict, code_text: str | None) -> str:
    lines: list[str] = []
    for key, value in idea.items():
        header = str(key).replace("_", " ").title()
        lines.append(f"## {header}\n")
        if isinstance(value, (list, tuple)):
            for item in value:
                lines.append(f"- {item}")
            lines.append("")
        elif isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"### {k}")
                lines.append(f"{v}\n")
        else:
            lines.append(f"{value}\n")
    if code_text:
        lines.append("## Code To Potentially Use\n")
        lines.append("Use the following code as context for your experiments:\n")
        lines.append("```python")
        lines.append(code_text.rstrip())
        lines.append("```\n")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert idea JSON to Markdown.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input idea JSON path.")
    ap.add_argument("--out", dest="out_path", required=True, help="Output markdown path.")
    ap.add_argument("--code-path", help="Optional code file to embed.")
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()

    obj = _load_json(in_path)
    ideas = _normalize_ideas(obj)
    if not ideas:
        raise SystemExit("[ERROR] No ideas found in input JSON.")

    code_text = None
    if args.code_path:
        code_path = Path(args.code_path).expanduser().resolve()
        if not code_path.exists():
            raise SystemExit(f"[ERROR] Code file not found: {code_path}")
        code_text = code_path.read_text(encoding="utf-8")

    # If multiple ideas, write all separated by ---
    blocks = []
    for idea in ideas:
        if not isinstance(idea, dict):
            raise SystemExit("[ERROR] Each idea must be a JSON object.")
        blocks.append(_render_one(idea, code_text))
    out_path.write_text("\n---\n\n".join(blocks), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
