#!/usr/bin/env python3
"""
Summarize a run directory into Markdown + JSON without executing code.

The script uses heuristics to discover common artifacts and extracts key fields
from summary JSONs when present.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MAX_FILES = 4000
DEFAULT_MAX_BYTES = 2_000_000  # 2 MB per file


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _read_text(path: Path, max_bytes: int) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if len(data) > max_bytes:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _read_json(path: Path, max_bytes: int) -> Any | None:
    raw = _read_text(path, max_bytes=max_bytes)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _walk_files(base: Path, max_files: int) -> list[Path]:
    paths: list[Path] = []
    count = 0
    for root, dirs, files in os.walk(base):
        # avoid descending into very large/irrelevant dirs
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv"}]
        for fn in files:
            paths.append(Path(root) / fn)
            count += 1
            if count >= max_files:
                return paths
    return paths


def _extract_summary_fields(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    for k in [
        "Experiment_description",
        "Significance",
        "Description",
        "current_findings",
        "stage",
        "best_metric",
        "total_nodes",
        "good_nodes",
        "buggy_nodes",
    ]:
        if k in obj:
            out[k] = obj[k]

    # AI-Scientist-style keys
    if "Key_numerical_results" in obj and isinstance(obj["Key_numerical_results"], list):
        out["Key_numerical_results"] = obj["Key_numerical_results"]
    if "List_of_included_plots" in obj and isinstance(obj["List_of_included_plots"], list):
        out["List_of_included_plots"] = obj["List_of_included_plots"]
    return out


def _render_md(report: dict[str, Any]) -> str:
    base_dir = report["base_dir"]
    scanned_at = report["scanned_at"]
    inv = report["inventory"]
    extracted = report["extracted"]

    lines: list[str] = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append(f"- Base directory: `{base_dir}`")
    lines.append(f"- Scanned at: `{scanned_at}`")
    lines.append("")

    lines.append("## Key Artifacts")
    lines.append("")
    for section in ["key_files", "summary_jsons", "pdfs", "figures"]:
        items = inv.get(section, [])
        lines.append(f"### {section}")
        if not items:
            lines.append("_None found._")
        else:
            for it in items:
                lines.append(f"- `{it}`")
        lines.append("")

    lines.append("## Extracted Summaries (Grounded)")
    lines.append("")
    summaries = extracted.get("summaries", [])
    if not summaries:
        lines.append("_No parseable summary JSONs found._")
    else:
        for s in summaries:
            lines.append(f"### `{s.get('path','')}`")
            for k in ["Experiment_description", "Significance", "Description", "current_findings"]:
                if k in s:
                    val = s[k]
                    if isinstance(val, str):
                        val = val.strip()
                    lines.append(f"**{k}:** {val}")
                    lines.append("")

            knr = s.get("Key_numerical_results")
            if isinstance(knr, list) and knr:
                lines.append("**Key numerical results:**")
                for r in knr[:30]:
                    if isinstance(r, dict):
                        res = r.get("result")
                        desc = r.get("description") or r.get("desc") or ""
                        lines.append(f"- {res}: {desc}")
                    else:
                        lines.append(f"- {r}")
                lines.append("")

            lop = s.get("List_of_included_plots")
            if isinstance(lop, list) and lop:
                lines.append("**Plots mentioned:**")
                for p in lop[:50]:
                    if isinstance(p, dict):
                        lines.append(f"- {p.get('path')}: {p.get('description','')}")
                    else:
                        lines.append(f"- {p}")
                lines.append("")

    lines.append("")
    lines.append("## Missing / Next Steps")
    lines.append("")
    missing = report.get("missing_recommended", [])
    if not missing:
        lines.append("_No obvious missing recommended artifacts detected._")
    else:
        for m in missing:
            lines.append(f"- {m}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This report is generated by file scanning only; it does not execute any code."
    )
    lines.append(
        "- If a file was too large or invalid JSON, it is skipped and recorded in the JSON output."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize a run directory (read-only).")
    ap.add_argument("--dir", required=True, help="Run directory to scan.")
    ap.add_argument("--out", required=True, help="Output Markdown path.")
    ap.add_argument("--json-out", required=True, help="Output JSON path.")
    ap.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"Max files to scan (default: {DEFAULT_MAX_FILES}).",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Max bytes to read per file (default: {DEFAULT_MAX_BYTES}).",
    )
    args = ap.parse_args(argv)

    base = Path(args.dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        print(f"[ERROR] Not a directory: {base}")
        return 2

    all_files = _walk_files(base, max_files=max(1, args.max_files))
    rels = [_safe_rel(p, base) for p in all_files]

    key_candidates = [
        "idea.md",
        "idea.json",
        "research_idea.md",
        "token_tracker.json",
        "review_text.txt",
        "review_img_cap_ref.json",
    ]
    key_files: list[str] = []
    for k in key_candidates:
        p = base / k
        if p.exists():
            key_files.append(_safe_rel(p, base))

    summary_jsons = sorted(
        [r for r in rels if r.endswith(".json") and ("summary" in Path(r).name.lower())]
    )
    pdfs = sorted([r for r in rels if r.lower().endswith(".pdf")])
    figures = sorted([r for r in rels if "/figures/" in ("/" + r).replace("\\", "/") and r.lower().endswith(".png")])

    extracted_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for rel in summary_jsons:
        p = base / rel
        obj = _read_json(p, max_bytes=args.max_bytes)
        if obj is None:
            skipped.append({"path": rel, "reason": "invalid_or_too_large"})
            continue
        fields = _extract_summary_fields(obj)
        fields["path"] = rel
        extracted_summaries.append(fields)

    missing_recommended: list[str] = []
    if not summary_jsons:
        missing_recommended.append("No *summary*.json files found (e.g., baseline_summary.json).")
    if not figures and not (base / "figures").exists():
        missing_recommended.append("No figures/ directory with PNGs found.")

    report: dict[str, Any] = {
        "base_dir": str(base),
        "scanned_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "max_files": args.max_files,
            "max_bytes": args.max_bytes,
        },
        "inventory": {
            "key_files": key_files,
            "summary_jsons": summary_jsons,
            "pdfs": pdfs,
            "figures": figures,
        },
        "extracted": {
            "summaries": extracted_summaries,
            "skipped": skipped,
        },
        "missing_recommended": missing_recommended,
    }

    out_md = _render_md(report)
    Path(args.out).write_text(out_md, encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {args.out}")
    print(f"[OK] Wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

