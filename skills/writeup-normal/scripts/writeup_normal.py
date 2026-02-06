#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_adapter import chat


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON: {path}: {e}")


def _first_idea(obj):
    if isinstance(obj, dict) and "ideas" in obj and isinstance(obj["ideas"], list):
        return obj["ideas"][0] if obj["ideas"] else None
    if isinstance(obj, dict) and "idea" in obj and isinstance(obj["idea"], dict):
        return obj["idea"]
    if isinstance(obj, list):
        return obj[0] if obj else None
    if isinstance(obj, dict):
        return obj
    return None


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_figures(manifest: dict) -> str:
    figures = manifest.get("figures", []) if isinstance(manifest, dict) else []
    blocks = []
    for fig in figures:
        filename = fig.get("filename")
        caption = fig.get("caption", "")
        if not filename:
            continue
        blocks.append(
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.9\\linewidth]{{figures/{filename}}}\n"
            f"\\caption{{{caption}}}\n"
            "\\end{figure}\n"
        )
    return "\n".join(blocks) if blocks else ""


def _safe_section(text: str, fallback: str) -> str:
    return text.strip() if text and text.strip() else fallback


def _format_experiments(value) -> str:
    if isinstance(value, list):
        return "\n".join([f"- {v}" for v in value])
    if isinstance(value, str):
        return value
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate normal-length LaTeX writeup.")
    ap.add_argument("--idea-json", help="Path to idea JSON.")
    ap.add_argument("--summary-json", help="Path to summary JSON.")
    ap.add_argument("--summary-md", help="Path to summary markdown.")
    ap.add_argument("--plots-manifest", help="Path to plot manifest JSON.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--title", help="Override title.")
    ap.add_argument("--authors", default="Anonymous", help="Author list.")
    ap.add_argument("--online", action="store_true", help="Enable LLM generation.")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tex = out_dir / "paper.tex"
    if out_tex.exists() and not args.overwrite:
        print(f"[ERROR] Output exists: {out_tex} (use --overwrite)")
        return 2

    idea = None
    if args.idea_json:
        idea = _first_idea(_load_json(Path(args.idea_json)))

    summary = None
    if args.summary_json:
        summary = _load_json(Path(args.summary_json))

    summary_md = ""
    if args.summary_md:
        summary_md = Path(args.summary_md).read_text(encoding="utf-8")

    manifest = None
    if args.plots_manifest:
        manifest = _load_json(Path(args.plots_manifest))

    title = args.title or (idea.get("Title") if isinstance(idea, dict) else None) or "Untitled"
    abstract = ""
    related = ""
    experiments = ""
    if isinstance(idea, dict):
        abstract = idea.get("Abstract", "") or ""
        related = idea.get("Related Work", "") or ""
        experiments = _format_experiments(idea.get("Experiments"))

    intro = ""
    methods = ""
    results = ""
    discussion = ""

    if summary:
        intro = summary.get("Description", "")
        results = summary.get("Experiment_description", "")
        discussion = summary.get("Significance", "")

    if not intro and summary_md:
        intro = summary_md

    if args.online:
        os.environ["ASV2_ONLINE"] = "1"
        prompt = (
            "Use the provided idea and summary to draft sections for an 8-page paper. "
            "Return JSON with keys: abstract, introduction, related, methods, experiments, results, discussion.\n\n"
            f"IDEA: {json.dumps(idea, ensure_ascii=False)}\n\nSUMMARY: {json.dumps(summary, ensure_ascii=False)}"
        )
        system = "You are a concise scientific writer."
        response = chat(prompt=prompt, system=system, model=args.model)
        try:
            data = json.loads(response)
            abstract = data.get("abstract", abstract)
            intro = data.get("introduction", intro)
            related = data.get("related", related)
            methods = data.get("methods", methods)
            experiments = data.get("experiments", experiments)
            results = data.get("results", results)
            discussion = data.get("discussion", discussion)
        except Exception:
            pass

    template_path = Path(__file__).parent.parent / "references" / "normal_template.tex"
    template = _load_template(template_path)

    figures_block = _render_figures(manifest) if manifest else ""

    filled = template
    filled = filled.replace("{{{TITLE}}}", title)
    filled = filled.replace("{{{AUTHORS}}}", args.authors)
    filled = filled.replace("{ABSTRACT}", _safe_section(abstract, "TBD."))
    filled = filled.replace("{INTRO}", _safe_section(intro, "TBD."))
    filled = filled.replace("{RELATED}", _safe_section(related, "TBD."))
    filled = filled.replace("{METHODS}", _safe_section(methods, "TBD."))
    filled = filled.replace("{EXPERIMENTS}", _safe_section(experiments, "TBD."))
    filled = filled.replace("{RESULTS}", _safe_section(results, "TBD."))
    filled = filled.replace("{DISCUSSION}", _safe_section(discussion, "TBD."))
    filled = filled.replace("{FIGURES}", figures_block)

    out_tex.write_text(filled, encoding="utf-8")
    print(f"[OK] Wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
