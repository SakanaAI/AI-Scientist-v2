"""
Pre-compile checks (chktex + structural heuristics) and incremental LaTeX repair
using the LLM when pdflatex fails — without regenerating the paper from scratch.
"""

from __future__ import annotations

import logging
import os.path as osp
import shutil
import subprocess

from ai_scientist.extract_latex_block import extract_latex_fenced_block
from ai_scientist.latex_compile import _tail_text_file, compile_latex
from ai_scientist.llm import get_writeup_response_with_length_continuations

logger = logging.getLogger(__name__)

LATEX_REPAIR_SYSTEM_MESSAGE = """You are an expert LaTeX editor. You receive a full LaTeX manuscript (template.tex) and diagnostics (chktex, structure checks, or pdflatex log).
Your job is to output ONE corrected COMPLETE .tex file that fixes syntax and compilation issues while preserving scientific content, citations, figure filenames, and the overall structure.
Make minimal edits. Do not invent new results or citations. Keep \\begin{filecontents}{references.bib} ... \\end{filecontents} and \\graphicspath unless a fix strictly requires a small change there.
Always return the full file inside a ```latex fenced code block."""

REPAIR_PROMPT_TEMPLATE = """Fix the LaTeX file `template.tex` for the following issue.

{instruction}

Diagnostics (may include chktex output and/or pdflatex log excerpts):
{diagnostics}

---BEGIN FULL CURRENT SOURCE---
{tex}
---END FULL CURRENT SOURCE---

Return the complete fixed source in a single ```latex ... ``` block. Do not truncate."""


def structural_latex_diagnostics(tex: str) -> str:
    """Lightweight checks that catch many LLM truncation / imbalance issues."""
    notes: list[str] = []
    if "\\documentclass" not in tex:
        notes.append("Missing \\documentclass.")
    if "\\begin{document}" not in tex:
        notes.append("Missing \\begin{document}.")
    if "\\end{document}" not in tex:
        notes.append("Missing \\end{document} (file may be truncated).")
    n_begin = tex.count("\\begin{")
    n_end = tex.count("\\end{")
    if n_begin != n_end:
        notes.append(
            f"Unbalanced \\begin/\\end counts: {n_begin} \\\\begin{{...}} vs {n_end} \\\\end{{...}}."
        )
    return "\n".join(notes) if notes else ""


def run_chktex_safe(tex_path: str) -> str:
    if not shutil.which("chktex"):
        return ""
    try:
        r = subprocess.run(
            ["chktex", "-q", "-n2", "-n24", "-n13", "-n1", tex_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.info("chktex not usable: %s", e)
        return ""


def _repair_tex_with_llm(
    client,
    model: str,
    tex: str,
    diagnostics: str,
    instruction: str,
) -> str | None:
    """Ask the model to return a full fixed file; return None if extraction fails."""
    diag = (diagnostics or "").strip()
    if len(diag) > 14000:
        diag = diag[:14000] + "\n… (truncated)"
    if len(tex) > 240000:
        logger.warning(
            "LaTeX repair: template.tex is very large (%s chars); truncating for prompt",
            len(tex),
        )
        tex = tex[:240000] + "\n% [TRUNCATED FOR REPAIR PROMPT]\n"
    prompt = REPAIR_PROMPT_TEMPLATE.format(
        instruction=instruction,
        diagnostics=diag or "(no diagnostic text)",
        tex=tex,
    )
    response, _ = get_writeup_response_with_length_continuations(
        prompt=prompt,
        client=client,
        model=model,
        system_message=LATEX_REPAIR_SYSTEM_MESSAGE,
        print_debug=False,
        temperature=0.3,
    )
    fixed = extract_latex_fenced_block(response)
    if not fixed:
        logger.warning(
            "LaTeX repair: could not extract ```latex from model response (preview %s chars)",
            min(400, len(response or "")),
        )
        return None
    return fixed


def compile_latex_with_incremental_repair(
    latex_folder: str,
    pdf_file: str,
    writeup_file: str,
    client,
    model: str,
    *,
    precheck_rounds: int = 2,
    compile_repair_rounds: int = 5,
    compile_timeout: int = 120,
) -> bool:
    """
    1) Optional pre-compile: structural diagnostics + chktex → LLM fix (up to ``precheck_rounds``).
    2) ``compile_latex``; on failure, feed ``template.log`` tail to LLM and rewrite ``writeup_file`` (up to ``compile_repair_rounds``).
    Does not restart the paper from scratch — only repairs the existing file.
    """
    def read_tex() -> str:
        with open(writeup_file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_tex(s: str) -> None:
        with open(writeup_file, "w", encoding="utf-8") as f:
            f.write(s)

    # --- Pre-compile pass ---
    for pre_i in range(precheck_rounds):
        tex = read_tex()
        parts: list[str] = []
        sdiag = structural_latex_diagnostics(tex)
        if sdiag:
            parts.append("Structural:\n" + sdiag)
        chk = run_chktex_safe(writeup_file)
        if chk:
            parts.append("chktex:\n" + chk)
        diagnostics = "\n\n".join(parts).strip()
        if not diagnostics:
            logger.info("LaTeX precheck: no structural/chktex issues reported")
            break
        logger.info("LaTeX precheck round %s: running LLM repair", pre_i + 1)
        fixed = _repair_tex_with_llm(
            client,
            model,
            tex,
            diagnostics,
            instruction="Fix syntax, environment balance, and chktex issues. Ensure \\end{document} is present.",
        )
        if fixed is None:
            logger.warning("LaTeX precheck: repair failed to produce fenced LaTeX; continuing to compile anyway")
            break
        if fixed.strip() == tex.strip():
            logger.info("LaTeX precheck: model returned unchanged text; stopping precheck")
            break
        write_tex(fixed)

    # --- Compile + log-based repair ---
    log_path = osp.join(latex_folder, "template.log")
    for comp_i in range(compile_repair_rounds):
        ok = compile_latex(latex_folder, pdf_file, timeout=compile_timeout)
        if ok:
            if comp_i > 0:
                logger.info("LaTeX compile succeeded after %s repair round(s)", comp_i)
            return True
        tex = read_tex()
        log_tail = _tail_text_file(log_path) if osp.exists(log_path) else "(no template.log)"
        if len(log_tail) > 16000:
            log_tail = log_tail[-16000:]
        diagnostics = "pdflatex/bibtex pipeline failed. Log tail:\n" + log_tail
        logger.info("LaTeX compile failed; repair round %s / %s", comp_i + 1, compile_repair_rounds)
        fixed = _repair_tex_with_llm(
            client,
            model,
            tex,
            diagnostics,
            instruction="Fix errors shown in the log so pdflatex can produce template.pdf. Preserve content; minimal edits.",
        )
        if fixed is None:
            logger.warning("LaTeX compile repair: no fenced LaTeX from model; aborting repair loop")
            break
        if fixed.strip() == tex.strip():
            logger.warning("LaTeX compile repair: model made no changes; stopping to avoid a loop")
            break
        write_tex(fixed)

    # One more compile after the last repair write (the loop exits without compiling that version).
    if compile_latex(latex_folder, pdf_file, timeout=compile_timeout):
        return True
    return False
