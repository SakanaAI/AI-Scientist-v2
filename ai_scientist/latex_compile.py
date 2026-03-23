"""Shared LaTeX build helpers for writeup pipelines."""

import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback

logger = logging.getLogger(__name__)


def _tail_text_file(path: str, max_bytes: int = 16000) -> str:
    """Return the last chunk of a text file (best-effort)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        return data
    except OSError as e:
        return f"(could not read {path}: {e})"


def _escape_unescaped_ampersands(text: str) -> str:
    """Escape bare '&' that break LaTeX/BibTeX in bibliography fields."""
    return re.sub(r"(?<!\\)&", r"\\&", text)


def _repair_misplaced_alignment_ampersands(cwd: str) -> bool:
    """
    Best-effort fix for 'Misplaced alignment tab character &' in bibliography.
    Repairs both standalone .bib files in cwd and filecontents bib blocks in template.tex.
    """
    changed = False

    # Fix external bib files used by bibtex.
    for name in os.listdir(cwd):
        if not name.lower().endswith(".bib"):
            continue
        path = osp.join(cwd, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                original = f.read()
            updated = _escape_unescaped_ampersands(original)
            if updated != original:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(updated)
                changed = True
                logger.info("Repaired unescaped '&' in bibliography file: %s", path)
        except OSError:
            continue

    # Fix embedded filecontents bibliography blocks in template.tex.
    template_path = osp.join(cwd, "template.tex")
    if osp.exists(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                tex = f.read()
            pattern = r"(\\begin\{filecontents\*?\}\{[^}]+\.bib\})(.*?)(\\end\{filecontents\*?\})"
            def _repl(m: re.Match) -> str:
                nonlocal changed
                body = m.group(2)
                fixed = _escape_unescaped_ampersands(body)
                if fixed != body:
                    changed = True
                return f"{m.group(1)}{fixed}{m.group(3)}"
            updated_tex = re.sub(pattern, _repl, tex, flags=re.DOTALL)
            if updated_tex != tex:
                with open(template_path, "w", encoding="utf-8") as f:
                    f.write(updated_tex)
                logger.info("Repaired unescaped '&' in template.tex filecontents bib block(s).")
        except OSError:
            pass

    return changed


def compile_latex(cwd: str, pdf_file: str, timeout: int = 30) -> bool:
    """
    Run pdflatex/bibtex/pdflatex/pdflatex on template.tex in ``cwd``.
    On success, moves ``cwd/template.pdf`` to ``pdf_file``.

    Returns True if ``pdf_file`` exists after the pipeline.
    Logs WARNING when tools are missing, commands fail, or no PDF is produced.
    """
    logger.info("GENERATING LATEX cwd=%s -> %s", cwd, pdf_file)

    if not shutil.which("pdflatex"):
        logger.warning(
            "LaTeX: 'pdflatex' not found in PATH. Install TeX Live or load a cluster module "
            "(e.g. `module load texlive`) so PDFs can be built. cwd=%s",
            cwd,
        )
        return False

    if not shutil.which("bibtex"):
        logger.warning(
            "LaTeX: 'bibtex' not found in PATH; bibliography step may fail. cwd=%s",
            cwd,
        )

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    saw_misplaced_alignment_ampersand = False
    for command in commands:
        cmd_s = " ".join(command)
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.warning(
                    "LaTeX command exited with code %s: %s",
                    result.returncode,
                    cmd_s,
                )
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                if stderr:
                    logger.warning("stderr (tail):\n%s", stderr[-6000:])
                if stdout and not stderr:
                    logger.warning("stdout (tail):\n%s", stdout[-6000:])
            combined_output = f"{result.stdout or ''}\n{result.stderr or ''}"
            if "Misplaced alignment tab character &" in combined_output:
                saw_misplaced_alignment_ampersand = True
        except subprocess.TimeoutExpired:
            logger.warning(
                "LaTeX command timed out after %ss: %s",
                timeout,
                cmd_s,
            )
            logger.warning("%s", traceback.format_exc())
        except OSError as e:
            logger.warning("LaTeX command could not run: %s — %s", cmd_s, e)
            return False

    if saw_misplaced_alignment_ampersand:
        logger.warning(
            "Detected LaTeX '&' alignment error in bibliography; applying automatic escape repair and rebuilding."
        )
        if _repair_misplaced_alignment_ampersands(cwd):
            for command in commands:
                cmd_s = " ".join(command)
                try:
                    result = subprocess.run(
                        command,
                        cwd=cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout,
                    )
                    if result.returncode != 0:
                        logger.warning(
                            "LaTeX command exited with code %s after ampersand repair: %s",
                            result.returncode,
                            cmd_s,
                        )
                except (subprocess.TimeoutExpired, OSError):
                    logger.warning("LaTeX command failed after ampersand repair: %s", cmd_s)
                    break

    template_pdf = osp.join(cwd, "template.pdf")
    if not osp.exists(template_pdf):
        log_path = osp.join(cwd, "template.log")
        logger.warning(
            "LaTeX: template.pdf was not produced in %s after the build sequence.",
            cwd,
        )
        if osp.exists(log_path):
            logger.warning("template.log (tail):\n%s", _tail_text_file(log_path))
        else:
            logger.warning(
                "No template.log in %s — pdflatex may not have run successfully.",
                cwd,
            )
        return False

    try:
        shutil.move(template_pdf, pdf_file)
    except OSError as e:
        logger.warning(
            "LaTeX: could not move %s to %s: %s",
            template_pdf,
            pdf_file,
            e,
        )
        return False

    logger.info("FINISHED GENERATING LATEX -> %s", pdf_file)
    return True
