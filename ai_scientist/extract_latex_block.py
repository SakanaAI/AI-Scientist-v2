"""
Extract LaTeX source from LLM responses that use markdown fenced blocks.

Models often omit the closing ```, use ```LaTeX, or put ```latex on the same
line as \\documentclass; the naive regex ```latex(.*?)``` then fails.
"""

from __future__ import annotations

import re
from typing import Optional

_LATEX_START = re.compile(r"```\s*latex\s*", re.IGNORECASE)


def _looks_like_latex(s: str) -> bool:
    t = s.lstrip()
    if len(t) < 12:
        return False
    return (
        "\\documentclass" in t
        or "\\begin{document}" in t
        or "\\begin{" in t[:800]
    )


def extract_latex_fenced_block(response: Optional[str]) -> Optional[str]:
    """
    Return LaTeX source from `` ```latex ... ``` `` (or unclosed `` ```latex ``),
    or from `` ```tex ... ``` `` when the body looks like a LaTeX document.

    Returns None if nothing usable is found.
    """
    if response is None:
        return None
    s = str(response)
    if not s.strip():
        return None

    flags = re.DOTALL | re.IGNORECASE

    # 1) Closed ```latex ... ``` (flexible whitespace; case-insensitive "latex")
    m = re.search(r"```\s*latex\s*(.*?)```", s, flags)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner

    # 2) Closed ```tex ... ``` (some models use this tag for LaTeX)
    m = re.search(r"```\s*tex\s*(.*?)```", s, flags)
    if m:
        inner = m.group(1).strip()
        if inner and _looks_like_latex(inner):
            return inner

    # 3) Opening ```latex but no closing fence (common model failure mode)
    m = _LATEX_START.search(s)
    if m:
        rest = s[m.end() :].strip()
        if rest.endswith("```"):
            rest = rest[:-3].rstrip()
        if rest and _looks_like_latex(rest):
            return rest

    return None
