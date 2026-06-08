"""Markdown Agent Dialects ported into AI-Scientist-v2 (Fase 2 of the upgrade plan).

A *dialect* is a compressed I/O format that lets the LLM emit only what changes
instead of re-deriving everything. The first dialect here is ``strict_patch`` — a
line-precise patch notation (ported from SkillOS's ``strict-patch.dialect.md``)
that replaces full-script rewrites in the improve/debug loop, cutting output
tokens by an order of magnitude on large experiment files.
"""
from .strict_patch import (
    PatchError,
    FilePatch,
    parse_patch,
    apply_patch_to_text,
    apply_patch,
    render_patch_prompt_spec,
)

__all__ = [
    "PatchError",
    "FilePatch",
    "parse_patch",
    "apply_patch_to_text",
    "apply_patch",
    "render_patch_prompt_spec",
]
