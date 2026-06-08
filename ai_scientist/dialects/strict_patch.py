"""strict-patch — a real, line-precise patch applier.

Ported from SkillOS's ``system/dialects/strict-patch.dialect.md`` (which specified
the format and the apply algorithm but left the applier as design). This is the
executable implementation, with verification and deterministic semantics.

Format (one block per file)::

    [FILE] path/to/file.py
    [DEL:N] <exact original content of line N>
    [ADD:N] <new content to place at line N>
    [EOF]

Operation semantics (all line numbers are 1-indexed into the *original* file):

  * ``[DEL:N] content`` — delete original line N. ``content`` (if given) must match
    the actual line N (used to verify the model is patching the file it thinks it
    is); a mismatch raises :class:`PatchError`.
  * ``[ADD:N] content`` — insert ``content`` so it lands *at* position N, pushing
    the original line N (and everything after) down. Multiple ``[ADD:N]`` lines
    insert in the order written.
  * ``[DEL:N]`` + ``[ADD:N]`` at the same N — replace line N.

Apply model (single deterministic pass, equivalent to the spec's
"DEL in reverse, ADD forward" but without index-shifting hazards): walk original
positions 1..L+1; at each position emit any ADDs anchored there (in order), then
emit the original line unless it was deleted. Position L+1 is the virtual
append slot, so ``[ADD:L+1]`` appends at end.

Why it matters: instead of regenerating a 500-line PyTorch script to change 3
lines (~1,500 output tokens, with hallucination risk), the model emits ~6 lines
(~25 tokens). The applier is exact, so correctness does not depend on the model
reproducing unchanged code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_FILE_RE = re.compile(r"^\[FILE\]\s+(.+?)\s*$")
_OP_RE = re.compile(r"^\[(DEL|ADD):(\d+)\]( ?)(.*)$")
_EOF_RE = re.compile(r"^\[EOF\]\s*$")


class PatchError(Exception):
    """Raised when a patch cannot be parsed or safely applied (e.g. a DEL whose
    content does not match the target line, or an out-of-range line number).
    Callers should treat this as 'fall back to a full rewrite'."""


@dataclass
class _Op:
    kind: str  # "DEL" | "ADD"
    line: int
    content: str


@dataclass
class FilePatch:
    path: str
    ops: list = field(default_factory=list)


def parse_patch(text: str) -> list:
    """Parse strict-patch text into a list of :class:`FilePatch`. Tolerant of
    blank lines and surrounding prose/code fences around the blocks."""
    patches: list = []
    current: Optional[FilePatch] = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            continue  # ignore markdown code fences
        m_file = _FILE_RE.match(line)
        if m_file:
            current = FilePatch(path=m_file.group(1).strip())
            patches.append(current)
            continue
        if _EOF_RE.match(line):
            current = None
            continue
        m_op = _OP_RE.match(line)
        if m_op:
            if current is None:
                raise PatchError(f"operation outside a [FILE] block: {line!r}")
            kind, num, _sp, content = m_op.groups()
            current.ops.append(_Op(kind=kind, line=int(num), content=content))
            continue
        # any other line outside a block is prose; inside a block it is illegal
        if current is not None and line.strip():
            raise PatchError(f"unexpected line inside [FILE] block: {line!r}")
    if not patches:
        raise PatchError("no [FILE] blocks found in patch text")
    return patches


def _apply_ops_to_lines(lines: list, ops: list, verify: bool = True) -> list:
    """Apply ops to a list of line strings (without trailing newlines)."""
    n = len(lines)
    dels: set = set()
    adds: dict = {}  # position -> list of content (in order)

    for op in ops:
        if op.line < 1:
            raise PatchError(f"line number must be >= 1, got {op.line}")
        if op.kind == "DEL":
            if op.line > n:
                raise PatchError(f"[DEL:{op.line}] out of range (file has {n} lines)")
            if verify and op.content != "":
                actual = lines[op.line - 1]
                if actual != op.content and actual.rstrip() != op.content.rstrip():
                    raise PatchError(
                        f"[DEL:{op.line}] content mismatch.\n  expected: {op.content!r}\n  actual:   {actual!r}"
                    )
            dels.add(op.line)
        else:  # ADD
            if op.line > n + 1:
                raise PatchError(
                    f"[ADD:{op.line}] out of range (can append at most at line {n + 1})"
                )
            adds.setdefault(op.line, []).append(op.content)

    out: list = []
    for pos in range(1, n + 2):  # +1 virtual append slot
        if pos in adds:
            out.extend(adds[pos])
        if pos <= n and pos not in dels:
            out.append(lines[pos - 1])
    return out


def apply_patch_to_text(text: str, patch: str, verify: bool = True) -> str:
    """Apply a single-file patch block to ``text`` and return the new text.
    Preserves a trailing newline if the original had one."""
    patches = parse_patch(patch)
    if len(patches) != 1:
        raise PatchError(f"expected exactly one [FILE] block, got {len(patches)}")
    had_trailing_nl = text.endswith("\n")
    lines = text.split("\n")
    if had_trailing_nl:
        lines = lines[:-1]  # drop the empty element after the final newline
    new_lines = _apply_ops_to_lines(lines, patches[0].ops, verify=verify)
    result = "\n".join(new_lines)
    if had_trailing_nl or new_lines:
        result += "\n"
    return result


def apply_patch(patch: str, root_dir=".", verify: bool = True,
                dry_run: bool = False) -> dict:
    """Apply a (possibly multi-file) patch on disk under ``root_dir``.
    Returns ``{relative_path: new_content}``. With ``dry_run`` nothing is written."""
    root = Path(root_dir)
    patches = parse_patch(patch)
    results: dict = {}
    for fp in patches:
        target = root / fp.path
        if not target.exists():
            raise PatchError(f"target file does not exist: {target}")
        original = target.read_text()
        had_trailing_nl = original.endswith("\n")
        lines = original.split("\n")
        if had_trailing_nl:
            lines = lines[:-1]
        new_lines = _apply_ops_to_lines(lines, fp.ops, verify=verify)
        new_text = "\n".join(new_lines) + ("\n" if (had_trailing_nl or new_lines) else "")
        results[fp.path] = new_text
        if not dry_run:
            target.write_text(new_text)
    return results


def render_patch_prompt_spec() -> str:
    """The instruction block to inject into an LLM prompt so it emits a patch
    instead of a full rewrite. Kept here so the format and its documentation
    cannot drift apart."""
    return (
        "Output your edit as a STRICT-PATCH, not a full file. Emit ONLY the lines that "
        "change, using exact 1-indexed line numbers from the current file:\n"
        "  [FILE] <path>\n"
        "  [DEL:N] <exact current content of line N>      # to delete/replace line N\n"
        "  [ADD:N] <new content to place at line N>        # to insert/replace at line N\n"
        "  [EOF]\n"
        "Rules: a replace is [DEL:N] then [ADD:N] at the same N. Multiple [ADD:N] insert "
        "in order. Preserve indentation EXACTLY. Do not emit unchanged lines. Line numbers "
        "refer to the file as it is NOW."
    )
