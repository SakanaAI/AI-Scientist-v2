"""Patch-mode improve/debug glue (Fase 2a wiring).

Bridges the strict-patch applier into AI-Scientist's improve/debug loop. Instead of
asking the LLM to regenerate the whole experiment script, we:

  1. show it the current script WITH line numbers, and
  2. ask it to emit a strict-patch (only changed lines), then
  3. apply that patch to the parent code deterministically — falling back to the
     model's full code (or the parent code) if the patch is absent or doesn't apply.

The applier is exact, so output-token cost drops ~an order of magnitude on large
files without trusting the model to reproduce unchanged lines. Default-off so the
stock pipeline is byte-for-byte unchanged until explicitly enabled.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .strict_patch import PatchError, parse_patch, apply_patch_to_text, render_patch_prompt_spec


def number_code(code: str) -> str:
    """Prefix each line with its 1-indexed number so the model can reference exact
    lines in [DEL:N]/[ADD:N]. Numbering is for display only; it is not part of the file."""
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1))


def patch_prompt_block(parent_code: str, file_label: str = "experiment.py") -> dict:
    """A prompt sub-dict (merge into the 'Instructions' of an improve/debug prompt)
    that puts the agent in patch mode."""
    return {
        "Editing mode (STRICT-PATCH)": [
            render_patch_prompt_spec(),
            f"Use the path `{file_label}` in the [FILE] header.",
            "The numbered current file is shown below. Line numbers are 1-indexed and "
            "must be used exactly as shown; the ` | ` separator and the numbers are NOT "
            "part of the file content.",
        ],
        "Current file (numbered)": f"```\n{number_code(parent_code)}\n```",
    }


def resolve_completion(
    parent_code: str,
    completion_text: str,
    fallback_code: Optional[str] = None,
    verify: bool = True,
    file_label: str = "experiment.py",
) -> Tuple[Optional[str], bool]:
    """Turn a patch-mode completion into final code.

    Returns ``(code, used_patch)``:
      * ``(patched_code, True)``  — a strict-patch was found and applied cleanly.
      * ``(fallback_code, False)`` — no patch / patch failed; caller should use its
        normal full-code extraction (passed here as ``fallback_code``) or the parent.
    """
    if not completion_text:
        return (fallback_code, False)
    try:
        blocks = parse_patch(completion_text)
    except PatchError:
        return (fallback_code, False)
    # find the block that targets our file (or just the first block)
    block = next((b for b in blocks if b.path.strip().endswith(file_label)), blocks[0])
    # reconstruct a single-file patch string and apply to parent_code in memory
    patch_text = _block_to_text(block)
    try:
        new_code = apply_patch_to_text(parent_code, patch_text, verify=verify)
        return (new_code, True)
    except PatchError:
        return (fallback_code, False)


def _block_to_text(block) -> str:
    lines = [f"[FILE] {block.path}"]
    for op in block.ops:
        if op.content:
            lines.append(f"[{op.kind}:{op.line}] {op.content}")
        else:
            lines.append(f"[{op.kind}:{op.line}]")
    lines.append("[EOF]")
    return "\n".join(lines)
