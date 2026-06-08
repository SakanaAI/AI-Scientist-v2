"""Tests for the patch-mode improve glue (Fase 2a wiring).

Run: PYTHONPATH=. python3.12 -m pytest tests/test_patch_improve.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_scientist.dialects.patch_improve import (  # noqa: E402
    number_code,
    patch_prompt_block,
    resolve_completion,
)


PARENT = "import torch\nlr = 0.01\nmodel = MLP()\ntrain(model, lr)\n"


def test_number_code_is_1_indexed():
    numbered = number_code("a\nb\nc")
    assert "1 | a" in numbered and "2 | b" in numbered and "3 | c" in numbered


def test_prompt_block_contains_spec_and_numbered_code():
    block = patch_prompt_block(PARENT)
    assert "Current file (numbered)" in block
    assert "lr = 0.01" in block["Current file (numbered)"]
    assert "Editing mode (STRICT-PATCH)" in block


def test_resolve_applies_patch_from_completion():
    completion = (
        "I will lower the learning rate.\n\n"
        "[FILE] experiment.py\n"
        "[DEL:2] lr = 0.01\n"
        "[ADD:2] lr = 0.001\n"
        "[EOF]\n"
    )
    code, used = resolve_completion(PARENT, completion)
    assert used is True
    assert "lr = 0.001" in code
    assert "lr = 0.01\n" not in code


def test_resolve_falls_back_when_no_patch():
    completion = "Here is the full new code:\n```python\nprint('x')\n```"
    code, used = resolve_completion(PARENT, completion, fallback_code="print('x')\n")
    assert used is False
    assert code == "print('x')\n"


def test_resolve_falls_back_on_bad_patch():
    # line content mismatch -> verification fails -> fallback
    completion = "[FILE] experiment.py\n[DEL:2] WRONG = 9\n[ADD:2] lr = 0.001\n[EOF]"
    code, used = resolve_completion(PARENT, completion, fallback_code=PARENT)
    assert used is False
    assert code == PARENT


def test_resolve_picks_block_for_target_file():
    completion = (
        "[FILE] other.py\n[ADD:1] noise\n[EOF]\n"
        "[FILE] experiment.py\n[DEL:1] import torch\n[ADD:1] import torch.nn as nn\n[EOF]\n"
    )
    code, used = resolve_completion(PARENT, completion)
    assert used is True
    assert code.startswith("import torch.nn as nn")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
