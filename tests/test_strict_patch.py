"""Tests for the strict-patch applier (Fase 2a).

Run: PYTHONPATH=. python3.12 -m pytest tests/test_strict_patch.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_scientist.dialects.strict_patch import (  # noqa: E402
    PatchError,
    parse_patch,
    apply_patch_to_text,
    apply_patch,
)


def test_single_line_replace():
    text = "a = 1\nb = 2\nc = 3\n"
    patch = "[FILE] x.py\n[DEL:2] b = 2\n[ADD:2] b = 20\n[EOF]\n"
    assert apply_patch_to_text(text, patch) == "a = 1\nb = 20\nc = 3\n"


def test_pure_insert_pushes_down():
    text = "line1\nline2\n"
    patch = "[FILE] x\n[ADD:2] inserted\n[EOF]"
    assert apply_patch_to_text(text, patch) == "line1\ninserted\nline2\n"


def test_multi_line_insert_in_order():
    text = "head\ntail\n"
    patch = "[FILE] x\n[ADD:2] mid1\n[ADD:2] mid2\n[EOF]"
    assert apply_patch_to_text(text, patch) == "head\nmid1\nmid2\ntail\n"


def test_delete_line():
    text = "keep1\ndrop\nkeep2\n"
    patch = "[FILE] x\n[DEL:2] drop\n[EOF]"
    assert apply_patch_to_text(text, patch) == "keep1\nkeep2\n"


def test_multi_line_delete():
    text = "a\nb\nc\nd\n"
    patch = "[FILE] x\n[DEL:2] b\n[DEL:3] c\n[EOF]"
    assert apply_patch_to_text(text, patch) == "a\nd\n"


def test_replace_one_line_with_two():
    text = "x\nold\ny\n"
    patch = "[FILE] f\n[DEL:2] old\n[ADD:2] new1\n[ADD:2] new2\n[EOF]"
    assert apply_patch_to_text(text, patch) == "x\nnew1\nnew2\ny\n"


def test_append_at_end():
    text = "a\nb\n"
    patch = "[FILE] f\n[ADD:3] c\n[EOF]"
    assert apply_patch_to_text(text, patch) == "a\nb\nc\n"


def test_preserves_indentation_exactly():
    text = "def f():\n    return 1\n"
    patch = "[FILE] f\n[DEL:2]     return 1\n[ADD:2]     return 2\n[EOF]"
    assert apply_patch_to_text(text, patch) == "def f():\n    return 2\n"


def test_verify_mismatch_raises():
    text = "a = 1\nb = 2\n"
    patch = "[FILE] f\n[DEL:2] WRONG CONTENT\n[ADD:2] b = 3\n[EOF]"
    with pytest.raises(PatchError):
        apply_patch_to_text(text, patch)


def test_verify_can_be_disabled():
    text = "a = 1\nb = 2\n"
    patch = "[FILE] f\n[DEL:2] WRONG\n[ADD:2] b = 3\n[EOF]"
    assert apply_patch_to_text(text, patch, verify=False) == "a = 1\nb = 3\n"


def test_del_out_of_range_raises():
    with pytest.raises(PatchError):
        apply_patch_to_text("a\n", "[FILE] f\n[DEL:5] a\n[EOF]")


def test_add_out_of_range_raises():
    with pytest.raises(PatchError):
        apply_patch_to_text("a\n", "[FILE] f\n[ADD:99] x\n[EOF]")


def test_ignores_code_fences_and_prose():
    text = "a\nb\n"
    patch = "Here is the fix:\n```\n[FILE] f\n[DEL:1] a\n[ADD:1] A\n[EOF]\n```\n"
    assert apply_patch_to_text(text, patch) == "A\nb\n"


def test_no_blocks_raises():
    with pytest.raises(PatchError):
        parse_patch("just some prose, no patch here")


def test_realistic_500_line_file_3_line_fix():
    # a 'large' file; the patch touches only 3 lines (the token-savings scenario)
    lines = [f"x{i} = {i}" for i in range(1, 501)]
    text = "\n".join(lines) + "\n"
    patch = (
        "[FILE] big.py\n"
        "[DEL:1] x1 = 1\n[ADD:1] x1 = 1000\n"
        "[DEL:250] x250 = 250\n[ADD:250] x250 = 0\n"
        "[ADD:501] x_new = 'appended'\n"
        "[EOF]\n"
    )
    out = apply_patch_to_text(text, patch).split("\n")
    assert out[0] == "x1 = 1000"
    assert out[249] == "x250 = 0"
    assert out[500] == "x_new = 'appended'"
    assert len(out) == 502  # 501 + trailing empty from final newline
    # patch is far smaller than the file it edits
    assert len(patch) < len(text) / 10


def test_multi_file_apply_on_disk(tmp_path):
    (tmp_path / "a.py").write_text("aa\nbb\n")
    (tmp_path / "b.py").write_text("cc\ndd\n")
    patch = (
        "[FILE] a.py\n[DEL:1] aa\n[ADD:1] AA\n[EOF]\n"
        "[FILE] b.py\n[ADD:3] ee\n[EOF]\n"
    )
    results = apply_patch(patch, root_dir=tmp_path)
    assert results["a.py"] == "AA\nbb\n"
    assert (tmp_path / "a.py").read_text() == "AA\nbb\n"
    assert (tmp_path / "b.py").read_text() == "cc\ndd\nee\n"


def test_dry_run_does_not_write(tmp_path):
    (tmp_path / "a.py").write_text("aa\n")
    apply_patch("[FILE] a.py\n[DEL:1] aa\n[ADD:1] AA\n[EOF]", root_dir=tmp_path, dry_run=True)
    assert (tmp_path / "a.py").read_text() == "aa\n"  # unchanged


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
