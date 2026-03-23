#!/usr/bin/env python3
"""One-off: replace ast.Call print(...) with logger.info(...), add module logger. Not for rich.print."""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)


def _node_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    if not hasattr(node, "end_lineno") or node.end_lineno is None:
        return None
    lines = source.splitlines(keepends=True)
    start = sum(len(lines[i]) for i in range(node.lineno - 1)) + node.col_offset
    end = sum(len(lines[i]) for i in range(node.end_lineno - 1)) + node.end_col_offset
    return start, end


def _collect_print_calls(tree: ast.AST) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        out.append(node)
    return out


def _replace_print_segments(source: str, tree: ast.AST) -> str:
    calls = _collect_print_calls(tree)
    if not calls:
        return source

    def sort_key(n: ast.Call) -> tuple[int, int, int]:
        return (n.end_lineno or 0, n.end_col_offset or 0, n.lineno)

    calls.sort(key=sort_key, reverse=True)
    out = source
    for call in calls:
        span = _node_span(out, call)
        if span is None:
            continue
        start, end = span
        segment = out[start:end]
        if not re.match(r"^\s*print\s*\(", segment):
            # e.g. weird formatting; try first line only
            new_seg = re.sub(r"^print\s*\(", "logger.info(", segment, count=1)
        else:
            new_seg = re.sub(r"^print\s*\(", "logger.info(", segment, count=1)
        if new_seg == segment:
            _log.warning("no replace in: %r", segment[:80])
            continue
        out = out[:start] + new_seg + out[end:]
    return out


def _insert_logger(source: str, tree: ast.Module) -> str:
    if re.search(r"^\s*logger\s*=\s*logging\.getLogger\(__name__\)", source, re.MULTILINE):
        return source

    body = tree.body
    insert_after = 0
    i = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            insert_after = body[0].end_lineno or body[0].lineno
            i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        insert_after = max(insert_after, body[i].end_lineno or body[i].lineno)
        i += 1
    while i < len(body) and isinstance(body[i], (ast.Import, ast.ImportFrom)):
        insert_after = max(insert_after, body[i].end_lineno or body[i].lineno)
        i += 1

    lines = source.splitlines(keepends=True)
    block_parts: list[str] = []
    if not re.search(r"^import logging\s*$|^import logging\s*#", source, re.M) and not re.search(
        r"^from logging\s+", source, re.M
    ):
        block_parts.append("import logging\n")
    block_parts.append("logger = logging.getLogger(__name__)\n\n")
    block = "".join(block_parts)

    if insert_after == 0:
        return block + source
    # insert after line insert_after (1-based end line of last header stmt)
    idx = insert_after  # slice index: keep lines[:idx] = first insert_after lines
    return "".join(lines[:idx]) + block + "".join(lines[idx:])


def migrate_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        _log.warning("SKIP (syntax): %s: %s", path, e)
        return False
    if not isinstance(tree, ast.Module):
        return False
    calls = _collect_print_calls(tree)
    if not calls:
        return False
    new_source = _replace_print_segments(source, tree)
    # re-parse for insert position based on original tree structure is OK
    new_source = _insert_logger(new_source, tree)
    path.write_text(new_source, encoding="utf-8")
    _log.info("OK %s (%s prints)", path, len(calls))
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "launch_scientist_bfts.py",
        root / "ai_scientist" / "vlm.py",
        root / "ai_scientist" / "llm.py",
        root / "ai_scientist" / "tools" / "semantic_scholar.py",
        root / "ai_scientist" / "perform_writeup.py",
        root / "ai_scientist" / "perform_vlm_review.py",
        root / "ai_scientist" / "perform_plotting.py",
        root / "ai_scientist" / "perform_llm_review.py",
        root / "ai_scientist" / "perform_icbinb_writeup.py",
        root / "ai_scientist" / "treesearch" / "utils" / "tree_export.py",
        root / "ai_scientist" / "treesearch" / "utils" / "metric.py",
        root / "ai_scientist" / "treesearch" / "utils" / "config.py",
        root / "ai_scientist" / "treesearch" / "perform_experiments_bfts_with_agentmanager.py",
        root / "ai_scientist" / "treesearch" / "parallel_agent.py",
        root / "ai_scientist" / "treesearch" / "log_summarization.py",
        root / "ai_scientist" / "treesearch" / "journal.py",
        root / "ai_scientist" / "treesearch" / "backend" / "backend_openai.py",
        root / "ai_scientist" / "treesearch" / "backend" / "backend_anthropic.py",
        root / "ai_scientist" / "treesearch" / "agent_manager.py",
        root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_betterrealworld.py",
        root / "ai_scientist" / "ideas" / "i_cant_believe_its_not_better.py",
    ]
    for f in files:
        if not f.is_file():
            _log.warning("MISSING %s", f)
            continue
        migrate_file(f)


if __name__ == "__main__":
    main()
