---
name: experiment-log-summarizer
description: Summarize an experiment run directory into a draft report (Markdown + JSON) by scanning logs, summary JSONs, figures, and artifacts without executing code. Use when you have a run folder (AI-Scientist, ML experiments, benchmarks) and need a grounded, source-linked summary for writeups or debugging.
---

# Experiment Log Summarizer

## Overview
Produce a grounded summary from a run directory by reading files only. The output is suitable as input for a paper writeup, lab notes, or an internal report.

This skill does not run experiments and does not call external services. It inventories artifacts and extracts key fields from common summary JSON formats.

## Workflow
1. Pick the run directory
   - Examples are described in references/run-layout.md.
2. Run the summarizer
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/summarize_dir.py --dir /path/to/run --out summary.md --json-out summary.json
   ~~~
3. Use the summary
   - Paste summary.md into your writeup workflow.
   - If you see missing artifacts, use the Missing section to locate or regenerate them.

## Output
- summary.md: human-readable report, with explicit source file paths.
- summary.json: structured extraction of key numbers, plots, and artifact inventory.

## Guardrails
Follow references/safeguards.md:
- Do not execute code.
- Do not infer missing results.
- Do not include secrets from logs.

## References
- Run layouts to expect: references/run-layout.md
- Suggested report structure: references/summary-template.md
- Safeguards: references/safeguards.md
- Summary schema: references/summary.schema.json
