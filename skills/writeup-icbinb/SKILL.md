---
name: writeup-icbinb
description: Draft a 4-page ICBINB-style LaTeX paper from idea JSON, summary artifacts, and plot manifest. Offline by default.
---

# ICBINB Writeup

## Overview
Generate a compact LaTeX paper skeleton for the ICBINB format using idea and summary inputs, plus optional plot manifests.

## Workflow
1. Prepare inputs
   - idea JSON (references/idea.schema.json)
   - summary JSON (references/summary.schema.json)
   - plot manifest JSON (references/plot_manifest.schema.json)
2. Generate LaTeX
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with openai -s scripts/writeup_icbinb.py \
     --idea-json idea.json --summary-json summary.json --plots-manifest plot_manifest.json \
     --out-dir writeup_icbinb
   ~~~
3. Optional LLM drafting (requires --online)
   ~~~bash
   uv run -s scripts/writeup_icbinb.py --idea-json idea.json --summary-json summary.json \
     --out-dir writeup_icbinb --online --model gpt-4o-mini
   ~~~

## Inputs
- --idea-json: path to idea JSON (optional)
- --summary-json: path to summary JSON (optional)
- --summary-md: path to summary markdown (optional)
- --plots-manifest: plot manifest JSON (optional)
- --out-dir: output directory (required)
- --online: enable network calls to LLMs (default offline)

## Outputs
- writeup_icbinb/paper.tex

## Safeguards
- Offline by default; --online required for network calls.
- Writes only within --out-dir.
- No file deletion unless --overwrite is set.

## References
- Template: references/icbinb_template.tex
- Idea schema: references/idea.schema.json
- Summary schema: references/summary.schema.json
- Plot manifest schema: references/plot_manifest.schema.json
