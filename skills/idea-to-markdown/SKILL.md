---
name: idea-to-markdown
description: Convert idea JSON into a structured idea.md (optionally embedding a code snippet). Use when preparing idea inputs for BFTS experiments or writeup workflows.
---

# Idea -> Markdown

## Overview
Convert a single idea JSON (or list) into `idea.md` with section headers. Optionally include a code snippet block.

## Workflow
1. **Prepare idea JSON**
   - Must follow `references/idea.schema.json`.
2. **Generate idea.md**
   - `UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/idea_to_markdown.py --in idea.json --out idea.md`
3. **(Optional) Include code**
   - `... --code-path baseline.py`

## Inputs
- `--in`: idea JSON file
- `--out`: output markdown file
- `--code-path` (optional): path to a code file to embed

## Outputs
- `idea.md`

## Safeguards
- Read-only on inputs.
- Writes only to `--out`.

## References
- Idea schema: `references/idea.schema.json`
