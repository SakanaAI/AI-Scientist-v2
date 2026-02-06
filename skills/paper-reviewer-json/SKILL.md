---
name: paper-reviewer-json
description: Produce structured peer-review JSON (NeurIPS-style fields) for a paper given PDF/LaTeX/Markdown text.
---

# Paper Reviewer JSON

## Overview
Extract paper text and produce a structured review JSON for automated or human review workflows.

## Workflow
1. Extract text from PDF
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with pypdf -s scripts/extract_pdf_text.py --pdf paper.pdf --out paper.txt
   ~~~
2. Generate review JSON with your LLM of choice using references/review-prompt.md and references/review.schema.json

## Outputs
- paper.txt
- review.json

## Safeguards
- Do not fabricate citations or claims not present in the input.
- Keep quotes short and avoid reproducing large sections of the paper.

## References
- Review JSON schema: references/review.schema.json
- Review prompt: references/review-prompt.md
- Safeguards: references/safeguards.md
