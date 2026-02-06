---
name: figure-caption-ref-audit
description: Extract figure screenshots, captions, and in-text figure references (figrefs) from a PDF into an audit bundle (images + figures.json). Use when checking whether figures match captions and main text, deciding what to move to appendix, or auditing figure redundancy and clarity.
---

# Figure / Caption / Figref Audit

## Overview
Create an auditable bundle for figure review: extracted figure-region images, caption text, and main-text figref snippets that mention each figure.

This is designed for *human/LLM-assisted auditing* (including vision models) and does not attempt to "understand" results beyond what is visible and referenced.

## Workflow
1. **Extract an audit bundle**
   - `UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with PyMuPDF -s scripts/extract_figures_and_refs.py --pdf paper.pdf --out-dir audit_out --max-pages 50 --dpi 150`
   - Output:
     - `audit_out/figures.json`
     - `audit_out/images/*.png`
2. **Audit with the checklist**
   - Use `references/audit-checklist.md` to review alignment and information density.
3. **Make edits to the paper**
   - Fix caption mismatches, missing figrefs, confusing axes/legends.
   - Decide what to move to appendix.

## References
- Output schema: `references/extraction-output-schema.md`
- Audit checklist: `references/audit-checklist.md`
- Safeguards: `references/safeguards.md`
