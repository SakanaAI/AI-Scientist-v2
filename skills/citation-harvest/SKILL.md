---
name: citation-harvest
description: Query Semantic Scholar to collect citations and generate a deduplicated BibTeX file. Offline by default.
---

# Citation Harvest

## Overview
Collect citations from Semantic Scholar using query strings and output a JSON bundle plus a BibTeX file.

## Workflow
1. Prepare queries (one per line)
2. Run the harvester
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/citation_harvest.py \
     --online --in queries.txt --out-json citations.json --out-bib citations.bib
   ~~~

## Inputs
- --in: text file with one query per line (optional)
- --query: repeatable query strings
- --limit: results per query (default 5)
- --online: enable network calls (required)

## Outputs
- citations.json
- citations.bib

## Safeguards
- Offline by default; --online required.
- No uploads; only queries sent to Semantic Scholar.
- API key must be provided via S2_API_KEY env var if needed.

## References
- Safeguards: references/safeguards.md
