---
name: token-cost-tracker
description: Summarize token usage and cost records from JSON or JSONL logs.
---

# Token Cost Tracker

## Overview
Aggregate token usage and costs by model from recorded LLM call logs.

## Workflow
1. Collect JSON or JSONL records with prompt_tokens, completion_tokens, total_tokens, and optional cost.
2. Run the summarizer
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/token_cost_tracker.py \
     --in token_records.jsonl --out token_summary.json
   ~~~

## Inputs
- --in: JSON or JSONL records
- --out: output summary JSON

## Outputs
- token_summary.json

## Safeguards
- Read-only on inputs; writes only to --out.

## References
- Schema: references/token_record_schema.md
