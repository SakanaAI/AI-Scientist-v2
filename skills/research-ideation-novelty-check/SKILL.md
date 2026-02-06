---
name: research-ideation-novelty-check
description: Generate and refine research ideas and run a lightweight novelty sanity-check using Semantic Scholar. Use when drafting proposal-style ideas from a topic brief, validating idea JSON outputs, or checking whether an idea seems close to existing work (without over-claiming novelty).
---

# Research Ideation + Novelty Check

## Overview
Turn a short topic brief into structured research ideas, then run targeted Semantic Scholar searches to sanity-check novelty and record what you searched.

This skill is intentionally tool-agnostic and does not depend on this repo. Use the scripts for Semantic Scholar search and idea JSON schema validation.

## Workflow
1. Draft a topic brief
   - Use the template in references/prompt-templates.md (keep it concrete: task, constraints, evaluation, compute budget).
2. Produce idea JSON
   - Output one or more ideas that match references/idea.schema.json.
3. Validate the schema
   - Run scripts/idea_schema_validate.py on the generated JSON.
4. Novelty sanity-check
   - For each idea, run 3-6 focused searches via scripts/s2_search.py.
   - Record the exact query strings and the search date/time in your notes.
   - Do not claim novelty just because search results look sparse; follow references/novelty-check.md.

## Scripts
### Semantic Scholar search
- Offline (default-safe; no network):
  ~~~bash
  UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/s2_search.py --query "your query"
  ~~~
- Online (explicit opt-in; optional S2_API_KEY in env):
  ~~~bash
  UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/s2_search.py --online --query "attention is all you need" --limit 10
  ~~~

### Idea schema validation
- ~~~bash
  UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run -s scripts/idea_schema_validate.py --in ideas.json
  ~~~

## References
- Idea schema: references/idea.schema.json
- Novelty check guidance: references/novelty-check.md
- Prompt templates: references/prompt-templates.md
- Safeguards: references/safeguards.md
