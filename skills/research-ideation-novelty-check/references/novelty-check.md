# Novelty Sanity-Check (Semantic Scholar)

This is a *sanity-check*, not a proof of novelty.

## Goals
- Find the closest prior work quickly.
- Avoid over-claiming novelty.
- Record what you searched (queries + date/time) so the process is auditable.

## Workflow
1. **Decompose the idea**
   - Write 3-6 key phrases: task + method + twist + evaluation setting.
2. **Write several queries**
   - One broad: `"<task>" + "<method family>"`
   - One specific: `"<core mechanism>" + "<task>"`
   - One baseline-centric: `"<closest baseline name>" + "<setting>"`
3. **Search and record**
   - For each query, record:
     - the exact query string
     - the time (local) you ran it
     - the top 5-10 results you considered most similar
4. **Interpret cautiously**
   - If you find a close match:
     - adjust the idea, or
     - narrow the claim, or
     - position as a replication / stress-test / negative result.
   - If you do *not* find a match:
     - write: "We did not find direct matches for [queries] as of [date]; we position this as ..."
     - do not write: "This is the first work to ..."

## Writing novelty claims safely
Prefer:
- "To our knowledge, based on searches for [queries] on [date], we did not find ..."
- "We study [X] in the under-explored setting of [Y] ..."
Avoid:
- "We are the first to ..."
- "No prior work has studied ..."

## Common failure modes
- Query is too narrow (misses synonyms).
- Query is too broad (returns irrelevant classics).
- You conflate "not found" with "non-existent".
