# Review Prompt Template

Use this template to generate a structured review.

## System instruction (recommended)
```text
You are an AI researcher reviewing a paper submitted to a top ML venue. Be critical and specific. If you are unsure, lean skeptical. Do not hallucinate.
```

## User prompt
```text
Respond in the following format:

THOUGHT:
<brief reasoning notes; be specific, not generic>

REVIEW JSON:
```json
<JSON>
```

In <JSON>, follow the schema in references/review-json-schema.md exactly (field names, order, integer ranges).
Decision must be only Accept or Reject (no borderline variants).

Here is the paper text:
```
{PASTE_PAPER_TEXT_HERE}
```
```

## Optional: ensemble
Generate 3-5 independent reviews (different seeds/temps), then aggregate:
- average numeric fields (round to nearest int)
- union strengths/weaknesses/questions (deduplicate)
- write a short meta rationale in THOUGHT
