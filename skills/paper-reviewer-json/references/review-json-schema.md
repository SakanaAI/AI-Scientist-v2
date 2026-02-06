# Review JSON Schema (NeurIPS-style)

Produce valid JSON with the following fields **in this order**:
- `Summary` (string)
- `Strengths` (list of strings)
- `Weaknesses` (list of strings)
- `Originality` (int 1-4)
- `Quality` (int 1-4)
- `Clarity` (int 1-4)
- `Significance` (int 1-4)
- `Questions` (list of strings)
- `Limitations` (list of strings)
- `Ethical Concerns` (boolean)
- `Soundness` (int 1-4)
- `Presentation` (int 1-4)
- `Contribution` (int 1-4)
- `Overall` (int 1-10)
- `Confidence` (int 1-5)
- `Decision` (string; must be `Accept` or `Reject`)

## Rating scales
- 1-4: low/poor to very high/excellent (use integers only)
- Overall 1-10: strong reject to award quality
- Confidence 1-5: low to absolute

## Guardrails
- Keep the review specific to the provided paper text.
- Do not invent results, claims, or citations not present in the paper.
- Avoid long verbatim quotes.
