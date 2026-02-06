# Idea JSON Schema

This schema is designed for proposal-style research ideas that can be executed with realistic academic resources.

## Accepted file shapes
Your ideas file may be:
- A list of idea objects: `[ { ... }, { ... } ]`
- A single idea object: `{ ... }`
- A wrapper object: `{ "idea": { ... } }` or `{ "ideas": [ ... ] }`

## Required fields (per idea)
- `Name` (string)
  - Short identifier; lowercase; no spaces.
  - Recommended pattern: `^[a-z0-9_-]+$`
- `Title` (string)
  - Human-readable title.
- `Short Hypothesis` (string)
  - One concise hypothesis or research question.
- `Related Work` (string)
  - Brief discussion of closest prior work and why this is not a trivial extension.
- `Abstract` (string)
  - ~200-300 words; motivation, method, expected outcome.
- `Experiments` (list or string)
  - Prefer a list of experiments (each as string or dict).
  - If dict, recommended keys: `Goal`, `Method`, `Metrics`, `Baselines`, `Datasets`.
- `Risk Factors and Limitations` (list or string)
  - Prefer a list of short bullets.

## Optional fields
- `Code` (string): starter code or constraints.
- `Datasets` (list): suggested datasets.
- `Metrics` (list): target metrics.

## Quality checklist
- Experiments are feasible, specific, and tied to the hypothesis.
- Related Work names at least 2-5 close baselines/lines of work.
- Risks include at least one technical risk and one evaluation risk.
