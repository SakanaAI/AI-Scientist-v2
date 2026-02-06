# Prompt Templates

Use these templates to keep ideation structured and easy to validate.

## Topic brief template (user input)
```markdown
# Title

# Keywords

# TL;DR

# Abstract / Background

# Constraints
- compute budget:
- data availability:
- timeline:

# Evaluation preferences
- primary metric:
- baselines to include:
- datasets to use:
```

## Ideation prompt (generate 1 idea)
Paste the topic brief, then:
```text
Generate ONE research proposal as JSON that matches the schema in references/idea-schema.md.
Requirements:
- Be specific about experiments and evaluation.
- Include a brief Related Work section that distinguishes this idea from close baselines.
- Keep the idea feasible for an academic lab.
- Do not claim novelty without evidence.
Output only JSON (no markdown fences).
```

## Reflection prompt (improve an idea)
```text
Review the idea for:
1) Clarity and falsifiability of the hypothesis
2) Feasibility of experiments
3) Whether Related Work is specific and non-trivial
4) Missing baselines/datasets/metrics
5) Obvious novelty risks

Rewrite the JSON to improve weak points, keeping the core idea intact unless it is flawed.
Output only JSON.
```

## Novelty check note template (what to write down)
```text
Novelty sanity-check (Semantic Scholar)
Date/time:
Queries:
- ...
Closest papers considered:
- [title] (year, venue) - why close
Conclusion:
- Positioning statement (no over-claiming)
```
