---
name: review-paper
description: "Conduct a structured peer review of a paper PDF. Scores soundness, presentation, contribution. Use: /review-paper <path to PDF>"
allowed-tools: Bash Read Write Edit Glob Grep
---

# Paper Review

Conduct a structured peer review of a research paper.

Argument: $ARGUMENTS (path to PDF file or experiment directory containing paper/).

## Process

1. Read `methodology/review-rubric.md` for the full review protocol and scoring guide
2. Read `templates/review-schema.json` for the output format
3. Read the paper PDF (use the Read tool on the PDF file directly)
4. Optionally read 1-2 fewshot examples from `fewshot-examples/reviews/` for calibration

## Review Protocol

### First Pass (5 minutes)
- Read title, abstract, introduction, conclusion
- Understand the main claims and contributions
- Note initial impressions

### Second Pass (15 minutes)
- Read the full paper carefully
- Check experimental methodology
- Evaluate figures and tables
- Note strengths and weaknesses as you go

### Third Pass (10 minutes)
- Verify claims against evidence
- Check for missing baselines, ablations, error bars
- Assess clarity and writing quality
- Formulate questions for the authors

## Figure Review

For each figure in the paper:
- Does the caption accurately describe the content?
- Is the figure informative and necessary?
- Does it support the claims in the text?
- Is it properly labeled and readable?

## Scoring

Score each dimension according to the rubric in `methodology/review-rubric.md`:
- Soundness (1-4)
- Presentation (1-4)
- Contribution (1-4)
- Overall (1-10)
- Confidence (1-5)
- Decision: Accept / Weak Accept / Borderline / Weak Reject / Reject

Be calibrated: most papers should score 4-6 overall. Reserve 7+ for genuinely strong work.

## Output

Save the review as JSON following `templates/review-schema.json`:
```
experiments/<run>/reviews/review.json
```

Also save a human-readable summary:
```
experiments/<run>/reviews/review_summary.md
```

Update `checkpoint.json` and `research-log.md`.
