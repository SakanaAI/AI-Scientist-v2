# Paper Review Rubric

How to conduct a structured peer review. Distilled from AI-Scientist-v2 review pipeline (NeurIPS format).

## Reviewer Identity

You are an AI researcher reviewing a paper submitted to a prestigious ML venue. Be critical and cautious. Evaluate both strengths and weaknesses thoroughly.

## Review Process

1. **First pass**: Read title, abstract, introduction, conclusion to understand claims
2. **Second pass**: Read the full paper carefully, noting strengths and weaknesses
3. **Third pass**: Check experiments, figures, and tables in detail
4. **Reflection**: Consider the overall contribution and write your review

## Review Format

Output reviews as JSON following `templates/review-schema.json`:

```json
{
  "summary": "Brief summary of the paper's contributions",
  "strengths": ["Strength 1", "Strength 2", ...],
  "weaknesses": ["Weakness 1", "Weakness 2", ...],
  "questions": ["Question 1", "Question 2", ...],
  "limitations": "Assessment of whether limitations are adequately addressed",
  "ethics": "Ethics review flag and reasoning",
  "soundness": 3,         // 1-4: poor, fair, good, excellent
  "presentation": 3,      // 1-4: poor, fair, good, excellent
  "contribution": 3,      // 1-4: poor, fair, good, excellent
  "overall": 6,           // 1-10: strong reject to award quality
  "confidence": 3,        // 1-5: low to absolute
  "decision": "Accept"    // Accept, Weak Accept, Borderline, Weak Reject, Reject
}
```

## Evaluation Dimensions

### Originality (1-4)
- Are the tasks or methods new?
- Is the work a novel combination of known techniques?
- Is it clear how this differs from prior work?
- Is related work adequately cited?

### Quality (1-4)
- Is the submission technically sound?
- Are claims well-supported by theory or experiments?
- Are methods appropriate for the problem?
- Is this complete work or work-in-progress?

### Clarity (1-4)
- Is the paper clearly written and well-organized?
- Does it provide enough information to reproduce results?
- Are figures and tables informative and well-labeled?

### Significance (1-4)
- Are results important to the community?
- Does it advance the state of the art?
- Would others build on this work?

### Soundness (1-4)
- Are technical claims valid?
- Is the experimental methodology appropriate?
- Are central claims adequately supported with evidence?

## Overall Score Guide

- **10**: Award quality — flawless with groundbreaking impact
- **8-9**: Strong Accept — excellent, novel contribution
- **7**: Accept — solid paper with high impact on at least one area
- **6**: Weak Accept — solid, moderate-to-high impact
- **5**: Borderline Accept — reasons to accept slightly outweigh reject
- **4**: Borderline Reject — reasons to reject slightly outweigh accept
- **3**: Reject — technical flaws, weak evaluation
- **1-2**: Strong Reject — major flaws or trivial results

## Figure Review

For each figure in the paper, evaluate:
- Does the caption accurately describe the content?
- Is the figure informative and necessary?
- Does it support the claims made in the text?
- Is it properly labeled and readable?
- Are there any misleading visual elements?

## Common Weaknesses to Check

- Missing baselines or unfair comparisons
- Insufficient ablation studies
- Only one random seed or no error bars
- Claims not supported by evidence
- Missing related work
- Unclear methodology
- Results that don't match claims in abstract/introduction
- Page limit violations

## Few-Shot Review Examples

See `fewshot-examples/reviews/` for examples of well-structured reviews:
- `132_automated_relational.json` — review of an automated relational learning paper
- `attention.json` — review of an attention mechanism paper
- `2_carpe_diem.json` — review of a carpe diem approach paper
