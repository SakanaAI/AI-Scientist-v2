---
name: review
description: Perform AI peer review of a scientific paper — uses Claude Code directly, no API keys required
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: <experiment-dir>
---

# AI Scientist v2 — Paper Review (Claude Code Native)

You are an expert peer reviewer for a top ML conference (NeurIPS). Review the paper thoroughly and critically. **No API keys required.**

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to experiment directory containing the PDF

## Step 1: Find and Load the Paper

```bash
ls <experiment-dir>/*.pdf
```

If multiple PDFs exist, prefer:
1. Files containing "reflection_final" in the name
2. Files with the highest reflection number
3. The most recently modified PDF

Convert the PDF to readable text:
```bash
pdftotext <pdf-path> -
```

Also read the PDF directly using the Read tool (it supports PDF files) to see figures.

Read the research idea from `<experiment-dir>/idea.md` or `<experiment-dir>/idea.json` for context.

## Step 2: Text Review

Review the paper following the **NeurIPS review form**:

1. **Summary**: Summarize contributions in 2-3 sentences
2. **Strengths**: Novelty, soundness, clarity, significance, experiment quality
3. **Weaknesses**: Missing baselines, questionable methodology, insufficient analysis, overclaims
4. **Questions**: Questions for the authors
5. **Limitations**: Whether limitations are adequately discussed
6. **Scores**:
   - **Soundness** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
   - **Presentation** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
   - **Contribution** (1-4): 1=poor, 2=fair, 3=good, 4=excellent
   - **Originality** (1-4): 1=low, 2=medium, 3=high, 4=very high
   - **Quality** (1-4): 1=low, 2=medium, 3=high, 4=very high
   - **Clarity** (1-4): 1=low, 2=medium, 3=high, 4=very high
   - **Significance** (1-4): 1=low, 2=medium, 3=high, 4=very high
   - **Overall** (1-10): 1=very strong reject, 3=clear reject, 5=marginally below, 6=marginally above, 8=clear accept, 10=award quality
   - **Confidence** (1-5): 1=low, 3=moderate, 5=absolute certainty
7. **Decision**: "Accept" or "Reject"

### Guidelines
- Be critical but fair — if unsure, lean toward rejection
- Check that all claims are supported by evidence
- Verify figures match the text discussion

## Step 3: Figure Review

Using the Read tool to view the PDF, evaluate each figure for description, quality, caption accuracy, relevance, and suggestions. Check for duplicates, unreferenced figures, missing labels.

## Step 4: Save Review Output

Save text review to `<experiment-dir>/review_text.txt` as JSON:
```json
{
  "Summary": "...", "Strengths": [...], "Weaknesses": [...],
  "Originality": 3, "Quality": 3, "Clarity": 3, "Significance": 2,
  "Soundness": 3, "Presentation": 3, "Contribution": 2,
  "Questions": [...], "Limitations": [...], "Ethical Concerns": false,
  "Overall": 5, "Confidence": 3, "Decision": "Reject"
}
```

Save figure review to `<experiment-dir>/review_img_cap_ref.json`.

## Step 5: Report Results

Present: Decision with score, key strengths, key weaknesses, figure assessment, and improvement recommendations.
