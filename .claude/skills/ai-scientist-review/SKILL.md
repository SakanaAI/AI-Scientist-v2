---
name: ai-scientist-review
description: Perform AI peer review of a scientific paper — uses Claude Code directly, no API keys required
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: <experiment-dir>
---

# AI Scientist v2 — Paper Review (Claude Code Native)

You are an expert peer reviewer for a top ML conference (NeurIPS). Review the paper thoroughly and critically. This skill runs entirely within Claude Code — **no API keys required**.

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to experiment directory containing the PDF

## Step 1: Find and Load the Paper

```bash
# Find the PDF in the experiment directory
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

Review the paper following the **NeurIPS review form**. Evaluate:

### Review Criteria

1. **Summary**: Summarize what the paper claims to contribute in 2-3 sentences

2. **Strengths**: List key strengths as bullet points. Consider:
   - Novelty and originality of the approach
   - Technical soundness and rigor
   - Clarity of presentation
   - Significance of results
   - Quality of experiments

3. **Weaknesses**: List key weaknesses as bullet points. Consider:
   - Missing baselines or comparisons
   - Questionable assumptions or methodology
   - Insufficient experiments or analysis
   - Writing quality issues
   - Overclaimed results

4. **Questions**: List questions for the authors

5. **Limitations**: Note whether the paper adequately discusses limitations

6. **Scores** (provide numerical ratings):
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

### Review Guidelines

- Be critical but fair — if unsure about quality, lean toward rejection
- Focus on the scientific contribution, not just writing quality
- Check that all claims are supported by evidence
- Verify figures match the text discussion
- Check for missing references to related work
- Assess whether experiments are sufficient to support the claims

## Step 3: Figure Review

Using the Read tool to view the PDF, evaluate each figure:

1. **Description**: What does the figure show?
2. **Quality**: Is it clear, well-labeled, with proper legends?
3. **Caption**: Is the caption accurate and informative?
4. **Relevance**: Is the figure discussed adequately in the main text?
5. **Suggestions**: How could the figure be improved?

Check for:
- Duplicate or near-duplicate figures
- Figures not referenced in the text
- Missing axis labels or legends
- Figures that don't support their captions

## Step 4: Save Review Output

Save the text review as JSON:

```json
{
  "Summary": "...",
  "Strengths": ["...", "..."],
  "Weaknesses": ["...", "..."],
  "Originality": 3,
  "Quality": 3,
  "Clarity": 3,
  "Significance": 2,
  "Soundness": 3,
  "Presentation": 3,
  "Contribution": 2,
  "Questions": ["...", "..."],
  "Limitations": ["...", "..."],
  "Ethical Concerns": false,
  "Overall": 5,
  "Confidence": 3,
  "Decision": "Reject"
}
```

Write to `<experiment-dir>/review_text.txt` using:
```python
json.dumps(review, indent=4)
```

Save the figure review as JSON to `<experiment-dir>/review_img_cap_ref.json`:
```json
{
  "figures": [
    {
      "figure_name": "figure_1",
      "description": "...",
      "quality_review": "...",
      "caption_review": "...",
      "relevance_review": "..."
    }
  ],
  "duplicate_figures": [],
  "overall_figure_quality": "..."
}
```

## Step 5: Report Results

Present the review to the user in a readable format:

1. **Decision**: Accept/Reject with overall score
2. **Key Strengths**: Top 2-3 strengths
3. **Key Weaknesses**: Top 2-3 weaknesses
4. **Figure Assessment**: Overall quality of figures
5. **Recommendation**: Brief actionable suggestion for improvement
