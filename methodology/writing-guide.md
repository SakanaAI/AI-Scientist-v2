# Paper Writing Guide

How to write a publication-quality ML paper. Distilled from AI-Scientist-v2 writeup pipeline.

## Paper Structure

### For ICBINB (4-page ICLR format)
Use template from `templates/icbinb/template.tex`. Sections:
1. **Introduction** (~1 page): Motivation, problem statement, contributions
2. **Related Work** (~0.5 page): Position relative to prior work
3. **Background** (~0.25 page): Necessary technical background
4. **Method** (~0.75 page): Proposed approach with mathematical detail
5. **Experimental Setup** (~0.25 page): Datasets, baselines, metrics, implementation details
6. **Results** (~1 page): Main results, scaling analysis, ablations with figures
7. **Conclusion** (~0.25 page): Summary, limitations, future work
8. **Appendix** (unlimited): Additional results, implementation details

### For ICML (8-page format)
Use template from `templates/icml/template.tex`. Same structure with more space.

## Writing Principles

- **Lead with results**: Every section should build toward the experimental findings
- **Be specific**: "Our method improves accuracy by 2.3% on PathMNIST" not "Our method improves accuracy"
- **Be honest**: Report negative results, limitations, and failure modes
- **Avoid hand-waving**: Every claim must be supported by evidence (experiment, citation, or proof)
- **One idea per paragraph**: Each paragraph should have a clear topic sentence

## Section-by-Section Guidance

### Introduction
- Hook: Why does this problem matter?
- Gap: What's missing from current approaches?
- Proposal: What do we do differently?
- Contributions: Bulleted list of 2-3 specific contributions
- Each contribution should be falsifiable and verified in experiments

### Related Work
- Organize by theme, not chronologically
- For each group of related papers, explain how your work differs
- Don't just list papers — synthesize and contrast
- Use `\citep{}` for parenthetical, `\citet{}` for textual citations

### Method
- Start with notation and problem setup
- Build up from simple to complex
- Use equations for precision (number all equations)
- Include an algorithm box if the method has clear steps
- Pseudocode should be self-contained

### Experiments
- State the exact research questions being answered
- Describe setup before results: datasets, splits, hyperparameters, seeds
- Present results with tables and figures
- Include error bars / standard deviations
- Discuss surprising or negative results

### Conclusion
- Summarize findings in 2-3 sentences
- State limitations explicitly
- Suggest 1-2 concrete future directions

## Citation Gathering Protocol

1. **Identify missing citations** by re-reading the paper
2. **Categories to cite**:
   - Background methods and datasets
   - Competing approaches
   - Theoretical foundations
   - Evaluation metrics
   - Related findings in other domains
3. **Search Semantic Scholar** for each citation need
4. **Verify relevance** — only add citations that genuinely support a point
5. **Format in BibTeX** and add to `references.bib`
6. **Don't over-cite** — quality over quantity

## LaTeX Best Practices

- Use `\label{}` and `\ref{}` for all figures, tables, and equations
- Figures: `\begin{figure}[t]` for top placement
- Tables: `\begin{table}[t]` with `\centering`
- Use `\textbf{}` for column headers in tables
- Number format: Use `\num{}` from siunitx or consistent decimal places
- Compile early and often to catch LaTeX errors

## Compilation

```bash
python lib/compile_latex.py <paper-directory>
```

Or manually:
```bash
cd <paper-dir>
pdflatex -interaction=nonstopmode template.tex
bibtex template
pdflatex -interaction=nonstopmode template.tex
pdflatex -interaction=nonstopmode template.tex
```

## Writing Iteration

1. Write a complete first draft (all sections)
2. Compile and verify PDF renders correctly
3. Re-read and refine: clarity, flow, consistency
4. Check all figures and tables are referenced in text
5. Verify all citations compile without warnings
6. Check page limit compliance
