---
name: ai-scientist-writeup
description: Generate a scientific paper from completed AI Scientist v2 experiments — uses Claude Code directly with Semantic Scholar for citations
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch
argument-hint: <experiment-dir> [--writeup-type normal|icbinb]
---

# AI Scientist v2 — Paper Writeup (Claude Code Native)

You are a scientific paper writer generating a complete LaTeX manuscript from experiment results. This skill runs within Claude Code using Semantic Scholar for citation gathering.

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to experiment directory (e.g., `experiments/2025-01-01_my_idea_attempt_0`)
- `--writeup-type TYPE`: `icbinb` (4-page workshop, default) or `normal` (8-page conference)

## Step 1: Load Experiment Data

Read these files from the experiment directory:

1. **Research idea**: `idea.md` or `research_idea.md`
2. **Idea JSON**: `idea.json`
3. **Experiment summaries** (in `logs/0-run/`):
   - `baseline_summary.json`
   - `research_summary.json`
   - `ablation_summary.json`
4. **Figures**: List all `.png` files in `figures/`
5. **Plot aggregator script**: `auto_plot_aggregator.py` (if exists)

Read all of these files. The experiment summaries are critical — they contain the full results.

## Step 2: Set Up LaTeX Workspace

```bash
cd <experiment-dir>

# Choose template based on writeup type
if [ "$WRITEUP_TYPE" = "normal" ]; then
  cp -r /home/user/AI-Scientist-v2/ai_scientist/blank_icml_latex latex
else
  cp -r /home/user/AI-Scientist-v2/ai_scientist/blank_icbinb_latex latex
fi
```

Read the template file at `<experiment-dir>/latex/template.tex`.

## Step 3: Gather Citations

Use **Semantic Scholar** to find relevant papers for citation. For each major topic, claim, or comparison in the experiment results, search for papers:

```bash
cd /home/user/AI-Scientist-v2
python -c "
from ai_scientist.tools.semantic_scholar import search_for_papers
import json

papers = search_for_papers('YOUR SEARCH QUERY', result_limit=5)
if papers:
    for p in papers:
        authors = ', '.join([a.get('name','') for a in p.get('authors',[])])
        print(f\"Title: {p.get('title')}\")
        print(f\"Authors: {authors}\")
        print(f\"Venue: {p.get('venue')}, Year: {p.get('year')}\")
        print(f\"Citations: {p.get('citationCount')}\")
        cite = p.get('citationStyles', {}).get('bibtex', '')
        if cite:
            print(f'BibTeX: {cite}')
        print('---')
"
```

Perform at least 5-8 separate searches covering different aspects of the paper (method, baselines, datasets, related work, etc.) to gather 10-15+ citations.

For each paper found:
1. Use the BibTeX from Semantic Scholar's `citationStyles.bibtex` field if available
2. Otherwise, create a BibTeX entry manually
3. Clean citation keys: lowercase, no accents/special chars, format `authorYYYYkeyword`
4. Add all citations to the `references.bib` filecontents block in `template.tex`

If Semantic Scholar is unavailable, fall back to **WebSearch**.

BibTeX entries should follow this format:
```bibtex
@article{authorYYYYkeyword,
  title={Paper Title},
  author={Last, First and Last2, First2},
  journal={Venue},
  year={YYYY}
}
```

## Step 4: Generate Paper Content

Replace the placeholder markers (`%%%%%%%%%SECTION%%%%%%%%%`) in `template.tex` with real content. The template has these sections:

### For ICBINB (4-page, single-column):
- **Title** and **Abstract** (from the idea)
- **Introduction**: Motivate the research question, state contributions
- **Related Work**: Position against existing literature (use citations from Step 3)
- **Background**: Necessary technical background
- **Method**: Describe the approach in detail
- **Experimental Setup**: Datasets, metrics, baselines, hyperparameters
- **Experiments**: Present results with figures and tables. Reference ALL relevant figures from `figures/`
- **Conclusion**: Summarize findings, limitations, future work
- **Appendix**: Additional results, ablations, extra figures

### For Normal (8-page, double-column):
Same sections plus:
- **Impact Statement**: Broader impact of the work

### Writing Guidelines:

1. **Figures**: Use `\includegraphics{filename.png}` (the template sets `\graphicspath{{../figures/}}`)
2. **Page limits**: 4 pages for ICBINB (single-column), 8 pages for normal (double-column), excluding references and appendix
3. **Accuracy**: Only report results that appear in the experiment summaries. Never hallucinate metrics.
4. **All results**: Include ALL experiment results truthfully — negative results are valuable, especially for ICBINB
5. **Figures in main text**: For ICBINB, keep at most 4 figures in main text; move rest to appendix
6. **No itemize/enumerate abuse**: Use flowing prose, minimize bullet lists
7. **Citations**: Use `\citep{}` for parenthetical and `\citet{}` for textual citations

### LaTeX Quality:
- Escape special characters: `%`, `_`, `&`, `#`, `$`
- Use `\textbf{}` and `\textit{}` for emphasis
- Tables should use `\begin{table}[h]` with `\centering`
- Figures should use `\begin{figure}[h]` or `\begin{figure*}` for full-width

## Step 5: Write the LaTeX File

Use the Edit tool to replace the template content section by section. Work through each `%%%%%%%%%MARKER%%%%%%%%%` pair, replacing the placeholder text between them with your generated content.

## Step 6: Compile the Paper

```bash
cd <experiment-dir>/latex
pdflatex -interaction=nonstopmode template.tex
bibtex template
pdflatex -interaction=nonstopmode template.tex
pdflatex -interaction=nonstopmode template.tex
```

## Step 7: Review & Fix Compilation

After compilation:

1. Check for LaTeX errors in the output
2. Run `chktex template.tex` to find common issues
3. Check page count: `pdftotext template.pdf - | grep -c "."` or examine the PDF
4. Fix any issues and recompile (up to 3 attempts)

Common fixes:
- `</end` → `\end` (common LLM mistake)
- Unescaped `%` or `_` in text
- Missing `\usepackage` declarations
- Overfull hbox warnings (rephrase text or adjust figure sizes)

## Step 8: Finalize

```bash
cd <experiment-dir>
FOLDER_NAME=$(basename $(pwd))
cp latex/template.pdf "${FOLDER_NAME}.pdf"
```

Report to the user:
- Path to the generated PDF
- Page count
- Any compilation warnings
- Suggestion to run `/ai-scientist-review <experiment-dir>` for peer review
