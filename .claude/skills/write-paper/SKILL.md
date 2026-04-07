---
name: write-paper
description: "Write a LaTeX paper from experiment results. Gathers citations, writes all sections, compiles to PDF. Use: /write-paper <path to experiment directory>"
allowed-tools: Bash Read Write Edit Glob Grep WebSearch
---

# Paper Writing

Write a complete LaTeX paper from experiment results.

Argument: $ARGUMENTS (path to experiment directory).

## Setup

1. Read `research.yaml` for venue (icbinb or icml) and page limit
2. Read `methodology/writing-guide.md` for the full writing protocol
3. Read the idea from `idea.json`
4. Read experiment results from `results/*.json`
5. Read `research-log.md` for the full research narrative
6. List available figures in `figures/`

## Template Setup

Copy the LaTeX template to the paper directory:
```bash
cp -r templates/<venue>/* experiments/<run>/paper/
```

Copy figures:
```bash
mkdir -p experiments/<run>/paper/figures
cp experiments/<run>/figures/* experiments/<run>/paper/figures/
```

## Writing Each Section

Edit `experiments/<run>/paper/template.tex` to fill in each section.

The template has placeholder comments like `%%%%%%%%%TITLE%%%%%%%%%`, `%%%%%%%%%ABSTRACT%%%%%%%%%`, etc. Replace each with the actual content.

### Section Order
1. **Title and Abstract**: From the idea, refined with actual results
2. **Introduction**: Motivation, gap, proposal, contributions
3. **Related Work**: Position vs prior work, using citations gathered below
4. **Background**: Technical prerequisites
5. **Method**: Proposed approach with math
6. **Experimental Setup**: Datasets, baselines, metrics, implementation
7. **Results**: Main findings with figures and tables
8. **Conclusion**: Summary, limitations, future work

### For each section:
- Write focused, specific content (no hand-waving)
- Reference figures with `\ref{fig:name}` and tables with `\ref{tab:name}`
- Use exact numbers from experiment results
- Be honest about negative results

## Citation Gathering

For each citation need:
1. Search using `python lib/semantic_scholar.py "<query>"` or web search
2. Get BibTeX entries for relevant papers
3. Add to `experiments/<run>/paper/references.bib`
4. Cite using `\citep{}` (parenthetical) or `\citet{}` (textual)

Categories to cite:
- Background methods and datasets
- Competing approaches
- Theoretical foundations
- Related findings in other domains

## Compilation

Compile the paper:
```bash
python lib/compile_latex.py experiments/<run>/paper/
```

Check for LaTeX errors and fix them. Iterate until clean compilation.

## Quality Check

After first draft:
- Verify all figures and tables are referenced in text
- Verify all citations compile without warnings
- Check page limit compliance
- Re-read for clarity, flow, and consistency
- Ensure abstract matches actual results (not the original idea abstract)

## Output

- Complete LaTeX source in `experiments/<run>/paper/`
- Compiled PDF at `experiments/<run>/paper/template.pdf`
- Update `checkpoint.json` and `research-log.md`
