---
name: plot
description: "Generate publication-quality figures from experiment results. Use: /plot <path to experiment directory>"
allowed-tools: Bash Read Write Edit Glob Grep
---

# Plot Generation

Create publication-quality figures for a research paper.

Argument: $ARGUMENTS (path to experiment directory containing results/).

## Process

1. Read `methodology/plotting-guide.md` for visual standards and best practices
2. Read the experiment results from `results/*.json` and list available `.npy` files
3. Read the idea from `idea.json` to understand what the paper is about
4. Read `research-log.md` for context on what results are most important

## Figure Planning

Determine what figures are needed based on the experiment phases:
- **From baselines**: Model comparison (bar chart or table)
- **From core hypothesis**: Main result (proposed vs baseline)
- **From extended experiments**: Scaling curves, cross-dataset comparison, Pareto frontiers
- **From ablations**: Component contribution analysis

Aim for 6-12 figures total (main paper + appendix).

## Figure Generation

Write a self-contained Python plotting script:
- Load data from `.npy` files and JSON summaries
- Each figure in a separate try-except block
- Save all figures to `figures/` directory as PDF (vector) or PNG
- Use 300 DPI, remove unnecessary spines, clear labels
- Font size at least 12pt for readability in paper

Save the plotting script as `code/generate_figures.py`.

## Quality Review

After generating figures, review each one:
- Does it convey a clear message?
- Are axes labeled with units?
- Is the legend visible and not overlapping data?
- Are error bars/confidence intervals shown where applicable?
- Would this look good at paper column width?

Regenerate any figures that don't meet the quality bar.

## Output

- All figures saved in `experiments/<run>/figures/`
- Plotting script saved in `experiments/<run>/code/generate_figures.py`
- Update `checkpoint.json` and `research-log.md`
