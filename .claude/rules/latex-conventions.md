# LaTeX Conventions

Applies to: `**/*.tex`, `**/*.bib`

## Compilation
Use this sequence to compile papers:
```bash
cd <paper-dir>
pdflatex -interaction=nonstopmode template.tex
bibtex template
pdflatex -interaction=nonstopmode template.tex
pdflatex -interaction=nonstopmode template.tex
```

Or use the utility: `python lib/compile_latex.py <paper-dir>`

## Figures
- Place all figures in the `figures/` subdirectory
- Use `\graphicspath{{figures/}}` in the LaTeX preamble
- Reference figures as `\includegraphics{filename}` (no path prefix needed)
- Use PDF or PNG format for figures (vector preferred)

## Citations
- Use BibTeX entries in `references.bib`
- Cite using `\citep{}` for parenthetical and `\citet{}` for textual citations
- Verify all citations compile without warnings

## Page Limits
- ICBINB (ICLR): 4 pages main text + unlimited appendix
- ICML: 8 pages main text + unlimited appendix
- Respect the venue's page limit configured in research.yaml
