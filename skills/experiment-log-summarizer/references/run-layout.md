# Common Run Layouts

This skill tries to work with many experiment folders by using heuristics.

## AI-Scientist-style (example)
Typical artifacts under a single run directory:
- `idea.md` / `idea.json` / `research_idea.md`
- `logs/` with stage subfolders containing:
  - `journal.json`
  - `*_summary.json` (baseline/research/ablation summaries)
  - `stage_progress.json`
- `figures/` (PNG figures used in the paper)
- `*.pdf` (compiled manuscript or drafts)
- `token_tracker.json` (optional)

## Generic ML experiment folder
Common artifacts:
- `README.md` or `notes.md`
- `metrics.json`, `results.json`, `history.csv`
- `figures/` or `plots/` with PNGs
- `checkpoints/` (not parsed by default)
- `wandb/` exports (not parsed unless you export JSON/CSV)

## What the summarizer extracts
- An artifact inventory (key files, summary JSONs, PDFs, figure PNGs)
- For recognized summary JSONs, it extracts:
  - key numerical results
  - included plot list (paths + descriptions)
  - high-level descriptions if present
