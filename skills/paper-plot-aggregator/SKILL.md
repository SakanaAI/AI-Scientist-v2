---
name: paper-plot-aggregator
description: Create publication-quality plots from experiment artifacts (.npy, JSON, CSV). Use when preparing final figures for a paper, consolidating plots across runs, or turning raw metrics into a clean figures/ folder.
---

# Paper Plot Aggregator

## Overview
Inventory metric files, generate a plotting skeleton, and produce publication-ready figures. This skill is read-only on inputs and writes outputs to a figures directory.

## Workflow
1. Inventory NPY files
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with numpy -s scripts/npy_inventory.py --dir /path/to/run --out inventory.json
   ~~~
2. Generate aggregator skeleton
   ~~~bash
   UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with numpy,matplotlib -s scripts/gen_aggregator_skeleton.py --inventory inventory.json --out auto_plot_aggregator.py --figures-dir figures
   ~~~
3. Run the generated script and inspect figures

## Inputs
- --dir: input directory
- --inventory: inventory.json from npy_inventory.py
- --figures-dir: output directory for plots

## Outputs
- figures/ directory
- auto_plot_aggregator.py
- plot_manifest.json (optional)

## Safeguards
- Read-only on inputs; writes only to provided output locations.
- Use --clean only if you explicitly want to overwrite existing figures.

## References
- Plot guidelines: references/plot-guidelines.md
- Plot manifest format: references/plot-manifest.md
- Plot manifest schema: references/plot_manifest.schema.json
- Safeguards: references/safeguards.md
