---
name: experiment-bfts-runner
description: Run the standalone BFTS experiment pipeline to execute multi-agent tree-search experiments from a prepared bfts_config.yaml. Use after idea-to-markdown and bfts-config-prep to produce logs, workspaces, and experiment artifacts.
---

# Experiment BFTS Runner

## Overview
Execute the full BFTS tree-search experiment workflow from a prepared bfts_config.yaml, producing logs, workspaces, and per-node experiment results.

## Workflow
1. Prepare a run directory
   - Use bfts-config-prep to create runs/<timestamp>_<idea_name>/ with bfts_config.yaml and idea.md.
2. Run the experiment
   - Offline default:
     ~~~bash
     UV_CACHE_DIR=/tmp/uv-cache XDG_CACHE_HOME=/tmp uv run --with pyyaml,omegaconf,openai,anthropic,backoff,rich,humanize -s scripts/run_bfts.py --config runs/<run>/bfts_config.yaml
     ~~~
   - Online (required for LLM calls):
     ~~~bash
     uv run -s scripts/run_bfts.py --config runs/<run>/bfts_config.yaml --online
     ~~~
3. Inspect outputs
   - Logs and workspaces are placed under the run directory; use experiment-log-summarizer for summaries.

## Inputs
- --config: path to bfts_config.yaml.
- --online: enable network calls to LLM providers (default: offline).

## Outputs
- logs/ and workspaces/ under the run directory.
- Per-node experiment results (e.g., experiment_results/ containing experiment_data.npy).

## Safeguards
- Offline by default; --online required for network calls.
- Reads only from the run directory; writes only within the run directory and its logs/ and workspaces/ subfolders.
- No file deletion unless you manually clean outputs.

## References
- Run manifest schema: references/run.manifest.json
- Idea schema: references/idea.schema.json
- Summary schema: references/summary.schema.json
