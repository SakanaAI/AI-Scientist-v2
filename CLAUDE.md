# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The AI Scientist-v2 is an agentic system that autonomously runs the full ML research
pipeline: idea generation → experiment execution via tree search → plotting → paper
writeup (LaTeX/PDF) → automated review. It is designed to run on Linux with an NVIDIA
GPU (CUDA/PyTorch), and it executes LLM-generated code, so treat any local run as
executing untrusted code (sandbox/container recommended per the README's caution note).

## Python Environment

- Do not use global `pip` or standard `python`.
- Always run files or tests using `uv run <command>` (e.g., `uv run pytest`).
- To add a dependency, use `uv add <package>` (base dependency list lives in `requirements.txt`).

## Running the pipeline

Two-stage workflow, run from the repo root:

1. **Ideation** — generate research ideas from a topic markdown file:
   ```bash
   uv run python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 --max-num-generations 20 --num-reflections 5
   ```
   Topic files follow the structure of `ai_scientist/ideas/i_cant_believe_its_not_better.md`
   (Title/Keywords/TL;DR/Abstract). Output is a JSON file of structured ideas next to the input.

2. **Experiments + writeup + review** — consumes the ideation JSON:
   ```bash
   uv run python launch_scientist_bfts.py \
     --load_ideas "ai_scientist/ideas/my_research_topic.json" \
     --load_code --add_dataset_ref \
     --model_writeup o1-preview-2024-09-12 \
     --model_citation gpt-4o-2024-11-20 \
     --model_review gpt-4o-2024-11-20 \
     --model_agg_plots o3-mini-2025-01-31 \
     --num_cite_rounds 20
   ```
   Results land in a timestamped `experiments/<timestamp>_<idea_name>_attempt_<id>/` folder;
   the tree search visualization is `logs/0-run/unified_tree_viz.html` inside it, and the
   final paper is `<timestamp>_<idea_name>.pdf`.

Tree-search behavior (workers, steps, debug depth, num_drafts, etc.) is controlled by
`bfts_config.yaml`, which `launch_scientist_bfts.py` copies/edits per-run via
`ai_scientist/treesearch/bfts_utils.py:edit_bfts_config_file`.

## Model / API configuration

Model dispatch is centralized in `ai_scientist/llm.py` (`create_client`, `get_response_from_llm`,
`get_batch_responses_from_llm`) and mirrored for tool-calling in
`ai_scientist/treesearch/backend/` (`backend_anthropic.py`, `backend_openai.py`, dispatched via
`backend/__init__.py:query`). Model selection is done by string-matching on the model name
(`"claude"`, `"gpt"`, `"o1"/"o3"`, `"gemini"`, `"ollama/"`, `bedrock/...`, `vertex_ai/...`), so
adding a new provider means extending the branching in both `llm.py` and `treesearch/backend/`,
plus the `AVAILABLE_LLMS` list in `llm.py`. Credentials are read from env vars
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `S2_API_KEY`, AWS creds for Bedrock, etc.) — see README's
"Supported Models and API Keys" section.

All LLM calls are wrapped with `@track_token_usage` (`ai_scientist/utils/token_tracker.py`),
which accumulates cost/usage; `launch_scientist_bfts.py` dumps this to
`token_tracker.json`/`token_tracker_interactions.json` in the idea's experiment directory.

## Tree search architecture (`ai_scientist/treesearch/`)

This package is a fork/extension of the AIDE tree-search framework, restructured into a
4-stage pipeline driven by an `AgentManager`:

- **`perform_experiments_bfts_with_agentmanager.py`** — entry point (`perform_experiments_bfts`).
  Loads config, preps the agent workspace, constructs `AgentManager`, and runs it to completion.
- **`agent_manager.py`** (`AgentManager`) — owns the stage machine. Four *main stages*, each
  potentially split into LLM-generated *sub-stages*:
  1. `initial_implementation` — get a basic working implementation.
  2. `baseline_tuning` — hyperparameter tuning, no architecture changes, add 2 more HF datasets.
  3. `creative_research` — novel experiments/ideas, 3 HF datasets total.
  4. `ablation_studies` — systematic component analysis.
  Each (sub-)stage completion/progression decision is itself an LLM call (structured output via
  `FunctionSpec`). The best node from one stage seeds the next. State is checkpointed to
  `logs/<run>/stage_<name>/checkpoint.pkl` and the whole `AgentManager` is pickled at the end
  (`logs/.../manager.pkl`).
- **`parallel_agent.py`** (`ParallelAgent`) — runs one stage: drafts/debugs/improves nodes in
  parallel (`num_workers`), executes code via the interpreter, and requests VLM feedback on
  generated plots.
- **`journal.py`** (`Journal`, `Node`) — the solution tree for one stage: each `Node` holds
  plan/code/plot code, execution result, metric, buggy/leaf flags, and VLM plot analysis.
  `Journal.get_best_node()` selects the winning implementation.
- **`interpreter.py`** — sandboxed execution of agent-written code in a subprocess
  (`multiprocessing.Process`), capturing stdout/stderr, exceptions, and timing under
  `exec.timeout` from the config.
- **`backend/`** — thin LLM client layer used specifically for structured/function-calling
  queries inside tree search (as opposed to the freeform calls in top-level `ai_scientist/llm.py`).
- **`utils/config.py`** — OmegaConf-based config loading (`load_cfg`, `load_task_desc`,
  `prep_agent_workspace`, `save_run`); `bfts_config.yaml` is the schema source of truth.
- **`log_summarization.py`** — produces the `draft_summary.json` / `baseline_summary.json` /
  `research_summary.json` / `ablation_summary.json` reports written at the end of a run.

## Writeup and review (`ai_scientist/`)

- **`perform_writeup.py`** / **`perform_icbinb_writeup.py`** — turn a completed experiment
  directory's journals/results into a LaTeX paper and compile to PDF (8-page "normal" vs.
  4-page "icbinb" workshop format, selected via `--writeup-type`). LaTeX templates live in
  `ai_scientist/blank_icml_latex/` and `ai_scientist/blank_icbinb_latex/`.
  `gather_citations` in `perform_icbinb_writeup.py` drives citation search via
  `ai_scientist/tools/semantic_scholar.py`.
- **`perform_plotting.py`** (`aggregate_plots`) — combines per-node plots into final figures
  for the paper using a VLM-capable model.
- **`perform_llm_review.py`** / **`perform_vlm_review.py`** — automated paper review: text-based
  review (`perform_review`) and figure/caption/reference consistency review
  (`perform_imgs_cap_ref_review`), run against the generated PDF.
- **`tools/`** — `BaseTool` (`tools/base_tool.py`) is the abstract interface for agent tools
  (subclass and implement `use_tool`); `tools/semantic_scholar.py` implements literature search
  used during both ideation and citation gathering.

## Key data flow

`ideas/*.json` (from ideation) → `AgentManager` task_desc → 4-stage tree search producing
per-stage `Journal`s of `Node`s → best nodes' plots aggregated → `perform_writeup`/
`perform_icbinb_writeup` → PDF → `perform_llm_review`/`perform_vlm_review`. Every stage
writes its own notes/summaries under `logs/<run>/stage_*/` and `logs/<run>/notes/`, which is
where to look when debugging why a run stalled or a stage never progressed.
