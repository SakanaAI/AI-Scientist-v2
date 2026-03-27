---
name: ai-scientist
description: Run the full AI Scientist v2 pipeline — autonomous scientific research from ideation through review. No external API keys required (uses Claude Code natively).
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch
argument-hint: <ideas-json-or-topic-md> [--idea_idx N] [--writeup-type normal|icbinb] [--skip_writeup] [--skip_review] [--skip_experiments]
effort: max
---

# AI Scientist v2 — Full Pipeline (Claude Code Native)

Orchestrate the complete AI Scientist v2 research pipeline. **All stages run natively in Claude Code — no external API keys required** (only `S2_API_KEY` for Semantic Scholar is recommended).

## Arguments

Parse `$ARGUMENTS`:

- **Positional arg 1**: Path to either:
  - A `.json` file with pregenerated ideas (skips ideation)
  - A `.md` topic file (runs ideation first)
- `--idea_idx N`: Which idea to run experiments on (default: 0)
- `--writeup-type TYPE`: `icbinb` (4-page, default) or `normal` (8-page)
- `--load_code`: Load companion .py starter code
- `--add_dataset_ref`: Add HuggingFace dataset reference
- `--skip_experiments`: Skip experiment execution (use existing results)
- `--skip_writeup`: Skip paper generation
- `--skip_review`: Skip peer review
- `--max-num-generations N`: Ideas to generate if running ideation (default: 5)

## Environment Check

Before starting, verify:
```bash
# Check for Semantic Scholar key (recommended for ideation & writeup)
echo "S2_API_KEY: ${S2_API_KEY:+set}"

# Check GPU availability (needed for experiments)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available"

# Check LaTeX (needed for writeup)
which pdflatex 2>/dev/null && echo "pdflatex: available" || echo "pdflatex: not found (needed for writeup)"
```

If `S2_API_KEY` is not set, inform the user that literature search will use WebSearch as fallback.

## Pipeline Stages

### Stage 1: Ideation (Claude Code Native)

**Only runs if a `.md` file is provided instead of `.json`.**

Invoke the `/ai-scientist-ideate` skill with the topic file. This generates structured research ideas using Claude Code's capabilities and Semantic Scholar for literature search.

After ideation completes, the JSON file path becomes input for subsequent stages.

### Stage 2: Experiments (Claude Code Native)

**Skipped if `--skip_experiments` is set.**

Invoke the `/ai-scientist-experiment` skill. Claude Code acts as both the code generator and analyzer, running the full 4-stage experiment process:
1. Initial implementation
2. Hyperparameter tuning
3. Creative research improvements
4. Ablation studies

This stage generates code, executes it, analyzes results, views plots, and iterates — all within Claude Code.

### Stage 3: Paper Writeup (Claude Code Native)

**Skipped if `--skip_writeup` is set.**

Invoke the `/ai-scientist-writeup` skill. Claude Code will:
1. Read all experiment results and summaries
2. Search Semantic Scholar for relevant citations
3. Generate a complete LaTeX paper
4. Compile to PDF

### Stage 4: Peer Review (Claude Code Native)

**Skipped if `--skip_review` or `--skip_writeup` is set.**

Invoke the `/ai-scientist-review` skill. Claude Code will:
1. Read the generated PDF
2. Perform structured peer review (NeurIPS format)
3. Review figures and captions
4. Save review results

## Output Summary

After the pipeline completes, report:
1. Number of ideas generated (if ideation ran)
2. Experiment results directory and key metrics per stage
3. Paper PDF path (if writeup ran)
4. Review scores and decision (if review ran)
5. Total pipeline status

## Pipeline Flexibility

The pipeline is modular — users can run any combination:

```
# Full pipeline from topic
/ai-scientist my_topic.md

# Ideas exist, run everything
/ai-scientist ideas.json --idea_idx 0

# Experiments done, just write paper and review
/ai-scientist ideas.json --skip_experiments --idea_idx 0

# Just review an existing paper
/ai-scientist-review experiments/my_experiment/
```
