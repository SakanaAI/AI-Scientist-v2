---
name: pipeline
description: Run the full AI Scientist v2 pipeline — autonomous scientific research from ideation through review. No external API keys required.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch
argument-hint: <ideas-json-or-topic-md> [--idea_idx N] [--writeup-type normal|icbinb] [--skip_writeup] [--skip_review] [--skip_experiments]
effort: max
---

# AI Scientist v2 — Full Pipeline (Claude Code Native)

Orchestrate the complete AI Scientist v2 research pipeline. **All stages run natively in Claude Code** — only `S2_API_KEY` for Semantic Scholar is recommended.

## Environment

- **PROJECT_DIR**: `${user_config.PROJECT_DIR}` — path to the AI-Scientist-v2 repository.
- **S2_API_KEY**: `${user_config.S2_API_KEY}` — Semantic Scholar API key (optional).

Before starting, verify system dependencies:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available"
which pdflatex 2>/dev/null && echo "pdflatex: available" || echo "pdflatex: not found (needed for writeup)"
```

## Arguments

- **Positional arg 1**: Path to `.json` (ideas, skips ideation) or `.md` (topic, runs ideation first)
- `--idea_idx N`: Which idea (default: 0)
- `--writeup-type TYPE`: `icbinb` (4-page, default) or `normal` (8-page)
- `--load_code`: Load companion .py starter code
- `--add_dataset_ref`: Add HuggingFace dataset reference
- `--skip_experiments`: Skip experiment execution
- `--skip_writeup`: Skip paper generation
- `--skip_review`: Skip peer review
- `--max-num-generations N`: Ideas to generate (default: 5)

## Pipeline Stages

### Stage 1: Ideation (if `.md` input)

Invoke `/ai-scientist:ideate` with the topic file.

### Stage 2: Experiments

**Skipped if `--skip_experiments`.**

Invoke `/ai-scientist:experiment`. Runs the full 4-stage process:
1. Initial implementation
2. Hyperparameter tuning
3. Creative research improvements
4. Ablation studies

### Stage 3: Paper Writeup

**Skipped if `--skip_writeup`.**

Invoke `/ai-scientist:writeup`. Generates LaTeX paper with Semantic Scholar citations.

### Stage 4: Peer Review

**Skipped if `--skip_review` or `--skip_writeup`.**

Invoke `/ai-scientist:review`. NeurIPS-format structured review.

## Output Summary

Report: ideas generated, experiment directory + metrics, PDF path, review scores.

## Examples

```
# Full pipeline from topic
/ai-scientist:pipeline my_topic.md

# Ideas exist, run everything
/ai-scientist:pipeline ideas.json --idea_idx 0

# Skip experiments, just write paper and review
/ai-scientist:pipeline ideas.json --skip_experiments --idea_idx 0

# Individual stages
/ai-scientist:ideate topic.md
/ai-scientist:experiment ideas.json --idea_idx 0
/ai-scientist:writeup experiments/dir/
/ai-scientist:review experiments/dir/
```
