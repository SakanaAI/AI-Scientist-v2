---
name: research
description: "Run the full AI research pipeline: ideation, experiments, plotting, writing, and review. Use when starting a new research project or resuming one."
allowed-tools: Bash Read Write Edit Glob Grep Agent WebSearch
---

# Full Research Pipeline

You are running the complete AI research pipeline. This orchestrates all 5 phases with human checkpoints as configured.

## Setup

1. Read `research.yaml` for topic, venue, constraints, and checkpoint settings
2. Check if an experiment directory already exists in `experiments/` for this topic
3. If resuming, read `checkpoint.json` to determine where to pick up

## Create Experiment Directory

If starting fresh:
```bash
mkdir -p experiments/$(date +%Y-%m-%d)_<idea-name>/{code,results,figures,paper,reviews}
```

Initialize `research-log.md` with the research topic and date.

## Phase 1: Ideation

If `experiment.idea_source` is set in `research.yaml`, load pre-generated ideas from that file.
Otherwise, invoke the `/ideate` skill to generate new ideas.

**Checkpoint**: If `checkpoints.after_ideation` is true:
- Present the generated ideas to the researcher
- Ask which idea to pursue (or accept modifications)
- Wait for human response before continuing

## Phase 2: Experiments

Invoke the `/experiment` skill with the selected idea.

Read `methodology/experiment-guide.md` for the structured experiment phases:
1. Baseline implementation
2. Core hypothesis testing
3. Optimization and extended experiments
4. Ablation studies

**Checkpoint**: If `checkpoints.after_experiments` is true:
- Summarize experiment results
- Present key findings and metrics
- Ask if the researcher wants to proceed to paper writing
- Wait for human response

## Phase 3: Plotting

Invoke the `/plot` skill to generate publication-quality figures from experiment results.

## Phase 4: Paper Writing

Invoke the `/write-paper` skill to write the LaTeX paper.

This includes:
- Setting up the LaTeX template (copy from `templates/<venue>/`)
- Writing each section
- Gathering citations via literature search
- Compiling to PDF

**Checkpoint**: If `checkpoints.after_writing` is true:
- Present the compiled PDF
- Ask for feedback before review

## Phase 5: Review

Invoke the `/review-paper` skill to get a structured peer review.

**Checkpoint**: If `checkpoints.after_review` is true:
- Present the review scores and feedback
- Ask if revisions are needed

## Completion

When all phases are complete:
1. Update `checkpoint.json` with final status
2. Summarize the entire research process
3. Point the researcher to the final PDF and review

## Resuming from Checkpoint

If `checkpoint.json` exists and has incomplete phases:
- Skip completed phases
- Resume from the last incomplete phase
- Re-read `research-log.md` for context on what was done
