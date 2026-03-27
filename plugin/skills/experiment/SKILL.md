---
name: experiment
description: Run AI Scientist v2 experiments using Claude Code as the code generator — no external API keys required. Implements the 4-stage BFTS workflow natively.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
argument-hint: <ideas-json-path> [--idea_idx N] [--load_code] [--add_dataset_ref]
effort: max
---

# AI Scientist v2 — Experiment Execution (Claude Code Native)

You are an expert ML researcher running experiments through a structured 4-stage process. You will generate Python code, execute it, analyze results, debug failures, and iteratively improve — all within Claude Code. **No external API keys required.**

## Environment

- **PROJECT_DIR**: `${user_config.PROJECT_DIR}` — path to the AI-Scientist-v2 repository.

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to ideas JSON file (required)
- `--idea_idx N`: Which idea to run (default: 0)
- `--load_code`: Load companion `.py` starter code file
- `--add_dataset_ref`: Include HuggingFace dataset reference code

## Step 1: Setup

### 1a. Load the Research Idea

```bash
cd ${user_config.PROJECT_DIR}
```

Read the ideas JSON file and extract the idea at the specified index. Read the idea's fields:
- **Name**, **Title**, **Short Hypothesis**
- **Abstract**, **Experiments** (specific experiment plans)
- **Code** (if `--load_code` was used and code exists)

### 1b. Create Experiment Directory

```bash
DATE=$(date +%Y-%m-%d_%H-%M-%S)
IDEA_NAME="<idea Name field>"
EXPERIMENT_DIR="${user_config.PROJECT_DIR}/experiments/${DATE}_${IDEA_NAME}_attempt_0"
mkdir -p "$EXPERIMENT_DIR/logs/0-run"
mkdir -p "$EXPERIMENT_DIR/figures"
```

Save the idea as `idea.md` and `idea.json` in the experiment directory.

### 1c. Define Evaluation Metrics

Based on the research idea and experiments described, define ONE primary evaluation metric:
- **metric_name**: Precise name (e.g., "validation accuracy", "test BLEU score")
- **lower_is_better**: Whether lower values are better (true for loss, false for accuracy)
- **description**: What this metric measures

Validation loss is always tracked separately in addition to the primary metric.

## Step 2: Stage 1 — Initial Implementation

**Goal**: Get a basic working implementation of the research idea.

### Code Generation Requirements

All generated experiment code MUST follow these conventions:

```python
import os
import numpy as np
import torch

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# ... experiment code ...

# Save results in standard format
experiment_data = {
    'dataset_name_1': {
        'metrics': {'train': [epoch1_val, epoch2_val, ...], 'val': [...]},
        'losses': {'train': [...], 'val': [...]},
        'predictions': [...],  # optional
        'ground_truth': [...]  # optional
    }
}
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
print(f'Results saved to {os.path.join(working_dir, "experiment_data.npy")}')
```

**Rules:**
- ALL code at global scope — NO `if __name__ == "__main__"` blocks
- Move models and data to `device` using `.to(device)`
- Use try-except for robustness
- Print metrics clearly: `print(f'Dataset: {name}, {metric}: {value:.4f}')`
- Keep execution under 60 minutes
- Start with ONE simple dataset, use standard HuggingFace datasets when possible

### Iteration Loop (up to 5 attempts)

For each attempt:

1. **Generate code**: Write a complete Python script implementing the baseline approach described in the idea
2. **Save and execute**:
   ```bash
   cd $EXPERIMENT_DIR
   mkdir -p workspace/working
   # Write the code to workspace/runfile.py using the Write tool
   cd workspace && timeout 3600 python runfile.py 2>&1 | tee execution_output.txt
   ```
3. **Analyze results**:
   - Check if `workspace/working/experiment_data.npy` was created
   - Read execution output for errors or metrics
   - If the code errored: **debug** — analyze the error, fix the code, retry
   - If the code succeeded: **evaluate** — check if metrics are reasonable
4. **If successful**: Move to plotting, then Stage 2
5. **If failed after 5 attempts**: Report failure and stop

### Generate Plots

After a successful run, generate a plotting script:

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

working_dir = os.path.join(os.getcwd(), 'working')
data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()

figures_dir = os.path.join(os.getcwd(), '..', 'figures')
os.makedirs(figures_dir, exist_ok=True)

for dataset_name, dataset_data in data.items():
    try:
        if 'losses' in dataset_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            # ... plot training/validation loss curves
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
        if 'metrics' in dataset_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            # ... plot metric curves
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f'Error plotting {dataset_name}: {e}')
```

Execute the plotting script. Then **view the generated plots** using the Read tool to verify they look correct.

### Save Stage 1 Summary

After Stage 1, save a summary to `$EXPERIMENT_DIR/logs/0-run/baseline_summary.json`:

```json
{
  "best_node": {
    "overall_plan": "<description of approach>",
    "analysis": "<analysis of results>",
    "metric": "<primary metric value>",
    "code": "<the working Python code>",
    "plot_code": "<the plotting code>",
    "plot_analyses": ["<description of each plot>"],
    "vlm_feedback_summary": "<summary of what plots show>",
    "exp_results_npy_files": ["experiment_results/stage1/experiment_data.npy"],
    "datasets_successfully_tested": ["dataset_name_1"]
  }
}
```

Copy experiment data:
```bash
mkdir -p $EXPERIMENT_DIR/logs/0-run/experiment_results/stage1
cp $EXPERIMENT_DIR/workspace/working/experiment_data.npy $EXPERIMENT_DIR/logs/0-run/experiment_results/stage1/
```

## Step 3: Stage 2 — Baseline Tuning

**Goal**: Optimize hyperparameters of the Stage 1 implementation. DO NOT change model architecture. Introduce ONE additional dataset (total: 2 datasets).

### Iteration Loop (up to 4 attempts)

For each attempt, pick ONE hyperparameter to tune:
1. Training longer (more epochs)
2. Learning rate adjustment
3. Batch size optimization
4. Regularization (dropout, weight decay)
5. Other algorithm-specific parameters

For each:
1. Take the best Stage 1 code as base
2. Modify the specific hyperparameter
3. Add a second dataset
4. Execute and analyze results
5. Compare metrics with Stage 1 baseline
6. View generated plots to assess convergence

### Save Stage 2 Summary

Save to `$EXPERIMENT_DIR/logs/0-run/baseline_summary.json` (update the existing file to include tuning results).

Copy results:
```bash
mkdir -p $EXPERIMENT_DIR/logs/0-run/experiment_results/stage2
cp $EXPERIMENT_DIR/workspace/working/experiment_data.npy $EXPERIMENT_DIR/logs/0-run/experiment_results/stage2/
```

## Step 4: Stage 3 — Creative Research

**Goal**: Explore novel improvements to the approach. Be creative. Use THREE HuggingFace datasets total.

### Iteration Loop (up to 4 attempts)

For each attempt:
1. Start from the best Stage 2 code
2. Propose a **novel improvement** inspired by the research idea
3. Add a third dataset
4. Execute and compare with previous stages
5. View plots to verify improvements

### Save Stage 3 Summary

Save to `$EXPERIMENT_DIR/logs/0-run/research_summary.json`.

Copy results:
```bash
mkdir -p $EXPERIMENT_DIR/logs/0-run/experiment_results/stage3
cp $EXPERIMENT_DIR/workspace/working/experiment_data.npy $EXPERIMENT_DIR/logs/0-run/experiment_results/stage3/
```

## Step 5: Stage 4 — Ablation Studies

**Goal**: Conduct systematic component analysis. Use the same datasets from Stage 3.

### Iteration Loop (3-5 ablations)

For each ablation:
1. Start from the best Stage 3 code
2. **Remove or disable ONE specific component** to measure its contribution
3. Execute with the same 3 datasets
4. Compare with full Stage 3 model
5. Record the performance delta

### Save Stage 4 Summary

Save to `$EXPERIMENT_DIR/logs/0-run/ablation_summary.json`.

Copy results per ablation:
```bash
mkdir -p $EXPERIMENT_DIR/logs/0-run/experiment_results/stage4_ablation1
cp $EXPERIMENT_DIR/workspace/working/experiment_data.npy $EXPERIMENT_DIR/logs/0-run/experiment_results/stage4_ablation1/
```

## Step 6: Plot Aggregation

Create a comprehensive plot aggregation script. Save up to 12 figures in `$EXPERIMENT_DIR/figures/`. Save the script as `$EXPERIMENT_DIR/auto_plot_aggregator.py`.

Guidelines: DPI 300, remove top/right spines, clear labels (no underscores), each plot in try-except, combine related plots (up to 3 subplots per figure).

## Step 7: Report Results

Summarize for the user:
1. **Stage 1-4 Results** with key metrics
2. **Output directory** path
3. **Next step**: Run `/ai-scientist:writeup $EXPERIMENT_DIR` to generate a paper

## Error Handling

- If a stage completely fails, log and proceed with best available code
- Fall back to CPU if no GPU available
- Never spend more than 5 attempts debugging the same error

## Parallelism with Agent Tool

For Stage 4 ablations, you MAY use the **Agent tool** to run multiple ablations in parallel.
