---
name: ai-scientist-experiment
description: Run AI Scientist v2 experiments using Claude Code as the code generator — no external API keys required. Implements the 4-stage BFTS workflow natively.
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
argument-hint: <ideas-json-path> [--idea_idx N] [--load_code] [--add_dataset_ref]
effort: max
---

# AI Scientist v2 — Experiment Execution (Claude Code Native)

You are an expert ML researcher running experiments through a structured 4-stage process. You will generate Python code, execute it, analyze results, debug failures, and iteratively improve — all within Claude Code. **No external API keys required.**

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to ideas JSON file (required)
- `--idea_idx N`: Which idea to run (default: 0)
- `--load_code`: Load companion `.py` starter code file
- `--add_dataset_ref`: Include HuggingFace dataset reference code

## Step 1: Setup

### 1a. Load the Research Idea

```bash
cd /home/user/AI-Scientist-v2
```

Read the ideas JSON file and extract the idea at the specified index. Read the idea's fields:
- **Name**, **Title**, **Short Hypothesis**
- **Abstract**, **Experiments** (specific experiment plans)
- **Code** (if `--load_code` was used and code exists)

### 1b. Create Experiment Directory

```bash
DATE=$(date +%Y-%m-%d_%H-%M-%S)
IDEA_NAME="<idea Name field>"
EXPERIMENT_DIR="experiments/${DATE}_${IDEA_NAME}_attempt_0"
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

# Generate plots for each dataset
for dataset_name, dataset_data in data.items():
    try:
        # Training curves
        if 'losses' in dataset_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            # ... plot training/validation loss curves
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Metrics over epochs
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

Save to `$EXPERIMENT_DIR/logs/0-run/baseline_summary.json` (update the existing file to include tuning results):

Include:
- Which hyperparameters were tuned
- Results for each configuration
- Best configuration found
- Comparison with Stage 1 baseline

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
2. Propose a **novel improvement** inspired by the research idea:
   - New loss functions or regularization techniques
   - Architectural modifications
   - Data augmentation strategies
   - Ensemble methods
   - Novel training procedures
3. Add a third dataset
4. Execute and compare with previous stages
5. View plots to verify improvements

### Save Stage 3 Summary

Save to `$EXPERIMENT_DIR/logs/0-run/research_summary.json`:

```json
{
  "best_node": {
    "overall_plan": "<description of novel approach>",
    "analysis": "<analysis comparing to baseline>",
    "metric": "<primary metric value>",
    "code": "<the working Python code>",
    "plot_code": "<plotting code>",
    "plot_analyses": ["<description of each plot>"],
    "vlm_feedback_summary": "<summary of improvements>",
    "exp_results_npy_files": ["experiment_results/stage3/experiment_data.npy"],
    "datasets_successfully_tested": ["dataset1", "dataset2", "dataset3"]
  }
}
```

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
2. **Remove or disable ONE specific component** to measure its contribution:
   - Remove the novel technique from Stage 3 (baseline comparison)
   - Disable data augmentation
   - Simplify the architecture
   - Remove regularization
   - Use different loss function
3. Execute with the same 3 datasets
4. Compare with full Stage 3 model
5. Record the performance delta

### Save Stage 4 Summary

Save to `$EXPERIMENT_DIR/logs/0-run/ablation_summary.json`:

```json
{
  "ablation_1_name": {
    "overall_plan": "<what was ablated>",
    "ablation_name": "<component removed>",
    "analysis": "<impact analysis>",
    "metric": "<metric values>",
    "code": "<ablation code>",
    "exp_results_npy_files": ["experiment_results/stage4_ablation1/experiment_data.npy"],
    "datasets_successfully_tested": ["dataset1", "dataset2", "dataset3"]
  },
  "ablation_2_name": { ... }
}
```

Copy results:
```bash
mkdir -p $EXPERIMENT_DIR/logs/0-run/experiment_results/stage4_ablation1
cp $EXPERIMENT_DIR/workspace/working/experiment_data.npy $EXPERIMENT_DIR/logs/0-run/experiment_results/stage4_ablation1/
# Repeat for each ablation
```

## Step 6: Plot Aggregation

After all stages, create a comprehensive plot aggregation script that:

1. Loads ALL `.npy` files from all stages
2. Creates publication-ready comparison plots:
   - **Cross-stage comparison**: Bar chart of primary metric across stages
   - **Training curves**: Overlaid loss/metric curves for all configurations
   - **Per-dataset comparison**: Performance across datasets
   - **Ablation chart**: Bar chart showing component contributions
3. Saves up to 12 figures in `$EXPERIMENT_DIR/figures/`

Guidelines:
- DPI: 300
- Remove top/right spines
- Clear labels (no underscores, use spaces)
- Legends always visible
- Each plot in its own try-except block
- Combine related plots into subplots (up to 3 per figure)

Save the aggregation script as `$EXPERIMENT_DIR/auto_plot_aggregator.py`.

View all generated plots with the Read tool to verify quality.

## Step 7: Save Research Idea Markdown

Create `$EXPERIMENT_DIR/research_idea.md` that is a copy of `$EXPERIMENT_DIR/idea.md`.

## Step 8: Report Results

Summarize for the user:

1. **Stage 1 Results**: Baseline performance on initial dataset
2. **Stage 2 Results**: Best hyperparameter configuration and improvement
3. **Stage 3 Results**: Novel improvements and their impact across 3 datasets
4. **Stage 4 Results**: Ablation study findings — which components matter most
5. **Output directory**: Full path to experiment results
6. **Figures generated**: List of publication-ready plots
7. **Next step suggestion**: Run `/ai-scientist-writeup $EXPERIMENT_DIR` to generate a paper

## Error Handling

- If a stage completely fails (all attempts produce errors), log the failure and proceed to the next stage with the best available code from previous stages
- If GPU is not available, fall back to CPU with a warning
- If a dataset download fails, try an alternative dataset
- Never spend more than 5 attempts debugging the same error — try a different approach

## Parallelism with Agent Tool

For Stage 4 ablations, you MAY use the **Agent tool** to run multiple ablation experiments in parallel. Each agent should:
1. Receive the base Stage 3 code
2. Implement a specific ablation
3. Execute and return results

This speeds up the ablation stage significantly.
