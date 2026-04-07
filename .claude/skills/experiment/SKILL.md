---
name: experiment
description: "Execute ML experiments for a research idea. Writes code, runs experiments, iterates on results. Use: /experiment <path to idea.json or experiment dir>"
allowed-tools: Bash Read Write Edit Glob Grep WebSearch
---

# Experiment Execution

Run ML experiments for a research idea. This is the most critical skill — it replaces the BFTS tree search with methodology-guided exploration.

Argument: $ARGUMENTS (path to idea.json or experiment directory).

## Setup

1. Read the idea from the provided path
2. Read `research.yaml` for constraints (runtime, seeds, GPU, datasets)
3. Read `methodology/experiment-guide.md` for the full experiment protocol
4. Set up the experiment directory structure if not already created:
   ```
   experiments/<run>/code/
   experiments/<run>/results/
   ```
5. Initialize or resume from `checkpoint.json`

## Phase 1: Baseline Implementation

**Goal**: Establish reference performance.

1. Read the starter code from `research.yaml → experiment.starter_code` if available
2. Implement the simplest reasonable baselines:
   - For classification: logistic regression, small MLP, standard CNN
   - Use the datasets specified in the idea
3. Run with `num_seeds` seeds (from `research.yaml`)
4. Save code as `code/baseline_v1.py`
5. Save results as `results/baseline.json` with:
   - Per-seed metrics (accuracy, AUC, etc.)
   - Mean and standard deviation
6. Create at least one diagnostic plot
7. Log to `research-log.md`
8. Update `checkpoint.json`: `{"phase": "1_baseline", "status": "complete"}`

**Exit criterion**: All baselines produce reasonable, reproducible results.

## Phase 2: Core Hypothesis

**Goal**: Implement and test the main research idea.

1. Write the simplest implementation of the hypothesis
2. Get it running end-to-end before optimizing
3. Compare directly against baselines (same seeds, same evaluation)
4. Save code as `code/hypothesis_v1.py`

If results are worse than baseline:
- Analyze the output to understand why
- Try a different implementation approach
- Save each attempt as `code/hypothesis_v2.py`, `v3`, etc.
- Maximum 5 attempts before reassessing the hypothesis

5. Save results as `results/hypothesis.json`
6. Log decisions and results to `research-log.md`
7. Update `checkpoint.json`: `{"phase": "2_hypothesis", "status": "complete"}`

**Exit criterion**: Working implementation that confirms or challenges the hypothesis.

## Phase 3: Optimization and Extended Experiments

**Goal**: Strengthen results through broader evaluation.

1. Tune key hyperparameters (learning rate, augmentation strength, etc.)
2. Test across multiple datasets (at least 2-3 from the idea)
3. Test across different conditions specified in the idea
4. Look for scaling laws and trends
5. Save as `code/full_experiment.py` and `results/extended.json`
6. Save numerical results as .npy files for plotting
7. Update `checkpoint.json`

**Exit criterion**: Results are robust across multiple settings.

## Phase 4: Ablation Studies

**Goal**: Understand which components matter.

1. Identify key components of the method
2. For each component, run experiments removing or varying it
3. Compare ablations against full method AND baseline
4. Save as `code/ablation_study.py` and `results/ablation.json`
5. Update `checkpoint.json`

**Exit criterion**: Every claimed contribution has ablation evidence.

## Throughout All Phases

- **Never delete or overwrite** previous code versions
- **Update research-log.md** after every significant result
- **Update checkpoint.json** after every phase transition
- **Save .npy files** with raw numerical data for plotting
- If an experiment **fails 3 times** with the same error, try a fundamentally different approach
- If a single run takes **more than 30 minutes**, consider running it in the background:
  ```bash
  nohup python code/experiment.py > results/run.log 2>&1 &
  echo "PID: $!"
  ```
  Then monitor with `tail -f results/run.log`

## Code Quality Standards

- Set random seeds at the top of every script:
  ```python
  import random, numpy as np, torch
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
  ```
- Save all hyperparameters in the results JSON
- Use `os.makedirs(dir, exist_ok=True)` before saving
- Handle CUDA out-of-memory gracefully (reduce batch size)
- Print progress during long training loops
