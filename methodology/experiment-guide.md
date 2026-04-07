# Experiment Guide

How to design and execute ML experiments for a research paper. Distilled from AI-Scientist-v2 tree search and agent manager.

## Experiment Phases

Follow this structured progression. Each phase builds on the previous one.

### Phase 1: Baseline Implementation
**Goal**: Establish reference performance with simple, well-understood methods.

- Implement the simplest reasonable baselines (e.g., logistic regression, standard training, random baseline)
- Use the datasets and metrics specified in the idea
- Run with at least 3 random seeds
- Record: accuracy, AUC, F1, or task-appropriate metrics + standard deviation
- Save results as JSON and .npy files
- Create at least one diagnostic plot (e.g., training curves, confusion matrix)
- **Exit criterion**: All baselines produce reasonable, reproducible results

### Phase 2: Core Hypothesis Implementation
**Goal**: Implement and test the main research idea.

- Start with the simplest possible implementation of the hypothesis
- Get it running end-to-end before optimizing
- Compare directly against Phase 1 baselines using the same evaluation protocol
- If results are worse than baseline, analyze why before iterating
- Try up to 5 different implementation approaches if needed
- **Exit criterion**: Have a working implementation that either confirms or challenges the hypothesis

### Phase 3: Optimization and Extended Experiments
**Goal**: Strengthen results through tuning and broader evaluation.

- Tune hyperparameters systematically (grid search or informed search)
- Test across multiple datasets (at least 2-3) to show generality
- Test across different conditions specified in the idea
- Look for scaling laws, trends, and patterns
- **Exit criterion**: Results are robust across multiple settings

### Phase 4: Ablation Studies
**Goal**: Understand which components of the method are important.

- For each key component, run an experiment removing or varying it
- Compare ablation results against the full method AND baseline
- If a component doesn't help, acknowledge this honestly
- **Exit criterion**: Every claimed contribution has ablation evidence

## Code Organization

```
experiments/<run>/code/
  baseline_v1.py          # Phase 1: initial baseline
  baseline_v2.py          # Phase 1: refined baseline (if needed)
  hypothesis_v1.py        # Phase 2: first implementation attempt
  hypothesis_v2.py        # Phase 2: improved version
  full_experiment.py      # Phase 3: complete experiment with all conditions
  ablation_study.py       # Phase 4: ablation experiments
```

Save every version of the code. Never overwrite previous versions.

## Results Organization

```
experiments/<run>/results/
  baseline.json           # Phase 1 summary
  hypothesis.json         # Phase 2 summary
  extended.json           # Phase 3 summary
  ablation.json           # Phase 4 summary
  *.npy                   # Raw numerical results for plotting
```

Each JSON summary should include:
- Method name and description
- Hyperparameters used
- Results per dataset, per seed (mean + std)
- Paths to .npy files with full results
- Code file used to generate results

## Debugging Failed Experiments

When an experiment fails:
1. Read the full error traceback
2. Check data loading — is the dataset accessible and in expected format?
3. Check dimensions — do tensor shapes match throughout the pipeline?
4. Check for NaN/Inf — add gradient clipping or reduce learning rate
5. Check memory — reduce batch size or model size
6. If the same error persists 3 times, try a fundamentally different approach

## Research Log

Maintain `research-log.md` throughout experiments:
```markdown
## Phase 1: Baselines (YYYY-MM-DD HH:MM)
- Implemented logistic regression baseline: 73.2% +/- 1.1% on PathMNIST
- ResNet-18 baseline: 89.1% +/- 0.5% on PathMNIST
- **Decision**: Use ResNet-18 as primary baseline

## Phase 2: Hypothesis (YYYY-MM-DD HH:MM)
- First attempt: TTA with 10 views → 90.3% (+1.2% over baseline)
- Bug found: augmentation was applied to training too. Fixed.
- ...
```

## Checkpointing

Update `checkpoint.json` after every significant step:
```json
{
  "phase": "2_hypothesis",
  "step": "implementation_v2",
  "status": "running",
  "last_updated": "2026-04-06T15:30:00",
  "baselines_complete": true,
  "hypothesis_attempts": 2,
  "results_so_far": {
    "baseline_accuracy": 0.891,
    "hypothesis_v2_accuracy": 0.903
  }
}
```

## Common Pitfalls

- Don't skip baselines — reviewers always ask "how does this compare to X?"
- Don't use only one random seed — results should be reproducible
- Don't optimize only for the best metric — report honestly
- Don't ignore failed experiments — negative results inform the paper
- Don't run experiments without saving code — you must be able to reproduce
