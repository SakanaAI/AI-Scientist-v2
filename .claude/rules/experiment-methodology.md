# Experiment Methodology Standards

When running experiments, always follow these principles:

## Reproducibility
- Set and log random seeds for all sources of randomness (torch, numpy, random, CUDA)
- Save exact hyperparameters in a JSON/YAML file alongside results
- Save all code versions with descriptive filenames in the `code/` directory
- Record the exact command used to run each experiment

## Baselines
- Every research claim must be compared against at least one baseline
- Baselines should be simple, well-understood methods (e.g., logistic regression, standard training)
- Run baselines with the same seeds and evaluation protocol as the proposed method

## Ablations
- For every proposed component, run an ablation removing or varying it
- Ablation results must be reported alongside main results
- If a component doesn't improve results in ablation, acknowledge this honestly

## Statistical Rigor
- Run all experiments with at least 3 different random seeds
- Report mean and standard deviation, not just best run
- Use appropriate metrics for the task (accuracy, AUC, F1, ECE as relevant)
- Don't cherry-pick results; report all runs

## Experiment Phases
Follow this progression for thorough research:
1. **Baseline**: Get simple methods working first. Establish reference performance.
2. **Core hypothesis**: Implement and test the main research idea.
3. **Optimization**: Tune hyperparameters, try variations.
4. **Ablation**: Systematically remove/vary components.
5. **Analysis**: Look for patterns, scaling laws, failure modes.

## Failure Handling
- If an experiment fails 3 times with the same error, try a fundamentally different approach
- Log all failures in research-log.md with analysis of what went wrong
- Negative results are valuable - document them thoroughly
