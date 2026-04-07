# Safety and Resource Limits

## Budget
- Respect the constraints in research.yaml (max runtime, GPU requirements)
- If experiments are taking significantly longer than expected, stop and reassess

## Forbidden Operations in Experiment Code
- Do not install packages within experiment scripts (use requirements.txt)
- Do not download large datasets without checking available disk space first
- Do not modify files outside the experiments/ directory from experiment code

## Data Safety
- Never delete experiment results or research logs
- Always checkpoint before long-running operations
- Save intermediate results, not just final metrics

## Code Execution
- Use timeouts for all experiment runs
- Monitor GPU memory usage before launching large models
- Kill orphan processes after failed experiments
