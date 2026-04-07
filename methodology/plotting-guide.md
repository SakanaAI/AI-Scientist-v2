# Plotting Guide

How to create publication-quality figures. Distilled from AI-Scientist-v2 plotting pipeline.

## General Principles

- Only create plots where the data is best presented visually, not as a table
- Each figure should convey one clear message
- Aim for ~6-12 figures total (main paper + appendix)
- Aggregate related plots into subplots (up to 3 per row using `fig, axes = plt.subplots(1, 3)`)
- Use existing .npy data files — never hallucinate data

## Visual Standards

- DPI: 300 for final figures
- Remove top and right spines: `ax.spines['top'].set_visible(False)` etc.
- Font size larger than default for labels and titles (at least 12pt)
- No underscores in labels (use "loss vs epoch" not "loss_vs_epoch")
- Legends must always be visible and not overlap data
- Use colorblind-friendly palettes (e.g., seaborn's default or tab10)
- Informative axis labels with units where applicable

## Figure Types (when to use each)

- **Line plots**: Scaling curves, training progress, trends over a continuous variable
- **Bar plots**: Comparing discrete conditions (only when differences are visually clear)
- **Scatter plots**: Correlation between two variables
- **Heatmaps**: Confusion matrices, hyperparameter grids
- **Box/violin plots**: Distribution comparisons across conditions
- **Image grids**: Showing example inputs/outputs (select diverse categories)

## Code Structure

Write each plot in a separate try-except block so one failure doesn't affect others:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

# Figure 1: TTA Scaling Curves
try:
    data = np.load("results/tta_scaling.npy", allow_pickle=True).item()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, dataset in zip(axes, ["PathMNIST", "DermaMNIST", "BloodMNIST"]):
        # ... plot scaling curves ...
        ax.set_xlabel("Number of TTA Views")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(dataset)
    plt.tight_layout()
    plt.savefig("figures/tta_scaling_curves.pdf", dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error generating Figure 1: {e}")

# Figure 2: ...
```

## What to Plot for Each Experiment Phase

### From Baselines
- Model comparison bar chart (accuracy across models)
- Training curves if relevant

### From Core Hypothesis
- Main result comparison (proposed vs baseline)
- The key relationship being studied (e.g., accuracy vs TTA views)

### From Extended Experiments
- Scaling curves across conditions
- Cross-dataset comparison
- Compute-accuracy Pareto frontier

### From Ablations
- Ablation comparison (full method vs each component removed)
- Sensitivity analysis plots

## File Naming

- `figures/main_result.pdf` — The main finding
- `figures/scaling_curves.pdf` — Scaling behavior
- `figures/ablation.pdf` — Ablation study
- `figures/appendix_*.pdf` — Additional figures for appendix
- Save as PDF (vector) for main figures, PNG for image examples

## Quality Checklist

Before finalizing figures:
- [ ] All axes labeled with units
- [ ] Legends present and not overlapping data
- [ ] Error bars or confidence intervals shown where applicable
- [ ] Font sizes readable at paper column width
- [ ] No unnecessary decorations (gridlines, 3D effects, etc.)
- [ ] Consistent color scheme across all figures
- [ ] Figure tells its story without needing to read the caption
