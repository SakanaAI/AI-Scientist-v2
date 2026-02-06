# Plot Guidelines (Paper-Ready)

Use these guidelines when creating the final `figures/` set.

## Grounding and correctness
- Load real data from files (`.npy`, JSON, CSV). Do not fabricate numbers.
- If you need a single number, copy it from an existing summary file and cite its path.

## Figure budgeting
- Prefer ~6-12 figures total for a full paper (adjust to venue).
- Combine related plots into subplots (e.g., 1x2 or 1x3).
- Keep only the most informative figures in the main text; move extras to appendix.

## Style and readability
- Use `dpi=300` for saved PNGs.
- Include axis labels, titles, and a legend where needed.
- Avoid underscores in labels; replace with spaces.
- Keep legends visible and non-overlapping.
- Consider removing top/right spines for a cleaner look (matplotlib style).

## Robust scripting
- Put each figure in its own `try/except` block so one failure doesn't block others.
- Validate file existence before loading.
- Use safe defaults and deterministic filenames.
