# Safeguards

## Safety posture
- Do not delete anything by default.
- Avoid out-of-memory by using `mmap_mode='r'` and element limits for stats.
- Do not reference files that do not exist.
- Do not use network access in plot scripts.

## Cleaning behavior
- Only clean output folders when an explicit `--clean` flag is provided.
- If cleaning is enabled, only delete the figures directory inside the current working directory.

## AI-use disclosure (if applicable)
- If your workflow uses tools or code with mandatory AI-use disclosure requirements (e.g., AI-Scientist-v2), comply with that license and with your venue's policies when submitting manuscripts.
