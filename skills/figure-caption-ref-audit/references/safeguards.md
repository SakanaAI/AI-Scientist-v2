# Safeguards

## Privacy
- Do not upload or share the PDF, extracted images, or figref text unless the user explicitly asks.

## No hallucinated results
- Do not infer experimental outcomes beyond what is visible in figures and written in the text.
- If a figure is ambiguous, note uncertainty and recommend clarifying edits.

## Safety of extraction
- The extractor is read-only and writes only under `--out-dir`.
- By default it will not overwrite existing outputs unless `--overwrite` is passed.

## AI-use disclosure (if applicable)
- If your workflow uses tools or code with mandatory AI-use disclosure requirements (e.g., AI-Scientist-v2), comply with that license and with your venue's policies when submitting manuscripts.
