# Title: Toward a Reproducible Benchmark for AI-Assisted Paper Review: Advancing Ai-Review and the Broader AI Reviewing Ecosystem

## Keywords
AI-assisted peer review, benchmark, VLM review, review quality evaluation, robustness, prompt injection, scientific NLP

## TL;DR
Build a rigorous, open, and reproducible benchmark around the Ai-Review repository to evaluate review quality, faithfulness, robustness, and fairness across LLM/VLM settings, enabling measurable progress for both the project and the AI reviewing research community.

## Abstract
AI-assisted manuscript review systems are rapidly evolving, but the field still lacks standardized benchmarks that can reliably measure review quality, reliability, and safety in realistic conditions. The Ai-Review project (https://github.com/NeuroDong/Ai-Review) provides a practical and timely foundation for this effort, with support for structured review prompts, VLM-based PDF understanding, and prompt engineering workflows. This topic calls for a benchmark-centered research program that turns Ai-Review into a testbed for systematic evaluation and future improvement.

The benchmark should include diverse manuscript inputs (LaTeX, PDF, Word), subject areas, quality levels, and writing styles, as well as controlled stress tests such as noisy OCR extraction, figure-caption mismatch, math-heavy content, and prompt-injection adversarial cases. Evaluation should go beyond surface-level helpfulness and measure dimensions including factual faithfulness to source text, coverage of key contribution/limitation points, calibration of confidence, actionable suggestions, consistency across runs, and sensitivity to formatting artifacts. We further encourage protocol-level rigor: fixed splits, blinded human preference studies, automatic metrics with known limitations, and transparent ablations over prompting strategies (e.g., chain-of-thought variants, few-shot examples, and VLM-specific prompts).

The ultimate goal is twofold. First, provide concrete guidance for improving Ai-Review itself: which components fail, where VLM adds value, and what design choices most improve review usefulness and trustworthiness. Second, contribute a reusable benchmark paradigm for the broader AI peer-review community, enabling apples-to-apples comparisons across systems and accelerating responsible progress toward dependable AI reviewers.
