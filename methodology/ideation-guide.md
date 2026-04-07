# Ideation Guide

How to generate publication-quality research ideas. Distilled from AI-Scientist-v2 ideation pipeline.

## Role

You are an experienced AI researcher proposing high-impact research ideas resembling exciting grant proposals. Be creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis.

## Process

1. **Read the topic description** from `ideas/` or `research.yaml` to understand the research area
2. **Search existing literature** via Semantic Scholar or web search to understand the state of the art
3. **Generate candidate ideas** that are novel, feasible, and impactful
4. **Check novelty** of each idea against existing literature
5. **Refine through reflection** — iterate on each idea considering quality, novelty, feasibility
6. **Output structured ideas** in the format defined by `templates/idea-schema.json`

## Idea Quality Criteria

- **Novelty**: The idea must clearly differ from existing work. Search literature before finalizing.
- **Feasibility**: Must be implementable with academic-lab resources (single GPU, standard datasets).
- **Simplicity**: The best ideas are simple and elegant. Avoid overcomplication.
- **Specificity**: Experiments must be precisely described — exact datasets, metrics, baselines.
- **Impact**: Should lead to a publishable paper at a top ML venue.

## Idea Structure

Each idea must include (see `templates/idea-schema.json`):

- **Name**: Short snake_case identifier (e.g., `tta_scaling_medimg`)
- **Title**: Publication-ready paper title
- **Short Hypothesis**: One-sentence hypothesis statement. Clarify why this direction is needed and that there aren't simpler alternatives.
- **Related Work**: How this differs from prior work. Must reference actual papers found via literature search.
- **Abstract**: Conference-style abstract (~250 words)
- **Experiments**: Specific, numbered experiments with exact details (datasets, metrics, baselines, hyperparameters)
- **Risk Factors and Limitations**: Honest assessment of what could go wrong

## Literature Search Protocol

Before finalizing any idea:
1. Search for the core concept (e.g., "test-time augmentation scaling")
2. Search for the application domain (e.g., "medical image classification MedMNIST")
3. Search for competing approaches (e.g., "foundation models vs simple baselines medical imaging")
4. Verify the idea is not a trivial extension of existing work
5. Incorporate relevant citations into the Related Work section

## Reflection Prompts

After generating each idea, ask yourself:
- Is this truly novel, or has someone already done this?
- Is the hypothesis testable with a clear success criterion?
- Are the experiments specific enough that someone could implement them without asking questions?
- Could a simpler approach achieve the same goal?
- Would this be interesting to reviewers at the target venue?

## Output

Save generated ideas as JSON arrays in the experiment directory, following `templates/idea-schema.json`.
See `fewshot-examples/ideas/` for examples of well-formed ideas.
