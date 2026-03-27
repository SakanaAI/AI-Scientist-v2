---
name: ai-scientist-ideate
description: Generate research ideas for AI Scientist v2 — uses Claude Code directly with Semantic Scholar for literature search
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch
argument-hint: <workshop-topic-file.md> [--max-num-generations N] [--num-reflections N]
---

# AI Scientist v2 — Idea Generation (Claude Code Native)

You are an experienced AI researcher generating high-impact research ideas. This skill runs within Claude Code using Semantic Scholar for literature search.

## Environment

- **S2_API_KEY**: Required for Semantic Scholar API access. Verify it is set before proceeding:
  ```bash
  echo "S2_API_KEY: ${S2_API_KEY:+set}"
  ```
  If not set, warn the user and fall back to WebSearch.

## Arguments

- **Positional arg 1** (`$ARGUMENTS` first word): Path to workshop topic markdown file
- `--max-num-generations N`: Number of ideas to generate (default: 5)
- `--num-reflections N`: Refinement rounds per idea (default: 3)

If no file is provided, ask the user for a research topic.

## Step 1: Load the Workshop Topic

Read the workshop topic markdown file. It should contain:
- Title
- Keywords
- TL;DR
- Abstract describing the research area

If the user provides a topic description but no file exists, create one at `ai_scientist/ideas/<topic_name>.md` with this structure:

```markdown
# Title: <Workshop/Topic Title>

## Keywords
keyword1, keyword2, keyword3

## TL;DR
<One-sentence summary>

## Abstract
<Detailed description ~250 words>
```

## Step 2: Generate Ideas Iteratively

For each idea (up to max-num-generations), follow this process:

### 2a. Literature Search

Use **Semantic Scholar** to find relevant papers. Run searches via the existing Python tool:

```bash
cd /home/user/AI-Scientist-v2
python -c "
from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
tool = SemanticScholarSearchTool(max_results=10)
result = tool.use_tool('YOUR SEARCH QUERY HERE')
print(result)
"
```

For each idea, perform at least 2-3 searches to ground it in existing literature:
- The main topic keywords
- Specific sub-areas relevant to the idea you're developing
- Recent advances and open problems

If Semantic Scholar is unavailable (no API key or rate limited), fall back to **WebSearch**.

### 2b. Idea Generation

As an experienced AI researcher, generate a research idea that:
- Stems from a simple, elegant question or hypothesis
- Is clearly novel and distinct from existing literature
- Is feasible for an academic lab (no massive compute requirements)
- Would be publishable at a top ML conference
- Includes specific, testable experiments

### 2c. Reflection & Refinement

For each reflection round:
1. Critically evaluate your idea for quality, novelty, and feasibility
2. Search for additional related work if needed
3. Refine the hypothesis, experiments, and framing
4. Ensure the idea isn't a trivial extension of existing work

### 2d. Finalize Each Idea

Each idea must be a JSON object with exactly these fields:

```json
{
  "Name": "lowercase_with_underscores",
  "Title": "Catchy and informative title",
  "Short Hypothesis": "Concise statement of the main hypothesis. Clarify the need for this direction and why it's the best setting to investigate.",
  "Related Work": "Discussion of relevant related work and how this proposal distinguishes from it.",
  "Abstract": "Conference-format abstract, approximately 250 words.",
  "Experiments": [
    "Experiment 1: Specific description with metrics...",
    "Experiment 2: ..."
  ],
  "Risk Factors and Limitations": [
    "Risk 1: ...",
    "Risk 2: ..."
  ]
}
```

**Important**: Each new idea should be distinct from all previously generated ideas. Avoid duplicating themes or approaches.

## Step 3: Save Output

Save all ideas as a JSON array to a file alongside the input markdown file with the same base name but `.json` extension.

For example: `ai_scientist/ideas/my_topic.md` → `ai_scientist/ideas/my_topic.json`

```python
# The output file is a JSON array of idea objects:
[
  { "Name": "idea_1", "Title": "...", ... },
  { "Name": "idea_2", "Title": "...", ... }
]
```

Write the file using the Write tool.

## Step 4: Report Results

After generating all ideas, list them with:
- Index number
- Title
- One-line hypothesis summary

Tell the user they can run experiments with:
```
/ai-scientist-experiment ai_scientist/ideas/<topic>.json --idea_idx <N>
```

## Guidelines

- Be very creative and think out of the box
- Each proposal should stem from a simple and elegant question
- Proposals should explore new possibilities or challenge existing assumptions
- Ensure feasibility — no resources beyond what an academic lab could afford
- Perform literature searches to ensure novelty
- Keep experiments specific: detail exact algorithmic changes and evaluation metrics
