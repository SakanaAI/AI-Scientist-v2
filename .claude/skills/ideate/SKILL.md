---
name: ideate
description: "Generate research ideas for a given topic. Searches literature, checks novelty, and outputs structured ideas. Use: /ideate <topic or path to topic.md>"
allowed-tools: Bash Read Write Edit Glob Grep WebSearch
---

# Research Ideation

Generate publication-quality research ideas. Argument: $ARGUMENTS (topic string or path to topic .md file).

## Process

### 1. Understand the Topic

If $ARGUMENTS is a file path, read it for the full topic description.
Otherwise, treat $ARGUMENTS as the topic string.

Also read `research.yaml` for additional context (domain, venue, constraints).

### 2. Read Methodology

Read `methodology/ideation-guide.md` for the complete ideation protocol.

### 3. Literature Search

Before generating ideas, search for existing work:
- Search the core topic on the web or via `python lib/semantic_scholar.py "<query>"`
- Search for competing approaches and recent advances
- Note key papers, gaps, and opportunities

### 4. Generate Ideas

For each idea (generate `ideation.num_ideas` from `research.yaml`, default 3):

1. Start from a simple, elegant observation or hypothesis
2. Ensure it's novel (not a trivial extension of existing work)
3. Ensure it's feasible with academic-lab resources
4. Make experiments specific and measurable
5. Write a complete abstract (~250 words)

### 5. Novelty Check

For each generated idea:
- Search Semantic Scholar for the exact hypothesis
- Search for similar approaches in the same domain
- If a very similar paper exists, modify the idea or discard it

### 6. Reflection and Refinement

For each idea, ask:
- Is this truly novel?
- Is the hypothesis testable with a clear success criterion?
- Are experiments specific enough to implement without questions?
- Would this be interesting to reviewers at the target venue?

Refine based on these reflections.

### 7. Output

Save ideas as a JSON array following `templates/idea-schema.json`:
```
experiments/<run>/idea.json
```

Also save a human-readable summary in the research log.

## Quality Bar

Each idea must have:
- A clear, falsifiable hypothesis
- At least 3 specific experiments with exact datasets and metrics
- Honest risk factors and limitations
- References to actual papers found during literature search
