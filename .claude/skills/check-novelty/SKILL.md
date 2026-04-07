---
name: check-novelty
description: "Check if a research idea is novel by searching existing literature. Use: /check-novelty <idea description or path to idea.json>"
allowed-tools: Bash Read WebSearch
---

# Novelty Check

Verify that a research idea hasn't been done before.

Argument: $ARGUMENTS (idea description string or path to idea.json).

## Process

1. If given a file path, read the idea JSON and extract the hypothesis and title
2. Construct 3-5 targeted search queries:
   - The exact hypothesis
   - The method name + domain
   - Key technical terms
   - Alternative phrasings

3. Search each query:
   ```bash
   python lib/semantic_scholar.py "<query>"
   ```
   Also use WebSearch for very recent work.

4. For each result, assess:
   - **Direct overlap**: Same hypothesis tested in same setting → NOT NOVEL
   - **Partial overlap**: Similar idea but different setting or method → POTENTIALLY NOVEL (needs differentiation)
   - **Tangential**: Related topic but different hypothesis → NOVEL

## Output

Report:
- **Novelty assessment**: Novel / Partially Novel / Not Novel
- **Most similar papers**: List with explanation of similarity
- **Differentiation suggestions**: How to make the idea more distinct (if partially novel)
- **Key references to cite**: Papers that must be discussed in related work
