---
name: literature-search
description: "Search academic literature via Semantic Scholar and web. Returns formatted results with BibTeX. Use: /literature-search <query>"
allowed-tools: Bash Read WebSearch
---

# Literature Search

Search for academic papers related to a query.

Argument: $ARGUMENTS (search query string).

## Search Strategy

1. **Semantic Scholar API** (primary):
   ```bash
   python lib/semantic_scholar.py "$ARGUMENTS"
   ```

2. **Web search** (supplementary): Use WebSearch for recent papers, preprints, or when S2 returns few results.

3. **Combine results**: Deduplicate and rank by relevance and citation count.

## Output Format

For each paper found, provide:
- Title
- Authors
- Venue and year
- Citation count
- Brief abstract summary
- BibTeX entry (if available from S2)

## Usage Tips

- Use specific queries: "test-time augmentation medical imaging" not just "augmentation"
- Search for both the method and the application domain
- Look for recent surveys to get a broad overview
- Check citation counts to identify seminal papers
