"""
Semantic Scholar API client.
Pure HTTP client — no LLM dependencies.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import requests


S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,venue,year,abstract,citationStyles,citationCount"


def search_papers(
    query: str,
    max_results: int = 10,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[List[Dict]]:
    """Search Semantic Scholar for papers matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        api_key: Optional S2 API key. Falls back to S2_API_KEY env var.
        max_retries: Number of retries on transient failures.

    Returns:
        List of paper dicts sorted by citation count, or None if no results.
    """
    if not query:
        return None

    api_key = api_key or os.getenv("S2_API_KEY")
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
    else:
        warnings.warn("No S2_API_KEY set. Requests may be rate-limited.")

    for attempt in range(max_retries):
        try:
            response = requests.get(
                S2_API_URL,
                headers=headers,
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": S2_FIELDS,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("total", 0) == 0:
                return None

            papers = data.get("data", [])
            papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
            return papers

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    return None


def format_papers(papers: List[Dict]) -> str:
    """Format a list of papers into a readable string."""
    lines = []
    for i, paper in enumerate(papers):
        authors = ", ".join(
            a.get("name", "Unknown") for a in paper.get("authors", [])
        )
        lines.append(
            f"{i+1}. {paper.get('title', 'Unknown')}\n"
            f"   Authors: {authors}\n"
            f"   Venue: {paper.get('venue', 'N/A')}, {paper.get('year', 'N/A')}\n"
            f"   Citations: {paper.get('citationCount', 'N/A')}\n"
            f"   Abstract: {paper.get('abstract', 'N/A')[:200]}..."
        )
    return "\n\n".join(lines)


def get_bibtex(paper: Dict) -> Optional[str]:
    """Extract BibTeX citation from a paper dict if available."""
    styles = paper.get("citationStyles", {})
    return styles.get("bibtex") if styles else None


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "test-time augmentation"
    print(f"Searching for: {query}")
    papers = search_papers(query)
    if papers:
        print(format_papers(papers))
    else:
        print("No results found.")
