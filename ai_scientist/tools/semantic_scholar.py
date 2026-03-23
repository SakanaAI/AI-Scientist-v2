import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import backoff

from ai_scientist.tools.base_tool import BaseTool
import logging
logger = logging.getLogger(__name__)

# Official Semantic Scholar Graph API default; override for mirrors (e.g. Ai4Scholar).
_DEFAULT_GRAPH_API_BASE = "https://api.semanticscholar.org"


def get_semantic_scholar_api_base() -> str:
    """Base URL without trailing slash (e.g. https://api.semanticscholar.org or https://ai4scholar.net)."""
    base = (
        os.getenv("S2_API_BASE_URL")
        or os.getenv("SEMANTIC_SCHOLAR_API_BASE")
        or os.getenv("AI4SCHOLAR_API_BASE")
        or _DEFAULT_GRAPH_API_BASE
    )
    return base.rstrip("/")


def build_paper_search_url(base: Optional[str] = None) -> str:
    b = (base or get_semantic_scholar_api_base()).rstrip("/")
    return f"{b}/graph/v1/paper/search"


def build_auth_headers(api_key: Optional[str], base_url: Optional[str] = None) -> Dict[str, str]:
    """
    Official S2 uses ``X-API-KEY``. Ai4Scholar (and similar proxies) use ``Authorization: Bearer <token>``.

    Set ``S2_AUTH_MODE`` to ``bearer``, ``x-api-key``, or ``auto`` (default).
    In ``auto`` mode, hosts containing ``ai4scholar`` use Bearer; otherwise ``X-API-KEY``.
    """
    if not api_key:
        return {}
    base = base_url or get_semantic_scholar_api_base()
    mode = (os.getenv("S2_AUTH_MODE") or "auto").strip().lower()
    host = (urlparse(base).hostname or base).lower()
    if mode == "bearer" or (mode == "auto" and "ai4scholar" in host):
        return {"Authorization": f"Bearer {api_key}"}
    if mode == "x-api-key":
        return {"X-API-KEY": api_key}
    # auto, non-ai4scholar → official Semantic Scholar
    return {"X-API-KEY": api_key}


def on_backoff(details: Dict) -> None:
    logger.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY") or os.getenv("AI4SCHOLAR_API_KEY")
        self._api_base = get_semantic_scholar_api_base()
        if not self.S2_API_KEY:
            warnings.warn(
                "No literature API key found (S2_API_KEY or AI4SCHOLAR_API_KEY). "
                "Requests may hit strict rate limits or fail. "
                "For Semantic Scholar use S2_API_KEY; for Ai4Scholar set S2_API_BASE_URL=https://ai4scholar.net and your Bearer token in S2_API_KEY."
            )

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None

        headers = build_auth_headers(self.S2_API_KEY, self._api_base)
        url = build_paper_search_url(self._api_base)

        rsp = requests.get(
            url,
            headers=headers,
            params={
                "query": query,
                "limit": self.max_results,
                "fields": "title,authors,venue,year,abstract,citationCount",
            },
        )
        logger.info(f"Response Status Code: {rsp.status_code}")
        logger.info(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        total = results.get("total", 0)
        if total == 0:
            return None

        papers = results.get("data", [])
        # Sort papers by citationCount in descending order
        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    api_key = os.getenv("S2_API_KEY") or os.getenv("AI4SCHOLAR_API_KEY")
    api_base = get_semantic_scholar_api_base()
    headers = build_auth_headers(api_key, api_base)
    if not api_key:
        warnings.warn(
            "No literature API key found (S2_API_KEY or AI4SCHOLAR_API_KEY). "
            "Requests may hit strict rate limits."
        )

    if not query:
        return None

    url = build_paper_search_url(api_base)
    rsp = requests.get(
        url,
        headers=headers,
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    logger.info(f"Response Status Code: {rsp.status_code}")
    logger.info(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers
