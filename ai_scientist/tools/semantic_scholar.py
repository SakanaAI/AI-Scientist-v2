import os
import time
import warnings
from typing import Dict, List, Optional, Union

from ai_scientist.tools.base_tool import BaseTool

# v3: Backend mode. Set AI_SCIENTIST_BACKEND=legacy to use direct HTTP API calls.
_BACKEND_MODE = os.environ.get("AI_SCIENTIST_BACKEND", "agent")

if _BACKEND_MODE == "legacy":
    import backoff
    import requests

    def on_backoff(details: Dict) -> None:
        print(
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

        if _BACKEND_MODE == "legacy":
            self.S2_API_KEY = os.getenv("S2_API_KEY")
            if not self.S2_API_KEY:
                warnings.warn(
                    "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
                    "Set the S2_API_KEY environment variable for higher limits."
                )

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None

        if _BACKEND_MODE == "legacy":
            return self._search_legacy(query)

        return _search_via_agent(query, result_limit=self.max_results)

    def _search_legacy(self, query: str) -> Optional[List[Dict]]:
        @backoff.on_exception(
            backoff.expo,
            (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
            on_backoff=on_backoff,
        )
        def _inner():
            headers = {}
            if self.S2_API_KEY:
                headers["X-API-KEY"] = self.S2_API_KEY

            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={
                    "query": query,
                    "limit": self.max_results,
                    "fields": "title,authors,venue,year,abstract,citationCount",
                },
            )
            print(f"Response Status Code: {rsp.status_code}")
            print(f"Response Content: {rsp.text[:500]}")
            rsp.raise_for_status()
            results = rsp.json()
            total = results.get("total", 0)
            if total == 0:
                return None

            papers = results.get("data", [])
            papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
            return papers

        return _inner()

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


def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    """Search for academic papers.

    In v3: uses agent-based web search.
    In legacy: uses Semantic Scholar HTTP API directly.
    """
    if not query:
        return None

    if _BACKEND_MODE == "legacy":
        return _search_for_papers_legacy(query, result_limit)

    return _search_via_agent(query, result_limit)


def _search_via_agent(query: str, result_limit: int = 10) -> Optional[List[Dict]]:
    """Search for papers using the agent backend's web search capability.

    The agent is asked to search for academic papers and return structured results.
    """
    from ai_scientist.agent_backend import AgentBackend

    backend = AgentBackend(timeout=120)

    search_prompt = (
        f"Search for academic papers related to: {query}\n\n"
        f"Return up to {result_limit} relevant papers. For each paper, provide:\n"
        f"- title\n"
        f"- authors (list of names)\n"
        f"- venue (conference or journal)\n"
        f"- year\n"
        f"- abstract\n"
        f"- citationCount (approximate number of citations)\n\n"
        f"If you also find a citationStyles/bibtex entry, include it.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                        },
                        "venue": {"type": "string"},
                        "year": {"type": "integer"},
                        "abstract": {"type": "string"},
                        "citationCount": {"type": "integer"},
                        "citationStyles": {
                            "type": "object",
                            "properties": {"bibtex": {"type": "string"}},
                        },
                    },
                    "required": ["title", "authors", "year"],
                },
            },
        },
        "required": ["papers"],
    }

    try:
        result = backend.query(
            system_message="You are a research assistant that searches for academic papers. Use web search to find real papers with accurate metadata.",
            user_message=search_prompt,
            json_schema=schema,
            timeout=120,
        )

        if isinstance(result, dict) and "papers" in result:
            papers = result["papers"]
            # Sort by citation count if available
            papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
            return papers if papers else None
        return None

    except Exception as e:
        print(f"Agent-based paper search failed: {e}")
        return None


def _search_for_papers_legacy(query, result_limit=10) -> Union[None, List[Dict]]:
    """Legacy: direct Semantic Scholar HTTP API call."""
    S2_API_KEY = os.getenv("S2_API_KEY")
    headers = {}
    if not S2_API_KEY:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits."
        )
    else:
        headers["X-API-KEY"] = S2_API_KEY

    @backoff.on_exception(
        backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
    )
    def _inner():
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)
        if not total:
            return None
        return results["data"]

    return _inner()
