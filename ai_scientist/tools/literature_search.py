import os
import re
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

import requests

from ai_scientist.tools.base_tool import BaseTool


OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def _request_headers(include_s2_key: bool = False) -> Dict[str, str]:
    email = os.getenv("OPENALEX_EMAIL") or os.getenv("CONTACT_EMAIL") or os.getenv("EMAIL")
    user_agent = "AutoResearch-VSC-Copilot/1.0"
    if email:
        user_agent = f"{user_agent} ({email})"

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if include_s2_key and os.getenv("S2_API_KEY"):
        headers["X-API-KEY"] = os.getenv("S2_API_KEY")
    return headers


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _safe_key_fragment(text: str, fallback: str = "ref") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text or "")
    return cleaned[:24] or fallback


def _build_bibtex(
    title: str,
    authors: List[Dict],
    venue: str,
    year: Union[str, int, None],
    doi: Optional[str],
    url: Optional[str],
) -> str:
    year_str = str(year) if year else "unknown"
    first_author = authors[0]["name"].split()[-1] if authors else "unknown"
    cite_key = _safe_key_fragment(first_author.lower()) + year_str

    author_field = " and ".join(author["name"] for author in authors) or "Unknown"
    entry_type = "article" if venue else "misc"
    lines = [f"@{entry_type}{{{cite_key},"]
    lines.append(f"  title = {{{title or 'Unknown Title'}}},")
    lines.append(f"  author = {{{author_field}}},")
    if venue:
        lines.append(f"  journal = {{{venue}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def _normalize_paper(
    *,
    title: str,
    authors: List[str],
    venue: str,
    year: Union[int, str, None],
    abstract: str,
    citation_count: Optional[int],
    doi: Optional[str],
    url: Optional[str],
    source: str,
    bibtex: Optional[str] = None,
) -> Dict:
    author_objs = [{"name": _clean_text(author)} for author in authors if _clean_text(author)]
    normalized_bibtex = bibtex or _build_bibtex(
        title=_clean_text(title),
        authors=author_objs,
        venue=_clean_text(venue),
        year=year,
        doi=doi,
        url=url,
    )
    return {
        "title": _clean_text(title),
        "authors": author_objs,
        "venue": _clean_text(venue),
        "year": year,
        "abstract": _clean_text(abstract),
        "citationCount": citation_count or 0,
        "citationStyles": {"bibtex": normalized_bibtex},
        "source": source,
        "url": url,
        "doi": doi,
    }


def _dedupe_papers(papers: List[Dict], limit: int) -> List[Dict]:
    seen = set()
    deduped = []
    for paper in sorted(papers, key=lambda item: item.get("citationCount", 0), reverse=True):
        key = (_clean_text(paper.get("title", "")).lower(), str(paper.get("year") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
        if len(deduped) >= limit:
            break
    return deduped


def _search_openalex(query: str, result_limit: int) -> List[Dict]:
    params = {"search": query, "per-page": result_limit}
    email = os.getenv("OPENALEX_EMAIL") or os.getenv("CONTACT_EMAIL")
    if email:
        params["mailto"] = email

    rsp = requests.get(
        OPENALEX_API,
        headers=_request_headers(),
        params=params,
        timeout=20,
    )
    rsp.raise_for_status()
    data = rsp.json().get("results", [])

    papers = []
    for item in data:
        papers.append(
            _normalize_paper(
                title=item.get("title", ""),
                authors=[author.get("author", {}).get("display_name", "") for author in item.get("authorships", [])],
                venue=(item.get("primary_location") or {}).get("source", {}).get("display_name", ""),
                year=item.get("publication_year"),
                abstract="",
                citation_count=item.get("cited_by_count", 0),
                doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
                url=(item.get("primary_location") or {}).get("landing_page_url") or item.get("id"),
                source="openalex",
            )
        )
    return papers


def _search_crossref(query: str, result_limit: int) -> List[Dict]:
    rsp = requests.get(
        CROSSREF_API,
        headers=_request_headers(),
        params={
            "query": query,
            "rows": result_limit,
            "select": "title,author,container-title,published-print,published-online,abstract,is-referenced-by-count,DOI,URL",
        },
        timeout=20,
    )
    rsp.raise_for_status()
    items = rsp.json().get("message", {}).get("items", [])

    papers = []
    for item in items:
        year = None
        for field in ("published-print", "published-online"):
            date_parts = item.get(field, {}).get("date-parts", [])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break
        papers.append(
            _normalize_paper(
                title=(item.get("title") or [""])[0],
                authors=[
                    " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part)
                    for author in item.get("author", [])
                ],
                venue=(item.get("container-title") or [""])[0],
                year=year,
                abstract=_clean_text(item.get("abstract", "")),
                citation_count=item.get("is-referenced-by-count", 0),
                doi=item.get("DOI"),
                url=item.get("URL"),
                source="crossref",
            )
        )
    return papers


def _search_arxiv(query: str, result_limit: int) -> List[Dict]:
    rsp = requests.get(
        ARXIV_API,
        headers={"User-Agent": _request_headers()["User-Agent"]},
        params={"search_query": f"all:{query}", "start": 0, "max_results": result_limit},
        timeout=20,
    )
    rsp.raise_for_status()
    root = ET.fromstring(rsp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        papers.append(
            _normalize_paper(
                title=entry.findtext("atom:title", default="", namespaces=ns),
                authors=[author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)],
                venue="arXiv",
                year=(entry.findtext("atom:published", default="", namespaces=ns) or "")[:4] or None,
                abstract=entry.findtext("atom:summary", default="", namespaces=ns),
                citation_count=0,
                doi=None,
                url=entry.findtext("atom:id", default="", namespaces=ns),
                source="arxiv",
            )
        )
    return papers


def _search_semantic_scholar(query: str, result_limit: int) -> List[Dict]:
    rsp = requests.get(
        SEMANTIC_SCHOLAR_API,
        headers=_request_headers(include_s2_key=True),
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,url,externalIds",
        },
        timeout=20,
    )
    rsp.raise_for_status()
    data = rsp.json().get("data", [])

    papers = []
    for item in data:
        external_ids = item.get("externalIds") or {}
        papers.append(
            _normalize_paper(
                title=item.get("title", ""),
                authors=[author.get("name", "") for author in item.get("authors", [])],
                venue=item.get("venue", ""),
                year=item.get("year"),
                abstract=item.get("abstract", ""),
                citation_count=item.get("citationCount", 0),
                doi=external_ids.get("DOI"),
                url=item.get("url"),
                source="semantic_scholar",
                bibtex=(item.get("citationStyles") or {}).get("bibtex"),
            )
        )
    return papers


def _search_all_sources(query: str, result_limit: int) -> List[Dict]:
    sources = [
        ("openalex", _search_openalex),
        ("crossref", _search_crossref),
        ("arxiv", _search_arxiv),
    ]
    if os.getenv("S2_API_KEY"):
        sources.append(("semantic_scholar", _search_semantic_scholar))
    else:
        warnings.warn("S2_API_KEY not set. Skipping Semantic Scholar and using free sources only.")

    all_papers: List[Dict] = []
    for source_name, search_fn in sources:
        try:
            papers = search_fn(query, result_limit)
            print(f"{source_name} returned {len(papers)} papers")
            all_papers.extend(papers)
        except Exception as exc:
            print(f"{source_name} search failed: {exc}")
        time.sleep(0.5)

    return _dedupe_papers(all_papers, result_limit)


class MultiSourceLiteratureSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchLiterature",
        description: str = (
            "Search for relevant literature using multiple free sources "
            "(OpenAlex, Crossref, arXiv, and optionally Semantic Scholar)."
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

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        return "No papers found."

    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None
        papers = _search_all_sources(query, self.max_results)
        return papers or None

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(author.get("name", "Unknown") for author in paper.get("authors", []))
            paper_strings.append(
                f"""{i + 1}: {paper.get('title', 'Unknown Title')}. {authors}. {paper.get('venue', 'Unknown Venue')}, {paper.get('year', 'Unknown Year')}.
Source: {paper.get('source', 'unknown')}
Number of citations: {paper.get('citationCount', 'N/A')}
Abstract: {paper.get('abstract', 'No abstract available.')}"""
            )
        return "\n\n".join(paper_strings)


def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    papers = _search_all_sources(query, result_limit)
    return papers or None
