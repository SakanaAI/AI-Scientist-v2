"""
OpenAlex API integration for AI-Scientist-v2

This module provides OpenAlex API integration as a replacement for Semantic Scholar.
OpenAlex is a free, open-source API for scholarly literature with no authentication required.

Features:
- Compatible interface with Semantic Scholar tools
- Free API with 100,000 requests/day limit
- Automatic data format conversion for backward compatibility
- Abstract reconstruction from inverted index
- Citation-based result ranking
"""

import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    """
    Callback function for backoff decorator to log retry attempts.
    
    Args:
        details: Dictionary containing backoff details including wait time and tries
    """
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class OpenAlexSearchTool(BaseTool):
    """
    OpenAlex search tool for finding relevant academic literature.
    
    This tool provides a compatible interface with SemanticScholarSearchTool,
    allowing for seamless replacement in existing code.
    
    Attributes:
        max_results (int): Maximum number of results to return per search
        email (str): Contact email for polite API usage (optional)
    """
    
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex API. "
            "Provide a search query to find relevant academic papers."
        ),
        max_results: int = 10,
    ):
        """
        Initialize the OpenAlex search tool.
        
        Args:
            name: Tool name for identification
            description: Tool description for LLM understanding
            max_results: Maximum number of results to return per search
        """
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        # OpenAlex doesn't require API key but we can set email for polite usage
        self.email = os.getenv("OPENALEX_EMAIL", "ai-scientist@example.com")

    def use_tool(self, query: str) -> Optional[str]:
        """
        Execute the search tool with the given query.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted string of search results or "No papers found." if empty
        """
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
        """
        Search for papers using OpenAlex API.
        
        Args:
            query: Search query string
            
        Returns:
            List of paper dictionaries or None if no results found
            
        Raises:
            requests.exceptions.HTTPError: If API request fails
            requests.exceptions.ConnectionError: If connection fails
        """
        if not query:
            return None
        
        # Set up headers with user agent for polite API usage
        headers = {
            "User-Agent": f"AI-Scientist-v2 (mailto:{self.email})"
        }
        
        # Configure API parameters
        params = {
            "search": query,
            "per-page": self.max_results,
            "sort": "cited_by_count:desc",  # Sort by citation count descending
            "select": "id,title,display_name,publication_year,publication_date,type,cited_by_count,abstract_inverted_index,authorships,primary_location,open_access"
        }
        
        # Make API request
        rsp = requests.get(
            "https://api.openalex.org/works",
            headers=headers,
            params=params,
        )
        print(f"OpenAlex API Response Status: {rsp.status_code}")
        print(f"Response Preview: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        
        if not results.get("results"):
            return None

        papers = results.get("results", [])
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        """
        Format papers into a readable string format.
        
        Args:
            papers: List of paper dictionaries from OpenAlex API
            
        Returns:
            Formatted string with paper details
        """
        paper_strings = []
        for i, paper in enumerate(papers):
            # Extract authors from authorships
            authors = []
            for authorship in paper.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])
            authors_str = ", ".join(authors) if authors else "Unknown Authors"
            
            # Extract venue from primary_location
            venue = "Unknown Venue"
            primary_location = paper.get("primary_location", {})
            if primary_location:
                source = primary_location.get("source", {})
                if source and source.get("display_name"):
                    venue = source["display_name"]
            
            # Reconstruct abstract from inverted index
            abstract = self.reconstruct_abstract(paper.get("abstract_inverted_index", {}))
            if not abstract:
                abstract = "No abstract available."
            
            # Format paper entry
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", paper.get("display_name", "Unknown Title"))}. {authors_str}. {venue}, {paper.get("publication_year", "Unknown Year")}.
Number of citations: {paper.get("cited_by_count", "N/A")}
Abstract: {abstract}"""
            )
        return "\n\n".join(paper_strings)

    def reconstruct_abstract(self, inverted_index: Dict) -> str:
        """
        Reconstruct abstract text from OpenAlex inverted index format.
        
        OpenAlex stores abstracts as inverted indices where each word maps to
        its positions in the text. This function reconstructs the original text.
        
        Args:
            inverted_index: Dictionary mapping words to their positions
            
        Returns:
            Reconstructed abstract text
        """
        if not inverted_index:
            return ""
        
        # Create a list to hold words at their positions
        word_positions = []
        
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort by position and join words
        word_positions.sort(key=lambda x: x[0])
        abstract = " ".join([word for pos, word in word_positions])
        
        return abstract


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
    """
    Standalone function for searching papers - compatible with Semantic Scholar interface.
    
    This function provides backward compatibility with existing code that uses
    the search_for_papers function from semantic_scholar module.
    
    Args:
        query: Search query string
        result_limit: Maximum number of results to return
        
    Returns:
        List of paper dictionaries in Semantic Scholar compatible format,
        or None if no results found
        
    Raises:
        requests.exceptions.HTTPError: If API request fails
    """
    if not query:
        return None
    
    # Set up polite API usage
    email = os.getenv("OPENALEX_EMAIL", "ai-scientist@example.com")
    headers = {
        "User-Agent": f"AI-Scientist-v2 (mailto:{email})"
    }
    
    # Configure API parameters
    params = {
        "search": query,
        "per-page": result_limit,
        "sort": "cited_by_count:desc",
        "select": "id,title,display_name,publication_year,publication_date,type,cited_by_count,abstract_inverted_index,authorships,primary_location,open_access"
    }
    
    # Make API request
    rsp = requests.get(
        "https://api.openalex.org/works",
        headers=headers,
        params=params,
    )
    print(f"OpenAlex API Response Status: {rsp.status_code}")
    print(f"Response Preview: {rsp.text[:500]}")
    rsp.raise_for_status()
    results = rsp.json()
    
    # Be polite to the API
    time.sleep(0.1)
    
    if not results.get("results"):
        return None

    papers = results.get("results", [])
    
    # Convert OpenAlex format to Semantic Scholar compatible format
    converted_papers = []
    for paper in papers:
        # Extract authors in Semantic Scholar format
        authors = []
        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append({"name": author["display_name"]})
        
        # Extract venue information
        venue = "Unknown Venue"
        primary_location = paper.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source", {})
            if source and source.get("display_name"):
                venue = source["display_name"]
        
        # Reconstruct abstract from inverted index
        abstract = reconstruct_abstract_standalone(paper.get("abstract_inverted_index", {}))
        
        # Create Semantic Scholar compatible paper object
        converted_paper = {
            "title": paper.get("title", paper.get("display_name", "Unknown Title")),
            "authors": authors,
            "venue": venue,
            "year": paper.get("publication_year", "Unknown Year"),
            "abstract": abstract if abstract else "No abstract available.",
            "citationCount": paper.get("cited_by_count", 0),
            "citationStyles": {},  # OpenAlex doesn't provide this, keeping for compatibility
        }
        converted_papers.append(converted_paper)
    
    return converted_papers


def reconstruct_abstract_standalone(inverted_index: Dict) -> str:
    """
    Standalone function to reconstruct abstract from inverted index.
    
    This is a utility function used by the search_for_papers function
    to reconstruct abstracts from OpenAlex inverted index format.
    
    Args:
        inverted_index: Dictionary mapping words to their positions in the text
        
    Returns:
        Reconstructed abstract text as a string
    """
    if not inverted_index:
        return ""
    
    # Collect all word-position pairs
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    # Sort by position and reconstruct text
    word_positions.sort(key=lambda x: x[0])
    abstract = " ".join([word for pos, word in word_positions])
    
    return abstract

