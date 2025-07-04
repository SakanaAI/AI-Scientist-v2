import os
import time
import warnings
from typing import Dict, List, Optional

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class OpenAlexSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex. "
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
        self.mail = os.getenv("OPENALEX_MAIL_ADDRESS", None)
        if self.mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        try:
            import pyalex
            self.pyalex = pyalex
            if self.mail:
                pyalex.config.email = self.mail
        except ImportError:
            self.pyalex = None
            print("[ERROR] pyalex is not installed. Please install pyalex to use OpenAlexSearchTool.")

    def use_tool(self, query: str) -> Optional[str]:
        if self.pyalex is None:
            return "pyalex is not installed. Please install pyalex to use this tool."
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        on_backoff=on_backoff,
        max_tries=3,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query or self.pyalex is None:
            return None
        try:
            from pyalex import Works
            # Используем более простой подход с get() для получения результатов
            search_results = Works().search(query).filter(is_paratext=False).get()
            
            if not search_results:
                return None
                
            # Сортируем результаты по количеству цитирований
            search_results.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
            
            # Берем только нужное количество результатов
            papers = []
            for work in search_results[:self.max_results]:
                papers.append(self.extract_info_from_work(work))
                
            return papers if papers else None
        except Exception as e:
            print(f"[ERROR] Failed to search OpenAlex: {e}")
            return None

    def extract_info_from_work(self, work, max_abstract_length: int = 1000) -> dict:
        try:
            venue = "Unknown"
            for location in work.get("locations", []):
                if location.get("source") is not None:
                    venue = location["source"].get("display_name", "Unknown")
                    if venue != "":
                        break
            title = work.get("title", "Unknown Title")
            abstract = work.get("abstract") or ""
            if len(abstract) > max_abstract_length:
                print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars.")
                abstract = abstract[:max_abstract_length]
            authors_list = [author["author"].get("display_name", "Unknown") for author in work.get("authorships", [])]
            authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            paper = dict(
                title=title,
                authors=authors,
                venue=venue,
                year=work.get("publication_year", "Unknown Year"),
                abstract=abstract,
                citationCount=work.get("cited_by_count", 0),
            )
            return paper
        except Exception as e:
            print(f"[ERROR] Failed to extract info from work: {e}")
            return dict(
                title="Unknown Title",
                authors="Unknown Authors",
                venue="Unknown Venue",
                year="Unknown Year",
                abstract="No abstract available.",
                citationCount=0,
            )

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            paper_strings.append(
                f"""{i + 1}: {paper.get('title', 'Unknown Title')}. {paper.get('authors', 'Unknown Authors')}. {paper.get('venue', 'Unknown Venue')}, {paper.get('year', 'Unknown Year')}.
Number of citations: {paper.get('citationCount', 'N/A')}
Abstract: {paper.get('abstract', 'No abstract available.')}"""
            )
        return "\n\n".join(paper_strings)
