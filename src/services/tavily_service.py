"""Tavily search API service for registry discovery."""

import logging
from typing import List, Dict, Any, Optional
import httpx
from src.config import Config

logger = logging.getLogger(__name__)


class TavilyService:
    """Async wrapper for Tavily search API."""

    BASE_URL = "https://api.tavily.com"

    def __init__(self, api_key: str = ""):
        """Initialize Tavily service."""
        self.api_key = api_key or Config.TAVILY_API_KEY

    async def search(
        self,
        query: str,
        include_answer: bool = True,
        num_results: int = 20,
        topic: str = "general",
        search_depth: str = "basic",
    ) -> Dict[str, Any]:
        """
        Perform async search via Tavily API.

        Args:
            query: Search query
            include_answer: Whether to include AI-generated answer
            num_results: Number of results to return (sent as max_results in the API payload)
            topic: Search topic (general, news, etc.)
            search_depth: Search depth level (basic or advanced)

        Returns:
            Search results dictionary with 'results' array
        """
        if Config.USE_MOCKS:
            logger.debug(f"Using mock Tavily results for: {query}")
            return self._mock_search_results(query)

        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "include_answer": include_answer,
                    "max_results": num_results,
                    "topic": topic,
                    "search_depth": search_depth,
                }

                response = await client.post(
                    f"{self.BASE_URL}/search",
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()

                result = response.json()
                logger.info(
                    f"Tavily search succeeded for query: {query} -- ({len(result.get('results', []))} results)"
                )
                return result

        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}")
            return {"results": [], "answer": None, "error": str(e)}

    def _mock_search_results(self, query: str) -> Dict[str, Any]:
        """Return hardcoded mock search results for testing."""
        return {
            "results": [
                {
                    "title": "HVAC Distributors - Ohio State Business Registry",
                    "url": "https://www.ohio.gov/business/hvac-distributors",
                    "content": "Example Company Corp LLC, registered 2010, status: active. Address: 1500 Industrial Dr, Cleveland, OH 44114. Owner: Jordan Lee.",
                    "score": 0.95,
                },
                {
                    "title": "Example Company Corp - Official Website",
                    "url": "https://www.example-company-corp.test",
                    "content": "We are a leading independent HVAC distributor serving Ohio since 2010. Family-owned business.",
                    "score": 0.89,
                },
                {
                    "title": "Ohio HVAC Business Directory",
                    "url": "https://www.ohiobiz.gov/hvac",
                    "content": "Search results for HVAC businesses in Ohio. Example Company Corp listed as active.",
                    "score": 0.82,
                },
            ],
            "answer": "Example Company Corp is an active HVAC distribution company registered in Ohio since 2010.",
            "query": query,
        }


# Global Tavily service instance
_tavily_service: Optional[TavilyService] = None


async def get_tavily_service() -> TavilyService:
    """Get or initialize the global Tavily service."""
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyService()
    return _tavily_service
