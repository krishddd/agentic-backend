"""Web search and news tools for market research."""

from typing import Optional
import requests
from datetime import datetime, timedelta
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Tool for web search using available search APIs."""
    
    name: str = "Web Search"
    description: str = (
        "Search the web for information about a company or stock. "
        "Input should be a search query (e.g., 'Tesla recent news' or 'TSLA stock analysis'). "
        "Returns relevant search results with titles, descriptions, and URLs."
    )
    
    def _run(self, query: str) -> str:
        """Perform web search.
        
        Args:
            query: Search query
        
        Returns:
            Formatted search results
        """
        try:
            logger.info(f"Web search: {query}")
            
            # Check cache
            cached_result = cache.get("web_search", query=query)
            if cached_result:
                return cached_result
            
            # Use Tavily if available
            if settings.tavily_api_key:
                result = self._search_tavily(query)
            # Use DuckDuckGo as fallback
            else:
                result = self._search_duckduckgo(query)
            
            # Cache result
            cache.set("web_search", result, expire_hours=6, query=query)
            
            return result
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _search_tavily(self, query: str) -> str:
        """Search using Tavily API.
        
        Args:
            query: Search query
        
        Returns:
            Formatted results
        """
        try:
            # Use Tavily API directly with httpx
            import httpx
            
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return f"No results found for: {query}"
            
            output = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                output += f"[{i}] {result.get('title', 'No title')}\n"
                output += f"    {result.get('content', 'No description')}\n"
                output += f"    URL: {result.get('url', 'No URL')}\n\n"
            
            return output
            
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}, falling back to DuckDuckGo")
            return self._search_duckduckgo(query)
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> str:
        """Search using DuckDuckGo (free, no API key required).
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            Formatted results
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return f"No results found for: {query}"
            
            output = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                output += f"[{i}] {result.get('title', 'No title')}\n"
                output += f"    {result.get('body', 'No description')}\n"
                output += f"    URL: {result.get('href', 'No URL')}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return f"Search failed: {str(e)}"


class NewsSearchTool(BaseTool):
    """Tool for searching news articles about stocks."""
    
    name: str = "News Search"
    description: str = (
        "Search for recent news articles about a stock or company. "
        "Input should be a stock ticker or company name (e.g., 'TSLA' or 'Tesla'). "
        "Returns recent news headlines, summaries, and sources."
    )
    
    def _run(self, search_term: str) -> str:
        """Search for news articles.
        
        Args:
            search_term: Stock ticker or company name
        
        Returns:
            Formatted news articles
        """
        try:
            logger.info(f"News search: {search_term}")
            
            # Check cache
            cached_result = cache.get("news_search", term=search_term)
            if cached_result:
                return cached_result
            
            # Use News API if available
            if settings.news_api_key:
                result = self._search_newsapi(search_term)
            else:
                # Fallback to DuckDuckGo news search
                result = self._search_ddg_news(search_term)
            
            # Cache result for shorter time (news is time-sensitive)
            cache.set("news_search", result, expire_hours=2, term=search_term)
            
            return result
            
        except Exception as e:
            error_msg = f"Error searching news: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _search_newsapi(self, search_term: str) -> str:
        """Search using News API.
        
        Args:
            search_term: Search term
        
        Returns:
            Formatted news articles
        """
        try:
            # News API endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Get news from last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            params = {
                'q': search_term,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': settings.news_api_key,
                'pageSize': 10
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                return f"No recent news found for: {search_term}"
            
            output = f"Recent news for '{search_term}':\n\n"
            for i, article in enumerate(articles[:10], 1):
                output += f"[{i}] {article.get('title', 'No title')}\n"
                output += f"    Source: {article.get('source', {}).get('name', 'Unknown')}\n"
                output += f"    Published: {article.get('publishedAt', 'Unknown date')}\n"
                output += f"    {article.get('description', 'No description')}\n"
                output += f"    URL: {article.get('url', 'No URL')}\n\n"
            
            return output
            
        except Exception as e:
            logger.warning(f"NewsAPI search failed: {e}, falling back to DuckDuckGo")
            return self._search_ddg_news(search_term)
    
    def _search_ddg_news(self, search_term: str, max_results: int = 10) -> str:
        """Search news using DuckDuckGo.
        
        Args:
            search_term: Search term
            max_results: Maximum number of results
        
        Returns:
            Formatted news articles
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(search_term, max_results=max_results))
            
            if not results:
                return f"No recent news found for: {search_term}"
            
            output = f"Recent news for '{search_term}':\n\n"
            for i, article in enumerate(results, 1):
                output += f"[{i}] {article.get('title', 'No title')}\n"
                output += f"    Source: {article.get('source', 'Unknown')}\n"
                output += f"    Date: {article.get('date', 'Unknown date')}\n"
                output += f"    {article.get('body', 'No description')}\n"
                output += f"    URL: {article.get('url', 'No URL')}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"DuckDuckGo news search failed: {e}")
            return f"News search failed: {str(e)}"
