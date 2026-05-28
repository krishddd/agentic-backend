"""RAG retrieval tools for querying SEC filings."""

from typing import Optional
from crewai.tools import BaseTool
from rag.vector_store import VectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetrievalTool(BaseTool):
    """Tool for retrieving information from SEC filings using RAG."""
    
    name: str = "RAG Filing Retrieval"
    description: str = (
        "Retrieve relevant information from SEC 10-Q filings using semantic search. "
        "Input should be a question or topic to search for in the filings. "
        "Optionally specify ticker as 'TICKER: question' (e.g., 'TSLA: revenue breakdown'). "
        "Returns relevant excerpts from the filings that answer the question."
    )
    
    def _run(self, query: str) -> str:
        """Retrieve information using RAG.
        
        Args:
            query: Search query, optionally with ticker prefix
        
        Returns:
            Formatted string with relevant excerpts
        """
        try:
            # Parse ticker if provided
            ticker = None
            if ':' in query:
                parts = query.split(':', 1)
                ticker = parts[0].strip().upper()
                query = parts[1].strip()
            
            logger.info(f"RAG query for {ticker or 'all tickers'}: {query[:50]}...")
            
            # Initialize vector store on-demand
            vector_store = VectorStoreManager()
            
            # Query vector store
            results = vector_store.query(
                query_text=query,
                ticker=ticker,
                n_results=5
            )
            
            if not results:
                return f"No relevant information found in SEC filings for: {query}"
            
            # Format results
            output = f"Relevant information from SEC filings:\n\n"
            
            for i, result in enumerate(results, 1):
                text = result['text']
                metadata = result['metadata']
                distance = result['distance']
                
                ticker_info = metadata.get('ticker', 'Unknown')
                filing_date = metadata.get('filing_date', 'Unknown date')
                
                output += f"[{i}] From {ticker_info} filing ({filing_date}) [relevance: {1 - distance:.2f}]:\n"
                output += f"{text[:500]}...\n\n"  # Limit to 500 chars per excerpt
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return output
            
        except Exception as e:
            error_msg = f"Error retrieving from RAG: {str(e)}"
            logger.error(error_msg)
            return error_msg

