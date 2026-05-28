"""Vector store manager using ChromaDB for storing and retrieving embeddings."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from utils.logger import get_logger
from rag.embedder import Embedder

logger = get_logger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store for SEC filings."""
    
    def __init__(
        self,
        collection_name: str = "sec_filings",
        persist_directory: Path = None
    ):
        """Initialize vector store manager.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_dir
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Initialize embedder
        self.embedder = Embedder()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SEC 10-Q filings embeddings"}
        )
        
        logger.info(
            f"Collection '{collection_name}' initialized. "
            f"Current count: {self.collection.count()}"
        )
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        ticker: str,
        filing_date: str = None
    ):
        """Add document chunks to vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            ticker: Stock ticker symbol
            filing_date: Filing date (optional)
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks for {ticker} to vector store")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts)
        
        # Prepare metadata
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{ticker}_{filing_date or 'unknown'}_{i}"
            ids.append(chunk_id)
            
            metadata = {
                'ticker': ticker,
                'chunk_id': i,
                **{k: v for k, v in chunk.items() if k != 'text'}
            }
            if filing_date:
                metadata['filing_date'] = filing_date
            
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query vector store for relevant chunks.
        
        Args:
            query_text: Query string
            ticker: Filter by ticker symbol (optional)
            n_results: Number of results to return (defaults to settings)
            filters: Additional metadata filters
        
        Returns:
            List of matching chunks with metadata and scores
        """
        if n_results is None:
            n_results = settings.top_k_results
        
        logger.info(f"Querying vector store: '{query_text[:50]}...' (n={n_results})")
        
        # Build where clause for filtering
        where = {}
        if ticker:
            where['ticker'] = ticker
        if filters:
            where.update(filters)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query_text)
        
        # Query collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'id': results['ids'][0][i]
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} matching chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise
    
    def delete_by_ticker(self, ticker: str):
        """Delete all documents for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
        """
        logger.info(f"Deleting all documents for ticker: {ticker}")
        
        try:
            self.collection.delete(
                where={"ticker": ticker}
            )
            logger.info(f"Deleted documents for {ticker}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get sample to analyze tickers
        sample = self.collection.get(limit=1000)
        tickers = set()
        if sample['metadatas']:
            for meta in sample['metadatas']:
                if 'ticker' in meta:
                    tickers.add(meta['ticker'])
        
        stats = {
            'total_chunks': count,
            'unique_tickers_sampled': len(tickers),
            'collection_name': self.collection_name
        }
        
        logger.info(f"Collection stats: {stats}")
        return stats
    
    def clear_collection(self):
        """Clear all data from the collection."""
        logger.warning("Clearing entire collection")
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SEC 10-Q filings embeddings"}
            )
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
