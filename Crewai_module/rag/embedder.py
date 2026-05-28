"""
Embedder for generating vector representations of text.

Uses Ollama's nomic-embed-text model for embeddings (consistent
with the Ollama-only pipeline design). Falls back to sentence-transformers
if Ollama embedding is not available.
"""

from typing import List, Optional
import numpy as np
import requests
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

# Ollama base URL from settings or environment
OLLAMA_BASE_URL = getattr(settings, 'ollama_base_url', 'http://localhost:11434')
EMBEDDING_MODEL = getattr(settings, 'embedding_model_name', 'nomic-embed-text:latest')


class Embedder:
    """Generate embeddings using Ollama's nomic-embed-text model."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedder.
        
        Args:
            model_name: Ollama embedding model name (default: nomic-embed-text:latest)
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.base_url = OLLAMA_BASE_URL
        self.embedding_dim = None
        self._use_ollama = True
        self._st_model = None  # sentence-transformers fallback
        
        # Test Ollama connectivity and get embedding dimension
        try:
            test_embedding = self._ollama_embed("test")
            self.embedding_dim = len(test_embedding)
            logger.info(
                f"Embedder ready: {self.model_name} via Ollama "
                f"(dim={self.embedding_dim})"
            )
        except Exception as e:
            logger.warning(f"Ollama embedding unavailable ({e}), falling back to sentence-transformers")
            self._use_ollama = False
            self._init_sentence_transformers()
    
    def _init_sentence_transformers(self):
        """Fallback: initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            fallback_model = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            self._st_model = SentenceTransformer(fallback_model)
            self.embedding_dim = self._st_model.get_sentence_embedding_dimension()
            logger.info(f"Fallback embedder ready: {fallback_model} (dim={self.embedding_dim})")
        except ImportError:
            raise RuntimeError(
                "Neither Ollama nor sentence-transformers available for embeddings. "
                "Start Ollama with: ollama serve"
            )
    
    def _ollama_embed(self, text: str) -> List[float]:
        """Get embedding from Ollama API."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            if self._use_ollama:
                embedding = self._ollama_embed(text)
                return np.array(embedding, dtype=np.float32)
            else:
                return self._st_model.encode(
                    text, convert_to_numpy=True, show_progress_bar=False
                )
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Embedding batch of {len(texts)} texts")
        
        try:
            if self._use_ollama:
                embeddings = []
                for i, text in enumerate(texts):
                    emb = self._ollama_embed(text)
                    embeddings.append(emb)
                    if show_progress and (i + 1) % 10 == 0:
                        logger.info(f"  Embedded {i + 1}/{len(texts)}")
                result = np.array(embeddings, dtype=np.float32)
            else:
                result = self._st_model.encode(
                    texts, batch_size=batch_size,
                    convert_to_numpy=True, show_progress_bar=show_progress
                )
            
            logger.info(f"Generated {len(result)} embeddings")
            return result
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
