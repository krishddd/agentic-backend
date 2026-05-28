"""Caching mechanism for expensive operations."""

from pathlib import Path
from typing import Any, Optional
import hashlib
import json
from datetime import datetime, timedelta
from diskcache import Cache as DiskCache
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Cache:
    """Disk-based cache for API responses and expensive operations."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.
        
        Args:
            cache_dir: Directory for cache storage. Defaults to data/cache
        """
        if cache_dir is None:
            cache_dir = settings.data_dir / "cache"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = DiskCache(cache_dir)
        self.enabled = settings.cache_enabled
        logger.info(f"Cache initialized at {cache_dir}, enabled={self.enabled}")
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from prefix and kwargs.
        
        Args:
            prefix: Key prefix (e.g., 'sec_filing', 'web_search')
            **kwargs: Key-value pairs to include in key
        
        Returns:
            SHA256 hash of the key components
        """
        key_data = json.dumps({prefix: kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            prefix: Cache key prefix
            **kwargs: Key components
        
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(prefix, **kwargs)
        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT: {prefix}")
            return value
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, prefix: str, value: Any, expire_hours: Optional[int] = None, **kwargs):
        """Set value in cache.
        
        Args:
            prefix: Cache key prefix
            value: Value to cache
            expire_hours: Expiration time in hours (defaults to settings)
            **kwargs: Key components
        """
        if not self.enabled:
            return
        
        key = self._generate_key(prefix, **kwargs)
        if expire_hours is None:
            expire_hours = settings.cache_expire_hours
        
        expire_seconds = expire_hours * 3600
        
        try:
            self.cache.set(key, value, expire=expire_seconds)
            logger.debug(f"Cache SET: {prefix} (expire in {expire_hours}h)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def close(self):
        """Close cache connection."""
        try:
            self.cache.close()
        except Exception as e:
            logger.warning(f"Cache close error: {e}")


# Global cache instance
cache = Cache()
