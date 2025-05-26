"""
Caching utilities for OpenInterpretability platform.
"""

import json
import pickle
import time
import asyncio
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    In-memory cache manager with TTL support.
    In production, this would be backed by Redis or similar.
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        await self._cleanup_expired()
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if entry["expires_at"] > time.time():
                self.stats["hits"] += 1
                return entry["value"]
            else:
                # Remove expired entry
                del self.cache[key]
                self.stats["evictions"] += 1
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        expires_at = time.time() + ttl
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        self.stats["sets"] += 1
        
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            self.stats["deletes"] += 1
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        value = await self.get(key)
        return value is not None
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        self.stats["evictions"] += count
        
    async def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self.cache.items():
            if entry["expires_at"] <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1
        
        self.last_cleanup = current_time
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "evictions": self.stats["evictions"],
            "total_entries": len(self.cache),
            "total_requests": total_requests
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.clear()
        logger.info("Cache manager closed")


class LRUCache:
    """
    Least Recently Used cache implementation.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: Dict[str, float] = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value and update access time."""
        if key in self.cache:
            self.access_order[key] = time.time()
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value and manage cache size."""
        current_time = time.time()
        
        # If cache is full and key is new, remove LRU item
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
        
        self.cache[key] = value
        self.access_order[key] = current_time
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_order:
            return
        
        # Find least recently used key
        lru_key = min(self.access_order.keys(), key=lambda k: self.access_order[k])
        
        # Remove from both caches
        del self.cache[lru_key]
        del self.access_order[lru_key]


class CacheKey:
    """Utility for generating consistent cache keys."""
    
    @staticmethod
    def generate(prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a string representation of all arguments
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex objects, use hash
                key_parts.append(str(hash(str(arg))))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (str, int, float, bool)):
                key_parts.append(f"{key}={value}")
            else:
                key_parts.append(f"{key}={hash(str(value))}")
        
        # Join with separator and hash if too long
        cache_key = ":".join(key_parts)
        
        if len(cache_key) > 250:  # Reasonable limit for cache keys
            # Hash the key if it's too long
            cache_key = f"{prefix}:{hashlib.sha256(cache_key.encode()).hexdigest()[:32]}"
        
        return cache_key
    
    @staticmethod
    def evaluation_key(text: str, evaluation_types: list, model: str) -> str:
        """Generate cache key for evaluation results."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        types_str = ":".join(sorted(evaluation_types))
        return f"eval:{text_hash}:{types_str}:{model}"
    
    @staticmethod
    def model_analysis_key(model: str, prompts_hash: str, depth: str) -> str:
        """Generate cache key for model analysis."""
        return f"analysis:{model}:{prompts_hash}:{depth}"


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(self, cache_manager: CacheManager, ttl: int = 3600, key_prefix: str = "func"):
        self.cache_manager = cache_manager
        self.ttl = ttl
        self.key_prefix = key_prefix
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = CacheKey.generate(f"{self.key_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await self.cache_manager.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper


# Global cache instance
cache_manager = CacheManager() 