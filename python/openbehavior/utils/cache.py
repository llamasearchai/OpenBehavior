"""
Caching utilities for model responses.
"""

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional
import aioredis
import pickle

class ModelCache:
    """Cache for model responses with TTL support."""
    
    def __init__(
        self,
        ttl: int = 3600,
        max_size: int = 10000,
        use_redis: bool = False,
        redis_url: Optional[str] = None
    ):
        self.ttl = ttl
        self.max_size = max_size
        self.use_redis = use_redis
        
        if use_redis:
            self.redis_url = redis_url or "redis://localhost:6379"
            self._redis_pool = None
        else:
            self._memory_cache = {}
            self._access_times = {}
    
    async def get_redis_pool(self):
        """Get or create Redis connection pool."""
        if self._redis_pool is None:
            self._redis_pool = await aioredis.from_url(self.redis_url)
        return self._redis_pool
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self.use_redis:
            return await self._get_redis(key)
        else:
            return await self._get_memory(key)
    
    async def set(self, key: str, value: str) -> None:
        """Set value in cache."""
        if self.use_redis:
            await self._set_redis(key, value)
        else:
            await self._set_memory(key, value)
    
    async def _get_redis(self, key: str) -> Optional[str]:
        """Get from Redis cache."""
        try:
            redis = await self.get_redis_pool()
            cached_data = await redis.get(f"model_cache:{key}")
            
            if cached_data:
                data = pickle.loads(cached_data)
                if data["expires"] > time.time():
                    return data["value"]
                else:
                    await redis.delete(f"model_cache:{key}")
        
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
        
        return None
    
    async def _set_redis(self, key: str, value: str) -> None:
        """Set in Redis cache."""
        try:
            redis = await self.get_redis_pool()
            
            data = {
                "value": value,
                "expires": time.time() + self.ttl
            }
            
            serialized_data = pickle.dumps(data)
            await redis.setex(f"model_cache:{key}", self.ttl, serialized_data)
        
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    async def _get_memory(self, key: str) -> Optional[str]:
        """Get from memory cache."""
        if key in self._memory_cache:
            data = self._memory_cache[key]
            if data["expires"] > time.time():
                self._access_times[key] = time.time()
                return data["value"]
            else:
                del self._memory_cache[key]
                if key in self._access_times:
                    del self._access_times[key]
        
        return None
    
    async def _set_memory(self, key: str, value: str) -> None:
        """Set in memory cache."""
        # Evict old entries if cache is full
        if len(self._memory_cache) >= self.max_size:
            await self._evict_lru()
        
        self._memory_cache[key] = {
            "value": value,
            "expires": time.time() + self.ttl
        }
        self._access_times[key] = time.time()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self._access_times:
            return
        
        # Remove oldest 20% of items
        items_to_remove = max(1, len(self._access_times) // 5)
        
        sorted_items = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self._memory_cache:
                del self._memory_cache[key]
            del self._access_times[key]