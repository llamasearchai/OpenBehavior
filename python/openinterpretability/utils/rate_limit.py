"""
Rate limiting utilities for OpenInterpretability API.
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
from .config import config_manager


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """
    
    def __init__(self, default_rate: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            default_rate: Default requests per window
            window_seconds: Time window in seconds
        """
        self.default_rate = default_rate
        self.window_seconds = window_seconds
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.last_cleanup = time.time()
    
    async def check_rate_limit(self, api_key: str, weight: int = 1) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            api_key: API key to check
            weight: Request weight (default 1)
            
        Returns:
            True if within limit, False otherwise
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > 300:  # Every 5 minutes
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get or create bucket for this API key
        bucket = self.buckets[api_key]
        
        # Remove old entries from this bucket
        cutoff_time = current_time - self.window_seconds
        while bucket and bucket[0] < cutoff_time:
            bucket.popleft()
        
        # Check if adding this request would exceed limit
        current_count = len(bucket)
        if current_count + weight > self.default_rate:
            return False
        
        # Add request timestamps
        for _ in range(weight):
            bucket.append(current_time)
        
        return True
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """Clean up old entries from all buckets."""
        cutoff_time = current_time - self.window_seconds
        
        keys_to_remove = []
        for api_key, bucket in self.buckets.items():
            # Remove old entries
            while bucket and bucket[0] < cutoff_time:
                bucket.popleft()
            
            # Remove empty buckets
            if not bucket:
                keys_to_remove.append(api_key)
        
        for key in keys_to_remove:
            del self.buckets[key]
    
    def get_remaining_requests(self, api_key: str) -> int:
        """
        Get remaining requests for an API key.
        
        Args:
            api_key: API key to check
            
        Returns:
            Number of remaining requests
        """
        current_time = time.time()
        bucket = self.buckets[api_key]
        
        # Remove old entries
        cutoff_time = current_time - self.window_seconds
        while bucket and bucket[0] < cutoff_time:
            bucket.popleft()
        
        return max(0, self.default_rate - len(bucket))
    
    def reset_bucket(self, api_key: str) -> None:
        """
        Reset rate limit bucket for an API key.
        
        Args:
            api_key: API key to reset
        """
        if api_key in self.buckets:
            self.buckets[api_key].clear()


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more precise control.
    """
    
    def __init__(self, window_size: int = 3600, max_requests: int = 100):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: Dict[str, deque] = defaultdict(deque)
        
    async def check_rate_limit(self, key: str, weight: int = 1) -> bool:
        """
        Check if request is within rate limits using sliding window.
        
        Args:
            key: Identifier for rate limiting
            weight: Weight of this request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Remove old requests outside the window
        request_queue = self.requests[key]
        while request_queue and request_queue[0] < cutoff_time:
            request_queue.popleft()
        
        # Check if adding this request would exceed the limit
        current_count = len(request_queue)
        if current_count + weight <= self.max_requests:
            # Add the request(s) to the queue
            for _ in range(weight):
                request_queue.append(current_time)
            return True
        
        return False
    
    def get_rate_limit_info(self, key: str) -> Dict:
        """Get current rate limit status for a key."""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Clean up old requests
        request_queue = self.requests[key]
        while request_queue and request_queue[0] < cutoff_time:
            request_queue.popleft()
        
        current_count = len(request_queue)
        
        # Calculate when the next request slot will be available
        reset_time = None
        if current_count >= self.max_requests and request_queue:
            reset_time = request_queue[0] + self.window_size
        
        return {
            "current_requests": current_count,
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "reset_time": reset_time,
            "requests_remaining": max(0, self.max_requests - current_count)
        }


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load.
    """
    
    def __init__(self):
        self.base_limiter = RateLimiter()
        self.system_load = 0.0
        self.load_history = deque(maxlen=60)  # Track load for last 60 seconds
        
    async def check_rate_limit(self, key: str, weight: int = 1) -> bool:
        """
        Check rate limit with adaptive adjustment based on system load.
        
        Args:
            key: Identifier for rate limiting
            weight: Weight of this request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Calculate current system load
        await self._update_system_load()
        
        # Adjust weight based on system load
        if self.system_load > 0.8:
            # High load - increase weight to reduce allowed requests
            adjusted_weight = weight * 2
        elif self.system_load > 0.6:
            # Medium load - slight increase in weight
            adjusted_weight = int(weight * 1.5)
        else:
            # Normal load - no adjustment
            adjusted_weight = weight
        
        return await self.base_limiter.check_rate_limit(key, adjusted_weight)
    
    async def _update_system_load(self) -> None:
        """Update system load metrics."""
        current_time = time.time()
        
        # Simple load calculation based on active rate limit buckets
        # In production, use actual system metrics
        active_buckets = len(self.base_limiter.buckets)
        load = min(1.0, active_buckets / 100.0)  # Normalize to 0-1
        
        self.load_history.append((current_time, load))
        
        # Calculate average load over last 60 seconds
        recent_loads = [load for timestamp, load in self.load_history if current_time - timestamp < 60]
        self.system_load = sum(recent_loads) / len(recent_loads) if recent_loads else 0.0
    
    def get_system_load(self) -> float:
        """Get current system load."""
        return self.system_load 