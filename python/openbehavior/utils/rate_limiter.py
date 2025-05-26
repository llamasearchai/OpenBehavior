"""
Rate limiting utilities for API calls and request management.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_limit: int = 10
    backoff_factor: float = 1.5
    max_retries: int = 3


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 3600,
        burst_limit: Optional[int] = None
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit or min(requests_per_minute, 10)
        
        # Token buckets
        self.minute_tokens = self.requests_per_minute
        self.hour_tokens = self.requests_per_hour
        self.burst_tokens = self.burst_limit
        
        # Timestamps
        self.last_minute_refill = time.time()
        self.last_hour_refill = time.time()
        self.last_burst_refill = time.time()
        
        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the rate limiter."""
        async with self._lock:
            self._refill_tokens()
            
            # Check if we have enough tokens in all buckets
            if (self.minute_tokens >= tokens and 
                self.hour_tokens >= tokens and 
                self.burst_tokens >= tokens):
                
                # Consume tokens
                self.minute_tokens -= tokens
                self.hour_tokens -= tokens
                self.burst_tokens -= tokens
                
                self.total_requests += tokens
                return True
            else:
                self.rejected_requests += tokens
                logger.warning(f"Rate limit exceeded. Available tokens: "
                             f"minute={self.minute_tokens}, hour={self.hour_tokens}, "
                             f"burst={self.burst_tokens}")
                return False
    
    async def acquire_with_wait(self, tokens: int = 1, max_wait: float = 60.0) -> bool:
        """Acquire tokens, waiting if necessary."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self.acquire(tokens):
                return True
            
            # Calculate wait time based on token refill rate
            wait_time = self._calculate_wait_time(tokens)
            await asyncio.sleep(min(wait_time, 1.0))
        
        return False
    
    def _refill_tokens(self):
        """Refill token buckets based on elapsed time."""
        current_time = time.time()
        
        # Refill minute bucket
        time_since_minute = current_time - self.last_minute_refill
        if time_since_minute >= 60.0:
            self.minute_tokens = self.requests_per_minute
            self.last_minute_refill = current_time
        else:
            # Gradual refill
            tokens_to_add = (time_since_minute / 60.0) * self.requests_per_minute
            self.minute_tokens = min(
                self.requests_per_minute,
                self.minute_tokens + tokens_to_add
            )
        
        # Refill hour bucket
        time_since_hour = current_time - self.last_hour_refill
        if time_since_hour >= 3600.0:
            self.hour_tokens = self.requests_per_hour
            self.last_hour_refill = current_time
        else:
            # Gradual refill
            tokens_to_add = (time_since_hour / 3600.0) * self.requests_per_hour
            self.hour_tokens = min(
                self.requests_per_hour,
                self.hour_tokens + tokens_to_add
            )
        
        # Refill burst bucket (refills faster)
        time_since_burst = current_time - self.last_burst_refill
        if time_since_burst >= 10.0:  # Refill every 10 seconds
            self.burst_tokens = self.burst_limit
            self.last_burst_refill = current_time
        else:
            tokens_to_add = (time_since_burst / 10.0) * self.burst_limit
            self.burst_tokens = min(
                self.burst_limit,
                self.burst_tokens + tokens_to_add
            )
    
    def _calculate_wait_time(self, tokens: int) -> float:
        """Calculate how long to wait for tokens to be available."""
        self._refill_tokens()
        
        # Calculate wait time for each bucket
        minute_wait = 0.0
        hour_wait = 0.0
        burst_wait = 0.0
        
        if self.minute_tokens < tokens:
            deficit = tokens - self.minute_tokens
            minute_wait = (deficit / self.requests_per_minute) * 60.0
        
        if self.hour_tokens < tokens:
            deficit = tokens - self.hour_tokens
            hour_wait = (deficit / self.requests_per_hour) * 3600.0
        
        if self.burst_tokens < tokens:
            deficit = tokens - self.burst_tokens
            burst_wait = (deficit / self.burst_limit) * 10.0
        
        # Return the maximum wait time needed
        return max(minute_wait, hour_wait, burst_wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        self._refill_tokens()
        
        return {
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": (
                (self.total_requests - self.rejected_requests) / self.total_requests
                if self.total_requests > 0 else 1.0
            ),
            "available_tokens": {
                "minute": self.minute_tokens,
                "hour": self.hour_tokens,
                "burst": self.burst_tokens
            },
            "limits": {
                "requests_per_minute": self.requests_per_minute,
                "requests_per_hour": self.requests_per_hour,
                "burst_limit": self.burst_limit
            }
        }
    
    def reset(self):
        """Reset the rate limiter."""
        current_time = time.time()
        
        self.minute_tokens = self.requests_per_minute
        self.hour_tokens = self.requests_per_hour
        self.burst_tokens = self.burst_limit
        
        self.last_minute_refill = current_time
        self.last_hour_refill = current_time
        self.last_burst_refill = current_time
        
        self.total_requests = 0
        self.rejected_requests = 0


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on API response patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.error_count = 0
        self.success_count = 0
        self.last_error_time = 0
        self.backoff_multiplier = 1.0
        self.max_backoff = 10.0
    
    async def acquire_with_backoff(self, tokens: int = 1) -> bool:
        """Acquire tokens with adaptive backoff."""
        # Apply backoff if we've had recent errors
        if self.backoff_multiplier > 1.0:
            effective_tokens = int(tokens * self.backoff_multiplier)
        else:
            effective_tokens = tokens
        
        return await self.acquire(effective_tokens)
    
    def record_success(self):
        """Record a successful API call."""
        self.success_count += 1
        
        # Gradually reduce backoff on success
        if self.backoff_multiplier > 1.0:
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)
    
    def record_error(self, error_type: str = "rate_limit"):
        """Record an API error."""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Increase backoff on errors
        if error_type == "rate_limit":
            self.backoff_multiplier = min(self.max_backoff, self.backoff_multiplier * 1.5)
        elif error_type == "server_error":
            self.backoff_multiplier = min(self.max_backoff, self.backoff_multiplier * 2.0)
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics."""
        base_stats = self.get_stats()
        
        base_stats.update({
            "error_count": self.error_count,
            "success_count": self.success_count,
            "error_rate": (
                self.error_count / (self.error_count + self.success_count)
                if (self.error_count + self.success_count) > 0 else 0.0
            ),
            "backoff_multiplier": self.backoff_multiplier,
            "time_since_last_error": time.time() - self.last_error_time
        })
        
        return base_stats


class MultiProviderRateLimiter:
    """Rate limiter that manages multiple API providers."""
    
    def __init__(self, provider_configs: Dict[str, RateLimitConfig]):
        self.limiters = {}
        
        for provider, config in provider_configs.items():
            self.limiters[provider] = AdaptiveRateLimiter(
                requests_per_minute=config.requests_per_minute,
                requests_per_hour=config.requests_per_hour,
                burst_limit=config.burst_limit
            )
    
    async def acquire(self, provider: str, tokens: int = 1) -> bool:
        """Acquire tokens for a specific provider."""
        if provider not in self.limiters:
            logger.warning(f"Unknown provider: {provider}")
            return False
        
        return await self.limiters[provider].acquire_with_backoff(tokens)
    
    async def acquire_with_wait(
        self,
        provider: str,
        tokens: int = 1,
        max_wait: float = 60.0
    ) -> bool:
        """Acquire tokens with waiting for a specific provider."""
        if provider not in self.limiters:
            logger.warning(f"Unknown provider: {provider}")
            return False
        
        return await self.limiters[provider].acquire_with_wait(tokens, max_wait)
    
    def record_success(self, provider: str):
        """Record success for a provider."""
        if provider in self.limiters:
            self.limiters[provider].record_success()
    
    def record_error(self, provider: str, error_type: str = "rate_limit"):
        """Record error for a provider."""
        if provider in self.limiters:
            self.limiters[provider].record_error(error_type)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        return {
            provider: limiter.get_adaptive_stats()
            for provider, limiter in self.limiters.items()
        }
    
    def get_best_provider(self) -> Optional[str]:
        """Get the provider with the best availability."""
        best_provider = None
        best_score = -1
        
        for provider, limiter in self.limiters.items():
            stats = limiter.get_adaptive_stats()
            
            # Calculate availability score
            available_tokens = min(
                stats["available_tokens"]["minute"],
                stats["available_tokens"]["hour"],
                stats["available_tokens"]["burst"]
            )
            
            # Factor in error rate and backoff
            error_penalty = stats["error_rate"] * 0.5
            backoff_penalty = (stats["backoff_multiplier"] - 1.0) * 0.3
            
            score = available_tokens - error_penalty - backoff_penalty
            
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider 