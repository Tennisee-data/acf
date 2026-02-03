"""Redis Caching Pattern for FastAPI.

Keywords: cache, redis, caching, ttl, performance

Use Redis caching for:
- Expensive database queries
- API responses
- Session data
- Rate limiting data

Requirements:
    pip install redis

Key points:
- Always set TTL (don't let cache grow forever)
- Handle Redis failures gracefully
- Use consistent key naming
- Consider cache stampede prevention
"""

from fastapi import FastAPI, Depends, HTTPException
from functools import wraps
from typing import Callable, Any
import redis
import json
import hashlib
import os

# Redis connection
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI()


def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    ttl_seconds: int = 300,
    prefix: str = "cache",
    key_builder: Callable[..., str] | None = None,
):
    """Decorator for caching function results.

    Args:
        ttl_seconds: Cache TTL in seconds (default 5 minutes)
        prefix: Key prefix for organization
        key_builder: Custom function to build cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key
            if key_builder:
                key_suffix = key_builder(*args, **kwargs)
            else:
                key_suffix = cache_key(*args, **kwargs)

            cache_key_full = f"{prefix}:{func.__name__}:{key_suffix}"

            # Try to get from cache
            try:
                cached_value = redis_client.get(cache_key_full)
                if cached_value is not None:
                    return json.loads(cached_value)
            except redis.RedisError:
                # Redis error - proceed without cache
                pass

            # Call the actual function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Store in cache
            try:
                redis_client.setex(
                    cache_key_full,
                    ttl_seconds,
                    json.dumps(result)
                )
            except redis.RedisError:
                # Failed to cache, but we have the result
                pass

            return result
        return wrapper
    return decorator


import asyncio


# Example usage
@cached(ttl_seconds=60, prefix="users")
async def get_user_data(user_id: int) -> dict:
    """Get user data (expensive operation)."""
    # Simulate database query
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": f"User {user_id}"}


@cached(ttl_seconds=300, prefix="products")
async def get_product_list(category: str, page: int = 1) -> list:
    """Get product list (expensive operation)."""
    await asyncio.sleep(0.2)
    return [{"id": i, "name": f"Product {i}"} for i in range((page-1)*10, page*10)]


# Manual cache management
class CacheManager:
    """Manual cache operations for fine-grained control."""

    def __init__(self, client: redis.Redis, prefix: str = "app"):
        self.client = client
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            value = self.client.get(self._key(key))
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL."""
        try:
            self.client.setex(
                self._key(key),
                ttl,
                json.dumps(value)
            )
            return True
        except redis.RedisError:
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            self.client.delete(self._key(key))
            return True
        except redis.RedisError:
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            keys = self.client.keys(f"{self.prefix}:{pattern}")
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError:
            return 0


# Dependency
def get_cache() -> CacheManager:
    return CacheManager(redis_client)


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user with caching."""
    return await get_user_data(user_id)


@app.get("/products")
async def list_products(category: str = "all", page: int = 1):
    """Get products with caching."""
    return await get_product_list(category, page)


@app.post("/cache/invalidate")
async def invalidate_cache(pattern: str, cache: CacheManager = Depends(get_cache)):
    """Invalidate cache entries matching pattern."""
    count = cache.invalidate_pattern(pattern)
    return {"invalidated": count}
