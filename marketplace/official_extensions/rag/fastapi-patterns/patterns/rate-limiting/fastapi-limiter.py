"""FastAPI-Limiter Rate Limiting (Alternative to SlowAPI).

Keywords: rate limit, fastapi-limiter, redis, async, dependency

FastAPI-Limiter is another good option, especially if you prefer
dependency injection style over decorators.

Requirements:
    pip install fastapi-limiter redis

Comparison with SlowAPI:
    - FastAPI-Limiter: Uses dependencies, async-first, requires Redis
    - SlowAPI: Uses decorators, sync/async, multiple backends

Choose based on your preference and existing patterns.
"""

from fastapi import FastAPI, Request, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as aioredis
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

app = FastAPI()


@app.on_event("startup")
async def startup():
    """Initialize rate limiter with Redis backend."""
    redis = aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await FastAPILimiter.init(redis)


@app.on_event("shutdown")
async def shutdown():
    """Close Redis connection."""
    await FastAPILimiter.close()


# Rate limiting via dependency injection
@app.get("/")
async def homepage():
    """Endpoint without rate limiting."""
    return {"message": "Welcome!"}


@app.get(
    "/api/data",
    dependencies=[Depends(RateLimiter(times=100, hours=1))]
)
async def get_data():
    """Endpoint with 100 requests per hour limit."""
    return {"data": "some data"}


@app.post(
    "/login",
    dependencies=[Depends(RateLimiter(times=5, minutes=15))]
)
async def login():
    """Login with 5 attempts per 15 minutes."""
    return {"status": "logged_in"}


@app.post(
    "/api/expensive",
    dependencies=[Depends(RateLimiter(times=5, minutes=1))]
)
async def expensive_operation():
    """Expensive operation with strict 5/minute limit."""
    return {"result": "processed"}


# Custom identifier function (rate limit by user ID instead of IP)
async def user_identifier(request: Request) -> str:
    """Custom rate limit key based on authenticated user."""
    # Get user from your auth system
    user = getattr(request.state, "user", None)
    if user:
        return f"user:{user.id}"
    # Fall back to IP for unauthenticated users
    return f"ip:{request.client.host}"


@app.get(
    "/api/user-limited",
    dependencies=[Depends(RateLimiter(
        times=100,
        hours=1,
        identifier=user_identifier
    ))]
)
async def user_limited_endpoint():
    """Rate limited by user ID, not IP."""
    return {"data": "per-user limited"}
