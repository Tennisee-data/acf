"""SlowAPI with Redis Backend for Distributed Rate Limiting.

Keywords: rate limit, slowapi, redis, distributed, cluster, multiple instances

Use Redis backend when:
    - Running multiple app instances (load balanced)
    - Need rate limits to persist across restarts
    - Need centralized rate limit tracking

Requirements:
    pip install slowapi redis

CRITICAL: In-memory rate limiting FAILS in multi-instance deployments.
Each instance would have its own counter, allowing N * limit requests.
"""

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os

# Redis connection URL from environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Create limiter with Redis backend for distributed rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL,  # This enables Redis backend
)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/login")
@limiter.limit("5/15minutes")  # 5 login attempts per 15 minutes
async def login(request: Request):
    """Login endpoint with strict rate limiting.

    Rate limit is tracked in Redis, so it works correctly
    even with multiple app instances behind a load balancer.
    """
    # Your login logic here
    return {"status": "logged_in"}


@app.post("/api/webhook")
@limiter.limit("1000/minute")  # High limit for webhooks
async def webhook(request: Request):
    """Webhook endpoint - still rate limited to prevent abuse."""
    return {"status": "received"}


# Redis connection with fallback
# If Redis is down, you can choose to:
# 1. Fail open (allow requests) - better UX, less secure
# 2. Fail closed (block requests) - more secure, worse UX

# For production, consider:
# - Redis Sentinel for high availability
# - Redis Cluster for horizontal scaling
# - Connection pooling for performance
