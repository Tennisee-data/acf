"""SlowAPI Basic Rate Limiting for FastAPI.

Keywords: rate limit, slowapi, throttle, 429, too many requests

This is the RECOMMENDED approach for rate limiting in FastAPI.
SlowAPI is battle-tested, handles edge cases, and integrates cleanly.

Requirements:
    pip install slowapi

Why use SlowAPI instead of custom code:
    - Handles TTL correctly (sets on every request, not just on exceed)
    - Proper sliding window algorithm
    - Returns correct Retry-After headers
    - Supports multiple backends (memory, Redis, Memcached)
    - Thread-safe and async-compatible
"""

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Create limiter with IP-based rate limiting
# key_func determines what to rate limit by (IP address here)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

# Register the limiter with the app
app.state.limiter = limiter

# Add custom handler for rate limit exceeded
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
@limiter.limit("10/minute")  # 10 requests per minute
async def homepage(request: Request):
    """Public endpoint with moderate rate limit."""
    return {"message": "Welcome!"}


@app.get("/api/data")
@limiter.limit("100/hour")  # 100 requests per hour
async def get_data(request: Request):
    """API endpoint with hourly limit."""
    return {"data": "some data"}


@app.post("/api/expensive")
@limiter.limit("5/minute")  # 5 requests per minute for expensive operations
async def expensive_operation(request: Request):
    """Expensive operation with strict limit."""
    return {"result": "processed"}


# Rate limit format examples:
# "10/minute" or "10/m" - 10 per minute
# "100/hour" or "100/h" - 100 per hour
# "1000/day" or "1000/d" - 1000 per day
# "5/second" or "5/s" - 5 per second
# "5/15minutes" - 5 per 15 minutes (custom window)
