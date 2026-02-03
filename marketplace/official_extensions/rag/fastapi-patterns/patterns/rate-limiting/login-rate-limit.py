"""Login-Specific Rate Limiting with Failed Attempt Tracking.

Keywords: login, rate limit, brute force, failed attempts, authentication, security

This pattern rate limits based on FAILED login attempts only,
and resets the counter on successful login.

Requirements:
    pip install slowapi redis

Key points:
    - Only count failed attempts (not successful logins)
    - Reset counter on successful login
    - Consider both IP and username for limiting
    - Return consistent timing to prevent timing attacks
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import redis.asyncio as redis
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Basic rate limiter for general requests
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

# Redis client for custom failed attempt tracking
redis_client: redis.Redis = None

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


class LoginRequest(BaseModel):
    username: str
    password: str


async def check_failed_attempts(ip: str, username: str) -> bool:
    """Check if too many failed attempts.

    Returns True if login should be blocked.
    """
    # Check both IP-based and username-based limits
    ip_key = f"login:failed:ip:{ip}"
    user_key = f"login:failed:user:{username}"

    ip_count = await redis_client.get(ip_key)
    user_count = await redis_client.get(user_key)

    # Block if either exceeds limit
    if ip_count and int(ip_count) >= 5:
        return True
    if user_count and int(user_count) >= 10:
        return True

    return False


async def record_failed_attempt(ip: str, username: str) -> None:
    """Record a failed login attempt.

    CRITICAL: Set TTL on EVERY increment, not just the first!
    This ensures the counter expires even if the first request
    didn't set a TTL.
    """
    ip_key = f"login:failed:ip:{ip}"
    user_key = f"login:failed:user:{username}"

    # Use pipeline for atomic operations
    async with redis_client.pipeline() as pipe:
        # Increment and set TTL atomically
        await pipe.incr(ip_key)
        await pipe.expire(ip_key, 900)  # 15 minutes
        await pipe.incr(user_key)
        await pipe.expire(user_key, 3600)  # 1 hour for username
        await pipe.execute()


async def clear_failed_attempts(ip: str, username: str) -> None:
    """Clear failed attempts on successful login."""
    ip_key = f"login:failed:ip:{ip}"
    user_key = f"login:failed:user:{username}"

    await redis_client.delete(ip_key, user_key)


def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials (placeholder)."""
    # Replace with actual credential verification
    return username == "admin" and password == "secret"


@app.post("/login")
@limiter.limit("20/minute")  # General rate limit as backup
async def login(request: Request, credentials: LoginRequest):
    """Login endpoint with intelligent rate limiting.

    - Blocks after 5 failed attempts per IP (15 min window)
    - Blocks after 10 failed attempts per username (1 hour window)
    - Resets counters on successful login
    """
    ip = request.client.host
    username = credentials.username

    # Check if blocked due to failed attempts
    if await check_failed_attempts(ip, username):
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please try again later.",
            headers={"Retry-After": "900"}  # 15 minutes
        )

    # Verify credentials
    if verify_credentials(username, credentials.password):
        # Success! Clear failed attempt counters
        await clear_failed_attempts(ip, username)
        return {"status": "success", "message": "Logged in"}

    # Failed! Record the attempt
    await record_failed_attempt(ip, username)

    # Return generic error (don't reveal if user exists)
    raise HTTPException(
        status_code=401,
        detail="Invalid credentials"
    )
