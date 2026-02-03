"""JWT with Refresh Tokens Pattern.

Keywords: jwt, refresh token, authentication, token rotation, security

This pattern implements access + refresh token flow:
- Short-lived access tokens (15-60 min)
- Long-lived refresh tokens (7-30 days)
- Refresh token rotation for security
- Server-side token storage for revocation

Requirements:
    pip install python-jose[cryptography] passlib[bcrypt] redis

Why refresh tokens:
- Access tokens can't be revoked (stateless)
- Refresh tokens are stored server-side, can be revoked
- Reduces exposure if access token is leaked (short lifetime)
"""

from datetime import datetime, timedelta
from typing import Optional
import uuid

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import redis.asyncio as redis
import os

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived
REFRESH_TOKEN_EXPIRE_DAYS = 7     # Longer-lived

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Redis client for refresh token storage
redis_client: redis.Redis = None


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


def create_access_token(username: str) -> str:
    """Create short-lived access token."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": username, "exp": expire, "type": "access"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def create_refresh_token(username: str) -> str:
    """Create refresh token and store in Redis.

    Refresh tokens are:
    - Stored server-side for revocation capability
    - Have unique IDs to track and rotate
    - Associated with user for validation
    """
    token_id = str(uuid.uuid4())
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Store in Redis with TTL
    key = f"refresh_token:{token_id}"
    await redis_client.setex(
        key,
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        username
    )

    # Also track user's active refresh tokens (for "logout all devices")
    user_tokens_key = f"user_tokens:{username}"
    await redis_client.sadd(user_tokens_key, token_id)
    await redis_client.expire(user_tokens_key, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    # Create JWT with token ID (the actual refresh token)
    to_encode = {"sub": username, "exp": expire, "jti": token_id, "type": "refresh"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def verify_refresh_token(token: str) -> Optional[str]:
    """Verify refresh token and return username if valid.

    Checks:
    1. JWT signature and expiration
    2. Token exists in Redis (not revoked)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        if payload.get("type") != "refresh":
            return None

        username = payload.get("sub")
        token_id = payload.get("jti")

        if not username or not token_id:
            return None

        # Check if token is still valid in Redis
        key = f"refresh_token:{token_id}"
        stored_username = await redis_client.get(key)

        if stored_username != username:
            return None

        return username

    except JWTError:
        return None


async def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_id = payload.get("jti")
        username = payload.get("sub")

        if token_id:
            await redis_client.delete(f"refresh_token:{token_id}")

        if username and token_id:
            await redis_client.srem(f"user_tokens:{username}", token_id)

        return True
    except JWTError:
        return False


async def revoke_all_user_tokens(username: str) -> int:
    """Revoke all refresh tokens for a user (logout all devices)."""
    user_tokens_key = f"user_tokens:{username}"
    token_ids = await redis_client.smembers(user_tokens_key)

    count = 0
    for token_id in token_ids:
        await redis_client.delete(f"refresh_token:{token_id}")
        count += 1

    await redis_client.delete(user_tokens_key)
    return count


@app.post("/token", response_model=TokenPair)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access + refresh token pair."""
    # Authenticate user (simplified - use real auth)
    if form_data.username != "admin" or form_data.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(form_data.username)
    refresh_token = await create_refresh_token(form_data.username)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@app.post("/token/refresh", response_model=TokenPair)
async def refresh_tokens(request: RefreshRequest):
    """Get new token pair using refresh token.

    Implements token rotation: old refresh token is revoked,
    new one is issued. This limits damage if refresh token leaks.
    """
    username = await verify_refresh_token(request.refresh_token)

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Revoke old refresh token (rotation)
    await revoke_refresh_token(request.refresh_token)

    # Issue new token pair
    access_token = create_access_token(username)
    refresh_token = await create_refresh_token(username)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@app.post("/logout")
async def logout(request: RefreshRequest):
    """Logout - revoke refresh token."""
    await revoke_refresh_token(request.refresh_token)
    return {"message": "Logged out"}


@app.post("/logout/all")
async def logout_all(current_user: str = Depends(oauth2_scheme)):
    """Logout from all devices - revoke all refresh tokens."""
    # In real app, get username from token
    payload = jwt.decode(current_user, SECRET_KEY, algorithms=[ALGORITHM])
    username = payload.get("sub")

    count = await revoke_all_user_tokens(username)
    return {"message": f"Revoked {count} sessions"}
