"""API Key Authentication Pattern.

Keywords: api key, authentication, header, x-api-key, service auth

Use API keys for:
- Service-to-service authentication
- Public APIs with rate limiting
- Simpler auth where OAuth is overkill

Requirements:
    pip install fastapi

Security considerations:
- Always use HTTPS (keys are sent in headers)
- Store hashed keys, not plaintext
- Rotate keys periodically
- Use different keys for different environments
"""

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional
import secrets
import os

app = FastAPI()

# API key can be in header or query parameter
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Password context for hashing API keys
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class APIKey(BaseModel):
    key_id: str
    name: str
    hashed_key: str
    scopes: list[str]
    is_active: bool = True


# In production, store in database
# Keys are stored HASHED, not plaintext
API_KEYS_DB: dict[str, APIKey] = {}


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (key_id, raw_key)
        - key_id: For referencing the key (safe to log)
        - raw_key: The actual secret (only shown once!)
    """
    key_id = secrets.token_urlsafe(8)  # Short ID for reference
    raw_key = secrets.token_urlsafe(32)  # Actual secret
    return key_id, raw_key


def hash_api_key(raw_key: str) -> str:
    """Hash API key for storage."""
    return pwd_context.hash(raw_key)


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """Verify API key against hash."""
    return pwd_context.verify(raw_key, hashed_key)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> APIKey:
    """Dependency to validate API key from header or query.

    Checks both X-API-Key header and ?api_key query parameter.
    Header takes precedence.
    """
    api_key = api_key_header or api_key_query

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check against stored keys
    for key_data in API_KEYS_DB.values():
        if key_data.is_active and verify_api_key(api_key, key_data.hashed_key):
            return key_data

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


def require_scope(scope: str):
    """Dependency factory to require specific scope."""
    async def check_scope(api_key: APIKey = Depends(get_api_key)) -> APIKey:
        if scope not in api_key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' required",
            )
        return api_key
    return check_scope


# Setup: Create initial API key
@app.on_event("startup")
async def create_initial_key():
    """Create a demo API key on startup."""
    key_id, raw_key = generate_api_key()

    API_KEYS_DB[key_id] = APIKey(
        key_id=key_id,
        name="Demo Key",
        hashed_key=hash_api_key(raw_key),
        scopes=["read", "write"],
    )

    # In production, show this only once during key creation!
    print(f"Demo API Key (use in X-API-Key header): {raw_key}")


@app.get("/public")
async def public_endpoint():
    """Public endpoint - no auth required."""
    return {"message": "Public data"}


@app.get("/protected")
async def protected_endpoint(api_key: APIKey = Depends(get_api_key)):
    """Protected endpoint - requires valid API key."""
    return {"message": "Protected data", "key_name": api_key.name}


@app.get("/admin")
async def admin_endpoint(api_key: APIKey = Depends(require_scope("admin"))):
    """Admin endpoint - requires 'admin' scope."""
    return {"message": "Admin data", "key_name": api_key.name}


@app.post("/data")
async def create_data(api_key: APIKey = Depends(require_scope("write"))):
    """Write endpoint - requires 'write' scope."""
    return {"message": "Data created", "key_name": api_key.name}


# Key management endpoints (admin only in production)
@app.post("/keys")
async def create_key(name: str, scopes: list[str]):
    """Create a new API key."""
    key_id, raw_key = generate_api_key()

    API_KEYS_DB[key_id] = APIKey(
        key_id=key_id,
        name=name,
        hashed_key=hash_api_key(raw_key),
        scopes=scopes,
    )

    # Return raw key only on creation - can't be recovered later!
    return {
        "key_id": key_id,
        "api_key": raw_key,
        "message": "Save this key - it won't be shown again!"
    }


@app.delete("/keys/{key_id}")
async def revoke_key(key_id: str):
    """Revoke an API key."""
    if key_id in API_KEYS_DB:
        API_KEYS_DB[key_id].is_active = False
        return {"message": "Key revoked"}
    raise HTTPException(status_code=404, detail="Key not found")
