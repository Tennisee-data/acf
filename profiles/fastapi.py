"""FastAPI Stack Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects FastAPI as their tech stack. It provides version-specific
patterns, correct import syntax, and common pitfalls to avoid.
"""

PROFILE_NAME = "fastapi"
PROFILE_VERSION = "1.1"

# Metadata
DESCRIPTION = "Modern FastAPI with Pydantic v2 and SQLAlchemy 2.0"
AUTHOR = "AgentCodeFactory"
LAST_UPDATED = "2025-01"
ICON = "fastapi"

# Technologies covered by this profile
TECHNOLOGIES = ["fastapi", "pydantic", "sqlalchemy", "uvicorn"]

# Conflict handling
CONFLICTS_WITH = ["django", "flask"]  # Mutually exclusive web frameworks
PRIORITY = 50  # Lower = higher priority (FastAPI preferred over Flask)

# Keyword matching (improved)
EXACT_KEYWORDS = ["fastapi", "pydantic", "uvicorn"]  # Match as whole words
SUBSTRING_KEYWORDS = ["fast api", "fast-api"]  # Match as substrings

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## FastAPI Expert Guidelines

You are generating FastAPI code. Follow these patterns exactly:

### Imports (CRITICAL - get these right)
```python
# FastAPI core - always use specific imports
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Pydantic v2 (CURRENT VERSION) - NOT v1 syntax
from pydantic import BaseModel, Field, ConfigDict, field_validator
# WRONG: from pydantic import validator  # This is v1 syntax
# CORRECT: from pydantic import field_validator  # This is v2 syntax

# SQLAlchemy 2.0 style
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, select
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase, relationship
# WRONG: from sqlalchemy.ext.declarative import declarative_base  # This is 1.x style
# CORRECT: from sqlalchemy.orm import DeclarativeBase  # This is 2.0 style

# Async SQLAlchemy (if using async)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
```

### Pydantic v2 Patterns (CRITICAL)
```python
# Model definition with v2 syntax
class UserCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # NOT: class Config: orm_mode = True

    email: str = Field(..., description="User email")
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator('email')  # NOT: @validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
```

### Dependency Injection
```python
# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Use in route
@router.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    ...
```

### Router Organization
```python
# In routers/users.py
router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/")
async def list_users(...):
    ...

# In main.py
from routers import users
app.include_router(users.router)
```

### Error Handling
```python
# Use HTTPException with status codes
from fastapi import HTTPException, status

raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="User not found"
)

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

### Authentication Pattern
```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token and return user
    ...

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
```

### Common Mistakes to Avoid
1. Using `@validator` instead of `@field_validator` (Pydantic v1 vs v2)
2. Using `orm_mode = True` instead of `model_config = ConfigDict(from_attributes=True)`
3. Using `declarative_base()` instead of `class Base(DeclarativeBase)`
4. Forgetting `async` on route handlers that use `await`
5. Not using `status.HTTP_*` constants for status codes
6. Forgetting to add router to app with `app.include_router()`
7. Implementing custom rate limiting instead of using slowapi or fastapi-limiter
8. Using in-memory dict for rate limiting (fails in multi-instance deployments)

### Rate Limiting (USE LIBRARIES, NOT CUSTOM CODE)
For rate limiting, ALWAYS use established libraries. Custom implementations have subtle bugs.

```python
# RECOMMENDED: slowapi (most popular, battle-tested)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/login")
@limiter.limit("5/15minutes")
async def login(request: Request, credentials: LoginRequest):
    ...

# For Redis backend (distributed):
# limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")

# ALTERNATIVE: fastapi-limiter
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    await FastAPILimiter.init(redis)

@app.post("/login", dependencies=[Depends(RateLimiter(times=5, minutes=15))])
async def login(...):
    ...
```

NEVER implement rate limiting manually with Redis INCR/EXPIRE - it has edge cases:
- TTL not set on first increment = permanent ban
- Counting successful logins instead of just failures
- Race conditions in distributed systems
- Missing Retry-After headers

### Required Dependencies (requirements.txt)
```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
sqlalchemy>=2.0.0
```

### Project Structure
```
project/
├── main.py              # FastAPI app, include routers
├── routers/
│   ├── __init__.py
│   └── users.py         # APIRouter for user endpoints
├── models/
│   ├── __init__.py
│   └── user.py          # SQLAlchemy models
├── schemas/
│   ├── __init__.py
│   └── user.py          # Pydantic schemas
├── services/
│   └── user_service.py  # Business logic
├── database.py          # DB session, engine
├── config.py            # Settings with pydantic-settings
├── .env.example         # REQUIRED: Environment template
├── requirements.txt     # Dependencies
└── README.md            # Setup instructions
```

### Environment Configuration (CRITICAL)
ALWAYS generate `.env.example` with all required variables:
```
# .env.example - Copy to .env and configure
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=change-me-in-production
DEBUG=true
```
If README mentions `.env.example`, the file MUST exist.
"""

# Injected into requirements.txt guidance
DEPENDENCIES = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-dotenv>=1.0.0",
]

OPTIONAL_DEPENDENCIES = {
    "database": ["sqlalchemy>=2.0.0", "alembic>=1.13.0"],
    "async_database": ["sqlalchemy[asyncio]>=2.0.0", "asyncpg>=0.29.0"],
    "auth": ["python-jose[cryptography]>=3.3.0", "passlib[bcrypt]>=1.7.4"],
    "testing": ["pytest>=7.4.0", "pytest-asyncio>=0.23.0", "httpx>=0.26.0"],
    "rate_limiting": ["slowapi>=0.1.9"],
    "rate_limiting_redis": ["slowapi>=0.1.9", "redis>=4.0.0"],
}

# Keywords that trigger this profile
TRIGGER_KEYWORDS = [
    "fastapi",
    "fast api",
    "pydantic",
    "uvicorn",
    "async api",
    "rest api python",
    "rate limit",
    "rate-limit",
    "throttle",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied.

    Args:
        tech_stack: User-selected technologies (e.g., ["python", "fastapi"])
        prompt: The feature description

    Returns:
        True if this profile should be applied
    """
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["fastapi", "fast-api"]):
            return True

    # Check prompt keywords
    return any(kw in prompt_lower for kw in TRIGGER_KEYWORDS)


def get_guidance() -> str:
    """Get the full guidance text to inject into prompts."""
    return SYSTEM_GUIDANCE


def get_dependencies(features: list[str] | None = None) -> list[str]:
    """Get recommended dependencies.

    Args:
        features: Optional list of features (e.g., ["database", "auth"])

    Returns:
        List of pip dependencies
    """
    deps = DEPENDENCIES.copy()

    if features:
        for feature in features:
            if feature in OPTIONAL_DEPENDENCIES:
                deps.extend(OPTIONAL_DEPENDENCIES[feature])

    return deps
