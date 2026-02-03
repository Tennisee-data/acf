# FastAPI Patterns RAG Extension

Production-ready FastAPI patterns for common use cases. This extension provides battle-tested code examples that the ACF pipeline can inject into code generation prompts.

## What's Included

| Category | Patterns | Description |
|----------|----------|-------------|
| **Rate Limiting** | slowapi-basic, slowapi-redis, fastapi-limiter, login-rate-limit | Proper rate limiting using proven libraries |
| **Authentication** | jwt-basic, jwt-refresh, api-key-auth | JWT and API key authentication patterns |
| **Webhooks** | stripe-webhook, github-webhook | Signature verification and idempotency |
| **Database** | sqlalchemy-async, sqlalchemy-sync | SQLAlchemy 2.0 with proper session handling |
| **File Uploads** | validated-upload | Secure file upload with magic byte validation |
| **Caching** | redis-cache | Redis caching with TTL and error handling |
| **Error Handling** | exception-handlers | Consistent error responses and logging |
| **Testing** | pytest-async | Async testing with httpx and fixtures |

## Why Use This Extension

The ACF pipeline can generate code, but without good examples, it may produce implementations with subtle bugs. This extension provides:

1. **Correct patterns**: Each example is production-tested
2. **Common pitfall avoidance**: Comments explain what NOT to do
3. **Library recommendations**: Use battle-tested libraries instead of custom code
4. **Best practices**: Security, performance, and maintainability

## Example: Rate Limiting

Without this extension, the pipeline might generate custom rate limiting code like:

```python
# BAD: Custom rate limiting with bugs
failed_attempts = await redis.incr(key)
if failed_attempts > RATE_LIMIT:
    await redis.expire(key, TTL)  # BUG: TTL not set on first increment!
    raise HTTPException(429, "Rate limited")
```

With this extension, the pipeline learns to use proven libraries:

```python
# GOOD: Using slowapi (battle-tested)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/login")
@limiter.limit("5/15minutes")
async def login(request: Request):
    ...
```

## Installation

```bash
acf extensions install fastapi-patterns
```

## Usage

Once installed, the extension automatically activates when your feature description matches trigger keywords like:

- "rate limit", "throttle", "429"
- "jwt", "authentication", "login"
- "webhook", "stripe", "github"
- "database", "sqlalchemy", "postgres"
- "file upload", "multipart"
- "cache", "redis"
- "error handling", "exception"
- "test", "pytest"

## License

Free (MIT) - Part of ACF official extensions.
