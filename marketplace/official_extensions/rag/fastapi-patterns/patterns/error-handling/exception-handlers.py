"""Exception Handling Patterns for FastAPI.

Keywords: exception, error, handler, middleware, logging, 500, 404

Proper error handling:
- Consistent error response format
- Logging for debugging
- Don't leak sensitive info in production
- Handle both expected and unexpected errors

Requirements:
    pip install fastapi
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from typing import Any
import logging
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

app = FastAPI()


# Standard error response model
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None
    path: str | None = None


class ValidationErrorResponse(BaseModel):
    error: str = "Validation Error"
    detail: list[dict[str, Any]]


# Custom exceptions
class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        status_code: int = 500,
        detail: str | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class NotFoundError(AppException):
    """Resource not found."""

    def __init__(self, resource: str, id: Any):
        super().__init__(
            message=f"{resource} not found",
            code="NOT_FOUND",
            status_code=404,
            detail=f"{resource} with id '{id}' does not exist",
        )


class AuthenticationError(AppException):
    """Authentication failed."""

    def __init__(self, detail: str = "Invalid credentials"):
        super().__init__(
            message="Authentication failed",
            code="AUTH_ERROR",
            status_code=401,
            detail=detail,
        )


class PermissionError(AppException):
    """Permission denied."""

    def __init__(self, action: str):
        super().__init__(
            message="Permission denied",
            code="FORBIDDEN",
            status_code=403,
            detail=f"You don't have permission to {action}",
        )


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    logger.warning(f"AppException: {exc.code} - {exc.message} - Path: {request.url.path}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail,
            code=exc.code,
            path=str(request.url.path),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPExceptions."""
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail} - Path: {request.url.path}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
            path=str(request.url.path),
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors (Pydantic)."""
    logger.warning(f"ValidationError: {exc.errors()} - Path: {request.url.path}")

    # Format validation errors nicely
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(detail=errors).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions.

    CRITICAL: Don't leak stack traces in production!
    """
    # Log the full error for debugging
    logger.exception(f"Unhandled exception: {request.url.path}")

    # In debug mode, include stack trace
    if DEBUG:
        detail = traceback.format_exc()
    else:
        detail = "An internal error occurred. Please try again later."

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=detail,
            code="INTERNAL_ERROR",
            path=str(request.url.path),
        ).model_dump(),
    )


# Example endpoints demonstrating error handling
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Example endpoint that raises NotFoundError."""
    # Simulate user lookup
    if user_id != 1:
        raise NotFoundError("User", user_id)

    return {"id": user_id, "name": "John Doe"}


@app.post("/admin/action")
async def admin_action():
    """Example endpoint that raises PermissionError."""
    # Simulate permission check
    is_admin = False
    if not is_admin:
        raise PermissionError("perform admin actions")

    return {"status": "done"}


@app.get("/error")
async def trigger_error():
    """Example endpoint that triggers unhandled exception."""
    # This will be caught by unhandled_exception_handler
    raise RuntimeError("Something went wrong!")
