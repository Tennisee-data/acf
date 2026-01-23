"""HTTP request tool for API calls and health checks."""

from typing import Any

import requests

from .base import BaseTool, ToolResult, ToolStatus


class HttpTool(BaseTool):
    """Tool for HTTP operations.

    Provides:
    - REST API calls (GET, POST, PUT, DELETE)
    - Health checks
    - Response validation
    """

    name = "http"
    description = "HTTP requests and API operations"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize HTTP tool.

        Args:
            base_url: Base URL for requests
            timeout: Default timeout in seconds
            headers: Default headers for all requests
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout = timeout
        self.default_headers = headers or {}

    def execute(self, operation: str, **kwargs: Any) -> ToolResult:
        """Execute an HTTP operation.

        Args:
            operation: Operation name (get, post, health_check, etc.)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with response data
        """
        operations = {
            "get": self._get,
            "post": self._post,
            "put": self._put,
            "delete": self._delete,
            "health_check": self._health_check,
            "wait_for_ready": self._wait_for_ready,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            return ToolResult(status=ToolStatus.FAILURE, error=str(e))

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        data: Any = None,
        timeout: int | None = None,
    ) -> ToolResult:
        """Make an HTTP request."""
        url = self._build_url(path)
        request_headers = {**self.default_headers, **(headers or {})}

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                json=json_data,
                data=data,
                timeout=timeout or self.timeout,
            )

            # Try to parse JSON response
            try:
                body = response.json()
            except ValueError:
                body = response.text

            result_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body,
                "url": response.url,
            }

            if response.ok:
                return ToolResult(status=ToolStatus.SUCCESS, output=result_data)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    output=result_data,
                    error=f"HTTP {response.status_code}: {response.reason}",
                )

        except requests.exceptions.Timeout:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Request timed out after {timeout or self.timeout}s",
            )
        except requests.exceptions.ConnectionError as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Connection error: {e}",
            )

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ToolResult:
        """HTTP GET request."""
        return self._request("GET", path, headers=headers, params=params)

    def _post(
        self,
        path: str,
        json_data: dict[str, Any] | None = None,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> ToolResult:
        """HTTP POST request."""
        return self._request("POST", path, headers=headers, json_data=json_data, data=data)

    def _put(
        self,
        path: str,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ToolResult:
        """HTTP PUT request."""
        return self._request("PUT", path, headers=headers, json_data=json_data)

    def _delete(
        self,
        path: str,
        headers: dict[str, str] | None = None,
    ) -> ToolResult:
        """HTTP DELETE request."""
        return self._request("DELETE", path, headers=headers)

    def _health_check(
        self,
        path: str = "/health",
        expected_status: int = 200,
        expected_body: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Check endpoint health."""
        result = self._get(path)

        if not result.success:
            return result

        status_code = result.output["status_code"]
        body = result.output["body"]

        checks = {"status_code_match": status_code == expected_status}

        if expected_body:
            checks["body_match"] = all(
                body.get(k) == v for k, v in expected_body.items() if isinstance(body, dict)
            )

        all_passed = all(checks.values())

        return ToolResult(
            status=ToolStatus.SUCCESS if all_passed else ToolStatus.FAILURE,
            output={
                "checks": checks,
                "response": result.output,
            },
            error=None if all_passed else "Health check failed",
        )

    def _wait_for_ready(
        self,
        path: str = "/health",
        max_attempts: int = 30,
        interval: float = 2.0,
        expected_status: int = 200,
    ) -> ToolResult:
        """Wait for endpoint to become ready."""
        import time

        for attempt in range(max_attempts):
            result = self._health_check(path, expected_status=expected_status)
            if result.success:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output={"attempts": attempt + 1, "response": result.output},
                )

            if attempt < max_attempts - 1:
                time.sleep(interval)

        return ToolResult(
            status=ToolStatus.TIMEOUT,
            error=f"Service not ready after {max_attempts} attempts",
            metadata={"attempts": max_attempts},
        )
