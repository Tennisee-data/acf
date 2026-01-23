"""API Probe Generator and Runner.

Generates and executes API probes based on acceptance criteria
to validate that a running container meets feature requirements.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class ProbeResult:
    """Result of a single API probe."""

    criterion_id: str
    criterion_description: str
    success: bool
    message: str
    response_data: dict | None = None
    duration_ms: float = 0.0


@dataclass
class ProbeReport:
    """Complete probe report for all acceptance criteria."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.passed > 0


class APIProbeGenerator:
    """Generates API probe scripts from acceptance criteria.

    Analyzes acceptance criteria and generates executable probes
    that test the running API against each criterion.
    """

    # Patterns to extract API calls from acceptance criteria
    ENDPOINT_PATTERNS = [
        r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}]+)',  # HTTP method + path
        r'endpoint\s+["\']?(/[\w/\-{}]+)',  # "endpoint /path"
        r'call(?:ing)?\s+["\']?(/[\w/\-{}]+)',  # "calling /path"
        r'request(?:ing)?\s+["\']?(/[\w/\-{}]+)',  # "requesting /path"
        r'/api/[\w/\-{}]+',  # Direct /api/ paths
        r'/v\d+/[\w/\-{}]+',  # Versioned paths /v1/...
    ]

    STATUS_CODE_PATTERNS = [
        r'(?:returns?|responds?|status)\s+(\d{3})',  # "returns 200"
        r'(\d{3})\s+(?:response|status|code)',  # "200 response"
        r'HTTP\s+(\d{3})',  # "HTTP 200"
    ]

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize probe generator.

        Args:
            base_url: Base URL of the API to probe
        """
        self.base_url = base_url.rstrip("/")

    def generate_probes(
        self,
        acceptance_criteria: list[dict],
        context: dict | None = None,
    ) -> list[dict]:
        """Generate probe definitions from acceptance criteria.

        Args:
            acceptance_criteria: List of acceptance criteria from feature spec
            context: Additional context (endpoints, schemas, etc.)

        Returns:
            List of probe definitions with method, path, expected results
        """
        probes = []
        context = context or {}

        for criterion in acceptance_criteria:
            if not criterion.get("testable", True):
                continue

            probe = self._generate_probe_from_criterion(criterion, context)
            if probe:
                probes.append(probe)

        return probes

    def _generate_probe_from_criterion(
        self,
        criterion: dict,
        context: dict,
    ) -> dict | None:
        """Generate a single probe from an acceptance criterion.

        Args:
            criterion: Single acceptance criterion
            context: Additional context

        Returns:
            Probe definition or None if can't generate
        """
        description = criterion.get("description", "")
        hint = criterion.get("verification_hint", "")
        criterion_id = criterion.get("id", "AC-???")

        # Combine description and hint for analysis
        text = f"{description} {hint}".lower()

        # Extract endpoint
        endpoint = self._extract_endpoint(text, context)

        # Extract expected status code
        expected_status = self._extract_status_code(text)

        # Determine HTTP method
        method = self._extract_method(text)

        # Extract expected response patterns
        response_checks = self._extract_response_checks(text)

        # Build probe
        probe = {
            "criterion_id": criterion_id,
            "criterion_description": description,
            "verification_hint": hint,
            "method": method,
            "endpoint": endpoint,
            "expected_status": expected_status,
            "response_checks": response_checks,
            "body": self._extract_request_body(text, context),
            "headers": {},
        }

        # Add auth header if auth-related
        if "auth" in text or "login" in text or "token" in text:
            probe["requires_auth"] = True

        return probe

    def _extract_endpoint(self, text: str, context: dict) -> str:
        """Extract API endpoint from text."""
        for pattern in self.ENDPOINT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)

        # Check context for known endpoints
        if "endpoints" in context:
            for ep in context["endpoints"]:
                if ep.lower() in text:
                    return ep

        # Default to health for generic checks
        if "health" in text or "status" in text:
            return "/health"

        # Default probe endpoint
        return "/"

    def _extract_status_code(self, text: str) -> int:
        """Extract expected HTTP status code."""
        for pattern in self.STATUS_CODE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        # Infer from text
        if "error" in text or "fail" in text or "reject" in text:
            if "unauthorized" in text or "forbidden" in text:
                return 403
            if "not found" in text:
                return 404
            return 400

        if "success" in text or "created" in text:
            return 201 if "created" in text else 200

        return 200  # Default

    def _extract_method(self, text: str) -> str:
        """Extract HTTP method from text."""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for method in methods:
            if method.lower() in text.lower():
                return method

        # Infer from action words
        if any(w in text for w in ["create", "add", "submit", "send"]):
            return "POST"
        if any(w in text for w in ["update", "modify", "change"]):
            return "PUT"
        if any(w in text for w in ["delete", "remove"]):
            return "DELETE"

        return "GET"

    def _extract_response_checks(self, text: str) -> list[dict]:
        """Extract expected response checks."""
        checks = []

        # Check for specific field expectations
        field_patterns = [
            r'(?:returns?|contains?)\s+["\']?(\w+)["\']?\s+(?:field|key)',
            r'(?:field|key)\s+["\']?(\w+)["\']?\s+(?:should|must|is)',
            r'response\s+(?:has|contains)\s+["\']?(\w+)["\']?',
        ]

        for pattern in field_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                checks.append({
                    "type": "field_exists",
                    "field": match.group(1),
                })

        # Check for value expectations
        value_patterns = [
            r'["\']?(\w+)["\']?\s+(?:equals?|is|==)\s+["\']?([^"\']+)["\']?',
        ]

        for pattern in value_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                checks.append({
                    "type": "field_equals",
                    "field": match.group(1),
                    "value": match.group(2),
                })

        # Check for list/array expectations
        if "list" in text or "array" in text or "items" in text:
            checks.append({"type": "is_array"})

        # Check for non-empty
        if "not empty" in text or "non-empty" in text:
            checks.append({"type": "not_empty"})

        return checks

    def _extract_request_body(self, text: str, context: dict) -> dict | None:
        """Extract request body if mentioned."""
        # Look for JSON-like patterns
        json_pattern = r'\{[^}]+\}'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Check for specific fields to send
        if "username" in text and "password" in text:
            return {"username": "testuser", "password": "testpass"}

        return None


class APIProbeRunner:
    """Executes API probes against a running service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        auth_token: str | None = None,
    ):
        """Initialize probe runner.

        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
            auth_token: Optional auth token for authenticated requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.auth_token = auth_token
        self.session = requests.Session()

    def run_probes(self, probes: list[dict]) -> ProbeReport:
        """Execute all probes and return results.

        Args:
            probes: List of probe definitions

        Returns:
            ProbeReport with all results
        """
        report = ProbeReport(total=len(probes))

        for probe in probes:
            result = self._run_single_probe(probe)
            report.results.append(result)

            if result.success:
                report.passed += 1
            else:
                report.failed += 1

        return report

    def _run_single_probe(self, probe: dict) -> ProbeResult:
        """Execute a single probe.

        Args:
            probe: Probe definition

        Returns:
            ProbeResult
        """
        criterion_id = probe.get("criterion_id", "unknown")
        criterion_desc = probe.get("criterion_description", "")
        method = probe.get("method", "GET").upper()
        endpoint = probe.get("endpoint", "/")
        expected_status = probe.get("expected_status", 200)
        response_checks = probe.get("response_checks", [])
        body = probe.get("body")
        headers = probe.get("headers", {}).copy()

        # Add auth if required
        if probe.get("requires_auth") and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        url = f"{self.base_url}{endpoint}"

        start_time = time.time()

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=body if body else None,
                headers=headers,
                timeout=self.timeout,
            )
            duration_ms = (time.time() - start_time) * 1000

            # Check status code
            if response.status_code != expected_status:
                return ProbeResult(
                    criterion_id=criterion_id,
                    criterion_description=criterion_desc,
                    success=False,
                    message=f"Expected status {expected_status}, got {response.status_code}",
                    response_data={"status_code": response.status_code},
                    duration_ms=duration_ms,
                )

            # Parse response
            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = {"text": response.text[:500]}

            # Run response checks
            for check in response_checks:
                check_result = self._run_response_check(check, response_data)
                if not check_result["success"]:
                    return ProbeResult(
                        criterion_id=criterion_id,
                        criterion_description=criterion_desc,
                        success=False,
                        message=check_result["message"],
                        response_data=response_data,
                        duration_ms=duration_ms,
                    )

            return ProbeResult(
                criterion_id=criterion_id,
                criterion_description=criterion_desc,
                success=True,
                message=f"{method} {endpoint} -> {response.status_code} OK",
                response_data=response_data,
                duration_ms=duration_ms,
            )

        except requests.Timeout:
            return ProbeResult(
                criterion_id=criterion_id,
                criterion_description=criterion_desc,
                success=False,
                message=f"Request timed out after {self.timeout}s",
                duration_ms=self.timeout * 1000,
            )
        except requests.ConnectionError as e:
            return ProbeResult(
                criterion_id=criterion_id,
                criterion_description=criterion_desc,
                success=False,
                message=f"Connection error: {str(e)[:100]}",
            )
        except Exception as e:
            return ProbeResult(
                criterion_id=criterion_id,
                criterion_description=criterion_desc,
                success=False,
                message=f"Error: {str(e)[:100]}",
            )

    def _run_response_check(self, check: dict, response_data: Any) -> dict:
        """Run a single response check.

        Args:
            check: Check definition
            response_data: Parsed response data

        Returns:
            Dict with success and message
        """
        check_type = check.get("type")

        if check_type == "field_exists":
            field = check.get("field")
            if isinstance(response_data, dict) and field in response_data:
                return {"success": True, "message": f"Field '{field}' exists"}
            return {"success": False, "message": f"Field '{field}' not found"}

        elif check_type == "field_equals":
            field = check.get("field")
            expected = check.get("value")
            if isinstance(response_data, dict):
                actual = response_data.get(field)
                if str(actual) == str(expected):
                    return {"success": True, "message": f"Field '{field}' equals '{expected}'"}
                return {"success": False, "message": f"Field '{field}' is '{actual}', expected '{expected}'"}
            return {"success": False, "message": f"Response is not an object"}

        elif check_type == "is_array":
            if isinstance(response_data, list):
                return {"success": True, "message": "Response is an array"}
            return {"success": False, "message": "Response is not an array"}

        elif check_type == "not_empty":
            if response_data:
                return {"success": True, "message": "Response is not empty"}
            return {"success": False, "message": "Response is empty"}

        return {"success": True, "message": "Unknown check type, skipped"}


def run_acceptance_probes(
    feature_spec_path: Path | str,
    base_url: str = "http://localhost:8000",
    auth_token: str | None = None,
) -> ProbeReport:
    """Run API probes from a feature spec file.

    Args:
        feature_spec_path: Path to feature_spec.json
        base_url: Base URL of the running API
        auth_token: Optional auth token

    Returns:
        ProbeReport with all results
    """
    spec_path = Path(feature_spec_path)

    if not spec_path.exists():
        return ProbeReport(total=0, skipped=1, results=[
            ProbeResult(
                criterion_id="ERROR",
                criterion_description="Feature spec not found",
                success=False,
                message=f"File not found: {spec_path}",
            )
        ])

    with open(spec_path) as f:
        spec_data = json.load(f)

    acceptance_criteria = spec_data.get("acceptance_criteria", [])

    if not acceptance_criteria:
        return ProbeReport(total=0, skipped=1, results=[
            ProbeResult(
                criterion_id="SKIP",
                criterion_description="No acceptance criteria",
                success=True,
                message="No acceptance criteria defined in feature spec",
            )
        ])

    # Generate probes
    generator = APIProbeGenerator(base_url=base_url)
    probes = generator.generate_probes(acceptance_criteria)

    if not probes:
        return ProbeReport(total=0, skipped=len(acceptance_criteria), results=[
            ProbeResult(
                criterion_id="SKIP",
                criterion_description="Could not generate probes",
                success=True,
                message="No testable probes could be generated from acceptance criteria",
            )
        ])

    # Run probes
    runner = APIProbeRunner(base_url=base_url, auth_token=auth_token)
    return runner.run_probes(probes)


def generate_probe_report_markdown(report: ProbeReport) -> str:
    """Generate markdown report from probe results.

    Args:
        report: ProbeReport with results

    Returns:
        Markdown formatted report
    """
    lines = [
        "## API Probe Results",
        "",
        f"**Total:** {report.total} | **Passed:** {report.passed} | **Failed:** {report.failed}",
        "",
    ]

    if report.success:
        lines.append("**Status:** :white_check_mark: All probes passed")
    else:
        lines.append("**Status:** :x: Some probes failed")
    lines.append("")

    if report.results:
        lines.append("| Criterion | Status | Message |")
        lines.append("|-----------|--------|---------|")

        for result in report.results:
            status = ":white_check_mark:" if result.success else ":x:"
            message = result.message[:50] + "..." if len(result.message) > 50 else result.message
            lines.append(f"| {result.criterion_id} | {status} | {message} |")

        lines.append("")

    # Details for failures
    failures = [r for r in report.results if not r.success]
    if failures:
        lines.append("### Failed Probes")
        lines.append("")
        for result in failures:
            lines.append(f"**{result.criterion_id}**: {result.criterion_description}")
            lines.append(f"- Error: {result.message}")
            if result.response_data:
                lines.append(f"- Response: `{json.dumps(result.response_data)[:200]}`")
            lines.append("")

    return "\n".join(lines)
