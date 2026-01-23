"""Docker Agent for container-based validation.

Builds and runs code in Docker to validate it works in an isolated,
reproducible environment. If it runs in a container, it can run anywhere.
"""

import json
import time
from pathlib import Path

from llm_backend import LLMBackend
from tools import ShellTool, HttpTool
from tools.api_probe import (
    APIProbeGenerator,
    APIProbeRunner,
    ProbeReport,
    run_acceptance_probes,
    generate_probe_report_markdown,
)

from .base import AgentInput, AgentOutput, BaseAgent


class DockerAgent(BaseAgent):
    """Agent for Docker-based validation.

    Validates implementation by:
    1. Building a Docker image
    2. Running the container
    3. Waiting for service readiness
    4. Running health checks
    5. Optionally running tests inside container

    If code works in Docker, it's portable and production-ready.
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize DockerAgent.

        Args:
            llm: LLM backend (used for error analysis)
            repo_path: Path to repository with Dockerfile
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)
        self.repo_path = repo_path or Path.cwd()
        self.shell = ShellTool(working_dir=self.repo_path)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return "You analyze Docker build and runtime errors to suggest fixes."

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Build and validate code in Docker container.

        Args:
            input_data: Context with 'feature_id', optionally:
                       'image_tag', 'container_name', 'port', 'health_endpoint',
                       'feature_spec_path' (for API probes from acceptance criteria),
                       'run_integration_tests' (run full integration test suite)

        Returns:
            AgentOutput with validation results
        """
        context = input_data.context
        feature_id = context.get("feature_id", "FEAT-000")

        # Configuration
        image_tag = context.get("image_tag", f"coding-factory:{feature_id.lower()}")
        container_name = context.get("container_name", f"cf-{feature_id.lower()}")
        port = context.get("port", 8000)
        health_endpoint = context.get("health_endpoint", "/health")
        run_tests = context.get("run_tests", True)
        dockerfile = context.get("dockerfile", "Dockerfile")

        # API probe configuration
        feature_spec_path = context.get("feature_spec_path")  # Path to feature_spec.json
        run_api_probes = context.get("run_api_probes", True)  # Run acceptance criteria probes

        # Integration test configuration
        run_integration_tests = context.get("run_integration_tests", True)
        integration_test_path = context.get("integration_test_path", "tests/integration")

        # Runtime environment variables (secrets) - passed at docker run, NOT in image
        # This mirrors how Render.com handles secrets:
        #   - Secrets are configured in Render dashboard
        #   - Injected as environment variables at runtime
        #   - Never baked into the Docker image
        env = context.get("env", {})
        env_file = context.get("env_file")  # Optional path to .env file

        results = {
            "feature_id": feature_id,
            "image_tag": image_tag,
            "container_name": container_name,
            "stages": {},
            "success": False,
            "container_id": None,
            "errors": [],
        }

        # Check if Dockerfile exists
        dockerfile_path = self.repo_path / dockerfile
        if not dockerfile_path.exists():
            results["errors"].append(f"Dockerfile not found at {dockerfile_path}")
            results["stages"]["dockerfile_check"] = {"status": "failed", "error": "Dockerfile not found"}
            return AgentOutput(
                success=False,
                data=results,
                errors=[f"Dockerfile not found at {dockerfile_path}"],
            )

        results["stages"]["dockerfile_check"] = {"status": "passed"}

        # Stage 1: Build Docker image
        build_result = self._build_image(image_tag, dockerfile)
        results["stages"]["build"] = build_result

        if not build_result["success"]:
            results["errors"].append(f"Build failed: {build_result.get('error', 'Unknown error')}")
            return AgentOutput(
                success=False,
                data=results,
                errors=results["errors"],
            )

        # Stage 2: Run container with runtime secrets
        # Example secrets that would be passed:
        #   - DATABASE_URL: postgres://user:pass@host:5432/db
        #   - REDIS_URL: redis://:password@host:6379
        #   - API_KEY: sk-xxx
        # These are injected at runtime, never in the image
        run_result = self._run_container(
            image_tag,
            container_name,
            port,
            env=env,
            env_file=env_file,
        )
        results["stages"]["run"] = run_result
        results["container_id"] = run_result.get("container_id")

        if not run_result["success"]:
            results["errors"].append(f"Run failed: {run_result.get('error', 'Unknown error')}")
            self._cleanup_container(container_name)
            return AgentOutput(
                success=False,
                data=results,
                errors=results["errors"],
            )

        # Stage 3: Wait for ready
        ready_result = self._wait_for_ready(port, health_endpoint)
        results["stages"]["ready"] = ready_result

        if not ready_result["success"]:
            results["errors"].append("Service did not become ready")
            # Get logs for debugging
            logs = self._get_container_logs(container_name)
            results["stages"]["ready"]["logs"] = logs
            self._cleanup_container(container_name)
            return AgentOutput(
                success=False,
                data=results,
                errors=results["errors"],
            )

        # Stage 4: Health check
        health_result = self._health_check(port, health_endpoint)
        results["stages"]["health"] = health_result

        # Stage 5: API probes from acceptance criteria
        if run_api_probes and feature_spec_path:
            probe_result = self._run_api_probes(
                feature_spec_path=feature_spec_path,
                port=port,
            )
            results["stages"]["api_probes"] = probe_result
            if not probe_result["success"] and probe_result.get("total", 0) > 0:
                results["errors"].append("API probes failed")

        # Stage 6: Run tests in container (optional)
        if run_tests:
            test_result = self._run_tests_in_container(container_name)
            results["stages"]["tests"] = test_result
            if not test_result["success"]:
                results["errors"].append("Tests failed in container")

        # Stage 7: Integration tests in container
        if run_integration_tests:
            integration_result = self._run_integration_tests(
                container_name=container_name,
                port=port,
                test_path=integration_test_path,
            )
            results["stages"]["integration_tests"] = integration_result
            if not integration_result["success"] and not integration_result.get("skipped"):
                results["errors"].append("Integration tests failed")

        # Cleanup
        self._cleanup_container(container_name)
        results["stages"]["cleanup"] = {"status": "completed"}

        # Determine overall success
        critical_stages = ["build", "run", "ready", "health"]
        results["success"] = all(
            results["stages"].get(s, {}).get("success", False)
            for s in critical_stages
        )

        # Check optional stages
        if run_tests and not results["stages"].get("tests", {}).get("success", True):
            results["success"] = False

        if run_api_probes and feature_spec_path:
            probe_stage = results["stages"].get("api_probes", {})
            if probe_stage.get("total", 0) > 0 and not probe_stage.get("success", True):
                results["success"] = False

        if run_integration_tests:
            int_stage = results["stages"].get("integration_tests", {})
            if not int_stage.get("skipped") and not int_stage.get("success", True):
                results["success"] = False

        return AgentOutput(
            success=results["success"],
            data=results,
            errors=results["errors"] if results["errors"] else None,
            artifacts=["docker_validation.json"],
        )

    def _build_image(self, tag: str, dockerfile: str = "Dockerfile") -> dict:
        """Build Docker image."""
        result = self.shell.execute(
            "docker_build",
            tag=tag,
            dockerfile=dockerfile,
            context=".",
        )

        if result.success:
            return {
                "success": True,
                "image_tag": tag,
                "output": result.output.get("stdout", "")[:500] if result.output else "",
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "output": result.output.get("stderr", "")[:1000] if result.output else "",
            }

    def _run_container(
        self,
        image: str,
        name: str,
        port: int,
        env: dict[str, str] | None = None,
        env_file: str | None = None,
    ) -> dict:
        """Run Docker container with runtime environment variables.

        Args:
            image: Docker image to run
            name: Container name
            port: Port to expose
            env: Environment variables (secrets passed at runtime, NOT in image)
            env_file: Path to .env file for secrets

        Note: Secrets are passed here at runtime via -e flags.
              They are NEVER baked into the Docker image.
              This is the same pattern Render.com uses.
        """
        # First, ensure no container with same name exists
        self._cleanup_container(name)

        result = self.shell.execute(
            "docker_run",
            image=image,
            name=name,
            ports={str(port): str(port)},
            env=env,  # Runtime secrets passed here
            env_file=env_file,
            detach=True,
            remove=False,  # Don't auto-remove so we can get logs
        )

        if result.success:
            # Get container ID
            container_id = None
            if result.output:
                stdout = result.output.get("stdout", "")
                if stdout:
                    container_id = stdout.strip()[:12]

            return {
                "success": True,
                "container_id": container_id,
                "port": port,
                "env_vars_passed": list(env.keys()) if env else [],
            }
        else:
            return {
                "success": False,
                "error": result.error,
            }

    def _wait_for_ready(
        self,
        port: int,
        health_endpoint: str,
        max_attempts: int = 30,
        interval: float = 2.0,
    ) -> dict:
        """Wait for service to be ready."""
        http = HttpTool(base_url=f"http://localhost:{port}", timeout=5)

        for attempt in range(max_attempts):
            result = http.execute("get", path=health_endpoint)
            if result.success:
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "response_time_ms": result.metadata.get("response_time_ms") if result.metadata else None,
                }
            time.sleep(interval)

        return {
            "success": False,
            "attempts": max_attempts,
            "error": f"Service not ready after {max_attempts} attempts",
        }

    def _health_check(self, port: int, health_endpoint: str) -> dict:
        """Run health check against running container."""
        http = HttpTool(base_url=f"http://localhost:{port}", timeout=10)

        result = http.execute("health_check", path=health_endpoint, expected_status=200)

        if result.success:
            return {
                "success": True,
                "endpoint": health_endpoint,
                "status_code": result.output.get("response", {}).get("status_code") if result.output else 200,
            }
        else:
            return {
                "success": False,
                "endpoint": health_endpoint,
                "error": result.error,
            }

    def _run_tests_in_container(self, container_name: str) -> dict:
        """Run tests inside the container."""
        result = self.shell.execute(
            "run",
            command=f"docker exec {container_name} pytest -v --tb=short",
            timeout=120,
        )

        if result.success:
            stdout = result.output.get("stdout", "") if result.output else ""
            # Parse test results
            passed = stdout.count(" PASSED")
            failed = stdout.count(" FAILED")

            return {
                "success": failed == 0,
                "passed": passed,
                "failed": failed,
                "output": stdout[:2000],
            }
        else:
            # pytest might not be installed, that's okay
            error = result.error or ""
            if "pytest" in error.lower() and "not found" in error.lower():
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "pytest not installed in container",
                }
            return {
                "success": False,
                "error": result.error,
                "output": result.output.get("stderr", "")[:1000] if result.output else "",
            }

    def _run_api_probes(
        self,
        feature_spec_path: str | Path,
        port: int,
        auth_token: str | None = None,
    ) -> dict:
        """Run API probes from acceptance criteria against running container.

        Generates probes from acceptance criteria in feature_spec.json and
        executes them against the running API to verify requirements are met.

        Args:
            feature_spec_path: Path to feature_spec.json
            port: Port the API is running on
            auth_token: Optional auth token for authenticated endpoints

        Returns:
            Dict with probe results
        """
        base_url = f"http://localhost:{port}"

        try:
            report = run_acceptance_probes(
                feature_spec_path=feature_spec_path,
                base_url=base_url,
                auth_token=auth_token,
            )

            return {
                "success": report.success,
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "results": [
                    {
                        "criterion_id": r.criterion_id,
                        "success": r.success,
                        "message": r.message,
                        "duration_ms": r.duration_ms,
                    }
                    for r in report.results
                ],
                "report_markdown": generate_probe_report_markdown(report),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total": 0,
                "passed": 0,
                "failed": 0,
            }

    def _run_integration_tests(
        self,
        container_name: str,
        port: int,
        test_path: str = "tests/integration",
    ) -> dict:
        """Run integration tests against the running container.

        Integration tests are different from unit tests - they test the actual
        API endpoints from outside the container, simulating real client requests.

        Args:
            container_name: Name of the running container
            port: Port the API is running on
            test_path: Path to integration tests inside container

        Returns:
            Dict with integration test results
        """
        # First, check if integration tests exist in the container
        check_result = self.shell.execute(
            "run",
            command=f"docker exec {container_name} test -d {test_path}",
            timeout=10,
        )

        if not check_result.success:
            # Try alternative paths
            alt_paths = ["tests/integration", "integration_tests", "tests/e2e"]
            found_path = None

            for alt in alt_paths:
                check = self.shell.execute(
                    "run",
                    command=f"docker exec {container_name} test -d {alt}",
                    timeout=10,
                )
                if check.success:
                    found_path = alt
                    break

            if not found_path:
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "No integration test directory found",
                }

            test_path = found_path

        # Run integration tests with API_URL environment variable
        # This allows tests to know where to point their requests
        result = self.shell.execute(
            "run",
            command=(
                f"docker exec -e API_URL=http://localhost:{port} "
                f"{container_name} pytest {test_path} -v --tb=short"
            ),
            timeout=300,  # 5 min for integration tests
        )

        if result.success:
            stdout = result.output.get("stdout", "") if result.output else ""
            passed = stdout.count(" PASSED")
            failed = stdout.count(" FAILED")
            errors = stdout.count(" ERROR")

            return {
                "success": failed == 0 and errors == 0,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "test_path": test_path,
                "output": stdout[:3000],
            }
        else:
            error = result.error or ""
            stderr = result.output.get("stderr", "") if result.output else ""

            # Check if it's just missing pytest
            if "pytest" in error.lower() or "pytest" in stderr.lower():
                if "not found" in error.lower() or "not found" in stderr.lower():
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": "pytest not installed in container",
                    }

            # Check for no tests found
            if "no tests ran" in stderr.lower() or "no tests collected" in stderr.lower():
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "No integration tests found",
                }

            return {
                "success": False,
                "error": error[:500],
                "output": stderr[:1000],
                "test_path": test_path,
            }

    def _get_container_logs(self, container_name: str, tail: int = 100) -> str:
        """Get container logs for debugging."""
        result = self.shell.execute(
            "run",
            command=f"docker logs --tail {tail} {container_name}",
        )

        if result.success and result.output:
            stdout = result.output.get("stdout", "")
            stderr = result.output.get("stderr", "")
            return f"{stdout}\n{stderr}".strip()[:2000]
        return ""

    def _cleanup_container(self, container_name: str) -> None:
        """Stop and remove container."""
        # Stop container (ignore errors if not running)
        self.shell.execute("run", command=f"docker stop {container_name}", timeout=30)
        # Remove container (ignore errors if doesn't exist)
        self.shell.execute("run", command=f"docker rm -f {container_name}", timeout=10)

    def generate_report(self, results: dict) -> str:
        """Generate markdown report from validation results."""
        lines = [
            f"# Docker Validation Report: {results.get('feature_id', 'Unknown')}",
            "",
        ]

        # Overall status
        if results.get("success"):
            lines.append("**Status:** ✅ PASSED - Code validated in Docker")
        else:
            lines.append("**Status:** ❌ FAILED - Validation issues found")
        lines.append("")

        # Image info
        lines.append(f"**Image:** `{results.get('image_tag', 'N/A')}`")
        if results.get("container_id"):
            lines.append(f"**Container:** `{results.get('container_id')}`")
        lines.append("")

        # Stage results
        lines.append("## Validation Stages")
        lines.append("")
        lines.append("| Stage | Status | Details |")
        lines.append("|-------|--------|---------|")

        stage_names = {
            "dockerfile_check": "Dockerfile Check",
            "build": "Image Build",
            "run": "Container Start",
            "ready": "Service Ready",
            "health": "Health Check",
            "api_probes": "API Probes",
            "tests": "Unit Tests",
            "integration_tests": "Integration Tests",
            "cleanup": "Cleanup",
        }

        for stage_key, stage_name in stage_names.items():
            stage = results.get("stages", {}).get(stage_key, {})
            if not stage:
                continue

            if stage.get("success") or stage.get("status") in ("passed", "completed"):
                status = "✅"
            elif stage.get("skipped"):
                status = "⏭️"
            else:
                status = "❌"

            details = ""
            if stage.get("error"):
                details = stage["error"][:50]
            elif stage.get("attempts"):
                details = f"{stage['attempts']} attempts"
            elif stage.get("passed") is not None:
                details = f"{stage['passed']} passed, {stage.get('failed', 0)} failed"
            elif stage.get("skipped"):
                details = stage.get("reason", "Skipped")

            lines.append(f"| {stage_name} | {status} | {details} |")

        lines.append("")

        # Errors
        errors = results.get("errors", [])
        if errors:
            lines.append("## Errors")
            lines.append("")
            for error in errors:
                lines.append(f"- ❌ {error}")
            lines.append("")

        # Logs if available
        ready_stage = results.get("stages", {}).get("ready", {})
        if ready_stage.get("logs"):
            lines.append("## Container Logs")
            lines.append("")
            lines.append("```")
            lines.append(ready_stage["logs"][:1000])
            lines.append("```")
            lines.append("")

        return "\n".join(lines)
