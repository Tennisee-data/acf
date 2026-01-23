"""Shell command execution tool."""

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult, ToolStatus


class ShellTool(BaseTool):
    """Tool for executing shell commands.

    Provides safe execution of:
    - Test runners (pytest, npm test)
    - Linters (ruff, flake8, eslint)
    - Type checkers (mypy, tsc)
    - Build tools (docker, npm, pip)
    - Custom scripts
    """

    name = "shell"
    description = "Shell command execution"

    # Commands that are allowed by default
    ALLOWED_COMMANDS = {
        # Python
        "python",
        "pip",
        "pytest",
        "ruff",
        "flake8",
        "mypy",
        "black",
        "isort",
        "bandit",
        "coverage",
        # Node.js
        "node",
        "npm",
        "npx",
        "yarn",
        "eslint",
        "prettier",
        "tsc",
        "jest",
        # Docker
        "docker",
        "docker-compose",
        # General
        "make",
        "curl",
        "wget",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "ls",
        "pwd",
        "echo",
    }

    def __init__(
        self,
        working_dir: Path | str | None = None,
        timeout: int = 300,
        allowed_commands: set[str] | None = None,
    ) -> None:
        """Initialize shell tool.

        Args:
            working_dir: Working directory for commands
            timeout: Default timeout in seconds
            allowed_commands: Override allowed command set
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.timeout = timeout
        self.allowed_commands = allowed_commands or self.ALLOWED_COMMANDS

    def execute(self, operation: str, **kwargs: Any) -> ToolResult:
        """Execute a shell operation.

        Args:
            operation: Operation name (run, pytest, ruff, docker, etc.)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with command output
        """
        operations = {
            "run": self._run,
            "pytest": self._pytest,
            "ruff": self._ruff,
            "mypy": self._mypy,
            "docker_build": self._docker_build,
            "docker_run": self._docker_run,
            "docker_compose": self._docker_compose,
            "npm": self._npm,
            "pip": self._pip,
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

    def _run(
        self,
        command: str | list[str],
        timeout: int | None = None,
        check: bool = False,
        capture: bool = True,
        env: dict[str, str] | None = None,
    ) -> ToolResult:
        """Run a shell command."""
        if isinstance(command, str):
            parts = shlex.split(command)
        else:
            parts = command

        if not parts:
            return ToolResult(status=ToolStatus.FAILURE, error="Empty command")

        # Check if command is allowed
        cmd_name = Path(parts[0]).name
        if cmd_name not in self.allowed_commands:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Command not allowed: {cmd_name}. Allowed: {self.allowed_commands}",
            )

        try:
            result = subprocess.run(
                parts,
                cwd=self.working_dir,
                capture_output=capture,
                text=True,
                timeout=timeout or self.timeout,
                check=check,
                env={**subprocess.os.environ, **(env or {})},
            )

            return ToolResult(
                status=ToolStatus.SUCCESS if result.returncode == 0 else ToolStatus.FAILURE,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Command timed out after {timeout or self.timeout}s",
            )
        except FileNotFoundError:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Command not found: {parts[0]}",
            )

    def _pytest(
        self,
        path: str = ".",
        verbose: bool = True,
        coverage: bool = False,
        markers: str | None = None,
    ) -> ToolResult:
        """Run pytest."""
        args = ["pytest", path]
        if verbose:
            args.append("-v")
        if coverage:
            args.extend(["--cov", "--cov-report=term-missing"])
        if markers:
            args.extend(["-m", markers])

        return self._run(args)

    def _ruff(
        self,
        path: str = ".",
        fix: bool = False,
        format_code: bool = False,
    ) -> ToolResult:
        """Run ruff linter."""
        if format_code:
            args = ["ruff", "format", path]
        else:
            args = ["ruff", "check", path]
            if fix:
                args.append("--fix")

        return self._run(args)

    def _mypy(self, path: str = ".", strict: bool = False) -> ToolResult:
        """Run mypy type checker."""
        args = ["mypy", path]
        if strict:
            args.append("--strict")
        return self._run(args)

    def _docker_build(
        self,
        tag: str,
        dockerfile: str = "Dockerfile",
        context: str = ".",
        build_args: dict[str, str] | None = None,
    ) -> ToolResult:
        """Build Docker image."""
        args = ["docker", "build", "-t", tag, "-f", dockerfile]
        if build_args:
            for key, value in build_args.items():
                args.extend(["--build-arg", f"{key}={value}"])
        args.append(context)

        return self._run(args, timeout=600)

    def _docker_run(
        self,
        image: str,
        command: str | None = None,
        ports: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        env_file: str | None = None,
        detach: bool = False,
        remove: bool = True,
        name: str | None = None,
    ) -> ToolResult:
        """Run Docker container.

        Args:
            image: Docker image to run
            command: Command to execute in container
            ports: Port mappings {host_port: container_port}
            volumes: Volume mounts {host_path: container_path}
            env: Environment variables {name: value} - for runtime secrets
            env_file: Path to env file for secrets
            detach: Run in background
            remove: Remove container after exit
            name: Container name
        """
        args = ["docker", "run"]
        if detach:
            args.append("-d")
        if remove:
            args.append("--rm")
        if name:
            args.extend(["--name", name])
        if ports:
            for host, container in ports.items():
                args.extend(["-p", f"{host}:{container}"])
        if volumes:
            for host, container in volumes.items():
                args.extend(["-v", f"{host}:{container}"])
        # Environment variables - passed at runtime, NOT baked into image
        if env:
            for key, value in env.items():
                args.extend(["-e", f"{key}={value}"])
        if env_file:
            args.extend(["--env-file", env_file])

        args.append(image)
        if command:
            args.extend(shlex.split(command))

        return self._run(args, timeout=300)

    def _docker_compose(
        self,
        command: str,
        file: str = "docker-compose.yml",
        project_name: str | None = None,
    ) -> ToolResult:
        """Run docker-compose command."""
        args = ["docker-compose", "-f", file]
        if project_name:
            args.extend(["-p", project_name])
        args.extend(shlex.split(command))

        return self._run(args, timeout=600)

    def _npm(self, command: str, path: str | None = None) -> ToolResult:
        """Run npm command."""
        args = ["npm"] + shlex.split(command)
        original_dir = self.working_dir
        if path:
            self.working_dir = Path(path)

        result = self._run(args)
        self.working_dir = original_dir
        return result

    def _pip(self, command: str) -> ToolResult:
        """Run pip command."""
        args = ["pip"] + shlex.split(command)
        return self._run(args)
