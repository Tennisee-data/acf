"""Hello Checker Agent - Test extension for ACF marketplace."""

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class AgentInput:
    """Input for the agent."""
    context: dict[str, Any]
    previous_outputs: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = None


@dataclass
class AgentOutput:
    """Output from the agent."""
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None


class HelloCheckerAgent:
    """Test agent that verifies implementation contains expected patterns.

    This is a simple marketplace extension agent that demonstrates
    the extension system by checking generated code for patterns.
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        """Initialize the agent.

        Args:
            llm: LLM backend (not used by this agent)
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.name = "hello-checker"

    def default_system_prompt(self) -> str:
        """Return default system prompt (not used)."""
        return "You are a code pattern checker."

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Check implementation for expected patterns.

        Args:
            input_data: Pipeline context with implementation data

        Returns:
            AgentOutput with check results
        """
        import os
        from pathlib import Path

        print("\n" + "=" * 60)
        print("MARKETPLACE EXTENSION: Hello Checker Agent")
        print("=" * 60)

        # Get run directory from context
        run_id = input_data.context.get("run_id", "")
        run_dir = input_data.context.get("run_dir", "")  # This is already the full path
        repo_path = input_data.context.get("repo_path", ".")

        # Try to find generated project files
        if run_dir:
            generated_project_dir = Path(repo_path) / run_dir / "generated_project"
        else:
            artifacts_dir = input_data.context.get("artifacts_dir", "artifacts")
            generated_project_dir = Path(repo_path) / artifacts_dir / run_id / "generated_project"

        checks_passed = []
        checks_failed = []
        all_content = ""

        # Read files from generated project directory
        if generated_project_dir.exists():
            print(f"  Reading from: {generated_project_dir}")
            for py_file in generated_project_dir.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    all_content += content + "\n"
                    print(f"  Checking file: {py_file.relative_to(generated_project_dir)}")
                except Exception as e:
                    print(f"  Error reading {py_file}: {e}")
        else:
            print(f"  Generated project not found at: {generated_project_dir}")
            # Fallback: try implementation data from context
            implementation = input_data.context.get("implementation", {})
            files = implementation.get("files", [])
            for file_info in files:
                if isinstance(file_info, dict):
                    content = file_info.get("content", "")
                    path = file_info.get("path", "unknown")
                    all_content += content
                    print(f"  Checking file: {path}")

        # Pattern checks
        patterns = {
            "flask_import": r"from flask import|import flask",
            "route_decorator": r"@app\.route|@router\.",
            "json_response": r"jsonify|json\.dumps|JSONResponse",
            "hello_endpoint": r"/hello",
        }

        for check_name, pattern in patterns.items():
            if re.search(pattern, all_content, re.IGNORECASE):
                checks_passed.append(check_name)
                print(f"  [PASS] {check_name}")
            else:
                checks_failed.append(check_name)
                print(f"  [FAIL] {check_name}")

        # Summary
        total = len(patterns)
        passed = len(checks_passed)
        success = passed >= total * 0.5  # Pass if at least 50% checks pass

        print("-" * 60)
        print(f"Results: {passed}/{total} checks passed")
        print("=" * 60 + "\n")

        return AgentOutput(
            success=success,
            data={
                "hello_checker": {
                    "checks_passed": checks_passed,
                    "checks_failed": checks_failed,
                    "pass_rate": passed / total if total > 0 else 0,
                    "message": f"Extension agent ran successfully! {passed}/{total} pattern checks passed."
                }
            },
            artifacts=["hello_checker_report.json"],
            agent_name=self.name,
        )
