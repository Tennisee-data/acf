"""Test Agent for running tests and quality checks."""

import re
from pathlib import Path

from llm_backend import LLMBackend
from schemas.test_report import (
    CoverageInfo,
    LintIssue,
    TestCase,
    TestReport,
    TestResult,
    TestStatus,
    TypeIssue,
)
from tools import ShellTool
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent


SYSTEM_PROMPT = """You are a test analysis expert. Analyze test results, linting issues, and type errors.
Output ONLY a JSON object with your analysis. No markdown, no explanation.

{"failure_analysis":"why tests failed","suggested_fixes":["fix 1","fix 2"],"anti_patterns_found":["pattern 1"],"idiom_suggestions":["suggestion 1"],"quality_score":85,"ready_for_deploy":false,"blocking_issues":["issue 1"]}

IMPORTANT: Output starts with { and ends with }. No other text allowed."""


class TestAgent(BaseAgent):
    """Agent for running tests and quality checks.

    Executes:
    - Unit tests via pytest
    - Linting via ruff
    - Type checking via mypy
    - Coverage analysis

    Uses LLM for:
    - Analyzing test failures
    - Suggesting fixes
    - Detecting anti-patterns
    - Computing quality score
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize TestAgent.

        Args:
            llm: LLM backend for failure analysis
            repo_path: Path to repository for running tests
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)
        self.repo_path = repo_path or Path.cwd()
        self.shell = ShellTool(working_dir=self.repo_path)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run tests and quality checks.

        Args:
            input_data: Must contain 'feature_id' in context

        Returns:
            AgentOutput with TestReport data
        """
        context = input_data.context
        feature_id = context.get("feature_id", "FEAT-000")

        # Run all quality checks
        test_result, test_cases = self._run_pytest()
        lint_issues = self._run_ruff()
        type_issues = self._run_mypy()
        coverage = self._get_coverage()

        # Count blocking issues
        blocking_issues = []
        if test_result.failed > 0:
            blocking_issues.append(f"{test_result.failed} test(s) failing")
        if test_result.errors > 0:
            blocking_issues.append(f"{test_result.errors} test(s) with errors")

        error_lints = [i for i in lint_issues if i.severity == "error"]
        if error_lints:
            blocking_issues.append(f"{len(error_lints)} linting error(s)")

        if type_issues:
            blocking_issues.append(f"{len(type_issues)} type error(s)")

        # Use LLM for analysis if there are issues
        analysis_data = {}
        if blocking_issues or lint_issues:
            analysis_data = self._analyze_with_llm(
                test_cases=test_cases,
                lint_issues=lint_issues,
                type_issues=type_issues,
                blocking_issues=blocking_issues,
            )

        # Compute quality score
        quality_score = self._compute_quality_score(
            test_result=test_result,
            lint_issues=lint_issues,
            type_issues=type_issues,
            coverage=coverage,
        )

        # Build test report
        ready_for_deploy = len(blocking_issues) == 0 and quality_score >= 70

        report = TestReport(
            feature_id=feature_id,
            test_results=test_result,
            test_cases=test_cases,
            coverage=coverage,
            lint_issues=lint_issues,
            lint_tool="ruff",
            type_issues=type_issues,
            type_checker="mypy",
            tests_generated=[],
            failure_analysis=analysis_data.get("failure_analysis"),
            suggested_fixes=analysis_data.get("suggested_fixes", []),
            anti_patterns_found=analysis_data.get("anti_patterns_found", []),
            idiom_suggestions=analysis_data.get("idiom_suggestions", []),
            quality_score=quality_score,
            ready_for_deploy=ready_for_deploy,
            blocking_issues=blocking_issues,
        )

        return AgentOutput(
            success=True,
            data=report.model_dump(),
            artifacts=["test_report.json", "test_report.md"],
        )

    def _run_pytest(self) -> tuple[TestResult, list[TestCase]]:
        """Run pytest and parse results."""
        result = self.shell.execute("pytest", path=".", verbose=True, coverage=False)

        test_cases = []
        total = passed = failed = skipped = errors = 0
        duration = 0.0

        if result.output:
            stdout = result.output.get("stdout", "") if isinstance(result.output, dict) else str(result.output)

            # Parse pytest output
            # Pattern: test_file.py::test_name PASSED/FAILED/SKIPPED/ERROR
            test_pattern = re.compile(
                r'(\S+\.py)::(\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[\s*\d+%\])?'
            )

            for match in test_pattern.finditer(stdout):
                file_path, test_name, status_str = match.groups()
                status_map = {
                    "PASSED": TestStatus.PASSED,
                    "FAILED": TestStatus.FAILED,
                    "SKIPPED": TestStatus.SKIPPED,
                    "ERROR": TestStatus.ERROR,
                }
                status = status_map.get(status_str, TestStatus.ERROR)

                test_cases.append(
                    TestCase(
                        name=test_name,
                        file_path=file_path,
                        status=status,
                    )
                )

                total += 1
                if status == TestStatus.PASSED:
                    passed += 1
                elif status == TestStatus.FAILED:
                    failed += 1
                elif status == TestStatus.SKIPPED:
                    skipped += 1
                else:
                    errors += 1

            # Parse duration from summary line
            duration_match = re.search(r'in\s+([\d.]+)s', stdout)
            if duration_match:
                duration = float(duration_match.group(1))

            # Parse summary line for counts if we didn't find individual tests
            if total == 0:
                summary_match = re.search(
                    r'(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+error',
                    stdout,
                    re.IGNORECASE
                )
                if summary_match:
                    counts = re.findall(r'(\d+)\s+(passed|failed|skipped|error)', stdout, re.IGNORECASE)
                    for count, status in counts:
                        count = int(count)
                        total += count
                        if status.lower() == "passed":
                            passed = count
                        elif status.lower() == "failed":
                            failed = count
                        elif status.lower() == "skipped":
                            skipped = count
                        elif status.lower() == "error":
                            errors = count

        test_result = TestResult(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_seconds=duration,
        )

        return test_result, test_cases

    def _run_ruff(self) -> list[LintIssue]:
        """Run ruff linter and parse results."""
        result = self.shell.execute("ruff", path=".")
        lint_issues = []

        if result.output:
            stdout = result.output.get("stdout", "") if isinstance(result.output, dict) else str(result.output)

            # Parse ruff output
            # Pattern: file.py:line:col: CODE message
            pattern = re.compile(r'(\S+):(\d+):(\d+):\s+(\w+)\s+(.*)')

            for match in pattern.finditer(stdout):
                file_path, line, col, code, message = match.groups()
                severity = "error" if code.startswith("E") else "warning"

                lint_issues.append(
                    LintIssue(
                        file_path=file_path,
                        line=int(line),
                        column=int(col),
                        rule=code,
                        message=message.strip(),
                        severity=severity,
                        fixable=code in self._get_fixable_rules(),
                    )
                )

        return lint_issues

    def _get_fixable_rules(self) -> set[str]:
        """Return set of auto-fixable ruff rules."""
        return {
            "F401",  # unused import
            "F841",  # unused variable
            "I001",  # import sorting
            "W291",  # trailing whitespace
            "W292",  # no newline at end
            "W293",  # blank line whitespace
            "E501",  # line too long (sometimes)
            "UP",    # pyupgrade rules
        }

    def _run_mypy(self) -> list[TypeIssue]:
        """Run mypy type checker and parse results."""
        # First, try to install dependencies if requirements.txt exists
        self._ensure_dependencies_installed()

        # Run mypy with --ignore-missing-imports to avoid false positives
        # from third-party packages that aren't installed or lack stubs
        result = self.shell.execute("mypy", path=".", ignore_missing_imports=True)
        type_issues = []

        if result.output:
            stdout = result.output.get("stdout", "") if isinstance(result.output, dict) else str(result.output)

            # Parse mypy output
            # Pattern: file.py:line: error: message [code]
            pattern = re.compile(r'(\S+):(\d+):\s+error:\s+(.*?)(?:\s+\[(\w+)\])?$', re.MULTILINE)

            for match in pattern.finditer(stdout):
                file_path, line, message, error_code = match.groups()

                type_issues.append(
                    TypeIssue(
                        file_path=file_path,
                        line=int(line),
                        message=message.strip(),
                        error_code=error_code,
                    )
                )

        return type_issues

    def _ensure_dependencies_installed(self) -> None:
        """Try to install project dependencies before running type checks."""
        # Check for requirements.txt
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            self.shell.execute("pip", command="install -q -r requirements.txt")
            return

        # Check for pyproject.toml with dependencies
        pyproject_file = self.repo_path / "pyproject.toml"
        if pyproject_file.exists():
            self.shell.execute("pip", command="install -q -e .")

    def _get_coverage(self) -> CoverageInfo | None:
        """Get coverage information if available."""
        # Try running coverage report
        result = self.shell.execute("run", command="coverage report --format=total")

        if result.success and result.output:
            stdout = result.output.get("stdout", "") if isinstance(result.output, dict) else str(result.output)

            # Try to parse total coverage percentage
            try:
                coverage_pct = float(stdout.strip())
                return CoverageInfo(
                    total_lines=0,
                    covered_lines=0,
                    coverage_percent=coverage_pct,
                )
            except ValueError:
                pass

        return None

    def _compute_quality_score(
        self,
        test_result: TestResult,
        lint_issues: list[LintIssue],
        type_issues: list[TypeIssue],
        coverage: CoverageInfo | None,
    ) -> float:
        """Compute overall quality score (0-100)."""
        score = 100.0

        # Test results (40 points max)
        if test_result.total > 0:
            test_score = (test_result.passed / test_result.total) * 40
            score = score - (40 - test_score)
        else:
            score -= 10  # No tests penalty

        # Linting (20 points max)
        error_count = sum(1 for i in lint_issues if i.severity == "error")
        warning_count = sum(1 for i in lint_issues if i.severity == "warning")
        lint_penalty = min(20, error_count * 5 + warning_count * 1)
        score -= lint_penalty

        # Type checking (20 points max)
        type_penalty = min(20, len(type_issues) * 3)
        score -= type_penalty

        # Coverage (20 points max)
        if coverage and coverage.coverage_percent > 0:
            coverage_score = (coverage.coverage_percent / 100) * 20
            score = score - (20 - coverage_score)
        else:
            score -= 5  # No coverage data penalty

        return max(0, min(100, score))

    def _analyze_with_llm(
        self,
        test_cases: list[TestCase],
        lint_issues: list[LintIssue],
        type_issues: list[TypeIssue],
        blocking_issues: list[str],
    ) -> dict:
        """Use LLM to analyze failures and suggest fixes."""
        # Build analysis prompt
        parts = ["## Issues to Analyze\n"]

        # Failed tests
        failed_tests = [t for t in test_cases if t.status in (TestStatus.FAILED, TestStatus.ERROR)]
        if failed_tests:
            parts.append("### Failed Tests")
            for t in failed_tests[:5]:  # Limit to 5
                parts.append(f"- {t.file_path}::{t.name}: {t.status.value}")
                if t.error_message:
                    parts.append(f"  Error: {t.error_message}")

        # Lint issues
        if lint_issues:
            parts.append("\n### Lint Issues")
            for issue in lint_issues[:10]:  # Limit to 10
                parts.append(f"- {issue.file_path}:{issue.line} [{issue.rule}]: {issue.message}")

        # Type issues
        if type_issues:
            parts.append("\n### Type Errors")
            for issue in type_issues[:10]:  # Limit to 10
                parts.append(f"- {issue.file_path}:{issue.line}: {issue.message}")

        parts.append("\n### Blocking Issues")
        for issue in blocking_issues:
            parts.append(f"- {issue}")

        parts.append("\n## Task")
        parts.append("Analyze these issues and provide:")
        parts.append("1. failure_analysis: Brief explanation of why tests/checks failed")
        parts.append("2. suggested_fixes: Specific fixes for the issues")
        parts.append("3. anti_patterns_found: Any anti-patterns in the code")
        parts.append("4. idiom_suggestions: Modern Python idioms to use")
        parts.append("5. quality_score: Your assessment 0-100")
        parts.append("6. ready_for_deploy: true/false")
        parts.append("7. blocking_issues: Issues that must be fixed")

        user_message = "\n".join(parts)

        try:
            response = self._chat(user_message, temperature=0.2)
            result = parse_llm_json(response, default={})
            return result
        except Exception:
            return {}

    def generate_markdown_report(self, report: TestReport) -> str:
        """Generate human-readable markdown from test report."""
        lines = [
            f"# Test Report: {report.feature_id}",
            "",
        ]

        # Summary badge
        if report.ready_for_deploy:
            lines.append("**Status:** :white_check_mark: Ready for Deploy")
        else:
            lines.append("**Status:** :x: Not Ready")

        if report.quality_score is not None:
            lines.append(f"**Quality Score:** {report.quality_score:.1f}/100")
        lines.append("")

        # Test Results
        lines.extend([
            "## Test Results",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total | {report.test_results.total} |",
            f"| Passed | {report.test_results.passed} |",
            f"| Failed | {report.test_results.failed} |",
            f"| Skipped | {report.test_results.skipped} |",
            f"| Errors | {report.test_results.errors} |",
            f"| Duration | {report.test_results.duration_seconds:.2f}s |",
            "",
        ])

        # Failed tests detail
        failed_tests = [t for t in report.test_cases if t.status in (TestStatus.FAILED, TestStatus.ERROR)]
        if failed_tests:
            lines.extend(["### Failed Tests", ""])
            for test in failed_tests:
                lines.append(f"- **{test.name}** (`{test.file_path}`)")
                if test.error_message:
                    lines.append(f"  - Error: {test.error_message}")
            lines.append("")

        # Coverage
        if report.coverage:
            lines.extend([
                "## Coverage",
                "",
                f"**Coverage:** {report.coverage.coverage_percent:.1f}%",
                "",
            ])

        # Lint Issues
        if report.lint_issues:
            lines.extend([
                f"## Lint Issues ({len(report.lint_issues)})",
                "",
            ])
            errors = [i for i in report.lint_issues if i.severity == "error"]
            warnings = [i for i in report.lint_issues if i.severity == "warning"]

            if errors:
                lines.append("### Errors")
                for issue in errors[:10]:
                    lines.append(f"- `{issue.file_path}:{issue.line}` [{issue.rule}]: {issue.message}")
                lines.append("")

            if warnings:
                lines.append("### Warnings")
                for issue in warnings[:10]:
                    lines.append(f"- `{issue.file_path}:{issue.line}` [{issue.rule}]: {issue.message}")
                if len(warnings) > 10:
                    lines.append(f"- ... and {len(warnings) - 10} more")
                lines.append("")

        # Type Issues
        if report.type_issues:
            lines.extend([
                f"## Type Errors ({len(report.type_issues)})",
                "",
            ])
            for issue in report.type_issues[:10]:
                code_str = f" [{issue.error_code}]" if issue.error_code else ""
                lines.append(f"- `{issue.file_path}:{issue.line}`{code_str}: {issue.message}")
            if len(report.type_issues) > 10:
                lines.append(f"- ... and {len(report.type_issues) - 10} more")
            lines.append("")

        # Blocking Issues
        if report.blocking_issues:
            lines.extend([
                "## Blocking Issues",
                "",
            ])
            for issue in report.blocking_issues:
                lines.append(f"- :warning: {issue}")
            lines.append("")

        # Failure Analysis
        if report.failure_analysis:
            lines.extend([
                "## Failure Analysis",
                "",
                report.failure_analysis,
                "",
            ])

        # Suggested Fixes
        if report.suggested_fixes:
            lines.extend([
                "## Suggested Fixes",
                "",
            ])
            for fix in report.suggested_fixes:
                lines.append(f"- {fix}")
            lines.append("")

        # Anti-patterns
        if report.anti_patterns_found:
            lines.extend([
                "## Anti-patterns Found",
                "",
            ])
            for pattern in report.anti_patterns_found:
                lines.append(f"- {pattern}")
            lines.append("")

        # Idiom Suggestions
        if report.idiom_suggestions:
            lines.extend([
                "## Idiom Suggestions",
                "",
            ])
            for suggestion in report.idiom_suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")

        return "\n".join(lines)
