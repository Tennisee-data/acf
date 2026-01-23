"""Self-healing health check system for codebase maintenance.

Extends check_dependency_deprecation() with:
- Code deprecation scanning (stdlib, syntax patterns)
- Dependency update checking
- Auto-fix capability (pyupgrade, ruff)
- Test verification after fixes
- PR creation for changes
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============ Deprecated Code Patterns ============

# Python stdlib deprecations (Python 3.11+)
STDLIB_DEPRECATIONS = {
    # datetime
    r"datetime\.utcnow\(\)": {
        "replacement": "datetime.now(timezone.utc)",
        "message": "datetime.utcnow() is deprecated in Python 3.12",
        "import_needed": "from datetime import timezone",
    },
    r"datetime\.utcfromtimestamp\(": {
        "replacement": "datetime.fromtimestamp(..., tz=timezone.utc)",
        "message": "datetime.utcfromtimestamp() is deprecated in Python 3.12",
        "import_needed": "from datetime import timezone",
    },
    # typing module (use built-in generics in 3.9+)
    r"typing\.List\[": {
        "replacement": "list[",
        "message": "typing.List is deprecated, use list[] (Python 3.9+)",
        "auto_fix": True,
    },
    r"typing\.Dict\[": {
        "replacement": "dict[",
        "message": "typing.Dict is deprecated, use dict[] (Python 3.9+)",
        "auto_fix": True,
    },
    r"typing\.Set\[": {
        "replacement": "set[",
        "message": "typing.Set is deprecated, use set[] (Python 3.9+)",
        "auto_fix": True,
    },
    r"typing\.Tuple\[": {
        "replacement": "tuple[",
        "message": "typing.Tuple is deprecated, use tuple[] (Python 3.9+)",
        "auto_fix": True,
    },
    r"typing\.Optional\[": {
        "replacement": "X | None",
        "message": "typing.Optional is deprecated, use X | None (Python 3.10+)",
        "auto_fix": False,  # More complex replacement
    },
    r"typing\.Union\[": {
        "replacement": "X | Y",
        "message": "typing.Union is deprecated, use X | Y (Python 3.10+)",
        "auto_fix": False,
    },
    # os.path vs pathlib
    r"os\.path\.join\(": {
        "replacement": "Path(...) / ...",
        "message": "Consider using pathlib.Path instead of os.path.join",
        "severity": "info",
    },
    r"os\.path\.exists\(": {
        "replacement": "Path(...).exists()",
        "message": "Consider using pathlib.Path instead of os.path.exists",
        "severity": "info",
    },
    # collections.abc
    r"from typing import (.*\b(?:Mapping|Sequence|Iterable|Iterator)\b)": {
        "replacement": "from collections.abc import ...",
        "message": "Import from collections.abc instead of typing (Python 3.9+)",
        "auto_fix": False,
    },
    # asyncio deprecations
    r"asyncio\.get_event_loop\(\)": {
        "replacement": "asyncio.get_running_loop() or asyncio.new_event_loop()",
        "message": "asyncio.get_event_loop() deprecated without running loop",
        "severity": "warning",
    },
    # pkg_resources (deprecated)
    r"import pkg_resources": {
        "replacement": "importlib.metadata or importlib.resources",
        "message": "pkg_resources is deprecated, use importlib.metadata",
        "severity": "warning",
    },
    r"from pkg_resources import": {
        "replacement": "from importlib.metadata import ...",
        "message": "pkg_resources is deprecated, use importlib.metadata",
        "severity": "warning",
    },
}

# Old syntax patterns that should be modernized
SYNTAX_PATTERNS = {
    r"\.format\(": {
        "replacement": "f-strings",
        "message": "Consider using f-strings instead of .format()",
        "severity": "info",
        "auto_fix": False,
    },
    r"%\s*\(": {
        "replacement": "f-strings",
        "message": "Consider using f-strings instead of % formatting",
        "severity": "info",
        "auto_fix": False,
    },
    r"except\s+(\w+)\s*,\s*(\w+)\s*:": {
        "replacement": "except Exception as e:",
        "message": "Old except syntax, use 'except X as e:'",
        "auto_fix": True,
    },
}


@dataclass
class DeprecationIssue:
    """A single deprecation or code health issue."""

    file: str
    line: int
    pattern: str
    message: str
    replacement: str
    severity: str = "warning"  # info, warning, error
    auto_fixable: bool = False
    fixed: bool = False


@dataclass
class HealthReport:
    """Complete health check report."""

    issues: list[DeprecationIssue] = field(default_factory=list)
    outdated_deps: list[dict] = field(default_factory=list)
    vulnerability_deps: list[dict] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    tests_passed: bool | None = None
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "pattern": i.pattern,
                    "message": i.message,
                    "replacement": i.replacement,
                    "severity": i.severity,
                    "auto_fixable": i.auto_fixable,
                    "fixed": i.fixed,
                }
                for i in self.issues
            ],
            "outdated_deps": self.outdated_deps,
            "vulnerability_deps": self.vulnerability_deps,
            "fixes_applied": self.fixes_applied,
            "tests_passed": self.tests_passed,
            "summary": self.summary,
        }


class HealthChecker:
    """Self-healing health check system."""

    def __init__(self, project_dir: Path | str):
        """Initialize health checker.

        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.report = HealthReport()

    def scan(self) -> HealthReport:
        """Run full health scan without applying fixes.

        Returns:
            HealthReport with all issues found
        """
        self._scan_code_deprecations()
        self._scan_outdated_dependencies()
        self._scan_vulnerabilities()
        self._compute_summary()
        return self.report

    def fix(self, dry_run: bool = False) -> HealthReport:
        """Scan and apply auto-fixes.

        Args:
            dry_run: If True, only report what would be fixed

        Returns:
            HealthReport with fixes applied
        """
        self.scan()

        if not dry_run:
            self._apply_pyupgrade()
            self._apply_ruff_fixes()

        self._compute_summary()
        return self.report

    def verify(self) -> bool:
        """Run tests to verify fixes didn't break anything.

        Returns:
            True if tests pass
        """
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-x", "-q"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.report.tests_passed = result.returncode == 0
            return self.report.tests_passed
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.report.tests_passed = None
            return False

    def _scan_code_deprecations(self) -> None:
        """Scan Python files for deprecated patterns."""
        for py_file in self.project_dir.rglob("*.py"):
            # Skip common non-source directories
            if any(
                part in py_file.parts
                for part in ["venv", ".venv", "node_modules", "__pycache__", ".git"]
            ):
                continue

            try:
                content = py_file.read_text()
                lines = content.split("\n")

                # Check stdlib deprecations
                for pattern, info in STDLIB_DEPRECATIONS.items():
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            self.report.issues.append(
                                DeprecationIssue(
                                    file=str(py_file.relative_to(self.project_dir)),
                                    line=i,
                                    pattern=pattern,
                                    message=info["message"],
                                    replacement=info["replacement"],
                                    severity=info.get("severity", "warning"),
                                    auto_fixable=info.get("auto_fix", False),
                                )
                            )

                # Check syntax patterns
                for pattern, info in SYNTAX_PATTERNS.items():
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            self.report.issues.append(
                                DeprecationIssue(
                                    file=str(py_file.relative_to(self.project_dir)),
                                    line=i,
                                    pattern=pattern,
                                    message=info["message"],
                                    replacement=info["replacement"],
                                    severity=info.get("severity", "info"),
                                    auto_fixable=info.get("auto_fix", False),
                                )
                            )

            except Exception:
                continue

    def _scan_outdated_dependencies(self) -> None:
        """Check for outdated dependencies using pip."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated:
                    self.report.outdated_deps.append(
                        {
                            "name": pkg.get("name"),
                            "current": pkg.get("version"),
                            "latest": pkg.get("latest_version"),
                            "type": pkg.get("latest_filetype", "wheel"),
                        }
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

    def _scan_vulnerabilities(self) -> None:
        """Check for known vulnerabilities using pip-audit."""
        req_file = self.project_dir / "requirements.txt"
        if not req_file.exists():
            return

        try:
            result = subprocess.run(
                [
                    "pip-audit",
                    "--requirement",
                    str(req_file),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    for dep in audit_data.get("dependencies", []):
                        for vuln in dep.get("vulns", []):
                            self.report.vulnerability_deps.append(
                                {
                                    "package": dep.get("name"),
                                    "version": dep.get("version"),
                                    "vuln_id": vuln.get("id"),
                                    "description": vuln.get("description", ""),
                                    "fix_versions": vuln.get("fix_versions", []),
                                }
                            )
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _apply_pyupgrade(self) -> None:
        """Apply pyupgrade to modernize Python syntax."""
        try:
            # Check if pyupgrade is available
            check = subprocess.run(
                ["pyupgrade", "--version"],
                capture_output=True,
                check=False,
            )
            if check.returncode != 0:
                return

            # Run pyupgrade on all Python files
            for py_file in self.project_dir.rglob("*.py"):
                if any(
                    part in py_file.parts
                    for part in ["venv", ".venv", "node_modules", "__pycache__"]
                ):
                    continue

                result = subprocess.run(
                    [
                        "pyupgrade",
                        "--py311-plus",
                        str(py_file),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    rel_path = str(py_file.relative_to(self.project_dir))
                    if rel_path not in self.report.fixes_applied:
                        self.report.fixes_applied.append(f"pyupgrade: {rel_path}")

        except FileNotFoundError:
            pass

    def _apply_ruff_fixes(self) -> None:
        """Apply ruff auto-fixes for linting issues."""
        try:
            # Check if ruff is available
            check = subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                check=False,
            )
            if check.returncode != 0:
                return

            # Run ruff with auto-fix
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(self.project_dir),
                    "--fix",
                    "--select",
                    "UP,F401,I",  # pyupgrade, unused imports, isort
                    "--ignore",
                    "E501",  # line length
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if "Fixed" in result.stdout or "Fixed" in result.stderr:
                self.report.fixes_applied.append("ruff: auto-fixed issues")

        except FileNotFoundError:
            pass

    def _compute_summary(self) -> None:
        """Compute summary statistics."""
        severity_counts = {"info": 0, "warning": 0, "error": 0}
        for issue in self.report.issues:
            severity_counts[issue.severity] = (
                severity_counts.get(issue.severity, 0) + 1
            )

        self.report.summary = {
            "total_issues": len(self.report.issues),
            "auto_fixable": sum(1 for i in self.report.issues if i.auto_fixable),
            "by_severity": severity_counts,
            "outdated_packages": len(self.report.outdated_deps),
            "vulnerabilities": len(self.report.vulnerability_deps),
            "fixes_applied": len(self.report.fixes_applied),
        }


def check_health(
    project_dir: Path | str,
    fix: bool = False,
    verify: bool = False,
) -> HealthReport:
    """Run health check on a project.

    Args:
        project_dir: Path to project directory
        fix: Apply auto-fixes
        verify: Run tests after fixes

    Returns:
        HealthReport with results
    """
    checker = HealthChecker(project_dir)

    if fix:
        checker.fix()
        if verify:
            checker.verify()
    else:
        checker.scan()

    return checker.report


def create_health_pr(
    project_dir: Path | str,
    report: HealthReport,
    branch_name: str = "auto/health-fixes",
) -> dict[str, Any]:
    """Create a PR with health fixes.

    Args:
        project_dir: Path to project directory
        report: Health report with fixes
        branch_name: Branch name for PR

    Returns:
        Dict with PR creation result
    """
    project_dir = Path(project_dir)

    if not report.fixes_applied:
        return {"success": False, "error": "No fixes to commit"}

    try:
        # Create branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=project_dir,
            capture_output=True,
            check=True,
        )

        # Stage changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=project_dir,
            capture_output=True,
            check=True,
        )

        # Commit
        commit_msg = "fix: apply automated code health improvements\n\n"
        commit_msg += "Fixes applied:\n"
        for fix in report.fixes_applied:
            commit_msg += f"- {fix}\n"

        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=project_dir,
            capture_output=True,
            check=True,
        )

        # Push
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=project_dir,
            capture_output=True,
            check=True,
        )

        # Create PR using gh CLI
        pr_body = "## Automated Health Fixes\n\n"
        pr_body += f"**Issues found:** {report.summary.get('total_issues', 0)}\n"
        pr_body += f"**Fixes applied:** {report.summary.get('fixes_applied', 0)}\n"

        if report.tests_passed is not None:
            status = "Passed" if report.tests_passed else "Failed"
            pr_body += f"**Tests:** {status}\n"

        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--title",
                "fix: automated code health improvements",
                "--body",
                pr_body,
            ],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        return {
            "success": True,
            "pr_url": result.stdout.strip(),
            "branch": branch_name,
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr if e.stderr else str(e),
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"Command not found: {e}",
        }
