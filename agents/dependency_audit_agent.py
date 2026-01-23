"""Dependency audit agent for vulnerability scanning and version management.

Scans dependencies for:
- Known CVEs via pip-audit
- Deprecated/abandoned packages
- Outdated versions

Can automatically:
- Calculate minimal safe version bumps
- Patch requirements.txt / pyproject.toml
- Run post-patch test verification
"""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from llm_backend import LLMBackend
from schemas.dependency_audit_report import (
    DependencyAuditReport,
    DependencySource,
    DeprecatedPackage,
    OutdatedPackage,
    TestResult,
    VersionBump,
    Vulnerability,
    VulnerabilitySeverity,
)

from .base import AgentInput, AgentOutput, BaseAgent

# Known deprecated packages with replacements
DEPRECATED_PACKAGES: dict[str, dict[str, str]] = {
    "pycrypto": {
        "reason": "Unmaintained since 2013, has known vulnerabilities",
        "replacement": "pycryptodome",
    },
    "nose": {
        "reason": "Unmaintained, use pytest instead",
        "replacement": "pytest",
    },
    "distribute": {
        "reason": "Merged into setuptools",
        "replacement": "setuptools",
    },
    "argparse": {
        "reason": "Included in Python stdlib since 2.7/3.2",
        "replacement": None,
    },
    "typing": {
        "reason": "Included in Python stdlib since 3.5",
        "replacement": None,
    },
    "pathlib": {
        "reason": "Included in Python stdlib since 3.4",
        "replacement": None,
    },
    "mock": {
        "reason": "Use unittest.mock from stdlib (Python 3.3+)",
        "replacement": "unittest.mock",
    },
    "futures": {
        "reason": "Use concurrent.futures from stdlib (Python 3.2+)",
        "replacement": "concurrent.futures",
    },
    "enum34": {
        "reason": "Use enum from stdlib (Python 3.4+)",
        "replacement": None,
    },
    "pyopenssl": {
        "reason": "Consider using ssl module or cryptography directly",
        "replacement": "cryptography",
    },
    "django-extensions": {
        "reason": "Many features now in Django core",
        "replacement": None,
    },
    "pylint-django": {
        "reason": "Consider using ruff with Django plugin",
        "replacement": "ruff",
    },
}

# Severity mapping from pip-audit aliases
SEVERITY_MAP: dict[str, VulnerabilitySeverity] = {
    "critical": VulnerabilitySeverity.CRITICAL,
    "high": VulnerabilitySeverity.HIGH,
    "medium": VulnerabilitySeverity.MEDIUM,
    "moderate": VulnerabilitySeverity.MEDIUM,
    "low": VulnerabilitySeverity.LOW,
    "unknown": VulnerabilitySeverity.UNKNOWN,
}


class DependencyAuditAgent(BaseAgent):
    """Agent for auditing dependencies and managing versions.

    Capabilities:
    - Scan requirements.txt and pyproject.toml
    - Detect CVEs via pip-audit
    - Find deprecated/outdated packages
    - Calculate minimal safe version bumps
    - Patch dependency files
    - Run post-patch test verification
    """

    def __init__(
        self,
        llm: LLMBackend,
        auto_patch: bool = True,
        run_tests_after_patch: bool = True,
        block_on_critical: bool = True,
    ) -> None:
        """Initialize the agent.

        Args:
            llm: LLM backend for analysis
            auto_patch: Whether to automatically patch requirements
            run_tests_after_patch: Whether to run tests after patching
            block_on_critical: Whether to block pipeline on critical CVEs
        """
        super().__init__(llm)
        self.auto_patch = auto_patch
        self.run_tests_after_patch = run_tests_after_patch
        self.block_on_critical = block_on_critical

    def default_system_prompt(self) -> str:
        """Return the default system prompt for dependency auditing."""
        return """You are a security-focused dependency auditor. Your role is to:
1. Analyze vulnerability scan results and assess risk
2. Prioritize fixes based on severity and exploitability
3. Recommend safe version upgrades
4. Identify deprecated packages and suggest replacements
5. Provide actionable remediation steps

Always prioritize security while considering compatibility and stability."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run dependency audit.

        Args:
            input_data: Agent input with repo_path

        Returns:
            Agent output with DependencyAuditReport
        """
        start_time = time.time()
        repo_path = Path(input_data.repo_path)

        report = DependencyAuditReport()

        # Find dependency files
        requirements_txt = repo_path / "requirements.txt"
        pyproject_toml = repo_path / "pyproject.toml"

        # Scan for vulnerabilities
        if requirements_txt.exists():
            report.sources_scanned.append(DependencySource.REQUIREMENTS_TXT)
            self._scan_vulnerabilities(requirements_txt, report)
            self._scan_deprecated(requirements_txt, report)
            self._scan_outdated(requirements_txt, report)

        if pyproject_toml.exists():
            report.sources_scanned.append(DependencySource.PYPROJECT_TOML)
            self._scan_pyproject_dependencies(pyproject_toml, report)

        # Update counts and determine status
        report.update_counts()
        report.determine_status()

        # Calculate version bumps for vulnerabilities
        self._calculate_version_bumps(report)

        # Auto-patch if enabled and there are safe bumps
        if self.auto_patch and report.version_bumps:
            self._apply_patches(repo_path, report)

        # Run tests if patches were applied
        if (
            self.run_tests_after_patch
            and report.bumps_applied > 0
            and report.files_modified
        ):
            self._run_post_patch_tests(repo_path, report)

        # Get LLM recommendations
        self._get_llm_analysis(report, input_data)

        # Final status check
        if self.block_on_critical and report.critical_count > 0:
            report.blocked = True
            report.block_reason = (
                f"Critical vulnerabilities require immediate attention: "
                f"{report.critical_count} critical CVEs found"
            )

        duration = time.time() - start_time

        return AgentOutput(
            success=not report.blocked,
            result=report.model_dump(),
            error=report.block_reason if report.blocked else None,
            duration_seconds=duration,
        )

    def _scan_vulnerabilities(
        self, requirements_file: Path, report: DependencyAuditReport
    ) -> None:
        """Scan for CVEs using pip-audit.

        Args:
            requirements_file: Path to requirements.txt
            report: Report to update
        """
        if not shutil.which("pip-audit"):
            return

        try:
            result = subprocess.run(
                [
                    "pip-audit",
                    "-r",
                    str(requirements_file),
                    "--format",
                    "json",
                    "--progress-spinner",
                    "off",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            # pip-audit returns non-zero if vulnerabilities found
            output = result.stdout or result.stderr
            if not output.strip():
                return

            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                return

            # Handle different pip-audit output formats
            dependencies = data if isinstance(data, list) else data.get("dependencies", [])

            for dep in dependencies:
                vulns = dep.get("vulns", [])
                for vuln in vulns:
                    severity_str = vuln.get("severity", "unknown").lower()
                    severity = SEVERITY_MAP.get(
                        severity_str, VulnerabilitySeverity.UNKNOWN
                    )

                    report.vulnerabilities.append(
                        Vulnerability(
                            package=dep.get("name", "unknown"),
                            installed_version=dep.get("version", "unknown"),
                            vuln_id=vuln.get("id", "unknown"),
                            severity=severity,
                            description=vuln.get("description", ""),
                            fix_versions=vuln.get("fix_versions", []),
                            references=vuln.get("references", []),
                        )
                    )

        except subprocess.TimeoutExpired:
            pass
        except (subprocess.SubprocessError, OSError):
            pass

    def _scan_deprecated(
        self, requirements_file: Path, report: DependencyAuditReport
    ) -> None:
        """Scan for deprecated packages.

        Args:
            requirements_file: Path to requirements.txt
            report: Report to update
        """
        content = requirements_file.read_text()

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Parse package name and version
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if not match:
                continue

            package = match.group(1).lower()

            if package in DEPRECATED_PACKAGES:
                info = DEPRECATED_PACKAGES[package]
                version_match = re.search(r"[=<>!]+(.+)$", line)
                version = version_match.group(1) if version_match else "unknown"

                report.deprecated.append(
                    DeprecatedPackage(
                        package=package,
                        installed_version=version,
                        reason=info["reason"],
                        replacement=info.get("replacement"),
                    )
                )

    def _scan_outdated(
        self, requirements_file: Path, report: DependencyAuditReport
    ) -> None:
        """Scan for outdated packages using pip list --outdated.

        Args:
            requirements_file: Path to requirements.txt (for context)
            report: Report to update
        """
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return

            packages = json.loads(result.stdout)

            for pkg in packages:
                # Check if it's a major version bump
                old_parts = pkg.get("version", "0").split(".")
                new_parts = pkg.get("latest_version", "0").split(".")
                is_major = (
                    len(old_parts) > 0
                    and len(new_parts) > 0
                    and old_parts[0] != new_parts[0]
                )

                report.outdated.append(
                    OutdatedPackage(
                        package=pkg.get("name", "unknown"),
                        installed_version=pkg.get("version", "unknown"),
                        latest_version=pkg.get("latest_version", "unknown"),
                        source=DependencySource.REQUIREMENTS_TXT,
                        is_major_update=is_major,
                    )
                )

        except subprocess.TimeoutExpired:
            pass
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            pass

    def _scan_pyproject_dependencies(
        self, pyproject_file: Path, report: DependencyAuditReport
    ) -> None:
        """Scan pyproject.toml for dependencies.

        Args:
            pyproject_file: Path to pyproject.toml
            report: Report to update
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                return

        try:
            content = pyproject_file.read_text()
            data = tomllib.loads(content)

            # Get dependencies from [project.dependencies] or [tool.poetry.dependencies]
            deps: list[str] = []

            if "project" in data and "dependencies" in data["project"]:
                deps.extend(data["project"]["dependencies"])

            if "tool" in data:
                if "poetry" in data["tool"]:
                    poetry_deps = data["tool"]["poetry"].get("dependencies", {})
                    deps.extend(poetry_deps.keys())

            # Check each dependency against deprecated list
            for dep in deps:
                # Parse package name from dependency string
                match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
                if not match:
                    continue

                package = match.group(1).lower()

                if package in DEPRECATED_PACKAGES:
                    info = DEPRECATED_PACKAGES[package]
                    report.deprecated.append(
                        DeprecatedPackage(
                            package=package,
                            installed_version="specified in pyproject.toml",
                            reason=info["reason"],
                            replacement=info.get("replacement"),
                        )
                    )

        except (OSError, ValueError):
            pass

    def _calculate_version_bumps(self, report: DependencyAuditReport) -> None:
        """Calculate minimal safe version bumps for vulnerabilities.

        Args:
            report: Report to update with version bumps
        """
        # Group vulnerabilities by package
        package_vulns: dict[str, list[Vulnerability]] = {}
        for vuln in report.vulnerabilities:
            if vuln.package not in package_vulns:
                package_vulns[vuln.package] = []
            package_vulns[vuln.package].append(vuln)

        for package, vulns in package_vulns.items():
            # Find the minimum fix version that addresses all vulns
            all_fix_versions: set[str] = set()
            vuln_ids: list[str] = []
            highest_severity = VulnerabilitySeverity.LOW

            for vuln in vulns:
                all_fix_versions.update(vuln.fix_versions)
                vuln_ids.append(vuln.vuln_id)
                if self._severity_rank(vuln.severity) > self._severity_rank(
                    highest_severity
                ):
                    highest_severity = vuln.severity

            if not all_fix_versions:
                continue

            # Sort versions and pick the minimum safe one
            sorted_versions = self._sort_versions(list(all_fix_versions))
            if not sorted_versions:
                continue

            min_safe_version = sorted_versions[0]
            current_version = vulns[0].installed_version

            # Determine if this is a breaking change
            is_breaking = self._is_major_bump(current_version, min_safe_version)

            report.version_bumps.append(
                VersionBump(
                    package=package,
                    old_version=current_version,
                    new_version=min_safe_version,
                    reason=f"Fix {len(vuln_ids)} vulnerabilities: {', '.join(vuln_ids[:3])}",
                    is_breaking=is_breaking,
                    applied=False,
                    source_file="requirements.txt",
                    vulnerabilities_fixed=vuln_ids,
                )
            )

    def _severity_rank(self, severity: VulnerabilitySeverity) -> int:
        """Get numeric rank for severity comparison."""
        ranks = {
            VulnerabilitySeverity.CRITICAL: 4,
            VulnerabilitySeverity.HIGH: 3,
            VulnerabilitySeverity.MEDIUM: 2,
            VulnerabilitySeverity.LOW: 1,
            VulnerabilitySeverity.UNKNOWN: 0,
        }
        return ranks.get(severity, 0)

    def _sort_versions(self, versions: list[str]) -> list[str]:
        """Sort version strings semantically."""
        try:
            from packaging.version import Version

            parsed = [(v, Version(v)) for v in versions if v]
            parsed.sort(key=lambda x: x[1])
            return [v[0] for v in parsed]
        except (ImportError, Exception):
            # Fallback to string sort
            return sorted(versions)

    def _is_major_bump(self, old: str, new: str) -> bool:
        """Check if version bump is a major version change."""
        old_parts = old.split(".")
        new_parts = new.split(".")
        if len(old_parts) > 0 and len(new_parts) > 0:
            return old_parts[0] != new_parts[0]
        return False

    def _apply_patches(
        self, repo_path: Path, report: DependencyAuditReport
    ) -> None:
        """Apply version bumps to requirements files.

        Args:
            repo_path: Repository root
            report: Report with version bumps to apply
        """
        requirements_file = repo_path / "requirements.txt"

        if not requirements_file.exists():
            return

        content = requirements_file.read_text()
        original_content = content
        modified = False

        for bump in report.version_bumps:
            # Skip breaking changes unless explicitly requested
            if bump.is_breaking:
                continue

            # Pattern to match package with version specifier
            patterns = [
                rf"^{re.escape(bump.package)}==[^\s]+",  # pkg==version
                rf"^{re.escape(bump.package)}>=[^\s]+",  # pkg>=version
                rf"^{re.escape(bump.package)}~=[^\s]+",  # pkg~=version
                rf"^{re.escape(bump.package)}<[^\s]+",  # pkg<version
            ]

            for pattern in patterns:
                new_content = re.sub(
                    pattern,
                    f"{bump.package}>={bump.new_version}",
                    content,
                    flags=re.MULTILINE | re.IGNORECASE,
                )
                if new_content != content:
                    content = new_content
                    bump.applied = True
                    report.bumps_applied += 1
                    modified = True
                    break

        if modified:
            # Backup original
            backup_file = repo_path / "requirements.txt.bak"
            backup_file.write_text(original_content)

            # Write patched version
            requirements_file.write_text(content)
            report.files_modified.append("requirements.txt")

    def _run_post_patch_tests(
        self, repo_path: Path, report: DependencyAuditReport
    ) -> None:
        """Run tests after patching to verify compatibility.

        Args:
            repo_path: Repository root
            report: Report to update with test results
        """
        report.tests_run = True

        # First, install updated dependencies
        try:
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=repo_path,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

        # Run pytest
        start_time = time.time()
        try:
            result = subprocess.run(
                ["pytest", "--tb=short", "-q"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            duration = time.time() - start_time
            passed = result.returncode == 0

            # Extract failed test names
            failed_tests: list[str] = []
            if not passed:
                for line in result.stdout.splitlines():
                    if "FAILED" in line:
                        match = re.match(r"FAILED\s+(\S+)", line)
                        if match:
                            failed_tests.append(match.group(1))

            report.test_result = TestResult(
                passed=passed,
                test_command="pytest --tb=short -q",
                output_summary=result.stdout[-500:] if result.stdout else None,
                duration_seconds=duration,
                failed_tests=failed_tests,
            )

            # Rollback if tests failed
            if not passed:
                self._rollback_patches(repo_path, report)

        except subprocess.TimeoutExpired:
            report.test_result = TestResult(
                passed=False,
                test_command="pytest --tb=short -q",
                output_summary="Test execution timed out",
                failed_tests=[],
            )
            self._rollback_patches(repo_path, report)

        except (subprocess.SubprocessError, OSError) as e:
            report.test_result = TestResult(
                passed=False,
                test_command="pytest --tb=short -q",
                output_summary=f"Failed to run tests: {e}",
                failed_tests=[],
            )

    def _rollback_patches(
        self, repo_path: Path, report: DependencyAuditReport
    ) -> None:
        """Rollback patches if tests failed.

        Args:
            repo_path: Repository root
            report: Report to update
        """
        backup_file = repo_path / "requirements.txt.bak"
        requirements_file = repo_path / "requirements.txt"

        if backup_file.exists():
            requirements_file.write_text(backup_file.read_text())
            backup_file.unlink()
            report.rollback_applied = True
            report.bumps_applied = 0
            report.files_modified.clear()

            # Mark all bumps as not applied
            for bump in report.version_bumps:
                bump.applied = False

    def _get_llm_analysis(
        self, report: DependencyAuditReport, input_data: AgentInput
    ) -> None:
        """Get LLM recommendations for the audit findings.

        Args:
            report: Audit report with findings
            input_data: Original agent input
        """
        if not report.vulnerabilities and not report.deprecated:
            report.llm_recommendations = ["All dependencies appear secure and up to date."]
            report.risk_assessment = "Low risk - no vulnerabilities detected"
            return

        # Build prompt for LLM
        findings_summary = []

        if report.vulnerabilities:
            findings_summary.append(
                f"Vulnerabilities: {report.critical_count} critical, "
                f"{report.high_count} high, {report.medium_count} medium"
            )
            for vuln in report.vulnerabilities[:5]:
                findings_summary.append(
                    f"  - {vuln.package} ({vuln.vuln_id}): {vuln.description[:100]}"
                )

        if report.deprecated:
            findings_summary.append(f"\nDeprecated packages: {len(report.deprecated)}")
            for dep in report.deprecated[:3]:
                repl = f" -> {dep.replacement}" if dep.replacement else ""
                findings_summary.append(f"  - {dep.package}: {dep.reason}{repl}")

        prompt = f"""Analyze these dependency audit findings and provide:
1. A brief risk assessment (1-2 sentences)
2. Top 3 prioritized recommendations

Findings:
{chr(10).join(findings_summary)}

Feature context: {input_data.feature_description or 'Not specified'}

Respond in this format:
RISK: <assessment>
RECOMMENDATIONS:
1. <recommendation>
2. <recommendation>
3. <recommendation>"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.get("content", "")

            # Parse risk assessment
            if "RISK:" in content:
                risk_line = content.split("RISK:")[1].split("\n")[0].strip()
                report.risk_assessment = risk_line

            # Parse recommendations
            if "RECOMMENDATIONS:" in content:
                recs_section = content.split("RECOMMENDATIONS:")[1]
                for line in recs_section.splitlines():
                    line = line.strip()
                    if line and line[0].isdigit():
                        # Remove leading number and punctuation
                        rec = re.sub(r"^\d+\.\s*", "", line)
                        if rec:
                            report.llm_recommendations.append(rec)

        except Exception:
            report.llm_recommendations = [
                "Review and update vulnerable packages",
                "Consider replacing deprecated dependencies",
                "Run full test suite after updates",
            ]
            report.risk_assessment = (
                f"Manual review recommended - {len(report.vulnerabilities)} "
                f"vulnerabilities found"
            )
