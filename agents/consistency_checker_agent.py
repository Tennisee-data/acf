"""Consistency Checker Agent - Validates implementation matches design.

Catches cases where the LLM generates completely wrong code by checking:
1. File paths match design proposal
2. Code content is semantically relevant to feature
3. Required imports/patterns are present

This is a "semantic sanity check" that runs BEFORE the fix loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyError:
    """A detected consistency error."""

    category: str  # "path_mismatch", "content_mismatch", "missing_import"
    message: str
    severity: str  # "error" or "warning"
    expected: str | None = None
    actual: str | None = None
    fix_hint: str | None = None


@dataclass
class ConsistencyReport:
    """Result of consistency check."""

    passed: bool
    errors: list[ConsistencyError] = field(default_factory=list)
    warnings: list[ConsistencyError] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.passed:
            return f"Consistency check passed ({len(self.checks_performed)} checks)"

        lines = [
            f"Consistency check FAILED: {self.error_count} error(s), {self.warning_count} warning(s)",
            "",
        ]

        for err in self.errors:
            lines.append(f"[ERROR] {err.category}: {err.message}")
            if err.expected:
                lines.append(f"  Expected: {err.expected}")
            if err.actual:
                lines.append(f"  Actual: {err.actual}")
            if err.fix_hint:
                lines.append(f"  Fix: {err.fix_hint}")
            lines.append("")

        for warn in self.warnings:
            lines.append(f"[WARN] {warn.category}: {warn.message}")

        return "\n".join(lines)


# Domain keywords - if feature mentions these, code should contain related imports
DOMAIN_KEYWORDS = {
    "stripe": {
        "keywords": ["stripe", "payment", "checkout", "subscription"],
        "expected_imports": ["stripe"],
        "expected_patterns": [r"stripe\.", r"Stripe"],
    },
    "webhook": {
        "keywords": ["webhook", "callback", "hook"],
        "expected_imports": [],
        "expected_patterns": [r"@app\.(post|put)", r"webhook", r"signature"],
    },
    "slack": {
        "keywords": ["slack", "notification"],
        "expected_imports": ["slack", "slack_sdk", "aiohttp", "requests"],
        # More lenient - any notification/send pattern works
        "expected_patterns": [r"slack", r"notification", r"send.*message", r"send.*notification", r"notify"],
    },
    "auth": {
        "keywords": ["auth", "login", "jwt", "oauth", "password"],
        "expected_imports": ["jwt", "oauth", "passlib", "bcrypt"],
        # More specific patterns to avoid false positives with "token"
        "expected_patterns": [r"authenticate", r"authorize", r"login", r"jwt\.", r"oauth"],
    },
    "database": {
        "keywords": ["database", "db", "sql", "model", "orm", "order status"],
        "expected_imports": ["sqlalchemy", "sqlmodel", "prisma", "tortoise"],
        # More lenient patterns - include common db operation names
        "expected_patterns": [r"session", r"query", r"model", r"update.*status", r"order", r"save", r"commit"],
    },
    "fastapi": {
        "keywords": ["fastapi", "api", "endpoint", "rest"],
        "expected_imports": ["fastapi"],
        "expected_patterns": [r"FastAPI", r"@app\.", r"APIRouter"],
    },
}


class ConsistencyCheckerAgent:
    """Agent that validates implementation matches design.

    Catches semantic mismatches like:
    - Design says "create app/webhooks.py" but impl creates "unknown"
    - Feature is "Stripe webhook" but code is Google OAuth
    - Missing required imports for the domain

    Example:
        checker = ConsistencyCheckerAgent()
        report = checker.check(
            feature_description="Add Stripe webhook endpoint",
            design_file_paths=["app/webhooks.py"],
            impl_file_changes=[{"path": "unknown", "new_code": "..."}],
        )
        if not report.passed:
            raise ValueError(report.summary())
    """

    def check(
        self,
        feature_description: str,
        design_file_paths: list[str],
        impl_file_changes: list[dict[str, Any]],
        design_summary: str = "",
    ) -> ConsistencyReport:
        """Check implementation consistency with design.

        Args:
            feature_description: Original feature request
            design_file_paths: Expected file paths from design proposal
            impl_file_changes: Actual file changes from implementation
            design_summary: Design proposal summary

        Returns:
            ConsistencyReport with errors/warnings
        """
        errors: list[ConsistencyError] = []
        warnings: list[ConsistencyError] = []
        checks: list[str] = []

        # Extract implementation paths and code
        impl_paths = [fc.get("path", "") for fc in impl_file_changes]
        impl_code = "\n".join(fc.get("new_code", "") for fc in impl_file_changes)

        # Check 1: Path consistency
        checks.append("path_consistency")
        path_errors = self._check_path_consistency(design_file_paths, impl_paths)
        errors.extend([e for e in path_errors if e.severity == "error"])
        warnings.extend([e for e in path_errors if e.severity == "warning"])

        # Check 2: Semantic content match
        checks.append("semantic_content")
        content_errors = self._check_semantic_content(
            feature_description, design_summary, impl_code
        )
        errors.extend([e for e in content_errors if e.severity == "error"])
        warnings.extend([e for e in content_errors if e.severity == "warning"])

        # Check 3: Required imports
        checks.append("required_imports")
        import_errors = self._check_required_imports(feature_description, impl_code)
        errors.extend([e for e in import_errors if e.severity == "error"])
        warnings.extend([e for e in import_errors if e.severity == "warning"])

        # Check 4: Code not empty
        checks.append("code_not_empty")
        if not impl_code.strip():
            errors.append(ConsistencyError(
                category="empty_code",
                message="Implementation generated no code",
                severity="error",
                fix_hint="Check LLM response parsing",
            ))

        # Check 5: README references match actual files
        checks.append("readme_file_references")
        readme_errors = self._check_readme_file_references(impl_file_changes)
        errors.extend([e for e in readme_errors if e.severity == "error"])
        warnings.extend([e for e in readme_errors if e.severity == "warning"])

        passed = len(errors) == 0

        return ConsistencyReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            checks_performed=checks,
        )

    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison (strip leading slashes, lowercase)."""
        return path.lstrip("/").lower()

    def _check_path_consistency(
        self,
        design_paths: list[str],
        impl_paths: list[str],
    ) -> list[ConsistencyError]:
        """Check that implementation paths match design."""
        errors = []

        # Invalid path detection
        invalid_paths = {"unknown", "file.py", "code.py", ""}
        for path in impl_paths:
            if path in invalid_paths:
                errors.append(ConsistencyError(
                    category="invalid_path",
                    message=f"Invalid file path: '{path}'",
                    severity="error",
                    expected="Valid path like 'app/main.py'",
                    actual=path,
                    fix_hint="LLM must specify real file paths",
                ))

        # Check design paths are covered (normalize for comparison)
        if design_paths and impl_paths:
            design_normalized = {self._normalize_path(p) for p in design_paths}
            impl_normalized = {
                self._normalize_path(p) for p in impl_paths if p not in invalid_paths
            }

            # Check if any normalized paths match
            matches = design_normalized & impl_normalized
            if not matches:
                # Fallback: check if basenames match (looser check)
                design_basenames = {p.split("/")[-1].lower() for p in design_paths}
                impl_basenames = {p.split("/")[-1].lower() for p in impl_paths if p not in invalid_paths}
                basename_matches = design_basenames & impl_basenames

                if not basename_matches:
                    errors.append(ConsistencyError(
                        category="path_mismatch",
                        message="Implementation paths don't match any design paths",
                        severity="error",
                        expected=", ".join(design_paths[:3]),
                        actual=", ".join(impl_paths[:3]),
                        fix_hint="Implementation should create files specified in design",
                    ))

        return errors

    def _check_semantic_content(
        self,
        feature_description: str,
        design_summary: str,
        impl_code: str,
    ) -> list[ConsistencyError]:
        """Check that code content matches feature domain."""
        errors = []
        feature_lower = feature_description.lower()
        design_lower = design_summary.lower()
        code_lower = impl_code.lower()

        # Detect which domains are relevant
        relevant_domains = []
        for domain, config in DOMAIN_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword in feature_lower or keyword in design_lower:
                    relevant_domains.append(domain)
                    break

        if not relevant_domains:
            return errors  # Can't determine expected domain

        # Check if code contains expected patterns
        for domain in relevant_domains:
            config = DOMAIN_KEYWORDS[domain]
            patterns = config["expected_patterns"]

            # Check if ANY pattern matches
            pattern_found = False
            for pattern in patterns:
                if re.search(pattern, impl_code, re.IGNORECASE):
                    pattern_found = True
                    break

            if not pattern_found:
                # Downgrade to warning - code might delegate to services or use different patterns
                errors.append(ConsistencyError(
                    category="content_mismatch",
                    message=f"Feature requires '{domain}' but expected patterns not found",
                    severity="warning",
                    expected=f"Code containing: {', '.join(patterns[:2])}",
                    actual=f"Code preview: {impl_code[:100]}..." if impl_code else "(empty)",
                    fix_hint=f"Verify {domain}-related code is present or delegated to services",
                ))

        # Check for wrong domain code (only if expected patterns are MISSING)
        # If we found expected patterns, don't flag wrong domain
        # Count actual errors (severity="error"), not warnings
        actual_errors = [e for e in errors if e.severity == "error"]
        expected_found = len(actual_errors) == 0

        if not expected_found:
            wrong_domains = []
            for domain, config in DOMAIN_KEYWORDS.items():
                if domain in relevant_domains:
                    continue
                # Check if this unrelated domain is strongly present
                matches = sum(
                    1 for pattern in config["expected_patterns"]
                    if re.search(pattern, impl_code, re.IGNORECASE)
                )
                # Only flag if multiple patterns match (strong signal)
                if matches >= 2:
                    wrong_domains.append(domain)

            if wrong_domains and relevant_domains:
                primary_domain = relevant_domains[0]
                # Downgrade to warning - might be valid related code
                errors.append(ConsistencyError(
                    category="wrong_domain",
                    message=f"Code may contain '{wrong_domains[0]}' patterns instead of '{primary_domain}'",
                    severity="warning",
                    expected=f"Code for {primary_domain}",
                    actual=f"Code contains {wrong_domains[0]} patterns",
                    fix_hint=f"Verify code is for the correct feature",
                ))

        return errors

    def _check_required_imports(
        self,
        feature_description: str,
        impl_code: str,
    ) -> list[ConsistencyError]:
        """Check that required imports are present."""
        errors = []
        feature_lower = feature_description.lower()

        for domain, config in DOMAIN_KEYWORDS.items():
            # Check if this domain is relevant
            domain_relevant = any(
                keyword in feature_lower
                for keyword in config["keywords"]
            )

            if not domain_relevant:
                continue

            # Check for expected imports
            expected_imports = config["expected_imports"]
            if not expected_imports:
                continue

            # Check if at least one expected import is present
            import_found = False
            for imp in expected_imports:
                if f"import {imp}" in impl_code or f"from {imp}" in impl_code:
                    import_found = True
                    break

            if not import_found:
                errors.append(ConsistencyError(
                    category="missing_import",
                    message=f"Expected import for '{domain}' not found",
                    severity="warning",  # Warning since import might be indirect
                    expected=f"import {expected_imports[0]}",
                    fix_hint=f"Add required import for {domain} functionality",
                ))

        return errors

    def _check_readme_file_references(
        self,
        impl_file_changes: list[dict[str, Any]],
    ) -> list[ConsistencyError]:
        """Check that files referenced in README actually exist.

        Catches cases like:
        - README says "cp .env.example .env" but .env.example doesn't exist
        - README says "bash setup.sh" but setup.sh doesn't exist
        """
        errors = []

        # Get all generated file paths and their content
        generated_paths = {fc.get("path", "") for fc in impl_file_changes}
        file_contents = {fc.get("path", ""): fc.get("new_code", "") for fc in impl_file_changes}

        # Find README content
        readme_content = ""
        for fc in impl_file_changes:
            path = fc.get("path", "").lower()
            if path in ("readme.md", "readme", "readme.txt"):
                readme_content = fc.get("new_code", "")
                break

        if not readme_content:
            return errors  # No README to check

        # Patterns for file references in instructions
        file_ref_patterns = [
            # cp source dest
            (r"cp\s+([^\s]+\.(?:example|sample|template))\s+", "copy command"),
            # cat/bash/python/sh file
            (r"(?:cat|bash|sh|python|python3|node)\s+([^\s\n|>]+\.[a-z]+)", "run command"),
            # source file
            (r"source\s+([^\s\n]+)", "source command"),
            # chmod +x file
            (r"chmod\s+\+x\s+([^\s\n]+)", "chmod command"),
        ]

        # Critical files that MUST exist if referenced
        critical_files = {".env.example", ".env.sample", "setup.sh", "install.sh", "start.sh"}

        for pattern, context in file_ref_patterns:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            for ref_file in matches:
                # Normalize the path
                ref_file_clean = ref_file.strip("'\"` ")

                # Check if file exists in generated paths
                file_exists = any(
                    ref_file_clean == p or
                    ref_file_clean == p.lstrip("./") or
                    p.endswith("/" + ref_file_clean) or
                    p == "./" + ref_file_clean
                    for p in generated_paths
                )

                if not file_exists:
                    severity = "error" if ref_file_clean in critical_files else "warning"
                    errors.append(ConsistencyError(
                        category="missing_referenced_file",
                        message=f"README references '{ref_file_clean}' ({context}) but file not generated",
                        severity=severity,
                        expected=f"File '{ref_file_clean}' should exist",
                        actual=f"Generated files: {', '.join(list(generated_paths)[:5])}...",
                        fix_hint=f"Generate '{ref_file_clean}' or update README instructions",
                    ))

        # Check .env.example has required variables if project uses database
        errors.extend(self._check_env_example_content(file_contents))

        return errors

    def _check_env_example_content(
        self,
        file_contents: dict[str, str],
    ) -> list[ConsistencyError]:
        """Check that .env.example contains required variables.

        If project uses database (imports sqlalchemy, alembic, prisma, etc.),
        .env.example MUST contain DATABASE_URL with a working default.
        """
        errors = []

        # Find .env.example content
        env_example_content = ""
        for path, content in file_contents.items():
            if path.endswith(".env.example") or path == ".env.example":
                env_example_content = content
                break

        # Check if project uses database
        all_code = "\n".join(file_contents.values())
        uses_database = any(
            pattern in all_code.lower()
            for pattern in [
                "sqlalchemy", "alembic", "prisma", "database_url",
                "create_engine", "sessionmaker", "asyncpg", "psycopg",
                "sqlite", "postgresql", "mysql"
            ]
        )

        if not uses_database:
            return errors  # No database, no need for DATABASE_URL

        # Project uses database - check .env.example
        if not env_example_content:
            errors.append(ConsistencyError(
                category="missing_env_example",
                message="Project uses database but .env.example not generated",
                severity="error",
                expected=".env.example with DATABASE_URL=sqlite:///./app.db",
                fix_hint="Generate .env.example with working DATABASE_URL default",
            ))
            return errors

        # Check for DATABASE_URL
        if "DATABASE_URL" not in env_example_content:
            errors.append(ConsistencyError(
                category="missing_database_url",
                message=".env.example missing DATABASE_URL variable",
                severity="error",
                expected="DATABASE_URL=sqlite:///./app.db (or other working default)",
                actual=f".env.example content: {env_example_content[:100]}...",
                fix_hint="Add DATABASE_URL with working default (sqlite:///./app.db)",
            ))
        # Check DATABASE_URL has a real value, not a placeholder
        elif "DATABASE_URL=" in env_example_content:
            # Extract the value
            for line in env_example_content.split("\n"):
                if line.strip().startswith("DATABASE_URL="):
                    value = line.split("=", 1)[1].strip()
                    # Check for placeholder values
                    placeholder_patterns = [
                        "your_", "changeme", "placeholder", "xxx", "driver://",
                        "user:password@", "<", ">", "${", "CHANGE_ME"
                    ]
                    is_placeholder = any(p in value.lower() for p in placeholder_patterns)
                    is_empty = not value or value in ('""', "''", '""', "''")

                    if is_placeholder or is_empty:
                        errors.append(ConsistencyError(
                            category="placeholder_database_url",
                            message="DATABASE_URL contains placeholder instead of working default",
                            severity="error",
                            expected="DATABASE_URL=sqlite:///./app.db",
                            actual=f"DATABASE_URL={value}",
                            fix_hint="Use sqlite:///./app.db as default - works without setup",
                        ))
                    break

        return errors
