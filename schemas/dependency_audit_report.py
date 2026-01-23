"""Dependency audit report schema.

Structured output for dependency vulnerability scanning,
outdated package detection, and automatic version bumping.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class VulnerabilitySeverity(str, Enum):
    """CVE severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class AuditStatus(str, Enum):
    """Overall audit status."""

    PASSING = "passing"
    WARNING = "warning"  # Has outdated/deprecated but no CVEs
    FAILING = "failing"  # Has CVEs or critical issues
    BLOCKED = "blocked"  # Cannot proceed until resolved


class DependencySource(str, Enum):
    """Where the dependency is defined."""

    REQUIREMENTS_TXT = "requirements.txt"
    PYPROJECT_TOML = "pyproject.toml"
    SETUP_PY = "setup.py"
    PIPFILE = "Pipfile"


class Vulnerability(BaseModel):
    """A security vulnerability in a dependency."""

    package: str = Field(..., description="Package name")
    installed_version: str = Field(..., description="Currently installed version")
    vuln_id: str = Field(..., description="CVE or vulnerability ID")
    severity: VulnerabilitySeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Vulnerability description")
    fix_versions: list[str] = Field(
        default_factory=list,
        description="Versions that fix this vulnerability",
    )
    cwe_ids: list[str] = Field(
        default_factory=list,
        description="CWE identifiers",
    )
    cvss_score: float | None = Field(None, description="CVSS score if available")
    published_date: str | None = Field(None, description="When CVE was published")
    references: list[str] = Field(
        default_factory=list,
        description="Links to CVE databases",
    )


class DeprecatedPackage(BaseModel):
    """A deprecated or abandoned package."""

    package: str = Field(..., description="Package name")
    installed_version: str = Field(..., description="Currently installed version")
    reason: str = Field(..., description="Why it's deprecated")
    replacement: str | None = Field(None, description="Recommended replacement")
    last_release_date: str | None = Field(
        None, description="When last version was released"
    )


class OutdatedPackage(BaseModel):
    """An outdated package with available updates."""

    package: str = Field(..., description="Package name")
    installed_version: str = Field(..., description="Currently installed version")
    latest_version: str = Field(..., description="Latest available version")
    source: DependencySource = Field(..., description="Where it's defined")
    is_major_update: bool = Field(
        False, description="Whether this is a major version bump"
    )
    changelog_url: str | None = Field(None, description="Link to changelog")


class VersionBump(BaseModel):
    """A proposed or applied version change."""

    package: str = Field(..., description="Package name")
    old_version: str = Field(..., description="Previous version")
    new_version: str = Field(..., description="New version")
    reason: str = Field(..., description="Why this bump is needed")
    is_breaking: bool = Field(
        False, description="Whether this might break compatibility"
    )
    applied: bool = Field(False, description="Whether the bump was applied")
    source_file: str = Field(..., description="File that was/will be modified")
    vulnerabilities_fixed: list[str] = Field(
        default_factory=list,
        description="CVE IDs fixed by this bump",
    )


class TestResult(BaseModel):
    """Result of post-patch test verification."""

    passed: bool = Field(..., description="Whether tests passed")
    test_command: str = Field(..., description="Command that was run")
    output_summary: str | None = Field(None, description="Brief test output")
    duration_seconds: float | None = Field(None, description="How long tests took")
    failed_tests: list[str] = Field(
        default_factory=list,
        description="Names of failed tests if any",
    )


class DependencyAuditReport(BaseModel):
    """Complete dependency audit report."""

    # Metadata
    scanned_at: datetime = Field(
        default_factory=datetime.now,
        description="When the scan was performed",
    )
    status: AuditStatus = Field(
        AuditStatus.PASSING,
        description="Overall audit status",
    )

    # Sources scanned
    sources_scanned: list[DependencySource] = Field(
        default_factory=list,
        description="Which dependency files were scanned",
    )
    total_dependencies: int = Field(0, description="Total packages scanned")

    # Findings
    vulnerabilities: list[Vulnerability] = Field(
        default_factory=list,
        description="Security vulnerabilities found",
    )
    deprecated: list[DeprecatedPackage] = Field(
        default_factory=list,
        description="Deprecated packages found",
    )
    outdated: list[OutdatedPackage] = Field(
        default_factory=list,
        description="Outdated packages found",
    )

    # Summary counts
    critical_count: int = Field(0, description="Critical vulnerabilities")
    high_count: int = Field(0, description="High severity vulnerabilities")
    medium_count: int = Field(0, description="Medium severity vulnerabilities")
    low_count: int = Field(0, description="Low severity vulnerabilities")
    deprecated_count: int = Field(0, description="Deprecated packages")
    outdated_count: int = Field(0, description="Outdated packages")

    # Actions taken
    version_bumps: list[VersionBump] = Field(
        default_factory=list,
        description="Version changes proposed or applied",
    )
    bumps_applied: int = Field(0, description="Number of bumps actually applied")
    files_modified: list[str] = Field(
        default_factory=list,
        description="Files that were modified",
    )

    # Post-patch verification
    tests_run: bool = Field(False, description="Whether post-patch tests were run")
    test_result: TestResult | None = Field(
        None, description="Result of post-patch testing"
    )
    rollback_applied: bool = Field(
        False, description="Whether changes were rolled back due to test failure"
    )

    # Pipeline control
    blocked: bool = Field(
        False,
        description="Whether pipeline should be blocked",
    )
    block_reason: str | None = Field(
        None,
        description="Why the pipeline is blocked",
    )

    # LLM analysis
    llm_recommendations: list[str] = Field(
        default_factory=list,
        description="AI-generated recommendations",
    )
    risk_assessment: str | None = Field(
        None,
        description="Overall risk assessment from LLM",
    )

    def update_counts(self) -> None:
        """Update summary counts from findings."""
        self.critical_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL
        )
        self.high_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH
        )
        self.medium_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM
        )
        self.low_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.LOW
        )
        self.deprecated_count = len(self.deprecated)
        self.outdated_count = len(self.outdated)
        self.total_dependencies = (
            len(set(v.package for v in self.vulnerabilities))
            + len(set(d.package for d in self.deprecated))
            + len(set(o.package for o in self.outdated))
        )

    def determine_status(self) -> None:
        """Determine overall audit status based on findings."""
        if self.critical_count > 0 or self.high_count > 0:
            self.status = AuditStatus.FAILING
            self.blocked = True
            self.block_reason = (
                f"Found {self.critical_count} critical and "
                f"{self.high_count} high severity vulnerabilities"
            )
        elif self.medium_count > 0 or self.deprecated_count > 0:
            self.status = AuditStatus.WARNING
        else:
            self.status = AuditStatus.PASSING

    class Config:
        json_schema_extra = {
            "example": {
                "status": "warning",
                "total_dependencies": 15,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 2,
                "outdated_count": 5,
                "bumps_applied": 2,
            }
        }
