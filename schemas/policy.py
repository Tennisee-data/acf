"""Schema for Policy enforcement.

Defines structured models for:
- Policy rules (YAML-defined)
- Policy context (input from pipeline)
- Policy decisions (allow/block/require_approval)
"""

from enum import Enum

from pydantic import BaseModel, Field


class PolicyAction(str, Enum):
    """Possible policy actions."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class PolicySeverity(str, Enum):
    """Severity levels for policy violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCondition(BaseModel):
    """Conditions that trigger a rule."""

    # Coverage conditions
    coverage_lt: float | None = Field(
        default=None, description="Trigger if coverage is less than this"
    )
    coverage_gt: float | None = Field(
        default=None, description="Trigger if coverage is greater than this"
    )

    # Security conditions
    has_secrets: bool | None = Field(
        default=None, description="Trigger if secrets are detected"
    )
    bandit_high_gt: int | None = Field(
        default=None, description="Trigger if bandit high issues exceed this"
    )
    bandit_medium_gt: int | None = Field(
        default=None, description="Trigger if bandit medium issues exceed this"
    )
    has_vulnerabilities: bool | None = Field(
        default=None, description="Trigger if CVEs are detected"
    )

    # Path-based conditions
    paths_match: list[str] | None = Field(
        default=None, description="Trigger if any changed file matches these patterns"
    )
    paths_subset_of: list[str] | None = Field(
        default=None, description="Trigger if ALL changed files match these patterns"
    )
    paths_exclude: list[str] | None = Field(
        default=None, description="Trigger if NO changed files match these patterns"
    )

    # Diff conditions
    files_changed_gt: int | None = Field(
        default=None, description="Trigger if files changed exceeds this"
    )
    lines_added_gt: int | None = Field(
        default=None, description="Trigger if lines added exceeds this"
    )
    lines_deleted_gt: int | None = Field(
        default=None, description="Trigger if lines deleted exceeds this"
    )

    # Test conditions
    tests_failed: bool | None = Field(
        default=None, description="Trigger if any tests failed"
    )
    tests_skipped_gt: int | None = Field(
        default=None, description="Trigger if skipped tests exceed this"
    )

    # Code review conditions
    ship_status: str | None = Field(
        default=None, description="Trigger if code review status matches"
    )
    review_issues_gt: int | None = Field(
        default=None, description="Trigger if review issues exceed this"
    )

    # Branch conditions
    branch_match: str | None = Field(
        default=None, description="Trigger if branch matches pattern"
    )
    is_main_branch: bool | None = Field(
        default=None, description="Trigger if targeting main/master branch"
    )


class PolicyRule(BaseModel):
    """A single policy rule."""

    id: str = Field(description="Unique rule identifier")
    description: str = Field(description="Human-readable description")
    when: RuleCondition = Field(description="Conditions that trigger this rule")
    action: PolicyAction = Field(description="Action to take when rule matches")
    severity: PolicySeverity = Field(
        default=PolicySeverity.WARNING, description="Severity of violation"
    )
    required_role: str | None = Field(
        default=None, description="Role required for approval (if require_approval)"
    )
    message: str | None = Field(
        default=None, description="Custom message to display when triggered"
    )
    enabled: bool = Field(default=True, description="Whether rule is active")


class PolicyRuleSet(BaseModel):
    """Collection of policy rules."""

    version: str = Field(default="1.0", description="Rule set version")
    description: str | None = Field(default=None, description="Rule set description")
    rules: list[PolicyRule] = Field(
        default_factory=list, description="List of policy rules"
    )


class DiffStats(BaseModel):
    """Statistics about code changes."""

    files_changed: int = Field(default=0, description="Number of files changed")
    lines_added: int = Field(default=0, description="Lines added")
    lines_deleted: int = Field(default=0, description="Lines deleted")


class TestResults(BaseModel):
    """Test execution results."""

    passed: bool = Field(default=True, description="Whether all tests passed")
    total: int = Field(default=0, description="Total tests")
    failed: int = Field(default=0, description="Failed tests")
    skipped: int = Field(default=0, description="Skipped tests")
    coverage: float = Field(default=0.0, description="Code coverage percentage (0-1)")


class SecurityResults(BaseModel):
    """Security scan results."""

    has_secrets: bool = Field(default=False, description="Secrets detected")
    secrets_count: int = Field(default=0, description="Number of secrets found")
    bandit_high: int = Field(default=0, description="Bandit high severity issues")
    bandit_medium: int = Field(default=0, description="Bandit medium severity issues")
    bandit_low: int = Field(default=0, description="Bandit low severity issues")
    has_vulnerabilities: bool = Field(default=False, description="CVEs detected")
    vulnerability_count: int = Field(default=0, description="Number of CVEs")


class CodeReviewResults(BaseModel):
    """Code review results."""

    ship_status: str = Field(default="unknown", description="ship/ship_with_nits/dont_ship")
    total_issues: int = Field(default=0, description="Total review issues")
    critical_issues: int = Field(default=0, description="Critical issues")
    major_issues: int = Field(default=0, description="Major issues")


class PolicyContext(BaseModel):
    """Context provided to PolicyAgent for evaluation."""

    run_id: str = Field(description="Pipeline run ID")
    actor: str = Field(default="unknown", description="Who triggered the run")
    branch: str = Field(default="main", description="Target branch")
    base_branch: str = Field(default="main", description="Base branch for comparison")
    files_changed: list[str] = Field(
        default_factory=list, description="List of changed file paths"
    )
    diff_stats: DiffStats = Field(
        default_factory=DiffStats, description="Diff statistics"
    )
    test_results: TestResults = Field(
        default_factory=TestResults, description="Test results"
    )
    security_results: SecurityResults = Field(
        default_factory=SecurityResults, description="Security scan results"
    )
    code_review: CodeReviewResults = Field(
        default_factory=CodeReviewResults, description="Code review results"
    )
    feature_description: str = Field(default="", description="Feature being built")


class MatchedRule(BaseModel):
    """A rule that matched during evaluation."""

    rule_id: str = Field(description="ID of the matched rule")
    action: PolicyAction = Field(description="Action from the rule")
    severity: PolicySeverity = Field(description="Severity level")
    description: str = Field(description="Rule description")
    message: str | None = Field(default=None, description="Custom message")
    required_role: str | None = Field(default=None, description="Required role")


class PolicyDecision(BaseModel):
    """Result of policy evaluation."""

    status: PolicyAction = Field(description="Final decision: allow/block/require_approval")
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable reasons"
    )
    matched_rules: list[MatchedRule] = Field(
        default_factory=list, description="All rules that matched"
    )
    required_role: str | None = Field(
        default=None, description="Role required for approval"
    )
    blocking_rules: list[str] = Field(
        default_factory=list, description="IDs of rules causing block"
    )
    approval_rules: list[str] = Field(
        default_factory=list, description="IDs of rules requiring approval"
    )
    summary: str = Field(default="", description="Summary of policy decision")
