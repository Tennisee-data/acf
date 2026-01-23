"""Schema for Rollback Strategy generation.

Defines structured output for deployment rollback strategies including:
- Rollback job configurations for CI/CD
- Canary deployment patterns (blue-green, rolling, etc.)
- Rollback playbook documentation
"""

from enum import Enum

from pydantic import BaseModel, Field


class DeploymentPattern(str, Enum):
    """Deployment strategy patterns."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class RollbackTrigger(str, Enum):
    """Triggers that initiate rollback."""

    MANUAL = "manual"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    LATENCY_THRESHOLD = "latency_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    CPU_THRESHOLD = "cpu_threshold"


class RollbackAction(str, Enum):
    """Actions to take during rollback."""

    REVERT_DEPLOYMENT = "revert_deployment"
    SCALE_DOWN_NEW = "scale_down_new"
    RESTORE_DATABASE = "restore_database"
    INVALIDATE_CACHE = "invalidate_cache"
    NOTIFY_TEAM = "notify_team"
    CREATE_INCIDENT = "create_incident"


class HealthCheck(BaseModel):
    """Health check configuration for deployment verification."""

    endpoint: str = Field(description="Health check endpoint path")
    expected_status: int = Field(default=200, description="Expected HTTP status")
    timeout_seconds: int = Field(default=30, description="Timeout for health check")
    interval_seconds: int = Field(default=10, description="Interval between checks")
    failure_threshold: int = Field(default=3, description="Failures before rollback")


class RollbackStep(BaseModel):
    """A step in the rollback procedure."""

    order: int = Field(description="Step order in rollback sequence")
    action: RollbackAction = Field(description="Action to perform")
    description: str = Field(description="Human-readable description")
    command: str | None = Field(default=None, description="Command to execute")
    timeout_seconds: int = Field(default=300, description="Step timeout")
    on_failure: str = Field(
        default="continue", description="Action on step failure: continue, abort, retry"
    )


class RollbackJob(BaseModel):
    """GitHub Actions rollback job configuration."""

    name: str = Field(description="Job name")
    trigger: RollbackTrigger = Field(description="What triggers this rollback")
    steps: list[RollbackStep] = Field(
        default_factory=list, description="Rollback steps"
    )
    environment: str = Field(default="production", description="Target environment")
    requires_approval: bool = Field(
        default=True, description="Requires manual approval"
    )
    notification_channels: list[str] = Field(
        default_factory=list, description="Slack/email channels to notify"
    )


class CanaryConfig(BaseModel):
    """Canary deployment configuration."""

    initial_percentage: int = Field(
        default=10, description="Initial traffic percentage to canary"
    )
    increment_percentage: int = Field(
        default=20, description="Traffic increment per step"
    )
    increment_interval_minutes: int = Field(
        default=15, description="Minutes between increments"
    )
    success_threshold: float = Field(
        default=0.99, description="Success rate threshold (0-1)"
    )
    error_rate_threshold: float = Field(
        default=0.01, description="Error rate threshold for rollback"
    )
    latency_p99_threshold_ms: int = Field(
        default=500, description="P99 latency threshold in ms"
    )


class BlueGreenConfig(BaseModel):
    """Blue-green deployment configuration."""

    blue_environment: str = Field(default="blue", description="Blue environment name")
    green_environment: str = Field(
        default="green", description="Green environment name"
    )
    switch_timeout_seconds: int = Field(
        default=60, description="Timeout for traffic switch"
    )
    validation_period_seconds: int = Field(
        default=300, description="Validation period after switch"
    )
    keep_previous_version: bool = Field(
        default=True, description="Keep previous version running"
    )


class RollingConfig(BaseModel):
    """Rolling deployment configuration."""

    max_surge: str = Field(
        default="25%", description="Max extra pods during update"
    )
    max_unavailable: str = Field(
        default="25%", description="Max unavailable pods during update"
    )
    min_ready_seconds: int = Field(
        default=30, description="Min seconds before pod is ready"
    )


class PlaybookSection(BaseModel):
    """A section of the rollback playbook."""

    title: str = Field(description="Section title")
    content: str = Field(description="Markdown content")


class RollbackPlaybook(BaseModel):
    """Rollback playbook documentation."""

    title: str = Field(default="Rollback Playbook", description="Document title")
    overview: str = Field(description="Overview of rollback procedures")
    prerequisites: list[str] = Field(
        default_factory=list, description="Prerequisites for rollback"
    )
    sections: list[PlaybookSection] = Field(
        default_factory=list, description="Playbook sections"
    )
    contacts: dict[str, str] = Field(
        default_factory=dict, description="Emergency contacts"
    )
    runbook_url: str | None = Field(
        default=None, description="Link to external runbook"
    )


class GeneratedWorkflow(BaseModel):
    """A generated GitHub Actions workflow file."""

    filename: str = Field(description="Workflow filename")
    content: str = Field(description="YAML content")
    description: str = Field(description="What this workflow does")


class RollbackStrategyReport(BaseModel):
    """Complete rollback strategy report."""

    deployment_pattern: DeploymentPattern = Field(
        description="Primary deployment pattern"
    )
    health_checks: list[HealthCheck] = Field(
        default_factory=list, description="Health check configurations"
    )
    rollback_jobs: list[RollbackJob] = Field(
        default_factory=list, description="Rollback job configurations"
    )

    # Pattern-specific configs
    canary_config: CanaryConfig | None = Field(
        default=None, description="Canary deployment config"
    )
    blue_green_config: BlueGreenConfig | None = Field(
        default=None, description="Blue-green deployment config"
    )
    rolling_config: RollingConfig | None = Field(
        default=None, description="Rolling deployment config"
    )

    # Generated artifacts
    workflows: list[GeneratedWorkflow] = Field(
        default_factory=list, description="Generated workflow files"
    )
    playbook: RollbackPlaybook | None = Field(
        default=None, description="Rollback playbook documentation"
    )

    # Summary
    recommendations: list[str] = Field(
        default_factory=list, description="Strategy recommendations"
    )
