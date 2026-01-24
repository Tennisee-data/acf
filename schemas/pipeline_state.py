"""Pipeline state schema.

State machine representation for pipeline orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Overall pipeline run status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Waiting for approval
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(str, Enum):
    """Pipeline stages."""

    INIT = "init"
    SPEC = "spec"
    DECOMPOSITION = "decomposition"  # Break spec into sub-tasks
    CONTEXT = "context"
    DESIGN = "design"
    DESIGN_APPROVAL = "design_approval"
    API_CONTRACT = "api_contract"  # Define API boundaries before implementation
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    COVERAGE = "coverage"  # Enforce test coverage thresholds
    SECRETS_SCAN = "secrets_scan"  # Detect hardcoded secrets
    DEPENDENCY_AUDIT = "dependency_audit"  # Scan for CVEs and outdated packages
    DOCKER_BUILD = "docker_build"
    ROLLBACK_STRATEGY = "rollback_strategy"  # CI/CD rollback and canary deployment
    OBSERVABILITY = "observability"  # Inject logging, metrics, tracing
    CONFIG = "config"  # Enforce 12-factor config layout
    DOCS = "docs"  # Generate and sync documentation
    CODE_REVIEW = "code_review"  # Senior engineer code review
    POLICY = "policy"  # Policy enforcement gatekeeper
    VERIFICATION = "verification"
    PR_PACKAGE = "pr_package"  # Build PR title, description, changelog
    FINAL_APPROVAL = "final_approval"
    DEPLOY = "deploy"
    DONE = "done"


class StageStatus(str, Enum):
    """Individual stage status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class StageResult(BaseModel):
    """Result of a single stage execution."""

    stage: Stage = Field(..., description="Stage name")
    status: StageStatus = Field(..., description="Stage status")
    started_at: datetime | None = Field(None, description="When stage started")
    completed_at: datetime | None = Field(None, description="When stage completed")
    duration_seconds: float | None = Field(None, description="Duration in seconds")

    # Output reference
    artifact_path: str | None = Field(None, description="Path to stage output artifact")
    output_summary: str | None = Field(None, description="Brief summary of output")

    # Error info
    error: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retries attempted")

    # Approval info (for approval stages)
    approved_by: str | None = Field(None, description="Who approved (if applicable)")
    approval_notes: str | None = Field(None, description="Approval notes")


class Checkpoint(BaseModel):
    """Human approval checkpoint."""

    stage: Stage = Field(..., description="Stage requiring approval")
    title: str = Field(..., description="Checkpoint title")
    description: str = Field(..., description="What needs to be reviewed")
    artifact_paths: list[str] = Field(
        default_factory=list,
        description="Artifacts to review",
    )
    auto_approve: bool = Field(
        False,
        description="Can this be auto-approved based on criteria?",
    )
    auto_approve_criteria: str | None = Field(
        None,
        description="Criteria for auto-approval",
    )


class PipelineState(BaseModel):
    """Complete pipeline state.

    This is the central state object that tracks pipeline execution.
    It's persisted to disk and updated as the pipeline progresses.
    """

    # Identity
    run_id: str = Field(..., description="Unique run identifier")
    feature_description: str = Field(..., description="Original feature request")

    # Status
    status: RunStatus = Field(RunStatus.PENDING, description="Overall status")
    current_stage: Stage = Field(Stage.INIT, description="Current stage")

    # Timing
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When run was created",
    )
    started_at: datetime | None = Field(None, description="When run started")
    completed_at: datetime | None = Field(None, description="When run completed")

    # Stage results
    stages: dict[str, StageResult] = Field(
        default_factory=dict,
        description="Results by stage name",
    )

    # Project and artifact paths
    project_dir: str = Field(..., description="Root directory of the generated project (where code lives)")
    artifacts_dir: str = Field(..., description="Directory for this run's artifacts (.acf/runs/{run_id})")
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Map of artifact name to path",
    )

    # Configuration used
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Config at time of run",
    )

    # Human interactions
    checkpoints: list[Checkpoint] = Field(
        default_factory=list,
        description="Approval checkpoints",
    )
    user_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="User decisions at checkpoints",
    )

    # Error handling
    last_error: str | None = Field(None, description="Most recent error")
    can_retry: bool = Field(True, description="Can the run be retried?")

    # Iteration mode (when improving existing projects)
    iteration_context: dict[str, Any] | None = Field(
        None,
        description="Context for iteration mode: base_run_id, original_feature, improvement_request",
    )

    # Runtime metadata (doc requirements, safety patterns, invariant violations)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime metadata passed between stages",
    )

    def get_stage_result(self, stage: Stage) -> StageResult | None:
        """Get result for a specific stage."""
        return self.stages.get(stage.value)

    def set_stage_result(self, result: StageResult) -> None:
        """Set result for a stage."""
        self.stages[result.stage.value] = result

    def mark_stage_started(self, stage: Stage) -> None:
        """Mark a stage as started."""
        self.current_stage = stage
        self.stages[stage.value] = StageResult(
            stage=stage,
            status=StageStatus.RUNNING,
            started_at=datetime.now(),
        )

    def mark_stage_completed(
        self,
        stage: Stage,
        artifact_path: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Mark a stage as completed."""
        result = self.stages.get(stage.value)
        if result:
            result.status = StageStatus.COMPLETED
            result.completed_at = datetime.now()
            if result.started_at:
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()
            result.artifact_path = artifact_path
            result.output_summary = summary

    def mark_stage_failed(self, stage: Stage, error: str) -> None:
        """Mark a stage as failed."""
        result = self.stages.get(stage.value)
        if result:
            result.status = StageStatus.FAILED
            result.completed_at = datetime.now()
            result.error = error
        self.last_error = error

    def mark_awaiting_approval(self, stage: Stage) -> None:
        """Mark a stage as awaiting approval."""
        result = self.stages.get(stage.value)
        if result:
            result.status = StageStatus.AWAITING_APPROVAL
        self.status = RunStatus.PAUSED

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "2026-01-04-143052",
                "feature_description": "Add login rate-limit",
                "status": "running",
                "current_stage": "design",
                "artifacts_dir": "artifacts/2026-01-04-143052",
            }
        }
