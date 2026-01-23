"""Verification report schema.

Output of Verification & Summary Agent: black-box testing results.
"""

from enum import Enum
from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    """Verification check status."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


class HealthCheck(BaseModel):
    """Health check result."""

    endpoint: str = Field(..., description="Endpoint checked")
    status: CheckStatus = Field(..., description="Check result")
    response_time_ms: float | None = Field(None, description="Response time")
    expected_status: int = Field(200, description="Expected HTTP status")
    actual_status: int | None = Field(None, description="Actual HTTP status")
    error: str | None = Field(None, description="Error if failed")


class CriterionCheck(BaseModel):
    """Check of a single acceptance criterion."""

    criterion_id: str = Field(..., description="Reference to AcceptanceCriteria.id")
    description: str = Field(..., description="Criterion description")
    status: CheckStatus = Field(..., description="Check result")
    evidence: str | None = Field(None, description="Evidence of pass/fail")
    notes: str | None = Field(None, description="Additional notes")


class APICheck(BaseModel):
    """API endpoint verification."""

    method: str = Field(..., description="HTTP method")
    endpoint: str = Field(..., description="Endpoint path")
    description: str = Field(..., description="What was tested")
    status: CheckStatus = Field(..., description="Check result")
    request_body: dict | None = Field(None, description="Request sent")
    expected_response: dict | None = Field(None, description="Expected response")
    actual_response: dict | None = Field(None, description="Actual response")
    error: str | None = Field(None, description="Error if failed")


class UICheck(BaseModel):
    """UI flow verification."""

    flow_name: str = Field(..., description="Name of the UI flow")
    steps: list[str] = Field(default_factory=list, description="Steps performed")
    status: CheckStatus = Field(..., description="Check result")
    screenshot_path: str | None = Field(None, description="Screenshot if captured")
    error: str | None = Field(None, description="Error if failed")


class VerificationReport(BaseModel):
    """Complete verification report.

    This is the output of the Verification & Summary Agent - black-box
    testing results against the running container.
    """

    # References
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")
    container_id: str | None = Field(None, description="Docker container ID tested")
    image_tag: str | None = Field(None, description="Docker image tag")

    # Health checks
    health_checks: list[HealthCheck] = Field(
        default_factory=list,
        description="Health check results",
    )

    # Acceptance criteria verification
    criteria_checks: list[CriterionCheck] = Field(
        default_factory=list,
        description="Acceptance criteria verification",
    )

    # API checks
    api_checks: list[APICheck] = Field(
        default_factory=list,
        description="API endpoint verifications",
    )

    # UI checks (if applicable)
    ui_checks: list[UICheck] = Field(
        default_factory=list,
        description="UI flow verifications",
    )

    # Summary
    all_criteria_met: bool = Field(False, description="All acceptance criteria passed")
    criteria_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count by status: {pass: N, fail: N, skip: N}",
    )

    # Technical summary
    technical_summary: str = Field(
        ...,
        description="Summary of technical changes",
    )

    # Behavioral summary
    behavioral_summary: str = Field(
        ...,
        description="Summary of observed behavior",
    )

    # Risks
    residual_risks: list[str] = Field(
        default_factory=list,
        description="Remaining risks or concerns",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions for human review",
    )

    # Recommendation
    recommendation: str = Field(
        ...,
        description="approve, reject, or needs_review",
    )
    recommendation_rationale: str = Field(
        ...,
        description="Why this recommendation",
    )

    # For PR/release
    pr_description: str | None = Field(
        None,
        description="Generated PR description",
    )
    release_notes: str | None = Field(
        None,
        description="Generated release notes entry",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "all_criteria_met": True,
                "criteria_summary": {"pass": 5, "fail": 0, "skip": 0},
                "recommendation": "approve",
                "recommendation_rationale": "All acceptance criteria verified, no blocking issues found.",
            }
        }
