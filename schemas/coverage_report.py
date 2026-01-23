"""Coverage report schema for tracking test coverage.

Output of CoverageAgent: coverage metrics, uncovered lines,
and recommendations for improving test coverage.
"""

from enum import Enum

from pydantic import BaseModel, Field


class CoverageStatus(str, Enum):
    """Overall coverage status."""

    PASSING = "passing"  # Meets threshold
    WARNING = "warning"  # Close to threshold
    FAILING = "failing"  # Below threshold
    UNKNOWN = "unknown"  # Could not determine


class FileCoverage(BaseModel):
    """Coverage data for a single file."""

    file_path: str = Field(..., description="Path to the source file")
    total_lines: int = Field(0, description="Total executable lines")
    covered_lines: int = Field(0, description="Lines covered by tests")
    missing_lines: list[int] = Field(
        default_factory=list,
        description="Line numbers not covered",
    )
    coverage_percent: float = Field(0.0, description="Coverage percentage")

    # Branch coverage (optional)
    total_branches: int | None = Field(None, description="Total branches")
    covered_branches: int | None = Field(None, description="Branches covered")
    branch_percent: float | None = Field(None, description="Branch coverage %")


class UncoveredBlock(BaseModel):
    """A block of uncovered code needing tests."""

    file_path: str = Field(..., description="Source file path")
    start_line: int = Field(..., description="Start line of uncovered block")
    end_line: int = Field(..., description="End line of uncovered block")
    code_snippet: str = Field("", description="The uncovered code")
    function_name: str | None = Field(None, description="Containing function")
    reason: str = Field("", description="Why this might be uncovered")
    test_suggestion: str = Field("", description="Suggested test approach")


class CoverageThreshold(BaseModel):
    """Coverage threshold configuration."""

    overall_min: float = Field(80.0, description="Minimum overall coverage %")
    diff_min: float = Field(80.0, description="Minimum coverage for changed files %")
    branch_min: float | None = Field(None, description="Minimum branch coverage %")
    fail_under: bool = Field(True, description="Fail if below threshold")


class DiffCoverage(BaseModel):
    """Coverage for changed/new files only."""

    changed_files: list[str] = Field(
        default_factory=list,
        description="Files that were changed",
    )
    new_lines: int = Field(0, description="New lines added")
    covered_new_lines: int = Field(0, description="New lines covered by tests")
    diff_coverage_percent: float = Field(0.0, description="Coverage of new code")
    uncovered_changes: list[UncoveredBlock] = Field(
        default_factory=list,
        description="Uncovered blocks in changed code",
    )


class TestGenerationRequest(BaseModel):
    """Request for TestGeneratorAgent to create more tests."""

    target_file: str = Field(..., description="File needing more tests")
    uncovered_blocks: list[UncoveredBlock] = Field(
        default_factory=list,
        description="Code blocks to cover",
    )
    priority: int = Field(1, description="Priority 1-3 (1=highest)")
    estimated_tests: int = Field(1, description="Estimated tests needed")


class CoverageReport(BaseModel):
    """Complete coverage report.

    Output of CoverageAgent - tracks test coverage and
    identifies areas needing additional tests.
    """

    # Overall metrics
    total_lines: int = Field(0, description="Total executable lines")
    covered_lines: int = Field(0, description="Lines covered by tests")
    overall_coverage: float = Field(0.0, description="Overall coverage %")

    # Branch coverage
    total_branches: int | None = Field(None, description="Total branches")
    covered_branches: int | None = Field(None, description="Branches covered")
    branch_coverage: float | None = Field(None, description="Branch coverage %")

    # Status
    status: CoverageStatus = Field(
        CoverageStatus.UNKNOWN,
        description="Whether coverage meets thresholds",
    )
    threshold: CoverageThreshold = Field(
        default_factory=CoverageThreshold,
        description="Applied thresholds",
    )

    # Per-file breakdown
    files: list[FileCoverage] = Field(
        default_factory=list,
        description="Coverage per file",
    )

    # Diff coverage (for changed files)
    diff_coverage: DiffCoverage | None = Field(
        None,
        description="Coverage for changed files only",
    )

    # Uncovered code analysis
    uncovered_blocks: list[UncoveredBlock] = Field(
        default_factory=list,
        description="Significant uncovered code blocks",
    )

    # Test generation requests
    test_requests: list[TestGenerationRequest] = Field(
        default_factory=list,
        description="Requests for TestGeneratorAgent",
    )

    # Iteration tracking
    iteration: int = Field(1, description="Coverage check iteration")
    previous_coverage: float | None = Field(
        None,
        description="Coverage from previous iteration",
    )
    improvement: float = Field(0.0, description="Coverage improvement")

    # Notes for other agents
    summary: str = Field("", description="Human-readable summary")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for improving coverage",
    )
    verify_notes: list[str] = Field(
        default_factory=list,
        description="Notes for VerifyAgent",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_lines": 1000,
                "covered_lines": 850,
                "overall_coverage": 85.0,
                "status": "passing",
                "threshold": {"overall_min": 80.0, "diff_min": 80.0},
                "files": [
                    {
                        "file_path": "src/api/routes.py",
                        "total_lines": 200,
                        "covered_lines": 180,
                        "coverage_percent": 90.0,
                    }
                ],
                "summary": "Coverage at 85%, above 80% threshold",
            }
        }
