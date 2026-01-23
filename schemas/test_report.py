"""Test report schema.

Output of Test & Quality Agent: test results and quality analysis.
"""

from enum import Enum
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCase(BaseModel):
    """Individual test case result."""

    name: str = Field(..., description="Test name")
    file_path: str = Field(..., description="Test file path")
    status: TestStatus = Field(..., description="Test result")
    duration_ms: float | None = Field(None, description="Execution time in ms")
    error_message: str | None = Field(None, description="Error message if failed")
    stack_trace: str | None = Field(None, description="Stack trace if failed")
    stdout: str | None = Field(None, description="Captured stdout")


class TestResult(BaseModel):
    """Aggregated test results."""

    total: int = Field(0, description="Total tests")
    passed: int = Field(0, description="Passed tests")
    failed: int = Field(0, description="Failed tests")
    skipped: int = Field(0, description="Skipped tests")
    errors: int = Field(0, description="Tests with errors")
    duration_seconds: float = Field(0, description="Total duration")

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0


class CoverageInfo(BaseModel):
    """Code coverage information."""

    total_lines: int = Field(0, description="Total lines")
    covered_lines: int = Field(0, description="Covered lines")
    coverage_percent: float = Field(0, description="Coverage percentage")
    uncovered_files: list[str] = Field(
        default_factory=list,
        description="Files with no coverage",
    )
    low_coverage_files: list[dict[str, float]] = Field(
        default_factory=list,
        description="Files with coverage below threshold",
    )


class LintIssue(BaseModel):
    """Linting issue found."""

    file_path: str = Field(..., description="File path")
    line: int = Field(..., description="Line number")
    column: int | None = Field(None, description="Column number")
    rule: str = Field(..., description="Rule ID")
    message: str = Field(..., description="Issue message")
    severity: str = Field("warning", description="error, warning, info")
    fixable: bool = Field(False, description="Can be auto-fixed")


class TypeIssue(BaseModel):
    """Type checking issue."""

    file_path: str = Field(..., description="File path")
    line: int = Field(..., description="Line number")
    message: str = Field(..., description="Type error message")
    error_code: str | None = Field(None, description="mypy/tsc error code")


class TestReport(BaseModel):
    """Complete test and quality report.

    This is the output of the Test & Quality Agent.
    """

    # References
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")

    # Test results
    test_results: TestResult = Field(
        default_factory=TestResult,
        description="Aggregated test results",
    )
    test_cases: list[TestCase] = Field(
        default_factory=list,
        description="Individual test results",
    )

    # Coverage
    coverage: CoverageInfo | None = Field(None, description="Coverage info")

    # Linting
    lint_issues: list[LintIssue] = Field(
        default_factory=list,
        description="Linting issues found",
    )
    lint_tool: str | None = Field(None, description="Linter used (ruff, eslint, etc.)")

    # Type checking
    type_issues: list[TypeIssue] = Field(
        default_factory=list,
        description="Type errors found",
    )
    type_checker: str | None = Field(None, description="Type checker used")

    # New tests generated
    tests_generated: list[str] = Field(
        default_factory=list,
        description="New test files created",
    )

    # Failure analysis
    failure_analysis: str | None = Field(
        None,
        description="LLM analysis of test failures",
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description="Suggested fixes for failures",
    )

    # Anti-pattern detection
    anti_patterns_found: list[str] = Field(
        default_factory=list,
        description="Anti-patterns detected in code",
    )
    idiom_suggestions: list[str] = Field(
        default_factory=list,
        description="Modern idiom suggestions",
    )

    # Overall assessment
    quality_score: float | None = Field(
        None,
        description="Overall quality score 0-100",
    )
    ready_for_deploy: bool = Field(
        False,
        description="Is the code ready for deployment?",
    )
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="Issues that block deployment",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "test_results": {
                    "total": 25,
                    "passed": 24,
                    "failed": 1,
                    "skipped": 0,
                    "errors": 0,
                    "duration_seconds": 3.5,
                },
                "ready_for_deploy": False,
                "blocking_issues": ["1 test failing: test_rate_limit_exceeded"],
            }
        }
