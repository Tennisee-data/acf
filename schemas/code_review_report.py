"""Code review report schema.

Defines output structure for CodeReviewAgent including review notes,
ship status, and auto-fix suggestions.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ShipStatus(str, Enum):
    """Review decision status."""

    SHIP = "ship"  # Ready to ship, no issues
    SHIP_WITH_NITS = "ship_with_nits"  # Minor issues, can ship
    DONT_SHIP = "dont_ship"  # Blocking issues, needs fixes


class IssueSeverity(str, Enum):
    """Severity of review issue."""

    CRITICAL = "critical"  # Must fix before shipping
    MAJOR = "major"  # Should fix, affects quality
    MINOR = "minor"  # Nice to fix, not blocking
    NIT = "nit"  # Style/preference, optional


class IssueCategory(str, Enum):
    """Category of review issue."""

    NAMING = "naming"  # Variable, function, class names
    CLARITY = "clarity"  # Code readability
    MODULARITY = "modularity"  # Code organization
    YAGNI = "yagni"  # Overengineering, unused code
    CONSISTENCY = "consistency"  # Style consistency
    COMPLEXITY = "complexity"  # Unnecessary complexity
    SECURITY = "security"  # Security concerns
    PERFORMANCE = "performance"  # Performance issues
    ERROR_HANDLING = "error_handling"  # Exception handling
    DOCUMENTATION = "documentation"  # Missing or wrong docs
    TESTING = "testing"  # Test coverage concerns
    BEST_PRACTICE = "best_practice"  # General best practices


class ReviewIssue(BaseModel):
    """A single review issue found in the code."""

    file_path: str = Field(..., description="File containing the issue")
    line_start: int = Field(..., description="Starting line number")
    line_end: int | None = Field(None, description="Ending line number if range")
    severity: IssueSeverity = Field(..., description="Issue severity")
    category: IssueCategory = Field(..., description="Issue category")
    title: str = Field(..., description="Brief issue title")
    description: str = Field(..., description="Detailed explanation")
    suggestion: str | None = Field(None, description="Suggested fix")
    code_snippet: str | None = Field(None, description="Relevant code snippet")
    auto_fixable: bool = Field(False, description="Can be auto-fixed")


class AutoFix(BaseModel):
    """An auto-applicable fix."""

    file_path: str = Field(..., description="File to modify")
    line_start: int = Field(..., description="Starting line")
    line_end: int = Field(..., description="Ending line")
    original_code: str = Field(..., description="Code to replace")
    fixed_code: str = Field(..., description="Replacement code")
    description: str = Field(..., description="What this fix does")
    fix_type: str = Field(..., description="Type: rename, docstring, format, etc.")
    applied: bool = Field(False, description="Whether fix was applied")


class FileReview(BaseModel):
    """Review summary for a single file."""

    file_path: str = Field(..., description="File path")
    lines_reviewed: int = Field(0, description="Number of lines reviewed")
    issues_count: int = Field(0, description="Number of issues found")
    critical_count: int = Field(0, description="Critical issues")
    major_count: int = Field(0, description="Major issues")
    minor_count: int = Field(0, description="Minor issues")
    nit_count: int = Field(0, description="Nit issues")
    summary: str = Field("", description="Brief file summary")


class StyleGuide(BaseModel):
    """Detected or configured style preferences."""

    naming_convention: str = Field(
        "snake_case",
        description="Function/variable naming: snake_case, camelCase",
    )
    class_naming: str = Field(
        "PascalCase",
        description="Class naming convention",
    )
    max_line_length: int = Field(100, description="Maximum line length")
    docstring_style: str = Field(
        "google",
        description="Docstring style: google, numpy, sphinx",
    )
    import_style: str = Field(
        "grouped",
        description="Import organization style",
    )
    type_hints: bool = Field(True, description="Type hints expected")


class CodeReviewReport(BaseModel):
    """Report from the CodeReviewAgent."""

    # Overall decision
    ship_status: ShipStatus = Field(
        ShipStatus.SHIP,
        description="Overall ship decision",
    )
    ship_status_reason: str = Field(
        "",
        description="Explanation for ship status",
    )

    # Issues found
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="All review issues",
    )

    # Per-file summaries
    file_reviews: list[FileReview] = Field(
        default_factory=list,
        description="Per-file review summaries",
    )

    # Auto-fixes
    auto_fixes: list[AutoFix] = Field(
        default_factory=list,
        description="Auto-applicable fixes",
    )
    auto_fixes_applied: int = Field(0, description="Number of auto-fixes applied")

    # Style analysis
    style_guide: StyleGuide = Field(
        default_factory=StyleGuide,
        description="Detected style guide",
    )
    style_violations: int = Field(0, description="Style violations count")

    # Counts by severity
    critical_count: int = Field(0, description="Critical issues")
    major_count: int = Field(0, description="Major issues")
    minor_count: int = Field(0, description="Minor issues")
    nit_count: int = Field(0, description="Nit issues")
    total_issues: int = Field(0, description="Total issues")

    # Counts by category
    issues_by_category: dict[str, int] = Field(
        default_factory=dict,
        description="Issue count per category",
    )

    # Review metadata
    files_reviewed: int = Field(0, description="Number of files reviewed")
    lines_reviewed: int = Field(0, description="Total lines reviewed")
    review_time_seconds: float = Field(0.0, description="Time spent reviewing")

    # Summary
    summary: str = Field("", description="Human-readable summary")
    top_concerns: list[str] = Field(
        default_factory=list,
        description="Top 3 concerns to address",
    )

    def update_counts(self) -> None:
        """Update summary counts from issues list."""
        self.critical_count = sum(
            1 for i in self.issues if i.severity == IssueSeverity.CRITICAL
        )
        self.major_count = sum(
            1 for i in self.issues if i.severity == IssueSeverity.MAJOR
        )
        self.minor_count = sum(
            1 for i in self.issues if i.severity == IssueSeverity.MINOR
        )
        self.nit_count = sum(
            1 for i in self.issues if i.severity == IssueSeverity.NIT
        )
        self.total_issues = len(self.issues)

        # Count by category
        self.issues_by_category = {}
        for issue in self.issues:
            cat = issue.category.value
            self.issues_by_category[cat] = self.issues_by_category.get(cat, 0) + 1

        # Update file reviews
        self.files_reviewed = len(self.file_reviews)
        self.lines_reviewed = sum(f.lines_reviewed for f in self.file_reviews)

    def determine_ship_status(self) -> None:
        """Determine ship status based on issues."""
        if self.critical_count > 0:
            self.ship_status = ShipStatus.DONT_SHIP
            self.ship_status_reason = (
                f"{self.critical_count} critical issue(s) must be fixed"
            )
        elif self.major_count > 2:
            self.ship_status = ShipStatus.DONT_SHIP
            self.ship_status_reason = (
                f"Too many major issues ({self.major_count}) to ship"
            )
        elif self.major_count > 0 or self.minor_count > 3:
            self.ship_status = ShipStatus.SHIP_WITH_NITS
            self.ship_status_reason = (
                f"{self.major_count} major and {self.minor_count} minor issues"
            )
        else:
            self.ship_status = ShipStatus.SHIP
            self.ship_status_reason = "Code looks good!"

    class Config:
        json_schema_extra = {
            "example": {
                "ship_status": "ship_with_nits",
                "total_issues": 5,
                "critical_count": 0,
                "major_count": 1,
                "files_reviewed": 3,
            }
        }
