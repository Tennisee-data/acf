"""Schema for PR Package generation.

Defines structured output for GitHub Pull Request packaging including:
- Title and description
- Acceptance criteria checklist
- Links to test/coverage/performance reports
- Suggested labels, reviewers, changelog entry
"""

from enum import Enum

from pydantic import BaseModel, Field


class PRLabel(str, Enum):
    """Standard PR labels."""

    FEATURE = "feature"
    BUG_FIX = "bug-fix"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    TESTS = "tests"
    BREAKING_CHANGE = "breaking-change"
    DEPENDENCIES = "dependencies"
    PERFORMANCE = "performance"
    SECURITY = "security"


class ChangeType(str, Enum):
    """Type of change for changelog."""

    ADDED = "Added"
    CHANGED = "Changed"
    DEPRECATED = "Deprecated"
    REMOVED = "Removed"
    FIXED = "Fixed"
    SECURITY = "Security"


class AcceptanceCriteriaItem(BaseModel):
    """An acceptance criterion with completion status."""

    criterion: str = Field(description="The acceptance criterion text")
    met: bool = Field(description="Whether this criterion is met")
    evidence: str | None = Field(
        default=None, description="Evidence or notes for this criterion"
    )


class ReportLink(BaseModel):
    """Link to a generated report artifact."""

    name: str = Field(description="Report name (e.g., 'Test Report')")
    path: str = Field(description="Relative path to the report file")
    summary: str | None = Field(
        default=None, description="Brief summary of report results"
    )


class ChangelogEntry(BaseModel):
    """A changelog entry for this PR."""

    change_type: ChangeType = Field(description="Type of change")
    description: str = Field(description="Description of the change")
    breaking: bool = Field(default=False, description="Whether this is breaking")


class SuggestedReviewer(BaseModel):
    """A suggested reviewer for the PR."""

    username: str = Field(description="GitHub username")
    reason: str | None = Field(default=None, description="Why they should review")


class PRPackage(BaseModel):
    """Complete PR package ready for GitHub submission."""

    # Core PR content
    title: str = Field(description="PR title")
    description: str = Field(description="PR description/body markdown")

    # Feature tracking
    feature_summary: str = Field(description="Brief summary of the feature")
    acceptance_criteria: list[AcceptanceCriteriaItem] = Field(
        default_factory=list, description="Acceptance criteria checklist"
    )

    # Report links
    report_links: list[ReportLink] = Field(
        default_factory=list, description="Links to generated reports"
    )

    # Metadata suggestions
    suggested_labels: list[PRLabel] = Field(
        default_factory=list, description="Suggested PR labels"
    )
    suggested_reviewers: list[SuggestedReviewer] = Field(
        default_factory=list, description="Suggested reviewers"
    )

    # Changelog
    changelog_entries: list[ChangelogEntry] = Field(
        default_factory=list, description="Changelog entries for this PR"
    )

    # Branch info
    source_branch: str = Field(description="Source branch name")
    target_branch: str = Field(default="main", description="Target branch")

    # Raw markdown body (fully rendered)
    rendered_body: str = Field(description="Fully rendered markdown PR body")
