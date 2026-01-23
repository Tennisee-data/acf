"""Base classes for issue tracker integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class IssuePriority(str, Enum):
    """Issue priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class IssueStatus(str, Enum):
    """Issue status."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass
class IssueData:
    """Normalized issue data from any tracker.

    This is the common format used by the pipeline regardless
    of whether the issue came from JIRA, GitHub, Linear, etc.
    """

    # Core fields
    id: str
    title: str
    description: str

    # Source info
    tracker: str  # "jira", "github", "linear"
    url: str

    # Optional metadata
    priority: IssuePriority = IssuePriority.UNKNOWN
    status: IssueStatus = IssueStatus.UNKNOWN
    labels: list[str] = field(default_factory=list)
    assignee: str | None = None
    reporter: str | None = None

    # Acceptance criteria (if available)
    acceptance_criteria: list[str] = field(default_factory=list)

    # Related issues
    parent_id: str | None = None
    subtasks: list[str] = field(default_factory=list)
    linked_issues: list[str] = field(default_factory=list)

    # Raw data for debugging
    raw: dict = field(default_factory=dict)

    def to_feature_description(self) -> str:
        """Convert issue to feature description for pipeline input.

        Returns a formatted string suitable for the 'feature' argument
        of the run command.
        """
        parts = [self.title]

        if self.description:
            parts.append("")
            parts.append(self.description)

        if self.acceptance_criteria:
            parts.append("")
            parts.append("Acceptance Criteria:")
            for i, criterion in enumerate(self.acceptance_criteria, 1):
                parts.append(f"  {i}. {criterion}")

        if self.labels:
            parts.append("")
            parts.append(f"Labels: {', '.join(self.labels)}")

        return "\n".join(parts)


class IssueTracker(ABC):
    """Base class for issue tracker clients.

    Subclasses implement fetching issues from specific trackers
    (JIRA, GitHub, Linear, etc.) and normalize them to IssueData.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tracker name (e.g., 'jira', 'github')."""
        ...

    @abstractmethod
    def fetch_issue(self, issue_id: str) -> IssueData:
        """Fetch a single issue by ID.

        Args:
            issue_id: Issue identifier (e.g., 'PROJ-123' for JIRA,
                     '42' for GitHub issues)

        Returns:
            Normalized IssueData

        Raises:
            ValueError: If issue not found
            ConnectionError: If tracker unavailable
        """
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """Check if connection to tracker is valid.

        Returns:
            True if connection works, False otherwise
        """
        ...
