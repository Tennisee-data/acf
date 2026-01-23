"""Issue tracker integrations for fetching issues as feature input.

Supports:
- JIRA (Atlassian)
- GitHub Issues
- Linear (planned)

Usage:
    coding-factory run --jira PROJ-123 --repo ./project
    coding-factory run --issue https://github.com/org/repo/issues/42
"""

from .base import IssueData, IssueTracker
from .github_issues import GitHubIssuesClient
from .jira import JiraClient
from .resolver import resolve_issue, detect_tracker_from_url

__all__ = [
    "IssueData",
    "IssueTracker",
    "JiraClient",
    "GitHubIssuesClient",
    "resolve_issue",
    "detect_tracker_from_url",
]
