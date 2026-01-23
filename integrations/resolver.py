"""Issue resolver - detects tracker type and fetches issues."""

import re
from urllib.parse import urlparse

from .base import IssueData, IssueTracker
from .github_issues import GitHubIssuesClient
from .jira import JiraClient


def detect_tracker_from_url(url: str) -> str | None:
    """Detect issue tracker type from URL.

    Args:
        url: Issue URL

    Returns:
        Tracker name ('github', 'jira', 'linear') or None if unknown
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # GitHub
    if host == "github.com" or host.endswith(".github.com"):
        if "/issues/" in parsed.path:
            return "github"

    # JIRA (Atlassian Cloud)
    if ".atlassian.net" in host:
        if "/browse/" in parsed.path:
            return "jira"

    # Self-hosted JIRA (check path pattern)
    if "/browse/" in parsed.path:
        # Could be JIRA - path like /browse/PROJ-123
        path_match = re.search(r"/browse/([A-Z][A-Z0-9]+-\d+)", parsed.path)
        if path_match:
            return "jira"

    # Linear
    if host == "linear.app" or host.endswith(".linear.app"):
        return "linear"

    return None


def resolve_issue(
    issue_ref: str,
    jira_url: str | None = None,
) -> IssueData:
    """Resolve an issue reference to IssueData.

    Accepts either:
    - Full URL: https://github.com/owner/repo/issues/42
    - JIRA key: PROJ-123 (requires jira_url or JIRA_URL env var)

    Args:
        issue_ref: Issue URL or JIRA key
        jira_url: Optional JIRA base URL for key-based lookups

    Returns:
        Normalized IssueData

    Raises:
        ValueError: If issue format not recognized or not found
        ConnectionError: If tracker unavailable
    """
    # Check if it's a URL
    if issue_ref.startswith(("http://", "https://")):
        return _resolve_from_url(issue_ref)

    # Check if it's a JIRA key (PROJ-123 format)
    if re.match(r"^[A-Z][A-Z0-9]+-\d+$", issue_ref.upper()):
        return _resolve_jira_key(issue_ref, jira_url)

    raise ValueError(
        f"Unrecognized issue format: {issue_ref}\n"
        "Expected: URL (https://...) or JIRA key (PROJ-123)"
    )


def _resolve_from_url(url: str) -> IssueData:
    """Resolve issue from URL."""
    tracker_type = detect_tracker_from_url(url)

    if tracker_type == "github":
        client = GitHubIssuesClient()
        return client.fetch_issue_from_url(url)

    elif tracker_type == "jira":
        # Extract JIRA key from URL
        match = re.search(r"/browse/([A-Z][A-Z0-9]+-\d+)", url)
        if not match:
            raise ValueError(f"Could not extract issue key from JIRA URL: {url}")

        # Extract base URL
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        client = JiraClient(base_url=base_url)
        return client.fetch_issue(match.group(1))

    elif tracker_type == "linear":
        raise NotImplementedError(
            "Linear integration not yet implemented. "
            "Coming soon! (See ROADMAP.md)"
        )

    else:
        raise ValueError(
            f"Unknown issue tracker for URL: {url}\n"
            "Supported: GitHub Issues, JIRA"
        )


def _resolve_jira_key(key: str, jira_url: str | None = None) -> IssueData:
    """Resolve JIRA issue from key."""
    try:
        client = JiraClient(base_url=jira_url)
        return client.fetch_issue(key)
    except ValueError as e:
        # Re-raise with more context
        if "credentials required" in str(e).lower():
            raise ValueError(
                f"Cannot fetch JIRA issue {key}: Missing credentials.\n"
                "Set environment variables: JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN"
            )
        raise


def get_tracker_client(tracker_type: str, **kwargs) -> IssueTracker:
    """Get a tracker client by type.

    Args:
        tracker_type: 'github', 'jira', or 'linear'
        **kwargs: Tracker-specific configuration

    Returns:
        Configured IssueTracker instance
    """
    if tracker_type == "github":
        return GitHubIssuesClient(**kwargs)
    elif tracker_type == "jira":
        return JiraClient(**kwargs)
    elif tracker_type == "linear":
        raise NotImplementedError("Linear integration coming soon")
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
