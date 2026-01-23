"""GitHub Issues tracker client."""

import os
import re

import httpx

from .base import IssueData, IssuePriority, IssueStatus, IssueTracker


class GitHubIssuesClient(IssueTracker):
    """Client for fetching issues from GitHub.

    Authentication via environment variables:
        GITHUB_TOKEN: Personal access token (or GH_TOKEN)
            Create at https://github.com/settings/tokens

    Usage:
        client = GitHubIssuesClient("owner/repo")
        issue = client.fetch_issue("42")
    """

    def __init__(
        self,
        repo: str | None = None,
        token: str | None = None,
    ):
        """Initialize GitHub Issues client.

        Args:
            repo: Repository in "owner/repo" format
            token: GitHub personal access token (or GITHUB_TOKEN/GH_TOKEN env var)
        """
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
        self.base_url = "https://api.github.com"

        self._headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"

    @property
    def name(self) -> str:
        return "github"

    def fetch_issue(self, issue_id: str, repo: str | None = None) -> IssueData:
        """Fetch a GitHub issue by number.

        Args:
            issue_id: Issue number (e.g., '42')
            repo: Optional repo override in "owner/repo" format

        Returns:
            Normalized IssueData
        """
        target_repo = repo or self.repo
        if not target_repo:
            raise ValueError(
                "Repository not specified. Pass repo to constructor or fetch_issue()."
            )

        # Validate issue number
        try:
            issue_num = int(issue_id)
        except ValueError:
            raise ValueError(
                f"Invalid GitHub issue number: {issue_id}. "
                "Expected a number like '42'."
            )

        url = f"{self.base_url}/repos/{target_repo}/issues/{issue_num}"

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=self._headers)

                if response.status_code == 404:
                    raise ValueError(f"Issue not found: {target_repo}#{issue_num}")
                elif response.status_code == 401:
                    raise ConnectionError(
                        "GitHub authentication failed. "
                        "Set GITHUB_TOKEN or GH_TOKEN environment variable."
                    )
                elif response.status_code == 403:
                    # Could be rate limiting or private repo without auth
                    if "rate limit" in response.text.lower():
                        raise ConnectionError(
                            "GitHub API rate limit exceeded. "
                            "Set GITHUB_TOKEN for higher limits."
                        )
                    raise ConnectionError(
                        "Access denied. For private repos, set GITHUB_TOKEN."
                    )
                elif response.status_code != 200:
                    raise ConnectionError(
                        f"GitHub API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                return self._parse_issue(data, target_repo)

        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to GitHub: {e}")

    def fetch_issue_from_url(self, url: str) -> IssueData:
        """Fetch a GitHub issue from its URL.

        Args:
            url: Full GitHub issue URL
                (e.g., 'https://github.com/owner/repo/issues/42')

        Returns:
            Normalized IssueData
        """
        # Parse URL
        match = re.match(
            r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)",
            url
        )
        if not match:
            raise ValueError(
                f"Invalid GitHub issue URL: {url}. "
                "Expected: https://github.com/owner/repo/issues/42"
            )

        repo = match.group(1)
        issue_num = match.group(2)
        return self.fetch_issue(issue_num, repo=repo)

    def validate_connection(self) -> bool:
        """Check if GitHub connection works."""
        url = f"{self.base_url}/user"
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(url, headers=self._headers)
                # 200 = authenticated, 401 = unauthenticated (still "connected")
                return response.status_code in (200, 401)
        except httpx.RequestError:
            return False

    def _parse_issue(self, data: dict, repo: str) -> IssueData:
        """Parse GitHub API response into IssueData."""
        # GitHub Issues API returns PRs too - check if it's actually an issue
        if data.get("pull_request"):
            raise ValueError(
                f"#{data['number']} is a pull request, not an issue."
            )

        # Extract labels
        labels = [label.get("name", "") for label in data.get("labels", [])]

        # Map priority from labels
        priority = self._extract_priority(labels)

        # Map status
        status = IssueStatus.OPEN if data.get("state") == "open" else IssueStatus.CLOSED

        # Extract assignee(s)
        assignees = data.get("assignees", [])
        assignee = assignees[0].get("login") if assignees else None

        # Extract body
        body = data.get("body") or ""

        # Extract acceptance criteria from body
        acceptance_criteria = self._extract_acceptance_criteria(body)

        return IssueData(
            id=str(data["number"]),
            title=data.get("title", ""),
            description=body,
            tracker="github",
            url=data.get("html_url", f"https://github.com/{repo}/issues/{data['number']}"),
            priority=priority,
            status=status,
            labels=labels,
            assignee=assignee,
            reporter=data.get("user", {}).get("login"),
            acceptance_criteria=acceptance_criteria,
            raw=data,
        )

    def _extract_priority(self, labels: list[str]) -> IssuePriority:
        """Extract priority from GitHub labels.

        Common label patterns:
        - priority/critical, P0
        - priority/high, P1
        - priority/medium, P2
        - priority/low, P3
        """
        labels_lower = [l.lower() for l in labels]

        # Check for priority labels
        for label in labels_lower:
            if any(x in label for x in ("critical", "p0", "priority-0", "urgent")):
                return IssuePriority.CRITICAL
            if any(x in label for x in ("high", "p1", "priority-1")):
                return IssuePriority.HIGH
            if any(x in label for x in ("medium", "p2", "priority-2")):
                return IssuePriority.MEDIUM
            if any(x in label for x in ("low", "p3", "priority-3", "minor")):
                return IssuePriority.LOW

        return IssuePriority.UNKNOWN

    def _extract_acceptance_criteria(self, body: str) -> list[str]:
        """Extract acceptance criteria from issue body.

        Looks for common patterns in GitHub issue templates:
        - "## Acceptance Criteria" section
        - "### AC" section
        - Task lists (- [ ])
        """
        criteria = []

        if not body:
            return criteria

        # Look for AC section (markdown headers)
        patterns = [
            r"##\s*acceptance criteria\s*\n([\s\S]*?)(?=\n##|\Z)",
            r"##\s*ac\s*\n([\s\S]*?)(?=\n##|\Z)",
            r"##\s*definition of done\s*\n([\s\S]*?)(?=\n##|\Z)",
            r"\*\*acceptance criteria\*\*\s*\n([\s\S]*?)(?=\n\*\*|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                section = match.group(1).strip()
                criteria.extend(self._parse_criteria_section(section))
                if criteria:
                    return criteria

        # Fallback: look for task lists anywhere
        task_items = re.findall(r"- \[[ x]\] (.+)", body)
        if task_items:
            return task_items

        return criteria

    def _parse_criteria_section(self, section: str) -> list[str]:
        """Parse acceptance criteria from a text section."""
        criteria = []
        lines = section.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and sub-headers
            if not line or line.startswith("#"):
                continue

            # Handle task list items
            task_match = re.match(r"- \[[ x]\] (.+)", line)
            if task_match:
                criteria.append(task_match.group(1))
                continue

            # Handle regular list items
            list_match = re.match(r"[-*•]\s+(.+)", line)
            if list_match:
                criteria.append(list_match.group(1))
                continue

            # Handle numbered items
            num_match = re.match(r"\d+[.)]\s+(.+)", line)
            if num_match:
                criteria.append(num_match.group(1))
                continue

        return criteria
