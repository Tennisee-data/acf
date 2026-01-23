"""JIRA issue tracker client."""

import os
import re
from base64 import b64encode

import httpx

from .base import IssueData, IssuePriority, IssueStatus, IssueTracker


class JiraClient(IssueTracker):
    """Client for fetching issues from JIRA.

    Authentication via environment variables:
        JIRA_URL: Base URL (e.g., https://company.atlassian.net)
        JIRA_EMAIL: User email
        JIRA_API_TOKEN: API token (create at https://id.atlassian.com/manage/api-tokens)

    Usage:
        client = JiraClient()
        issue = client.fetch_issue("PROJ-123")
    """

    def __init__(
        self,
        base_url: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
    ):
        """Initialize JIRA client.

        Args:
            base_url: JIRA instance URL (or JIRA_URL env var)
            email: User email (or JIRA_EMAIL env var)
            api_token: API token (or JIRA_API_TOKEN env var)
        """
        self.base_url = (base_url or os.environ.get("JIRA_URL", "")).rstrip("/")
        self.email = email or os.environ.get("JIRA_EMAIL", "")
        self.api_token = api_token or os.environ.get("JIRA_API_TOKEN", "")

        if not all([self.base_url, self.email, self.api_token]):
            raise ValueError(
                "JIRA credentials required. Set JIRA_URL, JIRA_EMAIL, and JIRA_API_TOKEN "
                "environment variables or pass them to constructor."
            )

        # Build auth header
        credentials = f"{self.email}:{self.api_token}"
        auth_bytes = b64encode(credentials.encode()).decode()
        self._headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def name(self) -> str:
        return "jira"

    def fetch_issue(self, issue_id: str) -> IssueData:
        """Fetch a JIRA issue by key (e.g., 'PROJ-123').

        Args:
            issue_id: JIRA issue key

        Returns:
            Normalized IssueData
        """
        # Validate issue key format
        if not re.match(r"^[A-Z][A-Z0-9]+-\d+$", issue_id.upper()):
            raise ValueError(
                f"Invalid JIRA issue key: {issue_id}. "
                "Expected format: PROJ-123"
            )

        issue_key = issue_id.upper()
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=self._headers)

                if response.status_code == 404:
                    raise ValueError(f"Issue not found: {issue_key}")
                elif response.status_code == 401:
                    raise ConnectionError("JIRA authentication failed. Check credentials.")
                elif response.status_code != 200:
                    raise ConnectionError(
                        f"JIRA API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                return self._parse_issue(data)

        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to JIRA: {e}")

    def validate_connection(self) -> bool:
        """Check if JIRA connection works."""
        url = f"{self.base_url}/rest/api/3/myself"
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(url, headers=self._headers)
                return response.status_code == 200
        except httpx.RequestError:
            return False

    def _parse_issue(self, data: dict) -> IssueData:
        """Parse JIRA API response into IssueData."""
        fields = data.get("fields", {})

        # Extract description (JIRA v3 uses ADF format)
        description = self._extract_description(fields.get("description"))

        # Extract acceptance criteria from description or custom field
        acceptance_criteria = self._extract_acceptance_criteria(description, fields)

        # Map priority
        priority = self._map_priority(fields.get("priority", {}).get("name"))

        # Map status
        status = self._map_status(fields.get("status", {}).get("statusCategory", {}).get("key"))

        # Extract labels
        labels = fields.get("labels", [])

        # Extract component names as additional labels
        components = [c.get("name") for c in fields.get("components", []) if c.get("name")]
        labels = labels + components

        # Extract assignee/reporter
        assignee = fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None
        reporter = fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None

        # Extract parent (for subtasks)
        parent_id = fields.get("parent", {}).get("key") if fields.get("parent") else None

        # Extract subtasks
        subtasks = [st.get("key") for st in fields.get("subtasks", []) if st.get("key")]

        # Extract linked issues
        linked_issues = []
        for link in fields.get("issuelinks", []):
            if link.get("outwardIssue"):
                linked_issues.append(link["outwardIssue"]["key"])
            if link.get("inwardIssue"):
                linked_issues.append(link["inwardIssue"]["key"])

        return IssueData(
            id=data["key"],
            title=fields.get("summary", ""),
            description=description,
            tracker="jira",
            url=f"{self.base_url}/browse/{data['key']}",
            priority=priority,
            status=status,
            labels=labels,
            assignee=assignee,
            reporter=reporter,
            acceptance_criteria=acceptance_criteria,
            parent_id=parent_id,
            subtasks=subtasks,
            linked_issues=linked_issues,
            raw=data,
        )

    def _extract_description(self, desc_field: dict | str | None) -> str:
        """Extract plain text from JIRA description field.

        JIRA API v3 uses Atlassian Document Format (ADF), a JSON structure.
        Earlier versions use plain text or wiki markup.
        """
        if not desc_field:
            return ""

        # Plain text (v2 API or simple content)
        if isinstance(desc_field, str):
            return desc_field

        # ADF format (v3 API)
        if isinstance(desc_field, dict):
            return self._adf_to_text(desc_field)

        return ""

    def _adf_to_text(self, node: dict) -> str:
        """Convert Atlassian Document Format to plain text."""
        if not isinstance(node, dict):
            return ""

        node_type = node.get("type", "")
        text_parts = []

        # Handle text nodes
        if node_type == "text":
            return node.get("text", "")

        # Handle inline nodes
        if node_type == "hardBreak":
            return "\n"

        if node_type == "mention":
            return f"@{node.get('attrs', {}).get('text', '')}"

        # Handle block nodes - recurse into content
        content = node.get("content", [])
        for child in content:
            text_parts.append(self._adf_to_text(child))

        # Add appropriate separators based on node type
        if node_type in ("paragraph", "heading", "bulletList", "orderedList", "codeBlock"):
            return "".join(text_parts) + "\n"

        if node_type == "listItem":
            return "• " + "".join(text_parts)

        return "".join(text_parts)

    def _extract_acceptance_criteria(
        self, description: str, fields: dict
    ) -> list[str]:
        """Extract acceptance criteria from description or custom field.

        Looks for common patterns:
        - "Acceptance Criteria:" section
        - "AC:" section
        - Numbered or bulleted lists after these headers
        """
        criteria = []

        # Try custom field first (common field names)
        for field_name in ["customfield_10100", "acceptance_criteria", "customfield_10020"]:
            if field_name in fields and fields[field_name]:
                value = fields[field_name]
                if isinstance(value, str):
                    criteria.extend(self._parse_criteria_text(value))
                elif isinstance(value, dict):
                    criteria.extend(self._parse_criteria_text(self._adf_to_text(value)))
                if criteria:
                    return criteria

        # Parse from description
        if description:
            # Look for AC section
            patterns = [
                r"(?:acceptance criteria|ac)[\s:]*\n([\s\S]*?)(?:\n\n|\Z)",
                r"(?:definition of done|dod)[\s:]*\n([\s\S]*?)(?:\n\n|\Z)",
            ]

            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    criteria.extend(self._parse_criteria_text(match.group(1)))
                    if criteria:
                        return criteria

        return criteria

    def _parse_criteria_text(self, text: str) -> list[str]:
        """Parse acceptance criteria from text block."""
        criteria = []
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Remove bullet points and numbers
            line = re.sub(r"^[\s•\-\*\d\.]+\s*", "", line)
            if line and len(line) > 3:  # Skip very short lines
                criteria.append(line)

        return criteria

    def _map_priority(self, jira_priority: str | None) -> IssuePriority:
        """Map JIRA priority to normalized priority."""
        if not jira_priority:
            return IssuePriority.UNKNOWN

        priority_lower = jira_priority.lower()
        if priority_lower in ("highest", "blocker", "critical"):
            return IssuePriority.CRITICAL
        elif priority_lower in ("high", "major"):
            return IssuePriority.HIGH
        elif priority_lower in ("medium", "normal"):
            return IssuePriority.MEDIUM
        elif priority_lower in ("low", "minor", "lowest", "trivial"):
            return IssuePriority.LOW
        return IssuePriority.UNKNOWN

    def _map_status(self, status_category: str | None) -> IssueStatus:
        """Map JIRA status category to normalized status."""
        if not status_category:
            return IssueStatus.UNKNOWN

        category_lower = status_category.lower()
        if category_lower == "new":
            return IssueStatus.OPEN
        elif category_lower == "indeterminate":
            return IssueStatus.IN_PROGRESS
        elif category_lower == "done":
            return IssueStatus.CLOSED
        return IssueStatus.UNKNOWN
