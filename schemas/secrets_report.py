"""Secrets report schema for tracking detected credentials.

Output of SecretsScanAgent: detected secrets, auto-fixes applied,
and blocking status for pipeline safety.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SecretType(str, Enum):
    """Types of secrets that can be detected."""

    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    GITHUB_PAT = "github_pat"
    GITLAB_TOKEN = "gitlab_token"
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    STRIPE_KEY = "stripe_key"
    TWILIO_KEY = "twilio_key"
    SENDGRID_KEY = "sendgrid_key"
    OPENAI_KEY = "openai_key"
    ANTHROPIC_KEY = "anthropic_key"
    GOOGLE_API_KEY = "google_api_key"
    FIREBASE_KEY = "firebase_key"
    JWT_SECRET = "jwt_secret"
    DATABASE_URL = "database_url"
    PRIVATE_KEY = "private_key"
    SSH_KEY = "ssh_key"
    PASSWORD = "password"
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"
    BEARER_TOKEN = "bearer_token"
    GENERIC_SECRET = "generic_secret"


class SecretSeverity(str, Enum):
    """Severity of detected secret."""

    CRITICAL = "critical"  # Active production keys
    HIGH = "high"  # API keys, tokens
    MEDIUM = "medium"  # Test keys, internal tokens
    LOW = "low"  # Potentially sensitive


class SecretStatus(str, Enum):
    """Status of a detected secret."""

    DETECTED = "detected"  # Found, not yet processed
    AUTO_FIXED = "auto_fixed"  # Moved to .env + os.getenv
    MANUAL_REQUIRED = "manual_required"  # Needs human intervention
    FALSE_POSITIVE = "false_positive"  # Marked as not a real secret
    IGNORED = "ignored"  # Intentionally skipped (e.g., test fixtures)


class DetectedSecret(BaseModel):
    """A single detected secret in code."""

    file_path: str = Field(..., description="Path to file containing secret")
    line_number: int = Field(..., description="Line number where found")
    column_start: int = Field(0, description="Column start position")
    column_end: int = Field(0, description="Column end position")

    secret_type: SecretType = Field(..., description="Type of secret")
    severity: SecretSeverity = Field(SecretSeverity.HIGH, description="Severity")
    status: SecretStatus = Field(SecretStatus.DETECTED, description="Current status")

    # The actual secret (redacted for safety)
    redacted_value: str = Field(
        "",
        description="Redacted secret value (e.g., sk-...abc)",
    )
    variable_name: str = Field(
        "",
        description="Variable name if identifiable",
    )
    context_snippet: str = Field(
        "",
        description="Code context around the secret (redacted)",
    )

    # Auto-fix details
    suggested_env_var: str = Field(
        "",
        description="Suggested environment variable name",
    )
    fix_applied: bool = Field(False, description="Whether auto-fix was applied")
    fix_description: str = Field("", description="Description of fix applied")


class AutoFix(BaseModel):
    """Record of an auto-fix applied."""

    file_path: str = Field(..., description="File that was modified")
    original_line: str = Field(..., description="Original line (redacted)")
    fixed_line: str = Field(..., description="Fixed line with os.getenv")
    env_var_name: str = Field(..., description="Environment variable name")
    env_var_added: bool = Field(False, description="Added to .env.example")


class SecretsReport(BaseModel):
    """Complete secrets scan report.

    Output of SecretsScanAgent - tracks detected secrets,
    applied fixes, and determines if pipeline should be blocked.
    """

    # Scan metadata
    scan_phase: str = Field(
        "pre_docker",
        description="When scan ran: pre_docker | pre_verify",
    )
    files_scanned: int = Field(0, description="Number of files scanned")
    scan_duration_ms: int = Field(0, description="Scan duration in milliseconds")

    # Detected secrets
    secrets: list[DetectedSecret] = Field(
        default_factory=list,
        description="All detected secrets",
    )

    # Auto-fixes applied
    auto_fixes: list[AutoFix] = Field(
        default_factory=list,
        description="Auto-fixes that were applied",
    )

    # Summary counts
    total_detected: int = Field(0, description="Total secrets detected")
    auto_fixed_count: int = Field(0, description="Secrets auto-fixed")
    manual_required_count: int = Field(0, description="Secrets needing manual fix")
    false_positive_count: int = Field(0, description="Marked as false positives")

    # Severity breakdown
    critical_count: int = Field(0, description="Critical severity count")
    high_count: int = Field(0, description="High severity count")
    medium_count: int = Field(0, description="Medium severity count")
    low_count: int = Field(0, description="Low severity count")

    # Pipeline control
    blocked: bool = Field(
        False,
        description="True if pipeline should be blocked",
    )
    block_reason: str = Field(
        "",
        description="Reason for blocking if blocked",
    )

    # Files modified
    env_example_updated: bool = Field(
        False,
        description="Whether .env.example was updated",
    )
    readme_updated: bool = Field(
        False,
        description="Whether README was updated with config instructions",
    )

    # Notes for other agents
    summary: str = Field("", description="Human-readable summary")
    verify_notes: list[str] = Field(
        default_factory=list,
        description="Notes for VerifyAgent",
    )

    def should_block(self) -> bool:
        """Determine if pipeline should be blocked."""
        # Block if any critical or high secrets remain unresolved
        unresolved = [
            s for s in self.secrets
            if s.status in (SecretStatus.DETECTED, SecretStatus.MANUAL_REQUIRED)
            and s.severity in (SecretSeverity.CRITICAL, SecretSeverity.HIGH)
        ]
        return len(unresolved) > 0

    class Config:
        json_schema_extra = {
            "example": {
                "scan_phase": "pre_docker",
                "files_scanned": 42,
                "total_detected": 2,
                "auto_fixed_count": 1,
                "manual_required_count": 1,
                "blocked": True,
                "block_reason": "1 high-severity secret requires manual review",
                "secrets": [
                    {
                        "file_path": "src/api/client.py",
                        "line_number": 15,
                        "secret_type": "openai_key",
                        "severity": "high",
                        "status": "auto_fixed",
                        "redacted_value": "sk-...xyz",
                        "variable_name": "OPENAI_API_KEY",
                    }
                ],
            }
        }
