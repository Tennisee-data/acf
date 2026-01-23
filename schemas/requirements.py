"""Requirements Tracker schema.

Tracks requirements throughout the pipeline to ensure all user requests
are addressed and verified before completion.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RequirementStatus(str, Enum):
    """Status of a requirement in the pipeline."""

    PENDING = "pending"  # Not yet addressed
    IN_PROGRESS = "in_progress"  # Currently being worked on
    ADDRESSED = "addressed"  # Agent claims to have addressed it
    VERIFIED = "verified"  # Verification confirmed it's done
    FAILED = "failed"  # Verification found it wasn't done


class RequirementType(str, Enum):
    """Type of requirement."""

    FUNCTIONAL = "functional"  # Feature behavior
    DOCUMENTATION = "documentation"  # Docs, README, comments
    TESTING = "testing"  # Tests, coverage
    SECURITY = "security"  # Security requirements
    PERFORMANCE = "performance"  # Performance requirements
    FIX = "fix"  # Bug fix or correction
    REMOVAL = "removal"  # Remove something
    STYLE = "style"  # UI/UX, styling


class Requirement(BaseModel):
    """A single trackable requirement extracted from user prompt."""

    id: str = Field(..., description="Unique requirement ID (e.g., REQ-001)")
    description: str = Field(..., description="What needs to be done")
    type: RequirementType = Field(
        RequirementType.FUNCTIONAL,
        description="Type of requirement"
    )
    status: RequirementStatus = Field(
        RequirementStatus.PENDING,
        description="Current status"
    )
    priority: int = Field(
        1,
        description="Priority (1=highest, 5=lowest)",
        ge=1,
        le=5
    )

    # Tracking
    addressed_by: str | None = Field(
        None,
        description="Agent that addressed this requirement"
    )
    addressed_at: datetime | None = Field(
        None,
        description="When the requirement was addressed"
    )
    evidence: str | None = Field(
        None,
        description="Evidence of completion (file path, code snippet, etc.)"
    )

    # Verification
    verified_by: str | None = Field(
        None,
        description="Agent that verified this requirement"
    )
    verified_at: datetime | None = Field(
        None,
        description="When the requirement was verified"
    )
    verification_method: str | None = Field(
        None,
        description="How it was verified (grep, file check, test, etc.)"
    )
    verification_result: str | None = Field(
        None,
        description="Result of verification"
    )

    # Original source
    source_text: str | None = Field(
        None,
        description="Original text from user prompt that led to this requirement"
    )
    acceptance_criteria_id: str | None = Field(
        None,
        description="Link to acceptance criteria if applicable"
    )

    def mark_addressed(
        self,
        agent_name: str,
        evidence: str | None = None
    ) -> None:
        """Mark this requirement as addressed by an agent."""
        self.status = RequirementStatus.ADDRESSED
        self.addressed_by = agent_name
        self.addressed_at = datetime.utcnow()
        if evidence:
            self.evidence = evidence

    def mark_verified(
        self,
        agent_name: str,
        method: str,
        result: str,
        success: bool = True
    ) -> None:
        """Mark this requirement as verified."""
        self.status = RequirementStatus.VERIFIED if success else RequirementStatus.FAILED
        self.verified_by = agent_name
        self.verified_at = datetime.utcnow()
        self.verification_method = method
        self.verification_result = result

    def mark_failed(self, reason: str) -> None:
        """Mark this requirement as failed verification."""
        self.status = RequirementStatus.FAILED
        self.verification_result = reason


class RequirementsTracker(BaseModel):
    """Tracks all requirements through the pipeline.

    This object flows through the entire pipeline, getting updated
    by each agent as work is done, and validated at the end.
    """

    requirements: list[Requirement] = Field(
        default_factory=list,
        description="List of tracked requirements"
    )
    original_prompt: str = Field(
        "",
        description="Original user prompt"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tracker was created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tracker was last updated"
    )

    # Summary stats
    _next_id: int = 1

    def add_requirement(
        self,
        description: str,
        req_type: RequirementType = RequirementType.FUNCTIONAL,
        priority: int = 1,
        source_text: str | None = None,
        acceptance_criteria_id: str | None = None,
    ) -> Requirement:
        """Add a new requirement to track.

        Args:
            description: What needs to be done
            req_type: Type of requirement
            priority: Priority (1=highest)
            source_text: Original text from prompt
            acceptance_criteria_id: Link to AC if applicable

        Returns:
            The created Requirement
        """
        req_id = f"REQ-{self._next_id:03d}"
        self._next_id += 1

        req = Requirement(
            id=req_id,
            description=description,
            type=req_type,
            priority=priority,
            source_text=source_text,
            acceptance_criteria_id=acceptance_criteria_id,
        )
        self.requirements.append(req)
        self.last_updated = datetime.utcnow()
        return req

    def get_pending(self) -> list[Requirement]:
        """Get all pending requirements."""
        return [r for r in self.requirements if r.status == RequirementStatus.PENDING]

    def get_addressed(self) -> list[Requirement]:
        """Get all addressed (but not verified) requirements."""
        return [r for r in self.requirements if r.status == RequirementStatus.ADDRESSED]

    def get_verified(self) -> list[Requirement]:
        """Get all verified requirements."""
        return [r for r in self.requirements if r.status == RequirementStatus.VERIFIED]

    def get_failed(self) -> list[Requirement]:
        """Get all failed requirements."""
        return [r for r in self.requirements if r.status == RequirementStatus.FAILED]

    def get_by_type(self, req_type: RequirementType) -> list[Requirement]:
        """Get requirements of a specific type."""
        return [r for r in self.requirements if r.type == req_type]

    def get_by_id(self, req_id: str) -> Requirement | None:
        """Get a requirement by ID."""
        for req in self.requirements:
            if req.id == req_id:
                return req
        return None

    def mark_addressed(
        self,
        req_id: str,
        agent_name: str,
        evidence: str | None = None
    ) -> bool:
        """Mark a requirement as addressed.

        Args:
            req_id: Requirement ID
            agent_name: Name of agent that addressed it
            evidence: Evidence of completion

        Returns:
            True if found and updated, False otherwise
        """
        req = self.get_by_id(req_id)
        if req:
            req.mark_addressed(agent_name, evidence)
            self.last_updated = datetime.utcnow()
            return True
        return False

    def mark_verified(
        self,
        req_id: str,
        agent_name: str,
        method: str,
        result: str,
        success: bool = True
    ) -> bool:
        """Mark a requirement as verified.

        Args:
            req_id: Requirement ID
            agent_name: Name of verifying agent
            method: Verification method used
            result: Verification result details
            success: Whether verification passed

        Returns:
            True if found and updated, False otherwise
        """
        req = self.get_by_id(req_id)
        if req:
            req.mark_verified(agent_name, method, result, success)
            self.last_updated = datetime.utcnow()
            return True
        return False

    def completion_summary(self) -> dict[str, Any]:
        """Get a summary of requirement completion status.

        Returns:
            Dict with counts and percentages
        """
        total = len(self.requirements)
        if total == 0:
            return {
                "total": 0,
                "pending": 0,
                "addressed": 0,
                "verified": 0,
                "failed": 0,
                "completion_rate": 0.0,
                "verification_rate": 0.0,
            }

        pending = len(self.get_pending())
        addressed = len(self.get_addressed())
        verified = len(self.get_verified())
        failed = len(self.get_failed())

        return {
            "total": total,
            "pending": pending,
            "addressed": addressed,
            "verified": verified,
            "failed": failed,
            "completion_rate": (addressed + verified) / total * 100,
            "verification_rate": verified / total * 100,
            "all_verified": verified == total,
            "has_failures": failed > 0,
        }

    def get_unmet_requirements(self) -> list[Requirement]:
        """Get requirements that are not yet verified.

        Returns:
            List of pending, addressed, or failed requirements
        """
        return [
            r for r in self.requirements
            if r.status != RequirementStatus.VERIFIED
        ]

    def to_checklist(self) -> str:
        """Generate a markdown checklist of requirements.

        Returns:
            Markdown formatted checklist
        """
        lines = ["## Requirements Checklist\n"]

        status_icons = {
            RequirementStatus.PENDING: "⬜",
            RequirementStatus.IN_PROGRESS: "🔄",
            RequirementStatus.ADDRESSED: "✅",
            RequirementStatus.VERIFIED: "✅✓",
            RequirementStatus.FAILED: "❌",
        }

        for req in self.requirements:
            icon = status_icons.get(req.status, "⬜")
            line = f"- {icon} **{req.id}**: {req.description}"
            if req.addressed_by:
                line += f" (by {req.addressed_by})"
            if req.status == RequirementStatus.FAILED and req.verification_result:
                line += f"\n  - ❌ Failed: {req.verification_result}"
            lines.append(line)

        summary = self.completion_summary()
        lines.append(f"\n**Progress**: {summary['verified']}/{summary['total']} verified")
        if summary['has_failures']:
            lines.append(f"**⚠️ {summary['failed']} requirement(s) failed verification**")

        return "\n".join(lines)

    class Config:
        json_schema_extra = {
            "example": {
                "requirements": [
                    {
                        "id": "REQ-001",
                        "description": "Fix README to use run.sh instead of main.py",
                        "type": "fix",
                        "status": "verified",
                        "priority": 1,
                        "addressed_by": "ImplementationAgent",
                        "evidence": "Updated doc_agent.py template",
                        "verified_by": "VerifyAgent",
                        "verification_method": "grep",
                        "verification_result": "No 'python main.py' found in generated README",
                    }
                ],
                "original_prompt": "Fix README to not reference main.py",
            }
        }
