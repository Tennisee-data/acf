"""Feature specification schema.

Output of Spec Agent: structured representation of a feature request.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Affected domain tags."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    INFRA = "infra"
    API = "api"
    AUTH = "auth"
    UI = "ui"
    CLI = "cli"
    TESTING = "testing"
    DOCS = "docs"


class Priority(str, Enum):
    """Feature priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AcceptanceCriteria(BaseModel):
    """Single acceptance criterion for the feature."""

    id: str = Field(..., description="Unique identifier (e.g., AC-001)")
    description: str = Field(..., description="What must be true for this to pass")
    testable: bool = Field(True, description="Can this be verified automatically?")
    verification_hint: str | None = Field(None, description="How to verify this criterion")


class Constraint(BaseModel):
    """Technical or business constraint."""

    description: str = Field(..., description="The constraint")
    type: str = Field("technical", description="Type: technical, business, security, etc.")
    impact: str | None = Field(None, description="Impact if violated")


class NonFunctionalRequirement(BaseModel):
    """Non-functional requirement (performance, security, etc.)."""

    category: str = Field(..., description="Category: performance, security, scalability, etc.")
    requirement: str = Field(..., description="The requirement statement")
    metric: str | None = Field(None, description="Measurable metric if applicable")
    threshold: str | None = Field(None, description="Acceptable threshold")


class FeatureSpec(BaseModel):
    """Complete feature specification.

    This is the output of the Spec Agent - a normalized, structured
    representation of a feature request.
    """

    # Identity
    id: str = Field(..., description="Unique feature ID")
    title: str = Field(..., description="Short feature title")
    original_description: str = Field(..., description="Original natural language input")

    # User story format
    user_story: str = Field(
        ...,
        description="As a [user], I want [feature], so that [benefit]",
    )

    # Detailed specs
    acceptance_criteria: list[AcceptanceCriteria] = Field(
        default_factory=list,
        description="List of acceptance criteria",
    )
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Technical and business constraints",
    )
    non_functional_requirements: list[NonFunctionalRequirement] = Field(
        default_factory=list,
        description="Non-functional requirements",
    )

    # Classification
    domains: list[Domain] = Field(
        default_factory=list,
        description="Affected domains/layers",
    )
    priority: Priority = Field(Priority.MEDIUM, description="Feature priority")

    # Assumptions and clarifications
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made during spec parsing",
    )
    clarifications_needed: list[str] = Field(
        default_factory=list,
        description="Questions that need human clarification",
    )

    # Metadata
    estimated_complexity: str | None = Field(
        None,
        description="Estimated complexity: trivial, simple, moderate, complex",
    )
    related_features: list[str] = Field(
        default_factory=list,
        description="IDs of related features",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "FEAT-001",
                "title": "Login Rate Limiting",
                "original_description": "Add login rate-limit",
                "user_story": "As a security admin, I want login attempts to be rate-limited, so that brute force attacks are prevented",
                "acceptance_criteria": [
                    {
                        "id": "AC-001",
                        "description": "After 5 failed attempts, user is locked out for 15 minutes",
                        "testable": True,
                        "verification_hint": "Attempt 6 logins with wrong password",
                    }
                ],
                "domains": ["backend", "auth"],
                "priority": "high",
            }
        }
