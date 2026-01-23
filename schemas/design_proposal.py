"""Design proposal schema.

Output of Research & Design Agent: architecture and implementation plan.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Type of file change."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class FileChange(BaseModel):
    """Proposed change to a file."""

    path: str = Field(..., description="File path")
    change_type: ChangeType = Field(..., description="Type of change")
    description: str = Field(..., description="What will be changed")
    estimated_lines: int | None = Field(None, description="Estimated lines affected")


class Dependency(BaseModel):
    """New dependency to add."""

    name: str = Field(..., description="Package name")
    version: str | None = Field(None, description="Version constraint")
    reason: str = Field(..., description="Why this dependency is needed")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative packages considered",
    )


class DesignOption(BaseModel):
    """A design option/approach."""

    name: str = Field(..., description="Option name")
    description: str = Field(..., description="Detailed description")
    pros: list[str] = Field(default_factory=list, description="Advantages")
    cons: list[str] = Field(default_factory=list, description="Disadvantages")
    estimated_effort: str | None = Field(
        None,
        description="Relative effort: low, medium, high",
    )
    recommended: bool = Field(False, description="Is this the recommended option?")


class DataFlowStep(BaseModel):
    """Step in the data flow."""

    order: int = Field(..., description="Step order")
    component: str = Field(..., description="Component/module involved")
    action: str = Field(..., description="What happens at this step")
    data_in: str | None = Field(None, description="Input data")
    data_out: str | None = Field(None, description="Output data")


class DesignProposal(BaseModel):
    """Complete design proposal for a feature.

    This is the output of the Research & Design Agent - architecture
    decisions and implementation plan.
    """

    # References
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")

    # High-level design
    summary: str = Field(..., description="One-paragraph design summary")
    architecture_sketch: str = Field(
        ...,
        description="ASCII diagram or structured description of the architecture",
    )

    # Options analysis
    options: list[DesignOption] = Field(
        default_factory=list,
        description="Design options considered",
    )
    chosen_approach: str = Field(..., description="Name of chosen option")
    rationale: str = Field(..., description="Why this approach was chosen")

    # Data flow
    data_flow: list[DataFlowStep] = Field(
        default_factory=list,
        description="Data flow through the system",
    )

    # File changes
    file_changes: list[FileChange] = Field(
        default_factory=list,
        description="Proposed file changes",
    )

    # Dependencies
    new_dependencies: list[Dependency] = Field(
        default_factory=list,
        description="New dependencies to add",
    )
    deprecation_notes: list[str] = Field(
        default_factory=list,
        description="Deprecation warnings for current patterns/libs",
    )

    # Best practices
    patterns_to_follow: list[str] = Field(
        default_factory=list,
        description="Patterns/idioms to follow",
    )
    patterns_to_avoid: list[str] = Field(
        default_factory=list,
        description="Anti-patterns to avoid",
    )

    # Risks and mitigations
    risks: list[str] = Field(default_factory=list, description="Implementation risks")
    mitigations: list[str] = Field(
        default_factory=list,
        description="Risk mitigations",
    )

    # Testing strategy
    testing_strategy: str = Field(
        ...,
        description="How to test this implementation",
    )

    # Approval required
    requires_approval: bool = Field(
        True,
        description="Does this design need human approval before implementation?",
    )
    approval_notes: str | None = Field(
        None,
        description="Notes for the human reviewer",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "summary": "Implement rate limiting using a sliding window algorithm with Redis as the backing store.",
                "architecture_sketch": """
                    [Request] -> [RateLimitMiddleware] -> [Redis Counter] -> [Allow/Deny]
                                         |
                                    [Config: 5 attempts/15 min]
                """,
                "chosen_approach": "Sliding Window with Redis",
                "rationale": "Provides accurate rate limiting with minimal memory footprint and supports distributed deployments.",
            }
        }
