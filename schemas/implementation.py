"""Implementation artifacts schema.

Output of Implementation Agent: code changes and notes.
"""

from enum import Enum
from pydantic import BaseModel, Field


class DiffType(str, Enum):
    """Type of diff content."""

    UNIFIED = "unified"
    GIT = "git"
    SIDE_BY_SIDE = "side_by_side"


class FileDiff(BaseModel):
    """Diff for a single file."""

    path: str = Field(..., description="File path")
    operation: str = Field(..., description="create, modify, delete, rename")
    old_path: str | None = Field(None, description="Original path if renamed")
    diff_content: str = Field(..., description="Unified diff content")
    language: str | None = Field(None, description="File language for syntax highlighting")


class ChangeSet(BaseModel):
    """Complete set of changes for a feature.

    This represents the diff.patch output from the Implementation Agent.
    """

    # References
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")
    design_id: str | None = Field(None, description="Reference to design proposal version")

    # The changes
    diffs: list[FileDiff] = Field(default_factory=list, description="File diffs")
    diff_type: DiffType = Field(DiffType.GIT, description="Diff format")

    # Statistics
    files_changed: int = Field(0, description="Number of files changed")
    insertions: int = Field(0, description="Lines added")
    deletions: int = Field(0, description="Lines removed")

    # Full patch (for easy application)
    combined_patch: str = Field(..., description="Combined patch content for git apply")

    @property
    def is_empty(self) -> bool:
        return len(self.diffs) == 0

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "files_changed": 3,
                "insertions": 45,
                "deletions": 2,
            }
        }


class NewFunction(BaseModel):
    """Documentation for a new function/method."""

    name: str = Field(..., description="Function name")
    file_path: str = Field(..., description="File where defined")
    signature: str = Field(..., description="Function signature")
    purpose: str = Field(..., description="What it does")
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter descriptions",
    )
    returns: str | None = Field(None, description="Return value description")


class NewClass(BaseModel):
    """Documentation for a new class."""

    name: str = Field(..., description="Class name")
    file_path: str = Field(..., description="File where defined")
    purpose: str = Field(..., description="What it does")
    key_methods: list[str] = Field(
        default_factory=list,
        description="Important method names",
    )
    inherits_from: list[str] = Field(
        default_factory=list,
        description="Parent classes",
    )


class TechDebtItem(BaseModel):
    """Technical debt introduced or discovered."""

    description: str = Field(..., description="What the debt is")
    location: str = Field(..., description="File/function where it exists")
    severity: str = Field("medium", description="low, medium, high")
    suggested_fix: str | None = Field(None, description="How to address it")


class ImplementationNotes(BaseModel):
    """Notes explaining the implementation.

    This is the implementation_notes.md output from the Implementation Agent.
    """

    # References
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")

    # Summary
    summary: str = Field(..., description="Brief summary of changes")

    # What changed
    new_functions: list[NewFunction] = Field(
        default_factory=list,
        description="New functions added",
    )
    new_classes: list[NewClass] = Field(
        default_factory=list,
        description="New classes added",
    )
    modified_functions: list[str] = Field(
        default_factory=list,
        description="Existing functions that were modified",
    )

    # Dependencies
    dependencies_added: list[str] = Field(
        default_factory=list,
        description="New dependencies added",
    )
    config_changes: list[str] = Field(
        default_factory=list,
        description="Configuration changes made",
    )

    # Migration notes
    migration_required: bool = Field(
        False,
        description="Does this require database/data migration?",
    )
    migration_notes: str | None = Field(
        None,
        description="Migration instructions if required",
    )

    # Tech debt
    tech_debt_items: list[TechDebtItem] = Field(
        default_factory=list,
        description="Technical debt introduced or noted",
    )

    # TODOs
    todos: list[str] = Field(
        default_factory=list,
        description="TODO items for follow-up",
    )

    # Rollback
    rollback_instructions: str | None = Field(
        None,
        description="How to rollback these changes",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "summary": "Added rate limiting middleware using Redis sliding window",
                "new_functions": [
                    {
                        "name": "check_rate_limit",
                        "file_path": "app/middleware/rate_limit.py",
                        "signature": "async def check_rate_limit(key: str, limit: int, window: int) -> bool",
                        "purpose": "Check if request should be rate limited",
                    }
                ],
            }
        }
