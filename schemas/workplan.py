"""Workplan schema for decomposed feature tasks.

Output of Decomposition Agent: breaks a feature spec into
traceable sub-tasks with priorities and dependencies.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    """Category of work for a sub-task."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    API = "api"
    DATABASE = "database"
    DATA_MIGRATION = "data_migration"
    INFRA = "infra"
    AUTH = "auth"
    TESTING = "testing"
    DOCS = "docs"
    CONFIG = "config"


class TaskPriority(str, Enum):
    """Priority for task ordering."""

    P0 = "p0"  # Must be done first (blocker)
    P1 = "p1"  # High priority
    P2 = "p2"  # Normal priority
    P3 = "p3"  # Nice to have


class TaskSize(str, Enum):
    """Estimated size/complexity."""

    XS = "xs"  # < 10 lines
    S = "s"    # 10-50 lines
    M = "m"    # 50-200 lines
    L = "l"    # 200-500 lines
    XL = "xl"  # > 500 lines (should be split further)


class SubTask(BaseModel):
    """A single sub-task within the workplan."""

    id: str = Field(..., description="Unique task ID (e.g., TASK-001)")
    title: str = Field(..., description="Short task title")
    description: str = Field(..., description="What needs to be done")

    # Classification
    category: TaskCategory = Field(..., description="Type of work")
    priority: TaskPriority = Field(TaskPriority.P1, description="Execution priority")
    size: TaskSize = Field(TaskSize.M, description="Estimated size")

    # Dependencies
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs this depends on (must complete first)",
    )
    blocks: list[str] = Field(
        default_factory=list,
        description="Task IDs blocked by this task",
    )

    # Acceptance criteria (inherited or new)
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria IDs from spec or new criteria",
    )

    # Implementation hints
    target_files: list[str] = Field(
        default_factory=list,
        description="Files likely to be modified",
    )
    implementation_notes: str | None = Field(
        None,
        description="Hints for ImplementationAgent",
    )

    # Tracking
    status: str = Field("pending", description="pending | in_progress | completed | skipped")


class WorkPlan(BaseModel):
    """Complete workplan for a feature.

    Output of DecompositionAgent - breaks a feature into
    ordered, dependent sub-tasks for incremental implementation.
    """

    # Reference back to source
    feature_id: str = Field(..., description="Source feature spec ID")
    feature_title: str = Field(..., description="Source feature title")

    # The sub-tasks
    tasks: list[SubTask] = Field(
        default_factory=list,
        description="Ordered list of sub-tasks",
    )

    # Execution plan
    execution_order: list[str] = Field(
        default_factory=list,
        description="Task IDs in recommended execution order",
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Groups of tasks that can run in parallel",
    )

    # Summary
    total_tasks: int = Field(0, description="Total number of tasks")
    estimated_complexity: str = Field(
        "moderate",
        description="Overall complexity: trivial, simple, moderate, complex, very_complex",
    )

    # Decomposition notes
    decomposition_rationale: str = Field(
        "",
        description="Why the feature was split this way",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Identified risks or blockers",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "feature_title": "User Authentication with OAuth",
                "tasks": [
                    {
                        "id": "TASK-001",
                        "title": "Add OAuth dependencies",
                        "description": "Add authlib and httpx to requirements",
                        "category": "config",
                        "priority": "p0",
                        "size": "xs",
                        "depends_on": [],
                    },
                    {
                        "id": "TASK-002",
                        "title": "Create OAuth config schema",
                        "description": "Add OAuth client ID/secret to config",
                        "category": "backend",
                        "priority": "p0",
                        "size": "s",
                        "depends_on": ["TASK-001"],
                    },
                    {
                        "id": "TASK-003",
                        "title": "Implement OAuth callback endpoint",
                        "description": "Handle /auth/callback for OAuth flow",
                        "category": "api",
                        "priority": "p1",
                        "size": "m",
                        "depends_on": ["TASK-002"],
                    },
                ],
                "execution_order": ["TASK-001", "TASK-002", "TASK-003"],
                "total_tasks": 3,
            }
        }
