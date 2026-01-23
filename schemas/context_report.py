"""Context report schema.

Output of Context Agent: repository analysis and relevant code context.
"""

from pydantic import BaseModel, Field


class CodeSnippet(BaseModel):
    """Relevant code snippet from the repository."""

    file_path: str = Field(..., description="Path to the file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    content: str = Field(..., description="The code content")
    relevance: str = Field(..., description="Why this snippet is relevant")
    language: str | None = Field(None, description="Programming language")


class FileContext(BaseModel):
    """Context about a specific file."""

    path: str = Field(..., description="File path relative to repo root")
    purpose: str = Field(..., description="What this file does")
    key_exports: list[str] = Field(
        default_factory=list,
        description="Key classes, functions, or exports",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Files this depends on",
    )
    dependents: list[str] = Field(
        default_factory=list,
        description="Files that depend on this",
    )


class RepoStructure(BaseModel):
    """High-level repository structure analysis."""

    framework: str | None = Field(None, description="Detected framework (FastAPI, React, etc.)")
    language: str = Field(..., description="Primary programming language")
    package_manager: str | None = Field(None, description="Package manager (pip, npm, etc.)")
    test_framework: str | None = Field(None, description="Test framework (pytest, jest, etc.)")
    entry_points: list[str] = Field(
        default_factory=list,
        description="Main entry points (API routes, CLI, etc.)",
    )
    key_directories: dict[str, str] = Field(
        default_factory=dict,
        description="Key directories and their purposes",
    )


class ExistingPattern(BaseModel):
    """Pattern already used in the codebase."""

    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="How it's used")
    examples: list[str] = Field(
        default_factory=list,
        description="File paths showing this pattern",
    )
    recommended: bool = Field(True, description="Should new code follow this pattern?")


class ContextReport(BaseModel):
    """Complete context report for a feature.

    This is the output of the Context Agent - analysis of the repository
    and identification of relevant code for the feature.
    """

    # Feature reference
    feature_id: str = Field(..., description="Reference to FeatureSpec.id")

    # Repository analysis
    repo_structure: RepoStructure = Field(..., description="High-level repo analysis")

    # Relevant files
    relevant_files: list[FileContext] = Field(
        default_factory=list,
        description="Files relevant to this feature",
    )
    files_to_modify: list[str] = Field(
        default_factory=list,
        description="Files that will likely need changes",
    )
    files_to_create: list[str] = Field(
        default_factory=list,
        description="New files that may need to be created",
    )

    # Code context
    snippets: list[CodeSnippet] = Field(
        default_factory=list,
        description="Relevant code snippets",
    )

    # Patterns
    existing_patterns: list[ExistingPattern] = Field(
        default_factory=list,
        description="Patterns already used in the codebase",
    )

    # Mental model
    mental_model: str = Field(
        ...,
        description="Short narrative of how the feature should integrate",
    )

    # Dependencies
    external_dependencies: list[str] = Field(
        default_factory=list,
        description="External packages/libraries in use",
    )
    potential_new_dependencies: list[str] = Field(
        default_factory=list,
        description="New dependencies that might be needed",
    )

    # Risks
    integration_risks: list[str] = Field(
        default_factory=list,
        description="Potential integration challenges",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "FEAT-001",
                "repo_structure": {
                    "framework": "FastAPI",
                    "language": "Python",
                    "package_manager": "pip",
                    "test_framework": "pytest",
                    "entry_points": ["app/main.py"],
                    "key_directories": {
                        "app/": "Main application code",
                        "tests/": "Test suite",
                    },
                },
                "mental_model": "Rate limiting should be implemented as middleware in the auth module, following the existing decorator pattern for endpoint protection.",
            }
        }
