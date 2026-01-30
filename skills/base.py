"""Base classes for ACF skills.

Skills are standalone code transformations that operate on files/directories
without requiring a full pipeline run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillInput:
    """Input data for a skill execution.

    Attributes:
        target_paths: Files or directories to process.
        config: Skill-specific configuration options.
        dry_run: If True, preview changes without applying them.
    """

    target_paths: list[Path]
    config: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False


@dataclass
class FileChange:
    """Represents a single file change produced by a skill.

    Attributes:
        path: Path to the file.
        original_content: Original file content (None for new files).
        modified_content: New file content.
        change_type: Type of change: "modified", "created", or "deleted".
    """

    path: Path
    original_content: str | None
    modified_content: str
    change_type: str  # "modified", "created", "deleted"


@dataclass
class SkillOutput:
    """Output from a skill execution.

    Attributes:
        success: Whether the skill completed successfully.
        changes: List of file changes produced.
        summary: Human-readable summary of what was done.
        errors: List of error messages, if any.
    """

    success: bool
    changes: list[FileChange]
    summary: str
    errors: list[str] | None = None


class BaseSkill(ABC):
    """Abstract base class for all skills.

    Skills transform code by operating on target files/directories.
    They can be run standalone via `acf skill run` or composed into
    chains for multi-step transformations.

    Example:
        class AddErrorHandlingSkill(BaseSkill):
            def run(self, input_data: SkillInput) -> SkillOutput:
                changes = []
                for path in input_data.target_paths:
                    content = path.read_text()
                    modified = self._add_error_handling(content)
                    changes.append(FileChange(
                        path=path,
                        original_content=content,
                        modified_content=modified,
                        change_type="modified",
                    ))
                return SkillOutput(
                    success=True,
                    changes=changes,
                    summary=f"Added error handling to {len(changes)} files",
                )
    """

    def __init__(self, llm: Any = None, config: dict[str, Any] | None = None):
        """Initialize the skill.

        Args:
            llm: Optional LLM backend for AI-powered skills.
            config: Optional skill configuration.
        """
        self.llm = llm
        self.config = config or {}

    @abstractmethod
    def run(self, input_data: SkillInput) -> SkillOutput:
        """Execute the skill on the given input.

        Args:
            input_data: Skill input with target paths and config.

        Returns:
            SkillOutput with changes and summary.
        """
        ...

    def preview(self, input_data: SkillInput) -> SkillOutput:
        """Preview changes without applying them.

        Default implementation calls run() with dry_run=True.
        Override for custom preview behavior.

        Args:
            input_data: Skill input with target paths and config.

        Returns:
            SkillOutput with proposed changes (not applied).
        """
        input_data.dry_run = True
        return self.run(input_data)
