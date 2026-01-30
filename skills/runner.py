"""Skill runner for executing individual skills.

Handles loading skill classes, resolving target files, and
applying or previewing changes.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Type

from skills.base import BaseSkill, FileChange, SkillInput, SkillOutput


class SkillRunner:
    """Execute a single skill on target files or directories.

    Example:
        >>> runner = SkillRunner(llm=my_llm)
        >>> output = runner.run(skill_class, target=Path("./src"), dry_run=False)
    """

    def __init__(self, llm: Any = None):
        """Initialize the skill runner.

        Args:
            llm: Optional LLM backend for AI-powered skills.
        """
        self.llm = llm

    def run(
        self,
        skill_class: Type[BaseSkill],
        target: Path,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        file_patterns: list[str] | None = None,
    ) -> SkillOutput:
        """Execute a skill on a target path.

        Args:
            skill_class: The skill class to instantiate and run.
            target: File or directory to process.
            config: Skill-specific configuration.
            dry_run: If True, preview changes without writing.
            file_patterns: Glob patterns to filter files (e.g. ["*.py"]).

        Returns:
            SkillOutput with changes and summary.
        """
        target = target.resolve()

        # Resolve target paths
        target_paths = self._resolve_targets(target, file_patterns)
        if not target_paths:
            return SkillOutput(
                success=False,
                changes=[],
                summary="No matching files found",
                errors=[f"No files matched in {target}"],
            )

        # Create and run the skill
        skill = skill_class(llm=self.llm, config=config)
        input_data = SkillInput(
            target_paths=target_paths,
            config=config or {},
            dry_run=dry_run,
        )

        output = skill.run(input_data)

        # Apply changes if not dry run
        if not dry_run and output.success:
            self._apply_changes(output.changes)

        return output

    def preview(
        self,
        skill_class: Type[BaseSkill],
        target: Path,
        config: dict[str, Any] | None = None,
        file_patterns: list[str] | None = None,
    ) -> SkillOutput:
        """Preview skill changes without applying them.

        Args:
            skill_class: The skill class to instantiate.
            target: File or directory to process.
            config: Skill-specific configuration.
            file_patterns: Glob patterns to filter files.

        Returns:
            SkillOutput with proposed changes (not applied).
        """
        return self.run(
            skill_class=skill_class,
            target=target,
            config=config,
            dry_run=True,
            file_patterns=file_patterns,
        )

    def _resolve_targets(
        self,
        target: Path,
        file_patterns: list[str] | None = None,
    ) -> list[Path]:
        """Resolve target path to a list of files.

        Args:
            target: File or directory path.
            file_patterns: Glob patterns to match (e.g. ["*.py", "*.js"]).

        Returns:
            List of resolved file paths.
        """
        if target.is_file():
            if file_patterns and not self._matches_patterns(target, file_patterns):
                return []
            return [target]

        if not target.is_dir():
            return []

        if not file_patterns:
            # Default: all files (non-hidden, non-__pycache__)
            return [
                f for f in target.rglob("*")
                if f.is_file()
                and not any(p.startswith(".") for p in f.relative_to(target).parts)
                and "__pycache__" not in f.parts
            ]

        matched: list[Path] = []
        for pattern in file_patterns:
            for f in target.rglob(pattern):
                if f.is_file() and f not in matched:
                    matched.append(f)
        return sorted(matched)

    def _matches_patterns(self, path: Path, patterns: list[str]) -> bool:
        """Check if a file matches any of the given patterns."""
        return any(fnmatch.fnmatch(path.name, p) for p in patterns)

    def _apply_changes(self, changes: list[FileChange]) -> None:
        """Write file changes to disk.

        Args:
            changes: List of FileChange objects to apply.
        """
        for change in changes:
            if change.change_type == "deleted":
                if change.path.exists():
                    change.path.unlink()
            else:
                change.path.parent.mkdir(parents=True, exist_ok=True)
                change.path.write_text(change.modified_content)
