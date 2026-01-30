"""Chain runner for executing multi-step skill chains.

Reads the `chain` field from a skill manifest and executes
sub-skills sequentially, passing output paths between steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skills.base import FileChange, SkillOutput
from skills.runner import SkillRunner


class ChainRunner:
    """Execute a chain of skills sequentially.

    Chained skills are defined in manifest.yaml:
        chain:
          - skill: add-error-handling
          - skill: add-type-hints
          - skill: generate-tests
            config:
              framework: pytest

    Each skill's output files become the input for the next skill.

    Example:
        >>> runner = ChainRunner(loader=ext_loader, llm=my_llm)
        >>> output = runner.run(chain_manifest, target=Path("./src"))
    """

    def __init__(self, loader: Any, llm: Any = None):
        """Initialize the chain runner.

        Args:
            loader: ExtensionLoader instance for resolving sub-skills.
            llm: Optional LLM backend for AI-powered skills.
        """
        self.loader = loader
        self.skill_runner = SkillRunner(llm=llm)

    def run(
        self,
        chain: list[dict[str, Any]],
        target: Path,
        dry_run: bool = False,
    ) -> SkillOutput:
        """Execute a chain of skills sequentially.

        Args:
            chain: List of chain step definitions from manifest.
            target: File or directory to process.
            dry_run: If True, preview changes without applying.

        Returns:
            Combined SkillOutput from all chain steps.
        """
        all_changes: list[FileChange] = []
        summaries: list[str] = []
        errors: list[str] = []

        for i, step in enumerate(chain, 1):
            skill_name = step.get("skill")
            step_config = step.get("config", {})

            if not skill_name:
                errors.append(f"Step {i}: missing 'skill' field")
                continue

            # Resolve the skill class
            skill_class = self.loader.get_skill(skill_name)
            if skill_class is None:
                errors.append(
                    f"Step {i}: skill '{skill_name}' not found. "
                    "Is it installed?"
                )
                return SkillOutput(
                    success=False,
                    changes=all_changes,
                    summary=f"Chain failed at step {i}/{len(chain)}: "
                    f"skill '{skill_name}' not found",
                    errors=errors,
                )

            # Get file patterns from the sub-skill manifest
            manifest = self.loader.get_manifest(skill_name)
            file_patterns = manifest.file_patterns if manifest else None

            # Run the skill
            output = self.skill_runner.run(
                skill_class=skill_class,
                target=target,
                config=step_config,
                dry_run=dry_run,
                file_patterns=file_patterns or None,
            )

            all_changes.extend(output.changes)
            summaries.append(f"[{i}/{len(chain)}] {skill_name}: {output.summary}")

            if output.errors:
                errors.extend(output.errors)

            if not output.success:
                return SkillOutput(
                    success=False,
                    changes=all_changes,
                    summary=f"Chain failed at step {i}/{len(chain)}: {skill_name}",
                    errors=errors,
                )

        return SkillOutput(
            success=True,
            changes=all_changes,
            summary="\n".join(summaries),
            errors=errors if errors else None,
        )
