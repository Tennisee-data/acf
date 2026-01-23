"""Implementation Agent for generating code changes."""

import logging
import re
from pathlib import Path

from llm_backend import LLMBackend

logger = logging.getLogger(__name__)

# Import stack profiles for tech-specific guidance
try:
    from profiles import get_profile_guidance
except ImportError:
    get_profile_guidance = None
    logger.warning("Stack profiles not available - running without tech-specific guidance")

from schemas.context_report import ContextReport
from schemas.design_proposal import DesignProposal, FileChange
from schemas.feature_spec import FeatureSpec
from schemas.implementation import (
    ChangeSet,
    DiffType,
    FileDiff,
    ImplementationNotes,
    NewClass,
    NewFunction,
    TechDebtItem,
)
from tools import FilesystemTool
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent
from .prompts import CODE_PRINCIPLES, IMPLEMENTATION_PRINCIPLES


SYSTEM_PROMPT = f"""{CODE_PRINCIPLES}

{IMPLEMENTATION_PRINCIPLES}

## Output Format

You are a code generator. Output ONLY a JSON object. No markdown, no explanation, no text before or after.

{{"file_changes":[{{"path":"app/main.py","operation":"create","language":"python","original_code":"","new_code":"# actual code here","description":"what changed"}}],"new_functions":[{{"name":"func","file_path":"app/main.py","signature":"def func()","purpose":"purpose","parameters":{{}},"returns":"return type"}}],"new_classes":[],"modified_functions":[],"dependencies_added":[],"config_changes":[],"migration_required":false,"migration_notes":null,"tech_debt_items":[],"todos":[],"rollback_instructions":"how to rollback","summary":"summary of changes"}}

CRITICAL RULES:
1. Output starts with {{ and ends with }}. No other text allowed.
2. The "path" field MUST be a real file path like "app/main.py" or "app/services/webhook.py". NEVER use "unknown" or "file.py".
3. Use paths from the "Planned File Changes" section if provided."""


class ImplementationAgent(BaseAgent):
    """Agent for generating code implementations.

    Takes feature spec, context, and design proposal to produce:
    - Actual code changes as diffs
    - New functions and classes documentation
    - Implementation notes
    - Technical debt tracking
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize ImplementationAgent.

        Args:
            llm: LLM backend for inference
            repo_path: Path to repository for file operations
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)
        self.repo_path = repo_path or Path.cwd()
        self.fs_tool = FilesystemTool(base_path=self.repo_path)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate code implementation from design.

        Args:
            input_data: Must contain 'feature_spec', 'context_report',
                       and 'design_proposal' in context

        Returns:
            AgentOutput with ChangeSet and ImplementationNotes data
        """
        context = input_data.context

        # Extract inputs
        feature_spec = self._extract_feature_spec(context)
        context_report = self._extract_context_report(context)
        design_proposal = self._extract_design_proposal(context)
        safety_invariants = context.get("safety_invariants")

        if safety_invariants:
            logger.info(
                "ImplementationAgent received safety invariants (%d chars)",
                len(safety_invariants),
            )

        feature_description = context.get("feature_description", "")
        feature_id = feature_spec.id if feature_spec else context.get("feature_id", "FEAT-000")

        # Extract workplan (from DecompositionAgent)
        workplan = context.get("workplan", {})

        # Extract tech stack for profile guidance
        tech_stack = context.get("tech_stack", [])

        if not design_proposal and not feature_description:
            return AgentOutput(
                success=False,
                data={},
                errors=["No design proposal or feature description provided"],
            )

        # Read files that need modification
        files_to_read = self._get_files_to_read(design_proposal, context_report)
        file_contents = self._read_files(files_to_read)

        # Build the prompt
        user_message = self._build_prompt(
            feature_spec=feature_spec,
            context_report=context_report,
            design_proposal=design_proposal,
            feature_description=feature_description,
            file_contents=file_contents,
            safety_invariants=safety_invariants,
            workplan=workplan,
            tech_stack=tech_stack,
        )

        try:
            # Call LLM with higher temperature for code generation
            # CRITICAL: max_tokens must be high enough for full code output
            # Code generation can be very long - use 32000 to avoid truncation
            response = self._chat(user_message, temperature=0.3, max_tokens=32000)

            # Debug: log raw response length and preview
            logger.info(
                "LLM response: %d chars, starts with: %s",
                len(response),
                response[:200].replace('\n', '\\n') if response else "(empty)"
            )
            # Check for truncation indicator (JSON should end with })
            if response and not response.rstrip().endswith('}'):
                logger.warning(
                    "IMPL: Response may be TRUNCATED - doesn't end with }. Ends with: %s",
                    response[-50:].replace('\n', '\\n') if len(response) > 50 else response
                )

            # Parse the response
            impl_data = self._parse_response(response)
            file_changes_count = len(impl_data.get("file_changes", []))
            logger.info(
                "IMPL: Parsed file_changes count: %d",
                file_changes_count
            )
            for i, fc in enumerate(impl_data.get("file_changes", [])[:3]):
                logger.info(
                    "IMPL: file_change[%d]: path=%s, new_code_len=%d",
                    i, fc.get("path"), len(fc.get("new_code", ""))
                )

            # If no files parsed, log for debugging
            if file_changes_count == 0:
                logger.error(
                    "IMPL: ZERO FILES PARSED! Full LLM response (%d chars):\n%s",
                    len(response),
                    response[:5000] if response else "(empty)"
                )

            # Fix missing/invalid paths using design proposal as fallback
            impl_data = self._fix_file_paths(impl_data, design_proposal)

            # Generate diffs from file changes
            change_set = self._create_change_set(impl_data, feature_id)
            impl_notes = self._create_implementation_notes(impl_data, feature_id)

            return AgentOutput(
                success=True,
                data={
                    "change_set": change_set.model_dump(),
                    "implementation_notes": impl_notes.model_dump(),
                    "raw_changes": impl_data.get("file_changes", []),
                },
                artifacts=["diff.patch", "implementation_notes.md"],
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={"raw_response": response if "response" in dir() else ""},
                errors=[f"ImplementationAgent error: {str(e)}"],
            )

    def _extract_feature_spec(self, context: dict) -> FeatureSpec | None:
        """Extract FeatureSpec from context."""
        data = context.get("feature_spec", {})
        if isinstance(data, FeatureSpec):
            return data
        if isinstance(data, dict) and data:
            try:
                return FeatureSpec(**data)
            except Exception:
                pass
        return None

    def _extract_context_report(self, context: dict) -> ContextReport | None:
        """Extract ContextReport from context."""
        data = context.get("context_report", {})
        if isinstance(data, ContextReport):
            return data
        if isinstance(data, dict) and data:
            try:
                return ContextReport(**data)
            except Exception:
                pass
        return None

    def _extract_design_proposal(self, context: dict) -> DesignProposal | None:
        """Extract DesignProposal from context."""
        data = context.get("design_proposal", {})
        if isinstance(data, DesignProposal):
            return data
        if isinstance(data, dict) and data:
            try:
                return DesignProposal(**data)
            except Exception:
                pass
        return None

    def _get_files_to_read(
        self,
        design_proposal: DesignProposal | None,
        context_report: ContextReport | None,
    ) -> list[str]:
        """Get list of files that need to be read for context."""
        files = set()

        # Files from design proposal
        if design_proposal:
            for fc in design_proposal.file_changes:
                if fc.change_type.value in ("modify", "delete", "rename"):
                    files.add(fc.path)

        # Relevant files from context
        if context_report and context_report.relevant_files:
            for rf in context_report.relevant_files[:5]:  # Limit to top 5
                files.add(rf.path)

        return list(files)

    def _read_files(self, file_paths: list[str]) -> dict[str, str]:
        """Read file contents for the given paths."""
        contents = {}
        for path in file_paths:
            result = self.fs_tool.execute("read", path=path)
            if result.success and result.output:
                # output is the file content string
                contents[path] = result.output if isinstance(result.output, str) else str(result.output)
        return contents

    def _build_prompt(
        self,
        feature_spec: FeatureSpec | None,
        context_report: ContextReport | None,
        design_proposal: DesignProposal | None,
        feature_description: str,
        file_contents: dict[str, str],
        safety_invariants: str | None = None,
        workplan: dict | None = None,
        tech_stack: list[str] | None = None,
    ) -> str:
        """Build the user prompt for code generation."""
        parts = []

        # Inject stack profile guidance at the start (sets expert context)
        if get_profile_guidance:
            profile_guidance = get_profile_guidance(
                tech_stack=tech_stack,
                prompt=feature_description,
            )
            if profile_guidance:
                parts.append(profile_guidance)
                parts.append("")  # Blank line separator
                logger.info("Injected stack profile guidance for tech_stack=%s", tech_stack)

        # Feature specification
        parts.append("## Feature Specification")
        if feature_spec:
            parts.append(f"**Title:** {feature_spec.title}")
            parts.append(f"**Description:** {feature_spec.original_description}")
            if feature_spec.acceptance_criteria:
                parts.append("\n**Acceptance Criteria:**")
                for ac in feature_spec.acceptance_criteria:
                    parts.append(f"- [{ac.id}] {ac.description}")
        else:
            parts.append(f"**Description:** {feature_description}")

        # Design proposal
        if design_proposal:
            parts.append("\n## Design Proposal")
            parts.append(f"**Approach:** {design_proposal.chosen_approach}")
            parts.append(f"**Summary:** {design_proposal.summary}")

            if design_proposal.file_changes:
                parts.append("\n**Planned File Changes:**")
                for fc in design_proposal.file_changes:
                    parts.append(f"- {fc.path} ({fc.change_type.value}): {fc.description}")

            if design_proposal.patterns_to_follow:
                parts.append("\n**Patterns to Follow:**")
                for p in design_proposal.patterns_to_follow:
                    parts.append(f"- {p}")

            if design_proposal.data_flow:
                parts.append("\n**Data Flow:**")
                for step in sorted(design_proposal.data_flow, key=lambda x: x.order):
                    parts.append(f"{step.order}. {step.component}: {step.action}")

        # Codebase context
        if context_report:
            parts.append("\n## Codebase Context")
            if context_report.repo_structure:
                rs = context_report.repo_structure
                parts.append(f"**Framework:** {rs.framework or 'Unknown'}")
                parts.append(f"**Language:** {rs.language or 'Unknown'}")

            if context_report.existing_patterns:
                parts.append("\n**Existing Patterns:**")
                for p in context_report.existing_patterns[:3]:
                    parts.append(f"- {p.name}: {p.description}")

        # Current file contents
        if file_contents:
            parts.append("\n## Current File Contents")
            for path, content in file_contents.items():
                # Truncate large files
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                parts.append(f"\n### {path}")
                parts.append(f"```\n{content}\n```")

        # Safety invariants - CRITICAL rules that must be followed
        if safety_invariants:
            parts.append("\n## CRITICAL SAFETY REQUIREMENTS")
            parts.append("The following rules are MANDATORY. Violating these will cause security vulnerabilities or bugs:")
            parts.append(safety_invariants)
            parts.append("\nYou MUST follow all MUST requirements above. Do NOT use any anti-patterns listed.")

        # Workplan tasks - ALL must be implemented
        if workplan and workplan.get("tasks"):
            tasks = workplan["tasks"]
            parts.append("\n## IMPLEMENTATION CHECKLIST (ALL REQUIRED)")
            parts.append(f"You MUST implement ALL {len(tasks)} tasks below. Missing any task is a failure.\n")
            for i, task in enumerate(tasks, 1):
                task_id = task.get("id", f"TASK-{i:03d}")
                title = task.get("title", "Untitled")
                desc = task.get("description", "")
                notes = task.get("implementation_notes", "")
                parts.append(f"### {task_id}: {title}")
                if desc:
                    parts.append(f"{desc}")
                if notes:
                    parts.append(f"*Implementation hint:* {notes}")
                if task.get("target_files"):
                    parts.append(f"*Files:* {', '.join(task['target_files'])}")
                parts.append("")

        parts.append("\n## Task")
        parts.append("Generate code implementation as JSON. Include file_changes with new_code containing actual Python code.")
        if workplan and workplan.get("tasks"):
            parts.append(f"CRITICAL: Generate code for ALL {len(workplan['tasks'])} tasks in the checklist above.")
        parts.append("Your response must be a valid JSON object starting with { and ending with }.")
        parts.append("Do NOT include any markdown, explanation, or text outside the JSON.")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response JSON with repair, falling back to markdown extraction."""
        import json
        import re

        # First try JSON parsing
        result = parse_llm_json(response, default=None)
        logger.info(
            "IMPL: parse_llm_json result: has_result=%s, has_file_changes=%s",
            result is not None,
            result.get("file_changes") is not None if result else False
        )

        if result is not None and result.get("file_changes"):
            # Handle dict format: {"file_changes": {"path": "code"}}
            fc = result.get("file_changes")
            if isinstance(fc, dict) and not isinstance(fc, list):
                # Convert dict format to list format
                file_changes = []
                for path, code in fc.items():
                    if isinstance(code, str):
                        file_changes.append({
                            "path": path,
                            "operation": "modify",
                            "language": self._detect_language(path),
                            "original_code": "",
                            "new_code": code,
                            "description": f"Update {path}",
                        })
                result["file_changes"] = file_changes
            else:
                # Normalize field names (some models use file_path instead of path)
                normalized = []
                for change in fc:
                    if isinstance(change, dict):
                        norm = dict(change)
                        # file_path -> path
                        if "file_path" in norm and "path" not in norm:
                            norm["path"] = norm.pop("file_path")
                        # code -> new_code
                        if "code" in norm and "new_code" not in norm:
                            norm["new_code"] = norm.pop("code")
                        # Strip leading slashes from paths (prevent /app -> absolute path)
                        if "path" in norm and isinstance(norm["path"], str):
                            norm["path"] = norm["path"].lstrip("/")
                        normalized.append(norm)
                result["file_changes"] = normalized
            return result

        # Fallback: extract code from markdown blocks
        # This handles cases where the LLM outputs explanatory markdown with code blocks
        logger.info("IMPL: JSON parsing failed or empty, falling back to markdown extraction")
        file_changes = []
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        logger.info("IMPL: Found %d markdown code blocks", len(code_blocks))

        # Try to detect file paths from the response
        file_refs = re.findall(r'[`"]?([a-zA-Z_/]+\.py)[`"]?', response)

        for i, code in enumerate(code_blocks):
            code = code.strip()
            if not code or len(code) < 20:
                continue

            # Try to match code block to a file path mentioned nearby
            file_path = file_refs[i] if i < len(file_refs) else f"generated_code_{i}.py"

            # Determine if this looks like a new file or modification
            operation = "create"
            if "def " in code and not code.startswith("def "):
                # Likely a snippet/modification
                operation = "modify"

            file_changes.append({
                "path": file_path,
                "operation": operation,
                "language": "python",
                "original_code": "",
                "new_code": code,
                "description": f"Extracted from markdown block {i+1}",
            })

        if file_changes:
            return {
                "file_changes": file_changes,
                "new_functions": [],
                "new_classes": [],
                "modified_functions": [],
                "dependencies_added": [],
                "config_changes": [],
                "migration_required": False,
                "migration_notes": None,
                "tech_debt_items": [],
                "todos": [],
                "rollback_instructions": None,
                "summary": "Implementation extracted from markdown response",
            }

        # If we still have nothing, return empty
        logger.warning(
            "Could not parse implementation response. Raw response preview: %s",
            response[:500].replace('\n', '\\n') if response else "(empty)"
        )
        return {
            "file_changes": [],
            "new_functions": [],
            "new_classes": [],
            "modified_functions": [],
            "dependencies_added": [],
            "config_changes": [],
            "migration_required": False,
            "migration_notes": None,
            "tech_debt_items": [],
            "todos": [],
            "rollback_instructions": None,
            "summary": "No implementation generated",
        }

    def _fix_file_paths(
        self,
        impl_data: dict,
        design_proposal: DesignProposal | None,
    ) -> dict:
        """Fix missing or invalid file paths using design proposal as fallback.

        Args:
            impl_data: Parsed implementation data
            design_proposal: Design proposal with expected file paths

        Returns:
            Updated impl_data with fixed paths
        """
        file_changes = impl_data.get("file_changes", [])
        if not file_changes:
            return impl_data

        # Get expected paths from design proposal
        expected_paths = []
        if design_proposal and design_proposal.file_changes:
            expected_paths = [fc.path for fc in design_proposal.file_changes]

        # Invalid path patterns
        invalid_paths = {"unknown", "file.py", "main.py", "code.py", ""}

        for i, fc in enumerate(file_changes):
            path = fc.get("path", "")

            # Check if path is invalid
            if not path or path in invalid_paths or not "/" in path:
                # Try to use expected path from design proposal
                if i < len(expected_paths):
                    fc["path"] = expected_paths[i]
                    logger.info("Fixed invalid path '%s' -> '%s'", path, expected_paths[i])
                elif expected_paths:
                    # Use first expected path as fallback
                    fc["path"] = expected_paths[0]
                    logger.info("Fixed invalid path '%s' -> '%s' (first expected)", path, expected_paths[0])
                else:
                    # Generate a reasonable default based on code content
                    code = fc.get("new_code", "")
                    fc["path"] = self._infer_path_from_code(code)
                    logger.info("Inferred path '%s' from code content", fc["path"])

        impl_data["file_changes"] = file_changes
        return impl_data

    def _infer_path_from_code(self, code: str) -> str:
        """Infer a file path from code content.

        Args:
            code: Python code content

        Returns:
            Inferred file path
        """
        code_lower = code.lower()

        # Check for common patterns
        if "fastapi" in code_lower or "@app." in code:
            return "app/main.py"
        if "webhook" in code_lower:
            return "app/webhooks.py"
        if "stripe" in code_lower:
            return "app/services/stripe_handler.py"
        if "slack" in code_lower:
            return "app/services/slack_notifier.py"
        if "class " in code and "model" in code_lower:
            return "app/models.py"
        if "def test_" in code:
            return "tests/test_main.py"

        # Default fallback
        return "app/generated_code.py"

    def _create_change_set(self, data: dict, feature_id: str) -> ChangeSet:
        """Create ChangeSet from parsed implementation data."""
        file_changes = data.get("file_changes", [])
        diffs = []
        combined_patch_parts = []
        total_insertions = 0
        total_deletions = 0

        for fc in file_changes:
            path = fc.get("path", "unknown")
            operation = fc.get("operation", "modify")
            original_code = fc.get("original_code", "")
            new_code = fc.get("new_code", "")
            language = fc.get("language", self._detect_language(path))

            # Generate unified diff
            diff_content = self._generate_diff(
                path=path,
                original=original_code,
                modified=new_code,
                operation=operation,
            )

            diffs.append(
                FileDiff(
                    path=path,
                    operation=operation,
                    diff_content=diff_content,
                    language=language,
                )
            )

            combined_patch_parts.append(diff_content)

            # Count changes (rough estimate)
            for line in diff_content.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    total_insertions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    total_deletions += 1

        combined_patch = "\n".join(combined_patch_parts)

        return ChangeSet(
            feature_id=feature_id,
            diffs=diffs,
            diff_type=DiffType.UNIFIED,
            files_changed=len(diffs),
            insertions=total_insertions,
            deletions=total_deletions,
            combined_patch=combined_patch,
        )

    def _generate_diff(
        self,
        path: str,
        original: str,
        modified: str,
        operation: str,
    ) -> str:
        """Generate unified diff format."""
        import difflib

        if operation == "create":
            # New file - all lines are additions
            lines = [
                f"--- /dev/null",
                f"+++ b/{path}",
                "@@ -0,0 +1,{} @@".format(len(modified.splitlines())),
            ]
            for line in modified.splitlines():
                lines.append(f"+{line}")
            return "\n".join(lines)

        elif operation == "delete":
            # Deleted file - all lines are deletions
            lines = [
                f"--- a/{path}",
                f"+++ /dev/null",
                "@@ -1,{} +0,0 @@".format(len(original.splitlines())),
            ]
            for line in original.splitlines():
                lines.append(f"-{line}")
            return "\n".join(lines)

        else:
            # Modify - use difflib for unified diff
            original_lines = original.splitlines(keepends=True)
            modified_lines = modified.splitlines(keepends=True)

            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
            return "".join(diff)

    def _detect_language(self, path: str) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".css": "css",
            ".html": "html",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".sql": "sql",
        }
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang
        return "text"

    def _create_implementation_notes(self, data: dict, feature_id: str) -> ImplementationNotes:
        """Create ImplementationNotes from parsed data."""
        # Parse new functions
        new_functions = []
        for func in data.get("new_functions", []):
            if isinstance(func, dict):
                new_functions.append(
                    NewFunction(
                        name=func.get("name", "unknown"),
                        file_path=func.get("file_path", "unknown"),
                        signature=func.get("signature", ""),
                        purpose=func.get("purpose", ""),
                        parameters=func.get("parameters", {}),
                        returns=func.get("returns"),
                    )
                )

        # Parse new classes
        new_classes = []
        for cls in data.get("new_classes", []):
            if isinstance(cls, dict):
                new_classes.append(
                    NewClass(
                        name=cls.get("name", "unknown"),
                        file_path=cls.get("file_path", "unknown"),
                        purpose=cls.get("purpose", ""),
                        key_methods=cls.get("key_methods", []),
                        inherits_from=cls.get("inherits_from", []),
                    )
                )

        # Parse tech debt items
        tech_debt = []
        for item in data.get("tech_debt_items", []):
            if isinstance(item, dict):
                tech_debt.append(
                    TechDebtItem(
                        description=item.get("description", ""),
                        location=item.get("location", "unknown"),
                        severity=item.get("severity", "medium"),
                        suggested_fix=item.get("suggested_fix"),
                    )
                )

        # Parse modified_functions - ensure strings
        modified_functions = []
        for func in data.get("modified_functions", []):
            if isinstance(func, str):
                modified_functions.append(func)
            elif isinstance(func, dict):
                # Extract name if it's a dict
                modified_functions.append(func.get("name", str(func)))

        # Parse dependencies - ensure strings
        dependencies = []
        for dep in data.get("dependencies_added", []):
            if isinstance(dep, str):
                dependencies.append(dep)
            elif isinstance(dep, dict):
                dependencies.append(dep.get("name", str(dep)))

        # Parse config_changes - ensure strings
        config_changes = []
        for change in data.get("config_changes", []):
            if isinstance(change, str):
                config_changes.append(change)
            elif isinstance(change, dict):
                config_changes.append(change.get("description", str(change)))

        # Parse todos - ensure strings
        todos = []
        for todo in data.get("todos", []):
            if isinstance(todo, str):
                todos.append(todo)
            elif isinstance(todo, dict):
                todos.append(todo.get("description", str(todo)))

        return ImplementationNotes(
            feature_id=feature_id,
            summary=data.get("summary", "Implementation completed"),
            new_functions=new_functions,
            new_classes=new_classes,
            modified_functions=modified_functions,
            dependencies_added=dependencies,
            config_changes=config_changes,
            migration_required=data.get("migration_required", False),
            migration_notes=data.get("migration_notes"),
            tech_debt_items=tech_debt,
            todos=todos,
            rollback_instructions=data.get("rollback_instructions"),
        )

    def generate_patch_file(self, change_set: ChangeSet) -> str:
        """Generate a complete patch file that can be applied with git apply."""
        return change_set.combined_patch

    def generate_notes_markdown(self, notes: ImplementationNotes) -> str:
        """Generate human-readable markdown from implementation notes."""
        lines = [
            f"# Implementation Notes: {notes.feature_id}",
            "",
            "## Summary",
            "",
            notes.summary,
            "",
        ]

        # New functions
        if notes.new_functions:
            lines.extend(["## New Functions", ""])
            for func in notes.new_functions:
                lines.append(f"### `{func.name}`")
                lines.append(f"**File:** `{func.file_path}`")
                lines.append(f"```\n{func.signature}\n```")
                lines.append(f"**Purpose:** {func.purpose}")
                if func.parameters:
                    lines.append("\n**Parameters:**")
                    for param, desc in func.parameters.items():
                        lines.append(f"- `{param}`: {desc}")
                if func.returns:
                    lines.append(f"\n**Returns:** {func.returns}")
                lines.append("")

        # New classes
        if notes.new_classes:
            lines.extend(["## New Classes", ""])
            for cls in notes.new_classes:
                lines.append(f"### `{cls.name}`")
                lines.append(f"**File:** `{cls.file_path}`")
                lines.append(f"**Purpose:** {cls.purpose}")
                if cls.inherits_from:
                    lines.append(f"**Inherits from:** {', '.join(cls.inherits_from)}")
                if cls.key_methods:
                    lines.append(f"**Key methods:** {', '.join(cls.key_methods)}")
                lines.append("")

        # Modified functions
        if notes.modified_functions:
            lines.extend(["## Modified Functions", ""])
            for func in notes.modified_functions:
                lines.append(f"- `{func}`")
            lines.append("")

        # Dependencies
        if notes.dependencies_added:
            lines.extend(["## Dependencies Added", ""])
            for dep in notes.dependencies_added:
                lines.append(f"- `{dep}`")
            lines.append("")

        # Config changes
        if notes.config_changes:
            lines.extend(["## Configuration Changes", ""])
            for change in notes.config_changes:
                lines.append(f"- {change}")
            lines.append("")

        # Migration
        if notes.migration_required:
            lines.extend([
                "## Migration Required",
                "",
                notes.migration_notes or "See migration instructions.",
                "",
            ])

        # Tech debt
        if notes.tech_debt_items:
            lines.extend(["## Technical Debt", ""])
            for item in notes.tech_debt_items:
                lines.append(f"### [{item.severity.upper()}] {item.location}")
                lines.append(item.description)
                if item.suggested_fix:
                    lines.append(f"\n*Suggested fix:* {item.suggested_fix}")
                lines.append("")

        # TODOs
        if notes.todos:
            lines.extend(["## TODOs", ""])
            for todo in notes.todos:
                lines.append(f"- [ ] {todo}")
            lines.append("")

        # Rollback
        if notes.rollback_instructions:
            lines.extend([
                "## Rollback Instructions",
                "",
                notes.rollback_instructions,
                "",
            ])

        return "\n".join(lines)
