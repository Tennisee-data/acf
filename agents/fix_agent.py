"""Fix Agent for correcting code errors."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from llm_backend import LLMBackend
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent
from .prompts import CODE_PRINCIPLES, FIX_PRINCIPLES


SYSTEM_PROMPT = f"""{CODE_PRINCIPLES}

{FIX_PRINCIPLES}

## Role

You are a code fixer. You receive code with errors and must fix them.

Output ONLY a JSON object with the fixed code. No markdown, no explanation.

{{"fixes":[{{"path":"file.py","original_code":"broken code","fixed_code":"working code","error_fixed":"what was wrong"}}],"summary":"brief summary of fixes","remaining_issues":[]}}

IMPORTANT: Output starts with {{ and ends with }}. No other text allowed."""


@dataclass
class FixLoopState:
    """Track state of the fix loop to detect infinite loops."""

    iteration: int = 0
    max_iterations: int = 5
    error_hashes: list[str] = field(default_factory=list)
    error_counts: list[int] = field(default_factory=list)
    timeout_seconds: int = 300

    def should_continue(self, current_errors: list[str]) -> tuple[bool, str]:
        """Determine if the fix loop should continue.

        Returns:
            Tuple of (should_continue, reason_if_stopping)
        """
        # Check iteration limit
        if self.iteration >= self.max_iterations:
            return False, f"Max iterations ({self.max_iterations}) reached"

        # No errors = success
        if not current_errors:
            return False, "All errors fixed"

        # Check for repeated identical errors
        error_hash = self._hash_errors(current_errors)
        if error_hash in self.error_hashes:
            return False, "Same errors repeating - fix loop stuck"

        # Check if error count is increasing
        current_count = len(current_errors)
        if len(self.error_counts) >= 2:
            if current_count > self.error_counts[-1] > self.error_counts[-2]:
                return False, "Error count increasing - making things worse"

        # Record this iteration
        self.error_hashes.append(error_hash)
        self.error_counts.append(current_count)
        self.iteration += 1

        return True, ""

    def _hash_errors(self, errors: list[str]) -> str:
        """Create hash of error list for comparison."""
        combined = "\n".join(sorted(errors))
        return hashlib.md5(combined.encode()).hexdigest()


class FixAgent(BaseAgent):
    """Agent for fixing code errors.

    Takes code with validation errors and produces fixes.
    Designed to be used in a loop with validation checks.
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo_path: Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize FixAgent.

        Args:
            llm: LLM backend for inference
            repo_path: Path to repository for file operations
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)
        self.repo_path = repo_path or Path.cwd()

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate fixes for code errors.

        Args:
            input_data: Must contain 'code_files' and 'errors' in context

        Returns:
            AgentOutput with fixes
        """
        context = input_data.context

        code_files = context.get("code_files", {})
        errors = context.get("errors", [])
        error_output = context.get("error_output", "")
        invariant_violations = context.get("invariant_violations", [])

        if not errors and not error_output and not invariant_violations:
            return AgentOutput(
                success=True,
                data={"fixes": [], "summary": "No errors to fix"},
            )

        # Build prompt
        user_message = self._build_prompt(
            code_files, errors, error_output, invariant_violations
        )

        try:
            response = self._chat(user_message, temperature=0.2)
            fix_data = self._parse_response(response)

            return AgentOutput(
                success=True,
                data={
                    "fixes": fix_data.get("fixes", []),
                    "summary": fix_data.get("summary", ""),
                    "remaining_issues": fix_data.get("remaining_issues", []),
                },
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"FixAgent error: {str(e)}"],
            )

    def _build_prompt(
        self,
        code_files: dict[str, str],
        errors: list[str],
        error_output: str,
        invariant_violations: list[dict] | None = None,
    ) -> str:
        """Build the prompt for fixing errors."""
        parts = ["## Code Files with Errors\n"]

        for path, content in code_files.items():
            parts.append(f"### {path}")
            parts.append(f"```\n{content}\n```\n")

        parts.append("## Errors to Fix\n")

        # Show syntax errors (excluding invariant markers for cleaner display)
        syntax_errors = [e for e in errors if not e.startswith("[INVARIANT]")]
        if syntax_errors:
            parts.append("### Syntax/Type Errors")
            for error in syntax_errors:
                parts.append(f"- {error}")

        # Show invariant violations with fix hints
        if invariant_violations:
            parts.append("\n### Security/Best Practice Violations (CRITICAL)")
            parts.append("These are domain-specific violations that MUST be fixed:")
            for v in invariant_violations:
                parts.append(f"- **{v.get('message', 'Unknown violation')}**")
                if v.get("fix_hint"):
                    parts.append(f"  - FIX: {v['fix_hint']}")
                if v.get("line"):
                    parts.append(f"  - Line: {v['line']}")

        if error_output:
            parts.append(f"\n### Full Error Output\n```\n{error_output}\n```")

        parts.append("\n## Task")
        parts.append("Fix all the errors above. Return JSON with the fixed code.")
        if invariant_violations:
            parts.append("PAY SPECIAL ATTENTION to the security/best practice violations - follow the FIX hints exactly.")
        parts.append("Your response must be valid JSON starting with { and ending with }.")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response."""
        result = parse_llm_json(response, default=None)

        if result and "fixes" in result:
            return result

        return {
            "fixes": [],
            "summary": "Could not parse fix response",
            "remaining_issues": ["Failed to generate fixes"],
        }

    def apply_fixes(self, fixes: list[dict]) -> dict[str, str]:
        """Apply fixes to get updated code.

        Args:
            fixes: List of fix dicts with 'path' and 'fixed_code'

        Returns:
            Dict mapping file paths to fixed code
        """
        result = {}
        for fix in fixes:
            path = fix.get("path", "")
            fixed_code = fix.get("fixed_code", "")
            if path and fixed_code:
                result[path] = fixed_code
        return result
