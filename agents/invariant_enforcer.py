"""Invariant Enforcer Agent - Post-generation validation.

Scans generated code for anti-pattern violations defined in invariants.
Triggers fix stage with specific feedback when violations are found.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import AgentInput, AgentOutput, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class AntiPattern:
    """An anti-pattern to detect in code."""

    pattern: str  # Regex pattern
    message: str
    fix_hint: str
    severity: str = "error"  # "error", "warning", "info"
    file_patterns: list[str] = field(default_factory=list)  # e.g., ["*.py", "*.js"]

    def matches(self, code: str) -> list[re.Match]:
        """Find all matches of this anti-pattern in code."""
        try:
            return list(re.finditer(self.pattern, code, re.MULTILINE))
        except re.error:
            logger.warning("Invalid regex pattern: %s", self.pattern)
            return []


@dataclass
class Invariant:
    """An invariant with anti-patterns to enforce."""

    topic: str
    category: str
    triggers: list[str]
    must: list[str]
    should: list[str]
    anti_patterns: list[AntiPattern]

    @classmethod
    def from_json(cls, data: dict) -> Invariant:
        """Load invariant from JSON dict."""
        anti_patterns = []
        for ap in data.get("anti_patterns", []):
            anti_patterns.append(AntiPattern(
                pattern=ap.get("pattern", ""),
                message=ap.get("message", ""),
                fix_hint=ap.get("fix_hint", ""),
                severity=ap.get("severity", "error"),
                file_patterns=ap.get("file_patterns", []),
            ))

        return cls(
            topic=data.get("topic", ""),
            category=data.get("category", ""),
            triggers=data.get("triggers", []),
            must=data.get("must", []),
            should=data.get("should", []),
            anti_patterns=anti_patterns,
        )


@dataclass
class Violation:
    """A detected invariant violation."""

    invariant_topic: str
    anti_pattern_message: str
    fix_hint: str
    severity: str
    file_path: str
    line_number: int | None
    matched_text: str
    context: str  # Surrounding code

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.invariant_topic,
            "message": self.anti_pattern_message,
            "fix": self.fix_hint,
            "severity": self.severity,
            "file": self.file_path,
            "line": self.line_number,
            "matched": self.matched_text[:100],  # Truncate
        }


class InvariantEnforcerAgent(BaseAgent):
    """Agent that enforces invariants by detecting anti-patterns.

    Scans generated code for violations of safety rules defined in
    invariant files. Returns detailed violation reports with fix hints.

    Example:
        agent = InvariantEnforcerAgent(
            llm=llm,
            invariants_dir=Path("invariants"),
        )
        result = agent.run(AgentInput(
            context={"files": [...], "query": "stripe webhook"},
        ))
        if not result.success:
            # Violations found - trigger fix stage
            violations = result.data["violations"]
    """

    def __init__(
        self,
        llm: Any,
        invariants_dir: Path | None = None,
        name: str = "invariant-enforcer",
        **kwargs: Any,
    ) -> None:
        """Initialize enforcer agent.

        Args:
            llm: LLM backend (may not be used if only pattern matching)
            invariants_dir: Path to invariants directory
            name: Agent name
            **kwargs: Additional BaseAgent args
        """
        super().__init__(llm=llm, name=name, **kwargs)
        self.invariants: list[Invariant] = []

        if invariants_dir:
            self.load_invariants(invariants_dir)

    def default_system_prompt(self) -> str:
        return """You are a code review assistant that checks for invariant violations.
Analyze the code and identify any patterns that violate the provided invariants."""

    def load_invariants(self, invariants_dir: Path) -> int:
        """Load invariants from directory.

        Args:
            invariants_dir: Path to invariants directory

        Returns:
            Number of invariants loaded
        """
        self.invariants.clear()

        if not invariants_dir.exists():
            logger.warning("Invariants directory not found: %s", invariants_dir)
            return 0

        for json_file in invariants_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue

            try:
                data = json.loads(json_file.read_text())
                invariant = Invariant.from_json(data)
                if invariant.anti_patterns:  # Only load if has patterns to check
                    self.invariants.append(invariant)
                    logger.debug("Loaded invariant: %s", invariant.topic)
            except Exception as e:
                logger.warning("Failed to load invariant %s: %s", json_file, e)

        logger.info("Loaded %d invariants with anti-patterns", len(self.invariants))
        return len(self.invariants)

    def get_active_invariants(self, query: str) -> list[Invariant]:
        """Get invariants that apply to the given query.

        Args:
            query: Feature description or task context

        Returns:
            List of matching invariants
        """
        query_lower = query.lower()
        active = []

        for invariant in self.invariants:
            for trigger in invariant.triggers:
                if trigger.lower() in query_lower:
                    active.append(invariant)
                    break

        return active

    def check_file(
        self,
        file_path: str,
        content: str,
        invariants: list[Invariant],
    ) -> list[Violation]:
        """Check a single file for violations.

        Args:
            file_path: Path to the file
            content: File content
            invariants: Invariants to check

        Returns:
            List of violations found
        """
        violations = []
        lines = content.split("\n")

        for invariant in invariants:
            for anti_pattern in invariant.anti_patterns:
                # Check file pattern filter
                if anti_pattern.file_patterns:
                    matches_filter = any(
                        self._matches_glob(file_path, pattern)
                        for pattern in anti_pattern.file_patterns
                    )
                    if not matches_filter:
                        continue

                # Find pattern matches
                for match in anti_pattern.matches(content):
                    # Calculate line number
                    start_pos = match.start()
                    line_number = content[:start_pos].count("\n") + 1

                    # Get context (surrounding lines)
                    context_start = max(0, line_number - 3)
                    context_end = min(len(lines), line_number + 2)
                    context = "\n".join(lines[context_start:context_end])

                    violations.append(Violation(
                        invariant_topic=invariant.topic,
                        anti_pattern_message=anti_pattern.message,
                        fix_hint=anti_pattern.fix_hint,
                        severity=anti_pattern.severity,
                        file_path=file_path,
                        line_number=line_number,
                        matched_text=match.group(),
                        context=context,
                    ))

        return violations

    def _matches_glob(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern)

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Check generated code for invariant violations.

        Args:
            input_data: Must contain "files" list and optionally "query"

        Returns:
            AgentOutput with violations if found
        """
        start_time = self._log_run_start(input_data)
        context = input_data.safe_context()

        files = context.get("files", [])
        query = context.get("query", "") or context.get("prompt", "")

        if not files:
            return self._create_output(
                success=True,
                data={"message": "No files to check", "violations": []},
                start_time=start_time,
            )

        # Get active invariants based on query
        active_invariants = self.get_active_invariants(query)
        if not active_invariants:
            # Check all invariants if no specific ones match
            active_invariants = self.invariants

        if not active_invariants:
            return self._create_output(
                success=True,
                data={"message": "No invariants loaded", "violations": []},
                start_time=start_time,
            )

        # Check each file
        all_violations: list[Violation] = []
        for file_info in files:
            file_path = file_info.get("path", "unknown")
            content = file_info.get("content", "")

            if not content:
                continue

            file_violations = self.check_file(file_path, content, active_invariants)
            all_violations.extend(file_violations)

        # Group by severity
        errors = [v for v in all_violations if v.severity == "error"]
        warnings = [v for v in all_violations if v.severity == "warning"]

        # Build result
        if errors:
            # Violations found - suggest fixes
            violation_dicts = [v.to_dict() for v in all_violations]
            fix_suggestions = self._build_fix_suggestions(errors)

            output = self._create_output(
                success=False,
                data={
                    "violations": violation_dicts,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "fix_suggestions": fix_suggestions,
                    "topics_violated": list(set(v.invariant_topic for v in errors)),
                },
                errors=[f"{len(errors)} invariant violation(s) found"],
                start_time=start_time,
            )
        elif warnings:
            # Only warnings - success but with notes
            output = self._create_output(
                success=True,
                data={
                    "violations": [v.to_dict() for v in warnings],
                    "error_count": 0,
                    "warning_count": len(warnings),
                    "message": f"{len(warnings)} warning(s) found",
                },
                start_time=start_time,
            )
        else:
            output = self._create_output(
                success=True,
                data={
                    "violations": [],
                    "error_count": 0,
                    "warning_count": 0,
                    "message": "No violations found",
                },
                artifacts=["invariant_check"],
                start_time=start_time,
            )

        self._log_run_end(output, start_time)
        return output

    def _build_fix_suggestions(self, violations: list[Violation]) -> str:
        """Build fix suggestion text from violations.

        Args:
            violations: List of error violations

        Returns:
            Formatted fix suggestions
        """
        suggestions = []
        seen_topics = set()

        for v in violations:
            if v.invariant_topic in seen_topics:
                continue
            seen_topics.add(v.invariant_topic)

            suggestions.append(f"""
### {v.invariant_topic}

**Issue in {v.file_path}** (line {v.line_number}):
{v.anti_pattern_message}

**Problematic code:**
```
{v.matched_text[:200]}
```

**Fix:**
{v.fix_hint}
""")

        return "\n".join(suggestions)

    def format_report(self, output: AgentOutput) -> str:
        """Format violations as human-readable report.

        Args:
            output: Agent output with violations

        Returns:
            Formatted report string
        """
        violations = output.data.get("violations", [])
        if not violations:
            return "No invariant violations found."

        lines = [
            "=" * 60,
            "INVARIANT VIOLATION REPORT",
            "=" * 60,
            "",
            f"Errors: {output.data.get('error_count', 0)}",
            f"Warnings: {output.data.get('warning_count', 0)}",
            "",
        ]

        for v in violations:
            severity_icon = "🔴" if v["severity"] == "error" else "🟡"
            lines.extend([
                f"{severity_icon} [{v['severity'].upper()}] {v['topic']}",
                f"   File: {v['file']}:{v['line']}",
                f"   {v['message']}",
                f"   Fix: {v['fix']}",
                "",
            ])

        return "\n".join(lines)
