"""Safety Patterns Agent - Injects domain-specific implementation invariants.

When dangerous patterns are detected (webhooks, crypto, auth), this agent
injects critical implementation rules that the LLM must follow.

This prevents hallucination of security-critical code by providing
explicit invariants learned from production experience.

Example:
    "Add Stripe webhook" triggers:
    - CRITICAL: Use raw bytes for signature verification
    - Use SDK's built-in construct_event method
    - Implement idempotency with event ID tracking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "safety_patterns.toml"


@dataclass
class SafetyPattern:
    """A detected safety pattern with its invariants."""

    name: str
    triggers_matched: list[str]
    force_premium: bool
    invariants: list[str]
    correct_pattern: str = ""


@dataclass
class SafetyReport:
    """Report of safety patterns detected in a feature."""

    feature_description: str
    patterns_detected: list[SafetyPattern] = field(default_factory=list)

    @property
    def has_safety_concerns(self) -> bool:
        """Return True if any safety patterns were detected."""
        return len(self.patterns_detected) > 0

    @property
    def requires_premium(self) -> bool:
        """Return True if any pattern requires premium model."""
        return any(p.force_premium for p in self.patterns_detected)

    @property
    def all_invariants(self) -> list[str]:
        """Get all invariants from all detected patterns."""
        invariants = []
        for pattern in self.patterns_detected:
            invariants.extend(pattern.invariants)
        return invariants

    def get_prompt_injection(self) -> str:
        """Generate text to inject into the implementation prompt.

        Returns:
            Formatted string with all safety invariants and patterns.
        """
        if not self.has_safety_concerns:
            return ""

        lines = [
            "",
            "=" * 60,
            "SAFETY-CRITICAL IMPLEMENTATION REQUIREMENTS",
            "=" * 60,
            "",
            "The following invariants MUST be followed. These are based on",
            "production security requirements, not suggestions.",
            "",
        ]

        for pattern in self.patterns_detected:
            lines.append(f"## {pattern.name.upper().replace('_', ' ')}")
            lines.append("")
            for inv in pattern.invariants:
                lines.append(f"  - {inv}")
            lines.append("")

            if pattern.correct_pattern.strip():
                lines.append("Correct implementation pattern:")
                lines.append("```python")
                lines.append(pattern.correct_pattern.strip())
                lines.append("```")
                lines.append("")

        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.has_safety_concerns:
            return "No safety-critical patterns detected."

        lines = [
            f"Detected {len(self.patterns_detected)} safety-critical pattern(s):",
            "",
        ]

        for pattern in self.patterns_detected:
            premium_flag = " [PREMIUM REQUIRED]" if pattern.force_premium else ""
            lines.append(f"  - {pattern.name}{premium_flag}")
            lines.append(f"    Triggers: {', '.join(pattern.triggers_matched)}")
            lines.append(f"    Invariants: {len(pattern.invariants)} rules")
            lines.append("")

        if self.requires_premium:
            lines.append("WARNING: Safety-critical code detected.")
            lines.append("Model will be upgraded to premium for this task.")

        return "\n".join(lines)


class SafetyPatternsAgent:
    """Agent that detects and injects safety-critical implementation patterns.

    Reads patterns from safety_patterns.toml and injects relevant
    invariants into the implementation prompt when triggered.
    """

    def __init__(self, config_path: Path | str | None = None):
        """Initialize with config file path.

        Args:
            config_path: Path to safety_patterns.toml. Uses default if None.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.patterns: dict[str, dict] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load safety patterns from TOML config."""
        if not self.config_path.exists():
            logger.warning(
                "Safety patterns config not found: %s",
                self.config_path
            )
            return

        try:
            # Use tomllib (Python 3.11+) or tomli
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib

            content = self.config_path.read_text(encoding="utf-8")
            self.patterns = tomllib.loads(content)

            logger.info(
                "Loaded %d safety patterns from %s",
                len(self.patterns),
                self.config_path
            )
        except Exception as e:
            logger.error("Failed to load safety patterns: %s", e)

    def analyze(
        self,
        feature_description: str,
        spec_content: str | None = None,
    ) -> SafetyReport:
        """Analyze feature for safety-critical patterns.

        Args:
            feature_description: The feature prompt/description
            spec_content: Optional parsed spec content

        Returns:
            SafetyReport with detected patterns and invariants
        """
        # Combine all text for analysis
        full_text = feature_description.lower()
        if spec_content:
            full_text += "\n" + spec_content.lower()

        detected = []

        for pattern_name, config in self.patterns.items():
            if not isinstance(config, dict):
                continue

            triggers = config.get("triggers", [])
            matched_triggers = []

            for trigger in triggers:
                # Use word boundary matching
                if re.search(rf'\b{re.escape(trigger.lower())}\b', full_text):
                    matched_triggers.append(trigger)

            if matched_triggers:
                detected.append(SafetyPattern(
                    name=pattern_name,
                    triggers_matched=matched_triggers,
                    force_premium=config.get("force_premium", False),
                    invariants=config.get("invariants", []),
                    correct_pattern=config.get("correct_pattern", ""),
                ))

        report = SafetyReport(
            feature_description=feature_description[:200],
            patterns_detected=detected,
        )

        if report.has_safety_concerns:
            logger.warning(
                "Safety patterns detected: %s (premium=%s)",
                [p.name for p in detected],
                report.requires_premium
            )

        return report

    def get_patterns_for_prompt(
        self,
        feature_description: str,
        spec_content: str | None = None,
    ) -> str:
        """Convenience method: analyze and return prompt injection.

        Args:
            feature_description: The feature prompt/description
            spec_content: Optional parsed spec content

        Returns:
            String to inject into implementation prompt (empty if no patterns)
        """
        report = self.analyze(feature_description, spec_content)
        return report.get_prompt_injection()
