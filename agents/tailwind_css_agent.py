"""Tailwind CSS Agent - Injects modern, responsive web design patterns.

When web/frontend keywords are detected (website, landing page, modern, etc.),
this agent injects Tailwind CSS best practices and responsive design patterns
into the implementation prompt.

This ensures all generated websites use:
- Tailwind CSS for styling
- Mobile-first responsive design
- Modern UI/UX patterns
- Dark mode support
- Accessible, semantic HTML

Example:
    "Build a modern website" triggers:
    - Use Tailwind CSS via CDN
    - Apply mobile-first responsive design
    - Use semantic HTML5 elements
    - Include dark mode support
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "tailwind_patterns.toml"


@dataclass
class TailwindPattern:
    """A detected Tailwind pattern with its invariants."""

    name: str
    triggers_matched: list[str]
    invariants: list[str]
    correct_pattern: str = ""


@dataclass
class TailwindReport:
    """Report of Tailwind patterns detected in a feature."""

    feature_description: str
    patterns_detected: list[TailwindPattern] = field(default_factory=list)

    @property
    def has_web_patterns(self) -> bool:
        """Return True if any web patterns were detected."""
        return len(self.patterns_detected) > 0

    @property
    def all_invariants(self) -> list[str]:
        """Get all invariants from all detected patterns."""
        invariants = []
        seen = set()
        for pattern in self.patterns_detected:
            for inv in pattern.invariants:
                if inv not in seen:
                    invariants.append(inv)
                    seen.add(inv)
        return invariants

    def get_prompt_injection(self) -> str:
        """Generate text to inject into the implementation prompt.

        Returns:
            Formatted string with all Tailwind CSS patterns and examples.
        """
        if not self.has_web_patterns:
            return ""

        lines = [
            "",
            "=" * 60,
            "TAILWIND CSS - MODERN WEB DESIGN REQUIREMENTS",
            "=" * 60,
            "",
            "This feature involves web/frontend development. The following",
            "Tailwind CSS patterns and best practices MUST be followed.",
            "",
        ]

        # Add all invariants first (deduplicated)
        lines.append("## CORE REQUIREMENTS")
        lines.append("")
        for inv in self.all_invariants:
            lines.append(f"  - {inv}")
        lines.append("")

        # Add pattern-specific examples
        for pattern in self.patterns_detected:
            if pattern.correct_pattern.strip():
                lines.append(f"## {pattern.name.upper().replace('_', ' ')} PATTERN")
                lines.append("")
                lines.append("Reference implementation:")
                lines.append("```html")
                lines.append(pattern.correct_pattern.strip())
                lines.append("```")
                lines.append("")

        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.has_web_patterns:
            return "No web/frontend patterns detected."

        lines = [
            f"Detected {len(self.patterns_detected)} web design pattern(s):",
            "",
        ]

        for pattern in self.patterns_detected:
            lines.append(f"  - {pattern.name}")
            lines.append(f"    Triggers: {', '.join(pattern.triggers_matched)}")
            lines.append(f"    Rules: {len(pattern.invariants)} design guidelines")
            lines.append("")

        lines.append("Tailwind CSS patterns will be injected into implementation.")

        return "\n".join(lines)


class TailwindCSSAgent:
    """Agent that detects web features and injects Tailwind CSS patterns.

    Reads patterns from tailwind_patterns.toml and injects relevant
    CSS patterns and responsive design guidelines when triggered.
    """

    def __init__(self, config_path: Path | str | None = None):
        """Initialize with config file path.

        Args:
            config_path: Path to tailwind_patterns.toml. Uses default if None.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.patterns: dict[str, dict] = {}
        self.load_error: str | None = None
        self._load_config()

    def _load_config(self) -> None:
        """Load Tailwind patterns from TOML config."""
        if not self.config_path.exists():
            logger.warning(
                "Tailwind patterns config not found: %s",
                self.config_path
            )
            self.load_error = f"Config file not found: {self.config_path}"
            return

        try:
            # Use tomllib (Python 3.11+) or tomli
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    logger.error("Neither tomllib nor tomli available for TOML parsing")
                    self.load_error = "TOML parser not available (install tomli)"
                    return

            content = self.config_path.read_text(encoding="utf-8")
            self.patterns = tomllib.loads(content)
            self.load_error = None

            logger.info(
                "Loaded %d Tailwind patterns from %s",
                len(self.patterns),
                self.config_path
            )
        except Exception as e:
            logger.error("Failed to load Tailwind patterns: %s", e)
            self.load_error = str(e)

    def analyze(
        self,
        feature_description: str,
        spec_content: str | None = None,
    ) -> TailwindReport:
        """Analyze feature for web/frontend patterns.

        Args:
            feature_description: The feature prompt/description
            spec_content: Optional parsed spec content

        Returns:
            TailwindReport with detected patterns and CSS guidelines
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
                # Use word boundary matching for accurate detection
                if re.search(rf'\b{re.escape(trigger.lower())}\b', full_text):
                    matched_triggers.append(trigger)

            if matched_triggers:
                detected.append(TailwindPattern(
                    name=pattern_name,
                    triggers_matched=matched_triggers,
                    invariants=config.get("invariants", []),
                    correct_pattern=config.get("correct_pattern", ""),
                ))

        report = TailwindReport(
            feature_description=feature_description[:200],
            patterns_detected=detected,
        )

        if report.has_web_patterns:
            logger.info(
                "Tailwind patterns detected: %s",
                [p.name for p in detected]
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

    def is_web_feature(
        self,
        feature_description: str,
        spec_content: str | None = None,
    ) -> bool:
        """Quick check if feature involves web/frontend development.

        Args:
            feature_description: The feature prompt/description
            spec_content: Optional parsed spec content

        Returns:
            True if web patterns are detected
        """
        report = self.analyze(feature_description, spec_content)
        return report.has_web_patterns
