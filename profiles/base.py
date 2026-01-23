"""Base profile protocol and types.

Defines the interface that all profiles must implement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ProfileMetadata:
    """Metadata about a profile for UI/tooling."""

    name: str
    version: str
    description: str = ""
    author: str = "community"
    last_updated: str = ""
    icon: str = ""
    min_python_version: str | None = None
    min_node_version: str | None = None


@dataclass
class GuidanceSections:
    """Structured guidance that can be selectively included.

    Allows pipeline to include only relevant sections or reorder them.
    """

    imports: str = ""
    project_structure: str = ""
    patterns: str = ""
    data_fetching: str = ""
    state_management: str = ""
    error_handling: str = ""
    testing: str = ""
    common_mistakes: str = ""
    dependencies: str = ""
    custom: dict[str, str] = field(default_factory=dict)

    def to_full_guidance(self, sections: list[str] | None = None) -> str:
        """Combine sections into full guidance string.

        Args:
            sections: List of section names to include (None = all)

        Returns:
            Combined guidance string
        """
        all_sections = {
            "imports": self.imports,
            "project_structure": self.project_structure,
            "patterns": self.patterns,
            "data_fetching": self.data_fetching,
            "state_management": self.state_management,
            "error_handling": self.error_handling,
            "testing": self.testing,
            "common_mistakes": self.common_mistakes,
            "dependencies": self.dependencies,
            **self.custom,
        }

        if sections is None:
            sections = list(all_sections.keys())

        parts = []
        for section in sections:
            content = all_sections.get(section, "")
            if content.strip():
                parts.append(content)

        return "\n\n".join(parts)


@runtime_checkable
class ProfileProtocol(Protocol):
    """Protocol that all profiles must implement."""

    PROFILE_NAME: str
    PROFILE_VERSION: str
    TECHNOLOGIES: list[str]
    TRIGGER_KEYWORDS: list[str]
    DEPENDENCIES: list[str]

    def should_apply(self, tech_stack: list[str] | None, prompt: str) -> bool:
        """Determine if this profile applies."""
        ...

    def get_guidance(self) -> str:
        """Get the full guidance text."""
        ...

    def get_dependencies(self, features: list[str] | None = None) -> list[str]:
        """Get recommended dependencies."""
        ...


# Feature keywords for auto-detection from prompt
FEATURE_KEYWORDS: dict[str, list[str]] = {
    "auth": ["auth", "login", "logout", "jwt", "oauth", "session", "password", "signup", "signin"],
    "database": ["sql", "postgres", "mysql", "sqlite", "orm", "database", "db", "query", "migration"],
    "api": ["api", "rest", "graphql", "endpoint", "route", "http", "fetch"],
    "websocket": ["websocket", "socket", "realtime", "real-time", "live", "streaming"],
    "testing": ["test", "testing", "unittest", "pytest", "jest", "vitest", "coverage"],
    "docker": ["docker", "container", "kubernetes", "k8s", "compose"],
    "cache": ["cache", "redis", "memcached", "caching"],
    "queue": ["queue", "celery", "rabbitmq", "kafka", "message"],
    "storage": ["upload", "file", "storage", "s3", "blob", "image"],
    "email": ["email", "smtp", "sendgrid", "mailgun", "notification"],
    "payment": ["payment", "stripe", "paypal", "checkout", "billing"],
    "search": ["search", "elasticsearch", "algolia", "fulltext"],
}


def detect_features(prompt: str) -> list[str]:
    """Auto-detect features from prompt text.

    Args:
        prompt: User's feature description

    Returns:
        List of detected feature names
    """
    prompt_lower = prompt.lower()
    prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))

    detected = []
    for feature, keywords in FEATURE_KEYWORDS.items():
        # Check both word boundaries and substrings for flexibility
        if any(kw in prompt_words for kw in keywords):
            detected.append(feature)
        elif any(kw in prompt_lower for kw in keywords if len(kw) > 4):
            # For longer keywords, allow substring matching
            detected.append(feature)

    return list(set(detected))


def match_keywords(
    text: str,
    exact_keywords: list[str],
    substring_keywords: list[str] | None = None,
) -> bool:
    """Match keywords with word boundary awareness.

    Args:
        text: Text to search in
        exact_keywords: Keywords that must match as whole words
        substring_keywords: Keywords that can match as substrings

    Returns:
        True if any keyword matches
    """
    text_lower = text.lower()
    text_words = set(re.findall(r'\b\w+\b', text_lower))

    # Check exact matches (word boundaries)
    if any(kw.lower() in text_words for kw in exact_keywords):
        return True

    # Check substring matches (for compound words like "fastapi" in "using fastapi")
    if substring_keywords:
        if any(kw.lower() in text_lower for kw in substring_keywords):
            return True

    return False
