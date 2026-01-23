"""Stack Profiles - Expert guidance for code generation.

Profiles provide technology-specific guidance that gets injected into
the implementation prompt. This helps the model generate correct code
with proper imports, patterns, and version-specific syntax.

Usage:
    from profiles import get_profile_guidance, get_manager

    # Simple API (backward compatible)
    guidance = get_profile_guidance(
        tech_stack=["python", "fastapi"],
        prompt="Create a REST API"
    )

    # Full API with ProfileManager
    manager = get_manager()
    result = manager.get_applicable(
        tech_stack=["python", "fastapi"],
        prompt="Create a REST API with authentication",
    )
    print(result.guidance)           # Merged guidance text
    print(result.dependencies)       # Merged dependencies
    print(result.detected_features)  # ['api', 'auth']
    print(result.profiles)           # ['fastapi']
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import (
    detect_features,
    GuidanceSections,
    match_keywords,
    ProfileMetadata,
    ProfileProtocol,
    FEATURE_KEYWORDS,
)
from .manager import (
    get_manager,
    ProfileConflictError,
    ProfileInfo,
    ProfileManager,
    ProfileResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    # Manager
    "ProfileManager",
    "ProfileResult",
    "ProfileInfo",
    "ProfileConflictError",
    "get_manager",
    # Base types
    "ProfileProtocol",
    "ProfileMetadata",
    "GuidanceSections",
    # Utilities
    "detect_features",
    "match_keywords",
    "FEATURE_KEYWORDS",
    # Convenience functions
    "get_profile_guidance",
    "get_profile_dependencies",
    "list_profiles",
]


def get_profile_guidance(
    tech_stack: list[str] | None = None,
    prompt: str = "",
    sections: list[str] | None = None,
) -> str | None:
    """Get combined guidance from all applicable profiles.

    This is a convenience function. For more control, use ProfileManager directly.

    Args:
        tech_stack: User-selected technologies (e.g., ["python", "fastapi"])
        prompt: The feature description
        sections: Optional list of guidance sections to include

    Returns:
        Combined guidance string, or None if no profiles apply
    """
    manager = get_manager()
    result = manager.get_applicable(
        tech_stack=tech_stack,
        prompt=prompt,
        sections=sections,
    )

    if not result:
        return None

    # Log applied profiles
    for profile_name in result.profiles:
        logger.info("Applied profile: %s", profile_name)

    # Log any warnings
    for warning in result.warnings:
        logger.warning(warning)

    return result.guidance


def get_profile_dependencies(
    tech_stack: list[str] | None = None,
    prompt: str = "",
    features: list[str] | None = None,
) -> list[str]:
    """Get recommended dependencies from applicable profiles.

    Args:
        tech_stack: User-selected technologies
        prompt: The feature description
        features: Optional features (e.g., ["database", "auth"])

    Returns:
        List of dependencies
    """
    manager = get_manager()
    result = manager.get_applicable(
        tech_stack=tech_stack,
        prompt=prompt,
        features=features,
    )
    return result.dependencies


def list_profiles() -> list[dict]:
    """List all available profiles with metadata.

    Returns:
        List of profile info dicts
    """
    manager = get_manager()
    return manager.list_profiles()
