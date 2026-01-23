"""Shared prompt components for ACF agents.

This module contains Claude Code-style sharp instructions about how to think
about code quality, not just output format.
"""

from .code_principles import (
    CODE_PRINCIPLES,
    DESIGN_PRINCIPLES,
    IMPLEMENTATION_PRINCIPLES,
    REVIEW_PRINCIPLES,
    FIX_PRINCIPLES,
)

__all__ = [
    "CODE_PRINCIPLES",
    "DESIGN_PRINCIPLES",
    "IMPLEMENTATION_PRINCIPLES",
    "REVIEW_PRINCIPLES",
    "FIX_PRINCIPLES",
]
