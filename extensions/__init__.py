"""Extension system for ACF Local Edition.

This module provides the extension loading, management, and marketplace
integration for the ACF extension ecosystem.

Extensions are organized into four types:
- agents: Custom pipeline agents that hook into specific stages
- profiles: Technology-specific profiles with guidance and dependencies
- rag: Custom RAG retrieval strategies
- skills: Standalone code transformations (run via acf skill)

Extensions are stored in ~/.coding-factory/extensions/ by default.
"""

from extensions.loader import ExtensionLoader
from extensions.manifest import (
    ExtensionManifest,
    ExtensionType,
    HookPoint,
    License,
)
from extensions.installer import ExtensionInstaller, MarketplaceClient

__all__ = [
    "ExtensionLoader",
    "ExtensionInstaller",
    "ExtensionManifest",
    "ExtensionType",
    "HookPoint",
    "License",
    "MarketplaceClient",
]
