"""Project scaffolding module for Coding Factory.

Creates new projects from templates with:
- Directory structure
- Git initialization
- Documentation
- Docker configuration
- CI/CD setup

Also supports:
- Custom templates (user-defined)
- Template marketplace (share and discover)
"""

from .generator import ProjectGenerator
from .templates import (
    TEMPLATES,
    BUILTIN_TEMPLATES,
    ProjectTemplate,
    get_template,
    list_templates,
    get_templates_dir,
    load_custom_templates,
    load_template_from_dir,
    save_template,
    delete_template,
)
from .marketplace import (
    MarketplaceClient,
    MarketplaceTemplate,
    MarketplaceSearchResult,
    MarketplaceError,
    LocalRegistry,
    DEFAULT_MARKETPLACE_URL,
)

__all__ = [
    # Generator
    "ProjectGenerator",
    # Templates
    "TEMPLATES",
    "BUILTIN_TEMPLATES",
    "ProjectTemplate",
    "get_template",
    "list_templates",
    "get_templates_dir",
    "load_custom_templates",
    "load_template_from_dir",
    "save_template",
    "delete_template",
    # Marketplace
    "MarketplaceClient",
    "MarketplaceTemplate",
    "MarketplaceSearchResult",
    "MarketplaceError",
    "LocalRegistry",
    "DEFAULT_MARKETPLACE_URL",
]
