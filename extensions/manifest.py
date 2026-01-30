"""Extension manifest schema for ACF extensions.

Defines the structure and validation for extension manifests (manifest.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ExtensionType(str, Enum):
    """Type of extension."""

    AGENT = "agent"
    PROFILE = "profile"
    RAG = "rag"
    SKILL = "skill"


class License(str, Enum):
    """Extension license type."""

    COMMERCIAL = "commercial"
    FREE = "free"
    MIT = "MIT"
    APACHE2 = "Apache-2.0"
    GPL3 = "GPL-3.0"


class HookPoint(str, Enum):
    """Pipeline hook points where agents can be inserted."""

    # Before stage hooks
    BEFORE_SPEC = "before:spec"
    BEFORE_CONTEXT = "before:context"
    BEFORE_DESIGN = "before:design"
    BEFORE_API_CONTRACT = "before:api_contract"
    BEFORE_IMPLEMENTATION = "before:implementation"
    BEFORE_TESTING = "before:testing"
    BEFORE_COVERAGE = "before:coverage"
    BEFORE_SECRETS_SCAN = "before:secrets_scan"
    BEFORE_DEPENDENCY_AUDIT = "before:dependency_audit"
    BEFORE_DOCKER_BUILD = "before:docker_build"
    BEFORE_ROLLBACK_STRATEGY = "before:rollback_strategy"
    BEFORE_OBSERVABILITY = "before:observability"
    BEFORE_CONFIG = "before:config"
    BEFORE_DOCS = "before:docs"
    BEFORE_CODE_REVIEW = "before:code_review"
    BEFORE_POLICY = "before:policy"
    BEFORE_VERIFICATION = "before:verification"
    BEFORE_PR_PACKAGE = "before:pr_package"
    BEFORE_DEPLOY = "before:deploy"

    # After stage hooks
    AFTER_SPEC = "after:spec"
    AFTER_CONTEXT = "after:context"
    AFTER_DESIGN = "after:design"
    AFTER_API_CONTRACT = "after:api_contract"
    AFTER_IMPLEMENTATION = "after:implementation"
    AFTER_TESTING = "after:testing"
    AFTER_COVERAGE = "after:coverage"
    AFTER_SECRETS_SCAN = "after:secrets_scan"
    AFTER_DEPENDENCY_AUDIT = "after:dependency_audit"
    AFTER_DOCKER_BUILD = "after:docker_build"
    AFTER_ROLLBACK_STRATEGY = "after:rollback_strategy"
    AFTER_OBSERVABILITY = "after:observability"
    AFTER_CONFIG = "after:config"
    AFTER_DOCS = "after:docs"
    AFTER_CODE_REVIEW = "after:code_review"
    AFTER_POLICY = "after:policy"
    AFTER_VERIFICATION = "after:verification"
    AFTER_PR_PACKAGE = "after:pr_package"
    AFTER_DEPLOY = "after:deploy"


class ManifestError(Exception):
    """Raised when manifest parsing or validation fails."""

    pass


class ModelTier(str, Enum):
    """Model size tiers for compatibility checking."""

    SMALL = "small"      # 7B models: 8K-16K context
    MEDIUM = "medium"    # 14B models: 32K context
    LARGE = "large"      # 32B+ models: 128K+ context
    ANY = "any"          # Works with any model


# Context window thresholds for each tier
MODEL_TIER_CONTEXT = {
    ModelTier.SMALL: 8192,     # 7B models (conservative)
    ModelTier.MEDIUM: 32768,   # 14B models
    ModelTier.LARGE: 131072,   # 32B+ models
    ModelTier.ANY: 4096,       # Minimal requirement
}


@dataclass
class ExtensionManifest:
    """Extension manifest containing metadata and configuration.

    Attributes:
        name: Unique extension identifier (lowercase, hyphens allowed).
        version: Semantic version string (e.g., "1.0.0").
        type: Extension type (agent, profile, rag).
        author: Extension author name or organization.
        description: Short description of what the extension does.
        license: License type for the extension.
        hook_point: For agents, the pipeline stage to hook into.
        agent_class: For agents, the class name to instantiate.
        profile_class: For profiles, the class name to instantiate.
        retriever_class: For RAG, the retriever class name.
        requires: List of Python package dependencies.
        price_usd: Price in USD (0 for free extensions).
        homepage: URL to extension homepage or documentation.
        repository: URL to source repository.
        keywords: List of keywords for search/discovery.
        conflicts_with: List of extension names that conflict.
        priority: Execution priority (lower = earlier, default 50).
        config_schema: JSON schema for extension configuration.
        context_tokens: Total tokens this extension adds to context window.
        min_model_tier: Minimum model tier required (small/medium/large/any).
    """

    name: str
    version: str
    type: ExtensionType
    author: str
    description: str
    license: License = License.FREE

    # Agent-specific
    hook_point: HookPoint | None = None
    agent_class: str | None = None

    # Profile-specific
    profile_class: str | None = None

    # RAG-specific
    retriever_class: str | None = None

    # Skill-specific
    skill_class: str | None = None
    input_type: str | None = None  # "files", "directory", "code_snippet"
    output_type: str | None = None  # "modified_files", "new_files", "report"
    file_patterns: list[str] = field(default_factory=list)  # e.g. ["*.py"]
    supports_dry_run: bool = False
    chain: list[dict[str, Any]] = field(default_factory=list)  # Sub-skills for chained skills

    # Dependencies
    requires: list[str] = field(default_factory=list)

    # Marketplace
    price_usd: float = 0.0

    # Metadata
    homepage: str | None = None
    repository: str | None = None
    keywords: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    priority: int = 50
    config_schema: dict[str, Any] | None = None

    # Token budget (for marketplace display)
    context_tokens: int = 0  # Total tokens added to context window
    min_model_tier: ModelTier = ModelTier.ANY  # Minimum model tier required

    def __post_init__(self) -> None:
        """Validate the manifest after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate manifest fields."""
        # Name validation
        if not self.name:
            raise ManifestError("Extension name is required")
        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ManifestError(
                f"Invalid extension name: {self.name}. "
                "Use only letters, numbers, hyphens, and underscores."
            )

        # Version validation
        if not self.version:
            raise ManifestError("Extension version is required")

        # Type-specific validation
        if self.type == ExtensionType.AGENT:
            if not self.hook_point:
                raise ManifestError("Agent extensions require a hook_point")
            if not self.agent_class:
                raise ManifestError("Agent extensions require an agent_class")

        elif self.type == ExtensionType.PROFILE:
            if not self.profile_class:
                raise ManifestError("Profile extensions require a profile_class")

        elif self.type == ExtensionType.RAG:
            if not self.retriever_class:
                raise ManifestError("RAG extensions require a retriever_class")

        elif self.type == ExtensionType.SKILL:
            # Chained skills don't need a skill_class (they compose sub-skills)
            if not self.skill_class and not self.chain:
                raise ManifestError(
                    "Skill extensions require a skill_class or chain"
                )
            if self.input_type and self.input_type not in (
                "files", "directory", "code_snippet",
            ):
                raise ManifestError(
                    f"Invalid input_type: {self.input_type}. "
                    "Use: files, directory, or code_snippet."
                )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ExtensionManifest:
        """Load manifest from a YAML file.

        Args:
            yaml_path: Path to manifest.yaml file.

        Returns:
            Parsed ExtensionManifest.

        Raises:
            ManifestError: If file is missing or invalid.
        """
        if not yaml_path.exists():
            raise ManifestError(f"Manifest not found: {yaml_path}")

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ManifestError(f"Invalid YAML in {yaml_path}: {e}")

        if not isinstance(data, dict):
            raise ManifestError(f"Manifest must be a YAML mapping: {yaml_path}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtensionManifest:
        """Create manifest from a dictionary.

        Args:
            data: Dictionary containing manifest fields.

        Returns:
            Parsed ExtensionManifest.

        Raises:
            ManifestError: If required fields are missing or invalid.
        """
        try:
            # Parse enums
            ext_type = ExtensionType(data.get("type", ""))
            license_type = License(data.get("license", "free"))

            hook_point = None
            if "hook_point" in data:
                hook_point = HookPoint(data["hook_point"])

            # Parse model tier
            min_tier = ModelTier.ANY
            if "min_model_tier" in data:
                min_tier = ModelTier(data["min_model_tier"])

            return cls(
                name=data.get("name", ""),
                version=data.get("version", ""),
                type=ext_type,
                author=data.get("author", ""),
                description=data.get("description", ""),
                license=license_type,
                hook_point=hook_point,
                agent_class=data.get("agent_class"),
                profile_class=data.get("profile_class"),
                retriever_class=data.get("retriever_class"),
                skill_class=data.get("skill_class"),
                input_type=data.get("input_type"),
                output_type=data.get("output_type"),
                file_patterns=data.get("file_patterns", []),
                supports_dry_run=bool(data.get("supports_dry_run", False)),
                chain=data.get("chain", []),
                requires=data.get("requires", []),
                price_usd=float(data.get("price_usd", 0.0)),
                homepage=data.get("homepage"),
                repository=data.get("repository"),
                keywords=data.get("keywords", []),
                conflicts_with=data.get("conflicts_with", []),
                priority=int(data.get("priority", 50)),
                config_schema=data.get("config_schema"),
                context_tokens=int(data.get("context_tokens", 0)),
                min_model_tier=min_tier,
            )
        except (ValueError, KeyError) as e:
            raise ManifestError(f"Invalid manifest data: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary.

        Returns:
            Dictionary representation of the manifest.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "type": self.type.value,
            "author": self.author,
            "description": self.description,
            "license": self.license.value,
        }

        if self.hook_point:
            result["hook_point"] = self.hook_point.value
        if self.agent_class:
            result["agent_class"] = self.agent_class
        if self.profile_class:
            result["profile_class"] = self.profile_class
        if self.retriever_class:
            result["retriever_class"] = self.retriever_class
        if self.skill_class:
            result["skill_class"] = self.skill_class
        if self.input_type:
            result["input_type"] = self.input_type
        if self.output_type:
            result["output_type"] = self.output_type
        if self.file_patterns:
            result["file_patterns"] = self.file_patterns
        if self.supports_dry_run:
            result["supports_dry_run"] = self.supports_dry_run
        if self.chain:
            result["chain"] = self.chain
        if self.requires:
            result["requires"] = self.requires
        if self.price_usd > 0:
            result["price_usd"] = self.price_usd
        if self.homepage:
            result["homepage"] = self.homepage
        if self.repository:
            result["repository"] = self.repository
        if self.keywords:
            result["keywords"] = self.keywords
        if self.conflicts_with:
            result["conflicts_with"] = self.conflicts_with
        if self.priority != 50:
            result["priority"] = self.priority
        if self.config_schema:
            result["config_schema"] = self.config_schema
        if self.context_tokens > 0:
            result["context_tokens"] = self.context_tokens
        if self.min_model_tier != ModelTier.ANY:
            result["min_model_tier"] = self.min_model_tier.value

        return result

    def to_yaml(self, yaml_path: Path) -> None:
        """Save manifest to a YAML file.

        Args:
            yaml_path: Path to write manifest.yaml.
        """
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @property
    def is_free(self) -> bool:
        """Check if extension is free."""
        return self.price_usd == 0.0

    @property
    def is_commercial(self) -> bool:
        """Check if extension is commercial."""
        return self.license == License.COMMERCIAL

    @property
    def context_tokens_formatted(self) -> str:
        """Format context tokens for display (e.g., '2.5K', '15K')."""
        if self.context_tokens == 0:
            return "minimal"
        elif self.context_tokens >= 1000:
            return f"{self.context_tokens / 1000:.1f}K"
        else:
            return str(self.context_tokens)

    @property
    def min_context_window(self) -> int:
        """Minimum context window required for this extension."""
        return MODEL_TIER_CONTEXT.get(self.min_model_tier, 4096)

    def is_compatible_with_model(self, context_window: int) -> bool:
        """Check if extension is compatible with a model's context window.

        Args:
            context_window: Model's context window size in tokens.

        Returns:
            True if extension can fit within the context window.
        """
        # Extension needs: its own tokens + buffer for prompts/output
        required = self.context_tokens + 4000  # 4K buffer for system/output
        return context_window >= required

    def get_compatible_tiers(self) -> list[ModelTier]:
        """Get list of compatible model tiers.

        Returns:
            List of ModelTier values this extension is compatible with.
        """
        compatible = []
        for tier in [ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]:
            tier_context = MODEL_TIER_CONTEXT[tier]
            if self.is_compatible_with_model(tier_context):
                compatible.append(tier)
        return compatible if compatible else [ModelTier.LARGE]

    def __repr__(self) -> str:
        return (
            f"ExtensionManifest(name={self.name!r}, version={self.version!r}, "
            f"type={self.type.value!r})"
        )
