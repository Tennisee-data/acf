"""Configuration management for Coding Factory.

Loads configuration from:
1. config.toml (defaults)
2. Environment variables (overrides)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class LLMConfig:
    """LLM backend configuration."""

    backend: str = "auto"  # "auto", "ollama", "openai", "anthropic", "lmstudio"
    base_url: str = "http://localhost:11434"
    model_general: str = "llama3.1:8b"
    model_code: str = "codellama:7b"
    timeout: int = 600

    # Lightweight mode for sophisticated API models
    # "auto" = enabled for openai/anthropic, disabled for ollama/lmstudio
    # true = always use lightweight prompts (skip RAG, concise prompts)
    # false = always use full prompts (RAG, invariants, verbose)
    lightweight_prompts: str | bool = "auto"

    def is_lightweight_mode(self, detected_backend: str | None = None) -> bool:
        """Check if lightweight mode should be used.

        Args:
            detected_backend: The actual backend being used (for "auto" mode)

        Returns:
            True if lightweight prompts should be used
        """
        if isinstance(self.lightweight_prompts, bool):
            return self.lightweight_prompts

        # Auto-detect based on backend
        backend = detected_backend or self.backend
        # API backends don't need RAG/verbose prompts
        # LM Studio is local like Ollama, so use full prompts
        return backend in ("openai", "anthropic")


@dataclass
class FixLoopConfig:
    """Fix loop configuration for iterative code correction."""

    enabled: bool = True
    max_iterations: int = 5
    timeout_seconds: int = 300
    stop_on_same_error: bool = True
    stop_on_increasing_errors: bool = True


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""

    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"
    fix_loop: FixLoopConfig = field(default_factory=FixLoopConfig)


@dataclass
class GitConfig:
    """Git integration configuration."""

    auto_commit: bool = True  # Auto-commit each iteration
    auto_push: bool = False
    remote_url: str = ""
    create_pr: bool = False


@dataclass
class RuntimeConfig:
    """Runtime environment configuration."""

    mode: str = "auto"  # "auto" | "docker" | "venv" | "local"
    docker_threshold_lines: int = 500  # Line count threshold for Docker recommendation
    fallback_to_local: bool = True  # If Docker fails, fallback to local execution


@dataclass
class DeployConfig:
    """Deployment configuration."""

    strategy: str = "docker-push"  # docker-push | render | fly | k8s | ssh | custom
    registry: str = ""
    image_name: str = ""
    health_check_url: str = ""
    health_check_timeout: int = 120
    rollback_on_failure: bool = True

    # Cloud platforms
    render_service_id: str = ""
    fly_app_name: str = ""
    railway_project_id: str = ""

    # Kubernetes
    k8s_namespace: str = "default"
    k8s_deployment: str = ""
    k8s_context: str = ""

    # SSH
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""
    ssh_deploy_path: str = "/app"

    # Custom
    custom_script: str = "deploy.sh"


@dataclass
class TemplatesConfig:
    """Template and marketplace configuration."""

    # Custom templates directory (default: ~/.coding-factory/templates)
    templates_dir: str = ""

    # Marketplace settings
    marketplace_url: str = "https://templates.coding-factory.dev/api/v1"
    marketplace_api_key: str = ""  # For publishing templates


@dataclass
class MemoryConfig:
    """Memory and past-run learning configuration."""

    enabled: bool = True
    store_location: str = "global"  # "global" (~/.coding-factory/memory) or "local"
    embedding_model: str = "nomic-embed-text"
    decay_half_life_days: int = 90
    auto_index: bool = True  # Auto-index runs on completion
    max_similar_features: int = 3
    max_patterns: int = 5
    max_error_hints: int = 3
    # Hybrid search settings
    search_mode: str = "hybrid"  # "semantic", "lexical", or "hybrid"
    hybrid_alpha: float = 0.7  # Weight for semantic in hybrid mode (0-1)


@dataclass
class RoutingConfig:
    """Multi-model routing configuration.

    Routes LLM calls to different model tiers based on task complexity.
    Tiers: cheap (7B), medium (14B), premium (30B+)
    """

    enabled: bool = True  # Enable model routing
    model_cheap: str = "qwen3:7b"  # Fast, low-cost model
    model_medium: str = "qwen3:14b"  # Balanced model
    model_premium: str = "qwen3:30b"  # High-quality model

    # Stage overrides (stage_name -> tier)
    stage_overrides: dict[str, str] = field(default_factory=dict)

    # Domains that always use premium model
    premium_domains: list[str] = field(
        default_factory=lambda: ["payments", "security", "auth", "infra", "database"]
    )


@dataclass
class PluginsConfig:
    """Plugin system configuration.

    Plugins extend the pipeline with custom agents that run
    at specific hook points (before/after stages).
    """

    enabled: bool = True  # Enable plugin system
    plugins_dir: str = ""  # Additional plugins directory (empty = default)
    enabled_plugins: list[str] = field(default_factory=list)  # Whitelist (empty = all)
    disabled_plugins: list[str] = field(default_factory=list)  # Blacklist
    plugin_config: dict[str, dict[str, Any]] = field(default_factory=dict)  # Per-plugin config


@dataclass
class IntegrationsConfig:
    """Issue tracker integrations configuration.

    Allows fetching issues from JIRA, GitHub, etc. as feature input.
    """

    # JIRA settings (also configurable via env: JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""

    # GitHub settings (also configurable via env: GITHUB_TOKEN or GH_TOKEN)
    github_token: str = ""

    # Linear settings (planned)
    linear_api_key: str = ""


@dataclass
class ExtensionsConfig:
    """Extensions configuration for ACF Local Edition.

    Extensions are loadable agents, profiles, and RAG retrievers from
    the marketplace or local development.
    """

    # Directory for installed extensions (default: ~/.coding-factory/extensions)
    extensions_dir: str = ""

    # Marketplace URL for downloading extensions
    marketplace_url: str = "https://marketplace.agentcodefactory.com/api/v1"

    # Enabled extensions (empty list = all installed enabled)
    agents: list[str] = field(default_factory=list)  # Agent extension names
    profiles: list[str] = field(default_factory=list)  # Profile extension names
    rag_kits: list[str] = field(default_factory=list)  # RAG extension names


@dataclass
class LocalStorageConfig:
    """Local git versioning configuration for ACF Local Edition.

    Provides automatic git-based versioning for pipeline iterations.
    """

    # Auto-commit each pipeline iteration to local git
    auto_commit: bool = True

    # Commit message prefix for ACF commits
    commit_prefix: str = "[ACF]"

    # Create tags for significant iterations
    auto_tag: bool = False


@dataclass
class Config:
    """Main configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    git: GitConfig = field(default_factory=GitConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    local_storage: LocalStorageConfig = field(default_factory=LocalStorageConfig)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        llm_data = data.get("llm", {})
        pipeline_data = data.get("pipeline", {})
        git_data = data.get("git", {})
        runtime_data = data.get("runtime", {})
        deploy_data = data.get("deploy", {})
        templates_data = data.get("templates", {})
        memory_data = data.get("memory", {})
        routing_data = data.get("routing", {})
        plugins_data = data.get("plugins", {})
        integrations_data = data.get("integrations", {})
        extensions_data = data.get("extensions", {})
        local_storage_data = data.get("local_storage", {})
        profiles_data = data.get("profiles", {})

        # Handle nested fix_loop config
        fix_loop_data = pipeline_data.pop("fix_loop", {})
        fix_loop_config = FixLoopConfig(**fix_loop_data) if fix_loop_data else FixLoopConfig()

        return cls(
            llm=LLMConfig(**llm_data),
            pipeline=PipelineConfig(**pipeline_data, fix_loop=fix_loop_config),
            git=GitConfig(**git_data),
            runtime=RuntimeConfig(**runtime_data),
            deploy=DeployConfig(**deploy_data),
            templates=TemplatesConfig(**templates_data),
            memory=MemoryConfig(**memory_data),
            routing=RoutingConfig(**routing_data),
            plugins=PluginsConfig(**plugins_data),
            integrations=IntegrationsConfig(**integrations_data),
            extensions=ExtensionsConfig(**extensions_data),
            local_storage=LocalStorageConfig(**local_storage_data),
            profiles=profiles_data,
        )


def find_config_file() -> Path | None:
    """Find config.toml in current or parent directories.

    Returns:
        Path to config.toml or None if not found.
    """
    current = Path.cwd()

    for directory in [current, *current.parents]:
        config_path = directory / "config.toml"
        if config_path.exists():
            return config_path

    return None


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from file and environment.

    Args:
        config_path: Optional explicit path to config.toml

    Returns:
        Config object with merged settings.
    """
    # Start with defaults
    config_data: dict[str, Any] = {}

    # Load from file if available
    if config_path is None:
        config_path = find_config_file()

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "rb") as f:
                config_data = tomllib.load(f)

    # Apply environment variable overrides
    env_overrides = {
        "llm": {
            "backend": os.getenv("LLM_BACKEND"),
            "base_url": os.getenv("OLLAMA_BASE_URL"),
            "model_general": os.getenv("OLLAMA_MODEL_GENERAL"),
            "model_code": os.getenv("OLLAMA_MODEL_CODE"),
            "timeout": _int_or_none(os.getenv("LLM_TIMEOUT")),
        },
        "pipeline": {
            "artifacts_dir": os.getenv("ARTIFACTS_DIR"),
            "log_level": os.getenv("LOG_LEVEL"),
        },
    }

    # Merge env overrides (only non-None values)
    for section, values in env_overrides.items():
        if section not in config_data:
            config_data[section] = {}
        for key, value in values.items():
            if value is not None:
                config_data[section][key] = value

    return Config.from_dict(config_data)


def _int_or_none(value: str | None) -> int | None:
    """Convert string to int, or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Config object (loaded once, cached).
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    """Force reload of configuration.

    Returns:
        Fresh Config object.
    """
    global _config
    _config = load_config()
    return _config
