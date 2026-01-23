"""Plugin manifest schema.

Defines the structure of plugin.yaml files that describe
plugin metadata, hook points, and configuration.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HookPoint(str, Enum):
    """Valid hook points in the pipeline."""

    # Before stages
    BEFORE_INIT = "before:init"
    BEFORE_SPEC = "before:spec"
    BEFORE_DECOMPOSITION = "before:decomposition"
    BEFORE_CONTEXT = "before:context"
    BEFORE_DESIGN = "before:design"
    BEFORE_DESIGN_APPROVAL = "before:design_approval"
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
    BEFORE_FINAL_APPROVAL = "before:final_approval"
    BEFORE_DEPLOY = "before:deploy"

    # After stages
    AFTER_INIT = "after:init"
    AFTER_SPEC = "after:spec"
    AFTER_DECOMPOSITION = "after:decomposition"
    AFTER_CONTEXT = "after:context"
    AFTER_DESIGN = "after:design"
    AFTER_DESIGN_APPROVAL = "after:design_approval"
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
    AFTER_FINAL_APPROVAL = "after:final_approval"
    AFTER_DEPLOY = "after:deploy"
    AFTER_DONE = "after:done"


class ConfigField(BaseModel):
    """Schema for a plugin configuration field."""

    type: str = Field(..., description="Field type: str, int, float, bool, path, list")
    default: Any = Field(None, description="Default value")
    description: str = Field("", description="Field description")
    required: bool = Field(False, description="Whether field is required")


class AgentConfig(BaseModel):
    """Agent class configuration."""

    class_name: str = Field(..., alias="class", description="Agent class path (module.ClassName)")

    @field_validator("class_name", mode="before")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Validate class name format."""
        if "." not in v:
            raise ValueError("class must be in format 'module.ClassName'")
        return v


class HookConfig(BaseModel):
    """Hook point configuration."""

    point: HookPoint = Field(..., description="When to run: before:X or after:X")
    priority: int = Field(100, description="Execution priority (lower = earlier)")
    required: bool = Field(False, description="If True, pipeline fails if plugin fails")
    skip_on_failure: bool = Field(True, description="Skip plugin if previous stage failed")


class PluginManifest(BaseModel):
    """Plugin manifest schema (plugin.yaml)."""

    # Metadata
    name: str = Field(..., description="Plugin name (alphanumeric, dashes, underscores)")
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    description: str = Field("", description="Plugin description")
    author: str = Field("", description="Plugin author")

    # Agent configuration
    agent: AgentConfig = Field(..., description="Agent class configuration")

    # Hook configuration
    hook: HookConfig = Field(..., description="Hook point configuration")

    # I/O
    inputs: list[str] = Field(
        default_factory=list,
        description="Required artifacts from pipeline state",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Artifacts this plugin produces",
    )

    # Plugin configuration schema
    config: dict[str, ConfigField] = Field(
        default_factory=dict,
        description="Configuration schema for plugin settings",
    )

    # Optional flags
    requires_llm: bool = Field(True, description="Whether plugin needs LLM backend")
    enabled_by_default: bool = Field(True, description="Enable plugin by default")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate plugin name format."""
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
            raise ValueError("name must start with letter and contain only alphanumeric, dashes, underscores")
        return v

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re

        if not re.match(r"^\d+\.\d+\.\d+", v):
            raise ValueError("version must be semantic (e.g., 1.0.0)")
        return v

    class Config:
        populate_by_name = True
