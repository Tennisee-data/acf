"""Extension loader for ACF Local Edition.

Discovers, loads, and manages extensions from the extensions directory.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Type

from extensions.manifest import (
    ExtensionManifest,
    ExtensionType,
    HookPoint,
    ManifestError,
)


class ExtensionLoadError(Exception):
    """Raised when extension loading fails."""

    pass


@dataclass
class LoadedExtension:
    """Represents a loaded extension."""

    manifest: ExtensionManifest
    path: Path
    module: Any | None = None
    agent_class: Type | None = None
    profile_class: Type | None = None
    retriever_class: Type | None = None
    enabled: bool = True


@dataclass
class ExtensionRegistry:
    """Registry of loaded extensions organized by type and hook point."""

    agents: dict[str, LoadedExtension] = field(default_factory=dict)
    profiles: dict[str, LoadedExtension] = field(default_factory=dict)
    rag: dict[str, LoadedExtension] = field(default_factory=dict)

    # Hook point index for fast lookup
    hooks: dict[HookPoint, list[LoadedExtension]] = field(default_factory=dict)

    def get_agents_for_hook(self, hook_point: HookPoint) -> list[LoadedExtension]:
        """Get all agents registered for a hook point, sorted by priority."""
        agents = self.hooks.get(hook_point, [])
        return sorted(agents, key=lambda x: x.manifest.priority)


class ExtensionLoader:
    """Load and register extensions from the extensions directory.

    Extensions are organized in:
    ~/.coding-factory/extensions/
    ├── agents/
    │   └── <extension-name>/
    │       ├── manifest.yaml
    │       ├── agent.py
    │       └── requirements.txt
    ├── profiles/
    │   └── <extension-name>/
    │       ├── manifest.yaml
    │       └── profile.py
    └── rag/
        └── <extension-name>/
            ├── manifest.yaml
            ├── retriever.py
            └── requirements.txt

    Example:
        >>> loader = ExtensionLoader(Path.home() / ".coding-factory/extensions")
        >>> loader.discover()
        >>> agent_cls = loader.get_agent("secrets-scan")
        >>> profile = loader.get_profile("vue")
    """

    DEFAULT_EXTENSIONS_DIR = Path.home() / ".coding-factory" / "extensions"

    def __init__(
        self,
        extensions_dir: Path | None = None,
        enabled_extensions: list[str] | None = None,
        disabled_extensions: list[str] | None = None,
    ):
        """Initialize the extension loader.

        Args:
            extensions_dir: Path to extensions directory.
            enabled_extensions: Whitelist of extension names (None = all).
            disabled_extensions: Blacklist of extension names.
        """
        self.extensions_dir = Path(extensions_dir or self.DEFAULT_EXTENSIONS_DIR)
        self.enabled_extensions = set(enabled_extensions) if enabled_extensions else None
        self.disabled_extensions = set(disabled_extensions or [])

        self.registry = ExtensionRegistry()
        self._manifests: dict[str, ExtensionManifest] = {}

    def discover(self) -> list[str]:
        """Discover and load all installed extensions.

        Returns:
            List of loaded extension names.

        Raises:
            ExtensionLoadError: If a critical extension fails to load.
        """
        loaded: list[str] = []

        if not self.extensions_dir.exists():
            return loaded

        for ext_type in ["agents", "profiles", "rag"]:
            type_dir = self.extensions_dir / ext_type
            if not type_dir.exists():
                continue

            for ext_dir in type_dir.iterdir():
                if not ext_dir.is_dir():
                    continue

                manifest_path = ext_dir / "manifest.yaml"
                if not manifest_path.exists():
                    continue

                try:
                    extension = self._load_extension(ext_dir, manifest_path)
                    if extension:
                        self._register(extension)
                        loaded.append(extension.manifest.name)
                except (ManifestError, ExtensionLoadError) as e:
                    # Log error but continue loading other extensions
                    print(f"Warning: Failed to load extension {ext_dir.name}: {e}")

        return loaded

    def _load_extension(
        self, ext_dir: Path, manifest_path: Path
    ) -> LoadedExtension | None:
        """Load a single extension.

        Args:
            ext_dir: Extension directory.
            manifest_path: Path to manifest.yaml.

        Returns:
            LoadedExtension or None if disabled/filtered.
        """
        manifest = ExtensionManifest.from_yaml(manifest_path)

        # Check enabled/disabled lists
        if self._is_disabled(manifest.name):
            return None

        # Store manifest for lookup
        self._manifests[manifest.name] = manifest

        # Create extension
        extension = LoadedExtension(
            manifest=manifest,
            path=ext_dir,
            enabled=True,
        )

        # Load the appropriate class based on type
        if manifest.type == ExtensionType.AGENT:
            extension.module, extension.agent_class = self._load_agent_class(
                ext_dir, manifest
            )
        elif manifest.type == ExtensionType.PROFILE:
            extension.module, extension.profile_class = self._load_profile_class(
                ext_dir, manifest
            )
        elif manifest.type == ExtensionType.RAG:
            extension.module, extension.retriever_class = self._load_retriever_class(
                ext_dir, manifest
            )

        return extension

    def _is_disabled(self, name: str) -> bool:
        """Check if an extension is disabled."""
        if name in self.disabled_extensions:
            return True
        if self.enabled_extensions is not None and name not in self.enabled_extensions:
            return True
        return False

    def _load_agent_class(
        self, ext_dir: Path, manifest: ExtensionManifest
    ) -> tuple[Any, Type]:
        """Load an agent class from extension directory."""
        agent_file = ext_dir / "agent.py"
        if not agent_file.exists():
            raise ExtensionLoadError(f"Agent file not found: {agent_file}")

        module = self._load_module(
            f"acf_ext_agent_{manifest.name.replace('-', '_')}", agent_file
        )

        if not hasattr(module, manifest.agent_class):
            raise ExtensionLoadError(
                f"Agent class '{manifest.agent_class}' not found in {agent_file}"
            )

        return module, getattr(module, manifest.agent_class)

    def _load_profile_class(
        self, ext_dir: Path, manifest: ExtensionManifest
    ) -> tuple[Any, Type]:
        """Load a profile class from extension directory."""
        profile_file = ext_dir / "profile.py"
        if not profile_file.exists():
            raise ExtensionLoadError(f"Profile file not found: {profile_file}")

        module = self._load_module(
            f"acf_ext_profile_{manifest.name.replace('-', '_')}", profile_file
        )

        if not hasattr(module, manifest.profile_class):
            raise ExtensionLoadError(
                f"Profile class '{manifest.profile_class}' not found in {profile_file}"
            )

        return module, getattr(module, manifest.profile_class)

    def _load_retriever_class(
        self, ext_dir: Path, manifest: ExtensionManifest
    ) -> tuple[Any, Type]:
        """Load a RAG retriever class from extension directory."""
        retriever_file = ext_dir / "retriever.py"
        if not retriever_file.exists():
            raise ExtensionLoadError(f"Retriever file not found: {retriever_file}")

        module = self._load_module(
            f"acf_ext_rag_{manifest.name.replace('-', '_')}", retriever_file
        )

        if not hasattr(module, manifest.retriever_class):
            raise ExtensionLoadError(
                f"Retriever class '{manifest.retriever_class}' not found in {retriever_file}"
            )

        return module, getattr(module, manifest.retriever_class)

    def _load_module(self, module_name: str, file_path: Path) -> Any:
        """Dynamically load a Python module from file.

        Args:
            module_name: Name to assign to the module.
            file_path: Path to the Python file.

        Returns:
            Loaded module.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ExtensionLoadError(f"Could not load module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _register(self, extension: LoadedExtension) -> None:
        """Register an extension in the registry."""
        manifest = extension.manifest

        if manifest.type == ExtensionType.AGENT:
            self.registry.agents[manifest.name] = extension
            # Index by hook point
            if manifest.hook_point:
                if manifest.hook_point not in self.registry.hooks:
                    self.registry.hooks[manifest.hook_point] = []
                self.registry.hooks[manifest.hook_point].append(extension)

        elif manifest.type == ExtensionType.PROFILE:
            self.registry.profiles[manifest.name] = extension

        elif manifest.type == ExtensionType.RAG:
            self.registry.rag[manifest.name] = extension

    def get_agent(self, name: str) -> Type | None:
        """Get an agent class by name.

        Args:
            name: Extension name.

        Returns:
            Agent class or None if not found.
        """
        ext = self.registry.agents.get(name)
        return ext.agent_class if ext else None

    def get_profile(self, name: str) -> Type | None:
        """Get a profile class by name.

        Args:
            name: Extension name.

        Returns:
            Profile class or None if not found.
        """
        ext = self.registry.profiles.get(name)
        return ext.profile_class if ext else None

    def get_retriever(self, name: str) -> Type | None:
        """Get a RAG retriever class by name.

        Args:
            name: Extension name.

        Returns:
            Retriever class or None if not found.
        """
        ext = self.registry.rag.get(name)
        return ext.retriever_class if ext else None

    def get_manifest(self, name: str) -> ExtensionManifest | None:
        """Get manifest for an extension.

        Args:
            name: Extension name.

        Returns:
            Manifest or None if not found.
        """
        return self._manifests.get(name)

    def get_hooks_for_stage(
        self, stage: str, position: str = "after"
    ) -> list[LoadedExtension]:
        """Get all extensions hooked to a pipeline stage.

        Args:
            stage: Stage name (e.g., "implementation").
            position: "before" or "after".

        Returns:
            List of extensions sorted by priority.
        """
        try:
            hook_point = HookPoint(f"{position}:{stage}")
            return self.registry.get_agents_for_hook(hook_point)
        except ValueError:
            return []

    def list_extensions(
        self, ext_type: ExtensionType | None = None
    ) -> list[ExtensionManifest]:
        """List all loaded extension manifests.

        Args:
            ext_type: Filter by extension type.

        Returns:
            List of manifests.
        """
        if ext_type == ExtensionType.AGENT:
            return [e.manifest for e in self.registry.agents.values()]
        elif ext_type == ExtensionType.PROFILE:
            return [e.manifest for e in self.registry.profiles.values()]
        elif ext_type == ExtensionType.RAG:
            return [e.manifest for e in self.registry.rag.values()]
        else:
            return list(self._manifests.values())

    def check_conflicts(self, extensions: list[str]) -> list[tuple[str, str]]:
        """Check for conflicts between extensions.

        Args:
            extensions: List of extension names to check.

        Returns:
            List of conflicting pairs.
        """
        conflicts: list[tuple[str, str]] = []
        ext_set = set(extensions)

        for name in extensions:
            manifest = self._manifests.get(name)
            if manifest:
                for conflict in manifest.conflicts_with:
                    if conflict in ext_set:
                        conflicts.append((name, conflict))

        return conflicts

    def enable_extension(self, name: str) -> bool:
        """Enable a disabled extension.

        Args:
            name: Extension name.

        Returns:
            True if extension was enabled.
        """
        for registry in [
            self.registry.agents,
            self.registry.profiles,
            self.registry.rag,
        ]:
            if name in registry:
                registry[name].enabled = True
                return True
        return False

    def disable_extension(self, name: str) -> bool:
        """Disable an enabled extension.

        Args:
            name: Extension name.

        Returns:
            True if extension was disabled.
        """
        for registry in [
            self.registry.agents,
            self.registry.profiles,
            self.registry.rag,
        ]:
            if name in registry:
                registry[name].enabled = False
                return True
        return False

    def get_requirements(self, names: list[str] | None = None) -> list[str]:
        """Get combined requirements for extensions.

        Args:
            names: Extension names (None = all loaded).

        Returns:
            List of pip requirement strings.
        """
        requirements: set[str] = set()

        if names is None:
            names = list(self._manifests.keys())

        for name in names:
            manifest = self._manifests.get(name)
            if manifest:
                requirements.update(manifest.requires)

        return sorted(requirements)

    def ensure_extensions_dir(self) -> None:
        """Create extensions directory structure if it doesn't exist."""
        for subdir in ["agents", "profiles", "rag"]:
            (self.extensions_dir / subdir).mkdir(parents=True, exist_ok=True)
