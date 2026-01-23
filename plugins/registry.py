"""Plugin registry for managing loaded plugins.

Stores loaded plugins and provides access by name and hook point.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent

from .manifest import HookPoint, PluginManifest

if TYPE_CHECKING:
    from llm_backend import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """A loaded and instantiated plugin."""

    manifest: PluginManifest
    agent_class: type[BaseAgent]
    plugin_path: Path
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def name(self) -> str:
        """Plugin name."""
        return self.manifest.name

    @property
    def hook_point(self) -> HookPoint:
        """Hook point for this plugin."""
        return self.manifest.hook.point

    @property
    def priority(self) -> int:
        """Execution priority."""
        return self.manifest.hook.priority

    def create_agent(self, llm: "LLMBackend | None" = None, **kwargs: Any) -> BaseAgent:
        """Create an instance of the plugin agent.

        Args:
            llm: LLM backend to use
            **kwargs: Additional arguments for agent constructor

        Returns:
            Instantiated agent
        """
        if self.manifest.requires_llm and llm is None:
            raise ValueError(f"Plugin {self.name} requires LLM backend")

        # Merge plugin config with kwargs
        agent_kwargs = {**self.config, **kwargs}
        if llm is not None:
            agent_kwargs["llm"] = llm

        return self.agent_class(**agent_kwargs)


class PluginRegistry:
    """Registry for loaded plugins.

    Provides access to plugins by name and hook point.
    Handles plugin ordering by priority.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._plugins: dict[str, LoadedPlugin] = {}
        self._by_hook: dict[HookPoint, list[LoadedPlugin]] = {}

    def register(self, plugin: LoadedPlugin) -> None:
        """Register a loaded plugin.

        Args:
            plugin: Plugin to register

        Raises:
            ValueError: If plugin with same name already registered
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already registered")

        self._plugins[plugin.name] = plugin

        # Add to hook point index
        hook = plugin.hook_point
        if hook not in self._by_hook:
            self._by_hook[hook] = []
        self._by_hook[hook].append(plugin)

        # Sort by priority (lower = earlier)
        self._by_hook[hook].sort(key=lambda p: p.priority)

        logger.info("Registered plugin: %s (hook: %s, priority: %d)", plugin.name, hook.value, plugin.priority)

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name.

        Args:
            name: Plugin name

        Returns:
            True if plugin was removed, False if not found
        """
        if name not in self._plugins:
            return False

        plugin = self._plugins.pop(name)

        # Remove from hook index
        hook = plugin.hook_point
        if hook in self._by_hook:
            self._by_hook[hook] = [p for p in self._by_hook[hook] if p.name != name]

        logger.info("Unregistered plugin: %s", name)
        return True

    def get(self, name: str) -> LoadedPlugin | None:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin or None if not found
        """
        return self._plugins.get(name)

    def get_for_hook(self, hook: HookPoint) -> list[LoadedPlugin]:
        """Get all enabled plugins for a hook point.

        Args:
            hook: Hook point

        Returns:
            List of plugins sorted by priority
        """
        plugins = self._by_hook.get(hook, [])
        return [p for p in plugins if p.enabled]

    def get_for_stage(self, stage: str, before: bool = True) -> list[LoadedPlugin]:
        """Get plugins for before or after a stage.

        Args:
            stage: Stage name (e.g., 'design', 'implementation')
            before: If True, get 'before:stage' plugins, else 'after:stage'

        Returns:
            List of enabled plugins for the hook
        """
        prefix = "before" if before else "after"
        hook_str = f"{prefix}:{stage.lower()}"

        try:
            hook = HookPoint(hook_str)
            return self.get_for_hook(hook)
        except ValueError:
            logger.warning("Unknown stage for hook: %s", stage)
            return []

    def list_all(self) -> list[LoadedPlugin]:
        """List all registered plugins.

        Returns:
            List of all plugins
        """
        return list(self._plugins.values())

    def enable(self, name: str) -> bool:
        """Enable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if enabled, False if not found
        """
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if disabled, False if not found
        """
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = False
            return True
        return False

    def __len__(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        """Check if plugin is registered."""
        return name in self._plugins
