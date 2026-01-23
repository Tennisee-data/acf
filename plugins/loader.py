"""Plugin loader for discovering and loading plugins.

Scans plugin directories, validates manifests, and loads agent classes.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from agents.base import BaseAgent

from .manifest import PluginManifest
from .registry import LoadedPlugin, PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Error loading a plugin."""

    pass


class PluginLoader:
    """Loads plugins from filesystem.

    Scans directories for plugin.yaml files, validates them,
    and loads the associated agent classes.
    """

    MANIFEST_FILE = "plugin.yaml"
    REQUIREMENTS_FILE = "requirements.txt"

    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        enabled_plugins: list[str] | None = None,
        disabled_plugins: list[str] | None = None,
        plugin_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize plugin loader.

        Args:
            plugin_dirs: Directories to scan for plugins
            enabled_plugins: Whitelist of plugins to enable (None = all)
            disabled_plugins: Blacklist of plugins to disable
            plugin_config: Per-plugin configuration overrides
        """
        self.plugin_dirs = plugin_dirs or []
        self.enabled_plugins = set(enabled_plugins) if enabled_plugins else None
        self.disabled_plugins = set(disabled_plugins) if disabled_plugins else set()
        self.plugin_config = plugin_config or {}

    def discover_plugins(self) -> list[Path]:
        """Discover all plugin directories.

        Returns:
            List of paths to plugin directories (containing plugin.yaml)
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.debug("Plugin directory does not exist: %s", plugin_dir)
                continue

            # Each subdirectory with plugin.yaml is a plugin
            for subdir in plugin_dir.iterdir():
                if subdir.is_dir():
                    manifest_path = subdir / self.MANIFEST_FILE
                    if manifest_path.exists():
                        discovered.append(subdir)
                        logger.debug("Discovered plugin: %s", subdir.name)

        return discovered

    def load_manifest(self, plugin_path: Path) -> PluginManifest:
        """Load and validate plugin manifest.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            Validated PluginManifest

        Raises:
            PluginLoadError: If manifest is invalid
        """
        manifest_path = plugin_path / self.MANIFEST_FILE

        try:
            with open(manifest_path) as f:
                data = yaml.safe_load(f)

            return PluginManifest(**data)

        except FileNotFoundError:
            raise PluginLoadError(f"Manifest not found: {manifest_path}")
        except yaml.YAMLError as e:
            raise PluginLoadError(f"Invalid YAML in manifest: {e}")
        except Exception as e:
            raise PluginLoadError(f"Invalid manifest: {e}")

    def load_agent_class(self, plugin_path: Path, class_path: str) -> type[BaseAgent]:
        """Load agent class from plugin.

        Args:
            plugin_path: Path to plugin directory
            class_path: Class path (e.g., 'agent.MyAgent')

        Returns:
            Agent class

        Raises:
            PluginLoadError: If class cannot be loaded
        """
        module_name, class_name = class_path.rsplit(".", 1)
        module_file = plugin_path / f"{module_name}.py"

        if not module_file.exists():
            raise PluginLoadError(f"Agent module not found: {module_file}")

        try:
            # Create unique module name to avoid conflicts
            unique_module_name = f"plugins.{plugin_path.name}.{module_name}"

            # Load module from file
            spec = importlib.util.spec_from_file_location(unique_module_name, module_file)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load module spec: {module_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module
            spec.loader.exec_module(module)

            # Get class from module
            if not hasattr(module, class_name):
                raise PluginLoadError(f"Class '{class_name}' not found in {module_file}")

            agent_class = getattr(module, class_name)

            # Validate it's a BaseAgent subclass
            if not isinstance(agent_class, type) or not issubclass(agent_class, BaseAgent):
                raise PluginLoadError(f"Class '{class_name}' must extend BaseAgent")

            return agent_class

        except PluginLoadError:
            raise
        except Exception as e:
            raise PluginLoadError(f"Failed to load agent class: {e}")

    def load_plugin(self, plugin_path: Path) -> LoadedPlugin:
        """Load a single plugin.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            LoadedPlugin instance

        Raises:
            PluginLoadError: If plugin cannot be loaded
        """
        # Load manifest
        manifest = self.load_manifest(plugin_path)

        # Check if plugin should be loaded
        if self.enabled_plugins is not None and manifest.name not in self.enabled_plugins:
            raise PluginLoadError(f"Plugin '{manifest.name}' not in enabled list")

        if manifest.name in self.disabled_plugins:
            raise PluginLoadError(f"Plugin '{manifest.name}' is disabled")

        # Load agent class
        agent_class = self.load_agent_class(plugin_path, manifest.agent.class_name)

        # Get plugin-specific config
        config = self.plugin_config.get(manifest.name, {})

        # Apply default values from manifest
        for field_name, field_def in manifest.config.items():
            if field_name not in config and field_def.default is not None:
                config[field_name] = field_def.default

        return LoadedPlugin(
            manifest=manifest,
            agent_class=agent_class,
            plugin_path=plugin_path,
            config=config,
            enabled=manifest.enabled_by_default,
        )

    def load_all(self) -> PluginRegistry:
        """Discover and load all plugins.

        Returns:
            PluginRegistry with all loaded plugins
        """
        registry = PluginRegistry()

        plugin_paths = self.discover_plugins()
        logger.info("Discovered %d plugins", len(plugin_paths))

        for plugin_path in plugin_paths:
            try:
                plugin = self.load_plugin(plugin_path)
                registry.register(plugin)
            except PluginLoadError as e:
                logger.warning("Failed to load plugin %s: %s", plugin_path.name, e)
            except Exception as e:
                logger.error("Unexpected error loading plugin %s: %s", plugin_path.name, e)

        return registry

    @staticmethod
    def get_default_plugin_dirs() -> list[Path]:
        """Get default plugin directories.

        Returns:
            List of default plugin directories:
            - ~/.coding-factory/plugins (global)
            - ./.plugins (local to project)
        """
        dirs = []

        # Global plugins
        global_dir = Path.home() / ".coding-factory" / "plugins"
        dirs.append(global_dir)

        # Local plugins
        local_dir = Path.cwd() / ".plugins"
        dirs.append(local_dir)

        return dirs
