"""Profile Manager - Discovery, selection, and merging of profiles.

Central registry that dynamically discovers profiles, handles conflicts,
and merges guidance from multiple applicable profiles.
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import detect_features, GuidanceSections, ProfileMetadata

logger = logging.getLogger(__name__)


@dataclass
class ProfileInfo:
    """Information about a loaded profile."""

    name: str
    version: str
    module: Any
    technologies: list[str]
    trigger_keywords: list[str]
    exact_keywords: list[str]
    substring_keywords: list[str]
    conflicts_with: list[str]
    priority: int
    metadata: ProfileMetadata
    guidance_sections: GuidanceSections | None = None

    def should_apply(self, tech_stack: list[str] | None, prompt: str) -> bool:
        """Check if profile should apply using improved matching."""
        # Use module's should_apply if available
        if hasattr(self.module, "should_apply"):
            return self.module.should_apply(tech_stack, prompt)

        # Fallback to improved keyword matching
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))

        # Check explicit tech stack
        if tech_stack:
            tech_lower = {t.lower() for t in tech_stack}
            if tech_lower & set(self.technologies):
                return True

        # Check exact keywords (word boundaries)
        if any(kw.lower() in prompt_words for kw in self.exact_keywords):
            return True

        # Check substring keywords
        if any(kw.lower() in prompt_lower for kw in self.substring_keywords):
            return True

        return False

    def get_guidance(self, sections: list[str] | None = None) -> str:
        """Get guidance text, optionally filtered by sections."""
        if self.guidance_sections is not None and sections:
            return self.guidance_sections.to_full_guidance(sections)

        if hasattr(self.module, "get_guidance"):
            return self.module.get_guidance()

        return getattr(self.module, "SYSTEM_GUIDANCE", "")

    def get_dependencies(self, features: list[str] | None = None) -> list[str]:
        """Get dependencies for profile."""
        if hasattr(self.module, "get_dependencies"):
            return self.module.get_dependencies(features)

        deps = list(getattr(self.module, "DEPENDENCIES", []))
        if features:
            optional = getattr(self.module, "OPTIONAL_DEPENDENCIES", {})
            for feature in features:
                if feature in optional:
                    deps.extend(optional[feature])
        return deps


@dataclass
class ProfileResult:
    """Result of profile selection and merging."""

    guidance: str
    dependencies: list[str]
    profiles: list[str]
    detected_features: list[str]
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.profiles)


class ProfileConflictError(Exception):
    """Raised when conflicting profiles are both applicable."""

    def __init__(self, profiles: list[str], message: str = ""):
        self.profiles = profiles
        self.message = message or f"Conflicting profiles: {', '.join(profiles)}"
        super().__init__(self.message)


class ProfileManager:
    """Manages profile discovery, selection, and merging.

    Features:
    - Automatic discovery of profile modules
    - Conflict detection between mutually exclusive profiles
    - Priority-based selection when conflicts occur
    - Feature auto-detection from prompts
    - Guidance merging from multiple profiles
    - Structured guidance sections

    Example:
        manager = ProfileManager()
        result = manager.get_applicable(
            tech_stack=["python", "fastapi"],
            prompt="Build a REST API with authentication",
        )
        print(result.guidance)
        print(result.dependencies)
        print(result.detected_features)  # ['auth', 'api']
    """

    # Known mutual exclusions
    KNOWN_CONFLICTS: list[set[str]] = [
        {"django", "fastapi", "flask"},  # Python web frameworks
        {"react", "vue", "angular", "svelte"},  # Frontend frameworks
        {"nextjs", "remix", "nuxt"},  # Meta-frameworks
    ]

    def __init__(
        self,
        profiles_dir: Path | None = None,
        extensions_dir: Path | None = None,
        auto_discover: bool = True,
        strict_conflicts: bool = False,
        load_extensions: bool = True,
    ) -> None:
        """Initialize profile manager.

        Args:
            profiles_dir: Directory containing profile modules
            extensions_dir: Directory containing extension profiles
            auto_discover: Automatically discover profiles on init
            strict_conflicts: Raise error on conflicts (vs. using priority)
            load_extensions: Load profiles from extensions directory
        """
        self.profiles_dir = profiles_dir or Path(__file__).parent
        self.extensions_dir = extensions_dir or (
            Path.home() / ".coding-factory" / "extensions" / "profiles"
        )
        self.strict_conflicts = strict_conflicts
        self.load_extensions = load_extensions
        self.profiles: dict[str, ProfileInfo] = {}

        if auto_discover:
            self.discover_profiles()
            if load_extensions:
                self.discover_extension_profiles()

    def discover_profiles(self) -> int:
        """Discover and load all profile modules.

        Returns:
            Number of profiles loaded
        """
        self.profiles.clear()
        count = 0

        for file in self.profiles_dir.glob("*.py"):
            if file.stem.startswith("_") or file.stem in ("base", "manager"):
                continue

            try:
                module = importlib.import_module(f"profiles.{file.stem}")

                # Must have PROFILE_NAME to be a valid profile
                if not hasattr(module, "PROFILE_NAME"):
                    continue

                profile_info = self._load_profile_info(module)
                self.profiles[profile_info.name] = profile_info
                count += 1
                logger.debug("Loaded profile: %s", profile_info.name)

            except ImportError as e:
                logger.warning("Failed to import profile %s: %s", file.stem, e)
            except Exception as e:
                logger.warning("Error loading profile %s: %s", file.stem, e)

        logger.info("Discovered %d profiles", count)
        return count

    def discover_extension_profiles(self) -> int:
        """Discover and load extension profiles from ~/.coding-factory/extensions/profiles/.

        Extension profiles follow the same format as core profiles but are loaded
        from the extensions directory, allowing marketplace-distributed profiles.

        Returns:
            Number of extension profiles loaded
        """
        if not self.extensions_dir.exists():
            return 0

        count = 0

        for ext_dir in self.extensions_dir.iterdir():
            if not ext_dir.is_dir():
                continue

            # Check for manifest.yaml and profile.py
            manifest_path = ext_dir / "manifest.yaml"
            profile_path = ext_dir / "profile.py"

            if not profile_path.exists():
                continue

            try:
                # Load the profile module dynamically
                import importlib.util
                import sys

                module_name = f"acf_ext_profile_{ext_dir.name.replace('-', '_')}"
                spec = importlib.util.spec_from_file_location(module_name, profile_path)

                if spec is None or spec.loader is None:
                    logger.warning("Could not load profile spec from %s", profile_path)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Must have PROFILE_NAME to be a valid profile
                if not hasattr(module, "PROFILE_NAME"):
                    logger.debug("Skipping %s - no PROFILE_NAME", ext_dir.name)
                    continue

                profile_info = self._load_profile_info(module)

                # Mark as extension profile
                profile_info.metadata.author = getattr(
                    module, "AUTHOR",
                    profile_info.metadata.author or "marketplace"
                )

                # Check for conflicts with existing profiles
                if profile_info.name in self.profiles:
                    logger.warning(
                        "Extension profile %s conflicts with existing profile - skipping",
                        profile_info.name,
                    )
                    continue

                self.profiles[profile_info.name] = profile_info
                count += 1
                logger.debug("Loaded extension profile: %s", profile_info.name)

            except Exception as e:
                logger.warning("Error loading extension profile %s: %s", ext_dir.name, e)

        if count > 0:
            logger.info("Discovered %d extension profiles", count)

        return count

    def _load_profile_info(self, module: Any) -> ProfileInfo:
        """Extract profile info from a module."""
        name = getattr(module, "PROFILE_NAME", module.__name__)
        version = getattr(module, "PROFILE_VERSION", "1.0")
        technologies = getattr(module, "TECHNOLOGIES", [])
        trigger_keywords = getattr(module, "TRIGGER_KEYWORDS", [])

        # Separate exact vs substring keywords
        exact_keywords = getattr(module, "EXACT_KEYWORDS", trigger_keywords)
        substring_keywords = getattr(module, "SUBSTRING_KEYWORDS", [])

        # Conflict and priority
        conflicts_with = getattr(module, "CONFLICTS_WITH", [])
        priority = getattr(module, "PRIORITY", 100)  # Lower = higher priority

        # Metadata
        metadata = ProfileMetadata(
            name=name,
            version=version,
            description=getattr(module, "DESCRIPTION", ""),
            author=getattr(module, "AUTHOR", "community"),
            last_updated=getattr(module, "LAST_UPDATED", ""),
            icon=getattr(module, "ICON", ""),
            min_python_version=getattr(module, "MIN_PYTHON_VERSION", None),
            min_node_version=getattr(module, "MIN_NODE_VERSION", None),
        )

        # Structured guidance sections (optional)
        guidance_sections = None
        if hasattr(module, "get_guidance_sections"):
            sections_dict = module.get_guidance_sections()
            guidance_sections = GuidanceSections(**sections_dict)

        return ProfileInfo(
            name=name,
            version=version,
            module=module,
            technologies=technologies,
            trigger_keywords=trigger_keywords,
            exact_keywords=exact_keywords,
            substring_keywords=substring_keywords,
            conflicts_with=conflicts_with,
            priority=priority,
            metadata=metadata,
            guidance_sections=guidance_sections,
        )

    def _detect_conflicts(self, profile_names: list[str]) -> list[tuple[str, str]]:
        """Detect conflicts between profiles.

        Args:
            profile_names: List of applicable profile names

        Returns:
            List of (profile1, profile2) conflict pairs
        """
        conflicts = []
        profile_set = set(profile_names)

        # Check known conflicts
        for conflict_group in self.KNOWN_CONFLICTS:
            matching = profile_set & conflict_group
            if len(matching) > 1:
                sorted_matching = sorted(matching)
                for i, p1 in enumerate(sorted_matching):
                    for p2 in sorted_matching[i + 1:]:
                        conflicts.append((p1, p2))

        # Check profile-declared conflicts
        for name in profile_names:
            profile = self.profiles.get(name)
            if profile:
                for conflict in profile.conflicts_with:
                    if conflict in profile_set and conflict != name:
                        pair = tuple(sorted([name, conflict]))
                        if pair not in conflicts:
                            conflicts.append(pair)

        return conflicts

    def _resolve_conflicts(
        self,
        profiles: list[ProfileInfo],
        conflicts: list[tuple[str, str]],
    ) -> tuple[list[ProfileInfo], list[str]]:
        """Resolve conflicts using priority.

        Args:
            profiles: List of applicable profiles
            conflicts: List of conflict pairs

        Returns:
            Tuple of (resolved profiles, warning messages)
        """
        if not conflicts:
            return profiles, []

        if self.strict_conflicts:
            conflict_names = list({p for pair in conflicts for p in pair})
            raise ProfileConflictError(conflict_names)

        warnings = []
        excluded = set()

        # Resolve each conflict by priority
        for p1_name, p2_name in conflicts:
            if p1_name in excluded or p2_name in excluded:
                continue

            p1 = self.profiles.get(p1_name)
            p2 = self.profiles.get(p2_name)

            if p1 and p2:
                # Lower priority wins
                if p1.priority <= p2.priority:
                    excluded.add(p2_name)
                    warnings.append(
                        f"Conflict: {p1_name} vs {p2_name} - using {p1_name} (priority {p1.priority})"
                    )
                else:
                    excluded.add(p1_name)
                    warnings.append(
                        f"Conflict: {p1_name} vs {p2_name} - using {p2_name} (priority {p2.priority})"
                    )

        resolved = [p for p in profiles if p.name not in excluded]
        return resolved, warnings

    def get_applicable(
        self,
        tech_stack: list[str] | None = None,
        prompt: str = "",
        features: list[str] | None = None,
        sections: list[str] | None = None,
        auto_detect_features: bool = True,
    ) -> ProfileResult:
        """Get applicable profiles and merged guidance.

        Args:
            tech_stack: User-selected technologies
            prompt: Feature description
            features: Explicit feature list (e.g., ["auth", "database"])
            sections: Guidance sections to include (None = all)
            auto_detect_features: Auto-detect features from prompt

        Returns:
            ProfileResult with guidance, dependencies, and metadata
        """
        # Find applicable profiles
        applicable = []
        for profile in self.profiles.values():
            if profile.should_apply(tech_stack, prompt):
                applicable.append(profile)

        if not applicable:
            return ProfileResult(
                guidance="",
                dependencies=[],
                profiles=[],
                detected_features=[],
            )

        # Detect features
        detected_features = []
        if auto_detect_features:
            detected_features = detect_features(prompt)

        all_features = list(set((features or []) + detected_features))

        # Check for conflicts
        profile_names = [p.name for p in applicable]
        conflicts = self._detect_conflicts(profile_names)

        # Resolve conflicts
        resolved, warnings = self._resolve_conflicts(applicable, conflicts)

        # Sort by priority
        resolved.sort(key=lambda p: p.priority)

        # Merge guidance
        guidance_parts = []
        for profile in resolved:
            header = f"### {profile.name.title()} Guidelines"
            content = profile.get_guidance(sections)
            if content.strip():
                guidance_parts.append(f"{header}\n\n{content}")

        merged_guidance = "\n\n---\n\n".join(guidance_parts)

        # Merge dependencies (deduplicated)
        all_deps = []
        seen_bases = set()
        for profile in resolved:
            for dep in profile.get_dependencies(all_features):
                # Normalize: "fastapi>=0.109.0" -> "fastapi"
                base = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0]
                if base not in seen_bases:
                    seen_bases.add(base)
                    all_deps.append(dep)

        return ProfileResult(
            guidance=merged_guidance,
            dependencies=all_deps,
            profiles=[p.name for p in resolved],
            detected_features=detected_features,
            conflicts=[f"{p1} <-> {p2}" for p1, p2 in conflicts],
            warnings=warnings,
        )

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all available profiles with metadata.

        Returns:
            List of profile info dicts
        """
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.metadata.description,
                "author": p.metadata.author,
                "technologies": p.technologies,
                "keywords": p.trigger_keywords,
                "priority": p.priority,
                "conflicts_with": p.conflicts_with,
            }
            for p in sorted(self.profiles.values(), key=lambda x: x.name)
        ]

    def get_profile(self, name: str) -> ProfileInfo | None:
        """Get a specific profile by name."""
        return self.profiles.get(name)

    def reload(self) -> int:
        """Reload all profiles from disk.

        Returns:
            Number of profiles loaded
        """
        # Clear module cache for profiles
        import sys
        to_remove = [
            key for key in sys.modules
            if (key.startswith("profiles.") and key != "profiles.manager" and key != "profiles.base")
            or key.startswith("acf_ext_profile_")
        ]
        for key in to_remove:
            del sys.modules[key]

        count = self.discover_profiles()
        if self.load_extensions:
            count += self.discover_extension_profiles()
        return count


# Singleton instance
_manager: ProfileManager | None = None


def get_manager() -> ProfileManager:
    """Get the global profile manager instance."""
    global _manager
    if _manager is None:
        _manager = ProfileManager()
    return _manager
