"""Extension installer and marketplace client for ACF Local Edition.

Handles downloading, installing, and managing extensions from the marketplace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from extensions.manifest import ExtensionManifest, ExtensionType, ManifestError


class MarketplaceError(Exception):
    """Raised when marketplace operations fail."""

    pass


class InstallError(Exception):
    """Raised when extension installation fails."""

    pass


@dataclass
class MarketplaceExtension:
    """Extension listing from the marketplace."""

    id: str
    name: str
    version: str
    type: str
    author: str
    description: str
    price_usd: float
    download_url: str | None
    checksum: str | None
    is_official: bool
    downloads: int
    rating: float | None
    created_at: datetime
    updated_at: datetime
    keywords: list[str]
    # Token budget info (helps users check model compatibility)
    context_tokens: int = 0
    min_model_tier: str = "any"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketplaceExtension:
        """Create from API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            type=data["type"],
            author=data["author"],
            description=data["description"],
            price_usd=float(data.get("price_usd", 0)),
            download_url=data.get("download_url"),
            checksum=data.get("checksum"),
            is_official=data.get("is_official", False),
            downloads=data.get("downloads", 0),
            rating=data.get("rating"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            keywords=data.get("keywords", []),
            context_tokens=int(data.get("context_tokens", 0)),
            min_model_tier=data.get("min_model_tier", "any"),
        )

    @property
    def is_free(self) -> bool:
        return self.price_usd == 0.0


class MarketplaceClient:
    """Client for the ACF extension marketplace.

    Example:
        >>> client = MarketplaceClient()
        >>> extensions = client.search("security")
        >>> client.get_extension_details("secrets-scan")
    """

    DEFAULT_MARKETPLACE_URL = "https://marketplace.agentcodefactory.com/api/v1"

    def __init__(
        self,
        marketplace_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the marketplace client.

        Args:
            marketplace_url: Base URL for the marketplace API.
            api_key: API key for authenticated requests.
        """
        self.marketplace_url = marketplace_url or self.DEFAULT_MARKETPLACE_URL
        self.api_key = api_key or os.environ.get("ACF_MARKETPLACE_API_KEY")

        # Also try loading from config file if not in environment
        if not self.api_key:
            self.api_key = self._load_api_key_from_config()

        if not HTTPX_AVAILABLE:
            raise MarketplaceError(
                "httpx is required for marketplace access. "
                "Install with: pip install httpx"
            )

    def _load_api_key_from_config(self) -> str | None:
        """Load API key from ~/.coding-factory/config.env file."""
        config_file = Path.home() / ".coding-factory" / "config.env"
        if not config_file.exists():
            return None

        try:
            for line in config_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "ACF_MARKETPLACE_API_KEY":
                        return value.strip()
        except Exception:
            pass
        return None

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request."""
        url = urljoin(self.marketplace_url + "/", endpoint.lstrip("/"))

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method, url, params=params, json=json_data, headers=headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                if not self.api_key:
                    raise MarketplaceError(
                        "API key required for paid extensions.\n\n"
                        "To configure your API key:\n"
                        "  1. Create a key at: https://marketplace.agentcodefactory.com\n"
                        "  2. Run: acf auth login --key YOUR_KEY\n"
                        "     Or: export ACF_MARKETPLACE_API_KEY=YOUR_KEY\n"
                        "  3. Then retry this command"
                    )
                else:
                    raise MarketplaceError(
                        "Invalid or expired API key.\n\n"
                        "Create a new key at: https://marketplace.agentcodefactory.com\n"
                        "Then run: acf auth login --key YOUR_NEW_KEY"
                    )
            elif e.response.status_code == 402:
                raise MarketplaceError(
                    "Payment required. Please purchase this extension first.\n\n"
                    "Visit: https://marketplace.agentcodefactory.com"
                )
            elif e.response.status_code == 404:
                raise MarketplaceError("Extension not found.")
            else:
                raise MarketplaceError(f"API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise MarketplaceError(f"Connection error: {e}")

    def list_extensions(
        self,
        ext_type: ExtensionType | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> list[MarketplaceExtension]:
        """List available extensions.

        Args:
            ext_type: Filter by extension type.
            page: Page number (1-indexed).
            per_page: Results per page.

        Returns:
            List of marketplace extensions.
        """
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if ext_type:
            params["type"] = ext_type.value

        data = self._request("GET", "/extensions", params=params)
        return [MarketplaceExtension.from_dict(ext) for ext in data.get("items", [])]

    def search(
        self,
        query: str,
        ext_type: ExtensionType | None = None,
        free_only: bool = False,
    ) -> list[MarketplaceExtension]:
        """Search for extensions.

        Args:
            query: Search query.
            ext_type: Filter by type.
            free_only: Only show free extensions.

        Returns:
            List of matching extensions.
        """
        params: dict[str, Any] = {"q": query}
        if ext_type:
            params["type"] = ext_type.value
        if free_only:
            params["free_only"] = "true"

        data = self._request("GET", "/extensions/search", params=params)
        return [MarketplaceExtension.from_dict(ext) for ext in data.get("items", [])]

    def get_extension(self, name: str) -> MarketplaceExtension:
        """Get details for a specific extension.

        Args:
            name: Extension name.

        Returns:
            Extension details.
        """
        data = self._request("GET", f"/extensions/{name}")
        return MarketplaceExtension.from_dict(data)

    def get_featured(self) -> list[MarketplaceExtension]:
        """Get featured/curated extensions.

        Returns:
            List of featured extensions.
        """
        data = self._request("GET", "/extensions/featured")
        return [MarketplaceExtension.from_dict(ext) for ext in data.get("items", [])]

    def get_purchases(self) -> list[MarketplaceExtension]:
        """Get user's purchased extensions.

        Returns:
            List of purchased extensions.
        """
        if not self.api_key:
            raise MarketplaceError("Authentication required to view purchases.")

        data = self._request("GET", "/purchases")
        return [MarketplaceExtension.from_dict(ext) for ext in data.get("items", [])]

    def purchase(self, extension_name: str) -> dict[str, Any]:
        """Initiate purchase for an extension.

        Args:
            extension_name: Name of extension to purchase.

        Returns:
            Dictionary with checkout URL or confirmation.
        """
        if not self.api_key:
            raise MarketplaceError("Authentication required to make purchases.")

        return self._request("POST", f"/purchases/{extension_name}")

    def get_download_url(self, extension_name: str) -> str:
        """Get download URL for an extension.

        Args:
            extension_name: Name of extension.

        Returns:
            Signed download URL.

        Raises:
            MarketplaceError: If not purchased or unauthorized.
        """
        data = self._request("GET", f"/extensions/{extension_name}/download")
        return data["download_url"]

    def submit_extension(
        self, tarball_path: Path, price_usd: float = 0.0
    ) -> dict[str, Any]:
        """Submit an extension for review.

        Args:
            tarball_path: Path to extension tarball.
            price_usd: Requested price (0 for free).

        Returns:
            Submission details including review ID.
        """
        if not self.api_key:
            raise MarketplaceError("Authentication required to submit extensions.")

        # Upload tarball and get submission ID
        with open(tarball_path, "rb") as f:
            files = {"file": (tarball_path.name, f, "application/gzip")}
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    urljoin(self.marketplace_url + "/", "contributions"),
                    files=files,
                    data={"price_usd": str(price_usd)},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                return response.json()

    def get_submission_status(self, submission_id: str) -> dict[str, Any]:
        """Get status of a submission.

        Args:
            submission_id: Submission ID.

        Returns:
            Submission status details.
        """
        return self._request("GET", f"/contributions/{submission_id}/status")


class ExtensionInstaller:
    """Install and manage extensions locally.

    Example:
        >>> installer = ExtensionInstaller()
        >>> installer.install_from_marketplace("secrets-scan")
        >>> installer.install_from_path(Path("./my-extension"))
        >>> installer.uninstall("secrets-scan")
    """

    def __init__(
        self,
        extensions_dir: Path | None = None,
        marketplace_client: MarketplaceClient | None = None,
    ):
        """Initialize the installer.

        Args:
            extensions_dir: Local extensions directory.
            marketplace_client: Marketplace client for remote installs.
        """
        self.extensions_dir = Path(
            extensions_dir or Path.home() / ".coding-factory" / "extensions"
        )
        self.marketplace = marketplace_client

    def install_from_marketplace(
        self, name: str, force: bool = False
    ) -> ExtensionManifest:
        """Install an extension from the marketplace.

        Args:
            name: Extension name.
            force: Overwrite if already installed.

        Returns:
            Installed extension manifest.

        Raises:
            InstallError: If installation fails.
        """
        if not self.marketplace:
            raise InstallError("Marketplace client not configured")

        # Get extension info
        ext = self.marketplace.get_extension(name)

        # Check if already installed
        ext_type = ExtensionType(ext.type)
        target_dir = self._get_target_dir(ext_type, name)
        if target_dir.exists() and not force:
            raise InstallError(
                f"Extension '{name}' is already installed. "
                "Use --force to overwrite."
            )

        # Check if purchase required
        if not ext.is_free:
            # Try to get download URL (will fail if not purchased)
            try:
                download_url = self.marketplace.get_download_url(name)
            except MarketplaceError as e:
                if "Payment required" in str(e):
                    raise InstallError(
                        f"Extension '{name}' requires purchase.\n"
                        f"Price: ${ext.price_usd:.2f}\n"
                        f"Purchase at: https://marketplace.agentcodefactory.com"
                    )
                raise
        else:
            download_url = self.marketplace.get_download_url(name)

        # Download and install
        return self._download_and_install(
            download_url, ext_type, name, ext.checksum, force
        )

    def _download_and_install(
        self,
        download_url: str,
        ext_type: ExtensionType,
        name: str,
        expected_checksum: str | None,
        force: bool,
    ) -> ExtensionManifest:
        """Download and install an extension tarball."""
        if not HTTPX_AVAILABLE:
            raise InstallError("httpx is required for marketplace installs")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tarball_path = tmp_path / f"{name}.tar.gz"

            # Download
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(download_url)
                response.raise_for_status()
                tarball_path.write_bytes(response.content)

            # Verify checksum
            if expected_checksum:
                actual_checksum = hashlib.sha256(
                    tarball_path.read_bytes()
                ).hexdigest()
                if actual_checksum != expected_checksum:
                    raise InstallError(
                        f"Checksum mismatch for {name}. "
                        "The download may be corrupted."
                    )

            # Extract and install
            return self.install_from_tarball(tarball_path, force=force)

    def install_from_tarball(
        self, tarball_path: Path, force: bool = False
    ) -> ExtensionManifest:
        """Install an extension from a tarball.

        Args:
            tarball_path: Path to .tar.gz file.
            force: Overwrite if exists.

        Returns:
            Installed extension manifest.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Extract tarball
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(tmp_path, filter="data")

            # Find manifest
            extracted_dirs = list(tmp_path.iterdir())
            if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
                ext_root = extracted_dirs[0]
            else:
                ext_root = tmp_path

            manifest_path = ext_root / "manifest.yaml"
            if not manifest_path.exists():
                raise InstallError("Tarball does not contain manifest.yaml")

            return self.install_from_path(ext_root, force=force)

    def install_from_path(
        self, source_path: Path, force: bool = False
    ) -> ExtensionManifest:
        """Install an extension from a local directory.

        Args:
            source_path: Path to extension directory.
            force: Overwrite if exists.

        Returns:
            Installed extension manifest.
        """
        manifest_path = source_path / "manifest.yaml"
        if not manifest_path.exists():
            raise InstallError(f"No manifest.yaml found in {source_path}")

        manifest = ExtensionManifest.from_yaml(manifest_path)

        # Determine target directory
        target_dir = self._get_target_dir(manifest.type, manifest.name)

        if target_dir.exists():
            if not force:
                raise InstallError(
                    f"Extension '{manifest.name}' already exists. "
                    "Use --force to overwrite."
                )
            shutil.rmtree(target_dir)

        # Create parent directories
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy extension
        shutil.copytree(source_path, target_dir)

        # Install Python dependencies
        self._install_requirements(target_dir, manifest)

        return manifest

    def _get_target_dir(self, ext_type: ExtensionType, name: str) -> Path:
        """Get installation directory for an extension."""
        type_dirs = {
            ExtensionType.AGENT: "agents",
            ExtensionType.PROFILE: "profiles",
            ExtensionType.RAG: "rag",
            ExtensionType.SKILL: "skills",
        }
        return self.extensions_dir / type_dirs[ext_type] / name

    def _install_requirements(
        self, ext_dir: Path, manifest: ExtensionManifest
    ) -> None:
        """Install Python dependencies for an extension."""
        requirements_file = ext_dir / "requirements.txt"

        if requirements_file.exists():
            requirements = requirements_file.read_text().strip().split("\n")
        elif manifest.requires:
            requirements = manifest.requires
        else:
            return

        if not requirements:
            return

        try:
            subprocess.run(
                ["pip", "install", "--quiet", *requirements],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise InstallError(
                f"Failed to install dependencies: {e.stderr.decode()}"
            )

    def uninstall(self, name: str) -> bool:
        """Uninstall an extension.

        Args:
            name: Extension name.

        Returns:
            True if uninstalled, False if not found.
        """
        for ext_type in ExtensionType:
            target_dir = self._get_target_dir(ext_type, name)
            if target_dir.exists():
                shutil.rmtree(target_dir)
                return True
        return False

    def list_installed(self) -> list[ExtensionManifest]:
        """List all installed extensions.

        Returns:
            List of installed extension manifests.
        """
        installed: list[ExtensionManifest] = []

        for type_dir in ["agents", "profiles", "rag", "skills"]:
            type_path = self.extensions_dir / type_dir
            if not type_path.exists():
                continue

            for ext_dir in type_path.iterdir():
                if not ext_dir.is_dir():
                    continue
                manifest_path = ext_dir / "manifest.yaml"
                if manifest_path.exists():
                    try:
                        installed.append(ExtensionManifest.from_yaml(manifest_path))
                    except ManifestError:
                        pass

        return installed

    def get_installed(self, name: str) -> ExtensionManifest | None:
        """Get manifest for an installed extension.

        Args:
            name: Extension name.

        Returns:
            Manifest or None if not installed.
        """
        for ext_type in ExtensionType:
            target_dir = self._get_target_dir(ext_type, name)
            manifest_path = target_dir / "manifest.yaml"
            if manifest_path.exists():
                try:
                    return ExtensionManifest.from_yaml(manifest_path)
                except ManifestError:
                    pass
        return None

    def is_installed(self, name: str) -> bool:
        """Check if an extension is installed."""
        return self.get_installed(name) is not None

    def check_updates(self) -> list[tuple[ExtensionManifest, MarketplaceExtension]]:
        """Check for available updates.

        Returns:
            List of (installed, latest) pairs where update is available.
        """
        if not self.marketplace:
            return []

        updates: list[tuple[ExtensionManifest, MarketplaceExtension]] = []

        for installed in self.list_installed():
            try:
                latest = self.marketplace.get_extension(installed.name)
                if self._version_compare(latest.version, installed.version) > 0:
                    updates.append((installed, latest))
            except MarketplaceError:
                pass

        return updates

    def _version_compare(self, v1: str, v2: str) -> int:
        """Compare two version strings.

        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
        """
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]

            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                if p1 > p2:
                    return 1

            if len(parts1) < len(parts2):
                return -1
            if len(parts1) > len(parts2):
                return 1

            return 0
        except ValueError:
            # Fallback to string comparison
            return -1 if v1 < v2 else (1 if v1 > v2 else 0)

    def create_extension_scaffold(
        self,
        name: str,
        ext_type: ExtensionType,
        output_dir: Path | None = None,
    ) -> Path:
        """Create a scaffold for a new extension.

        Args:
            name: Extension name.
            ext_type: Extension type.
            output_dir: Output directory (default: current directory).

        Returns:
            Path to created extension directory.
        """
        output_dir = output_dir or Path.cwd()
        ext_dir = output_dir / name
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest_data = {
            "name": name,
            "version": "0.1.0",
            "type": ext_type.value,
            "author": "Your Name",
            "description": f"A custom {ext_type.value} extension",
            "license": "MIT",
        }

        if ext_type == ExtensionType.AGENT:
            manifest_data["hook_point"] = "after:implementation"
            manifest_data["agent_class"] = f"{self._to_class_name(name)}Agent"
            self._create_agent_scaffold(ext_dir, manifest_data)
        elif ext_type == ExtensionType.PROFILE:
            manifest_data["profile_class"] = f"{self._to_class_name(name)}Profile"
            self._create_profile_scaffold(ext_dir, manifest_data)
        elif ext_type == ExtensionType.RAG:
            manifest_data["retriever_class"] = f"{self._to_class_name(name)}Retriever"
            self._create_rag_scaffold(ext_dir, manifest_data)
        elif ext_type == ExtensionType.SKILL:
            manifest_data["skill_class"] = f"{self._to_class_name(name)}Skill"
            manifest_data["input_type"] = "files"
            manifest_data["output_type"] = "modified_files"
            manifest_data["file_patterns"] = ["*.py"]
            manifest_data["supports_dry_run"] = True
            self._create_skill_scaffold(ext_dir, manifest_data)

        # Write manifest
        import yaml

        manifest_path = ext_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest_data, f, default_flow_style=False, sort_keys=False)

        return ext_dir

    def _to_class_name(self, name: str) -> str:
        """Convert extension name to class name."""
        return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))

    def _create_agent_scaffold(
        self, ext_dir: Path, manifest_data: dict[str, Any]
    ) -> None:
        """Create scaffold files for an agent extension."""
        class_name = manifest_data["agent_class"]
        agent_code = f'''"""Custom agent extension."""

from agents.base import BaseAgent, AgentInput, AgentOutput


class {class_name}(BaseAgent):
    """Custom agent that hooks into the pipeline.

    This agent runs {manifest_data.get("hook_point", "after:implementation")}.
    """

    async def run(self, input: AgentInput) -> AgentOutput:
        """Execute the agent logic.

        Args:
            input: Agent input with context and previous outputs.

        Returns:
            Agent output with results.
        """
        # Access pipeline context
        context = input.context

        # Your agent logic here
        result = {{"status": "success", "message": "Agent executed successfully"}}

        return AgentOutput(
            content=result,
            metadata={{"agent": "{manifest_data["name"]}"}},
        )
'''
        (ext_dir / "agent.py").write_text(agent_code)
        (ext_dir / "requirements.txt").write_text("# Add your dependencies here\n")

    def _create_profile_scaffold(
        self, ext_dir: Path, manifest_data: dict[str, Any]
    ) -> None:
        """Create scaffold files for a profile extension."""
        class_name = manifest_data["profile_class"]
        profile_code = f'''"""Custom profile extension."""

PROFILE_NAME = "{manifest_data["name"]}"
PROFILE_VERSION = "0.1"
TECHNOLOGIES = []  # Add technology keywords
TRIGGER_KEYWORDS = []  # Add trigger keywords
CONFLICTS_WITH = []  # Add conflicting profile names
PRIORITY = 50  # Lower = higher priority

SYSTEM_GUIDANCE = """
Add your technology-specific guidance here.
This text will be injected into agent prompts.
"""

DEPENDENCIES = []  # Add pip dependencies

OPTIONAL_DEPENDENCIES = {{
    # "feature": ["dependency1", "dependency2"],
}}


class {class_name}:
    """Custom profile class."""

    @staticmethod
    def get_guidance(sections: list[str] | None = None) -> str:
        """Get guidance text, optionally filtered by sections."""
        return SYSTEM_GUIDANCE

    @staticmethod
    def get_dependencies(features: list[str] | None = None) -> list[str]:
        """Get dependencies, optionally including optional ones for features."""
        deps = list(DEPENDENCIES)
        if features:
            for feature in features:
                if feature in OPTIONAL_DEPENDENCIES:
                    deps.extend(OPTIONAL_DEPENDENCIES[feature])
        return deps
'''
        (ext_dir / "profile.py").write_text(profile_code)

    def _create_rag_scaffold(
        self, ext_dir: Path, manifest_data: dict[str, Any]
    ) -> None:
        """Create scaffold files for a RAG extension."""
        class_name = manifest_data["retriever_class"]
        retriever_code = f'''"""Custom RAG retriever extension."""

from pathlib import Path
from typing import Any


class {class_name}:
    """Custom RAG retriever.

    Implement custom retrieval logic for code context.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the retriever.

        Args:
            config: Retriever configuration.
        """
        self.config = config or {{}}

    def index(self, project_dir: Path) -> None:
        """Index a project directory.

        Args:
            project_dir: Path to project to index.
        """
        # Your indexing logic here
        pass

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        file_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant code snippets.

        Args:
            query: Search query.
            top_k: Number of results to return.
            file_filter: Optional file path filters.

        Returns:
            List of relevant code snippets with metadata.
        """
        # Your retrieval logic here
        return []

    def clear(self) -> None:
        """Clear the index."""
        pass
'''
        (ext_dir / "retriever.py").write_text(retriever_code)
        (ext_dir / "requirements.txt").write_text("# Add your dependencies here\n")

    def _create_skill_scaffold(
        self, ext_dir: Path, manifest_data: dict[str, Any]
    ) -> None:
        """Create scaffold files for a skill extension."""
        class_name = manifest_data["skill_class"]
        skill_code = f'''"""Custom skill extension."""

from pathlib import Path
from typing import Any

from skills.base import BaseSkill, FileChange, SkillInput, SkillOutput


class {class_name}(BaseSkill):
    """Custom skill for code transformation.

    Processes files matching the configured file_patterns.
    """

    def run(self, input_data: SkillInput) -> SkillOutput:
        """Execute the skill on target files.

        Args:
            input_data: Skill input with target paths and config.

        Returns:
            SkillOutput with changes and summary.
        """
        changes: list[FileChange] = []

        for path in input_data.target_paths:
            content = path.read_text()
            modified = self.transform(content, path)

            if modified != content:
                changes.append(FileChange(
                    path=path,
                    original_content=content,
                    modified_content=modified,
                    change_type="modified",
                ))

        return SkillOutput(
            success=True,
            changes=changes,
            summary=f"Transformed {{len(changes)}} files",
        )

    def transform(self, content: str, path: Path) -> str:
        """Transform a single file's content.

        Args:
            content: File content to transform.
            path: Path to the file.

        Returns:
            Transformed content.
        """
        # Your transformation logic here
        return content
'''
        (ext_dir / "skill.py").write_text(skill_code)

    def package_extension(self, ext_dir: Path, output_path: Path | None = None) -> Path:
        """Package an extension into a tarball for submission.

        Args:
            ext_dir: Extension directory.
            output_path: Output tarball path.

        Returns:
            Path to created tarball.
        """
        manifest_path = ext_dir / "manifest.yaml"
        if not manifest_path.exists():
            raise InstallError(f"No manifest.yaml found in {ext_dir}")

        manifest = ExtensionManifest.from_yaml(manifest_path)

        if output_path is None:
            output_path = ext_dir.parent / f"{manifest.name}-{manifest.version}.tar.gz"

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(ext_dir, arcname=manifest.name)

        return output_path
