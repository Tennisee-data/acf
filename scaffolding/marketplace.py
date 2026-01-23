"""Template marketplace for sharing and discovering templates.

Supports:
- Searching for templates from a central registry
- Downloading and installing templates
- Publishing templates to the marketplace
"""

import hashlib
import io
import json
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from .templates import (
    get_templates_dir,
    load_template_from_dir,
    ProjectTemplate,
    load_custom_templates,
)


# Default marketplace URL (can be overridden via config)
DEFAULT_MARKETPLACE_URL = "https://templates.coding-factory.dev/api/v1"


@dataclass
class MarketplaceTemplate:
    """Metadata for a marketplace template."""

    id: str
    name: str
    description: str
    author: str
    version: str
    language: str
    framework: str
    downloads: int
    stars: int
    created_at: str
    updated_at: str
    tags: list[str] = field(default_factory=list)
    download_url: str = ""
    checksum: str = ""


@dataclass
class MarketplaceSearchResult:
    """Search results from the marketplace."""

    templates: list[MarketplaceTemplate]
    total: int
    page: int
    per_page: int


class MarketplaceClient:
    """Client for interacting with the template marketplace.

    The marketplace provides:
    - Template discovery and search
    - Template downloads
    - Template publishing (with API key)

    Usage:
        client = MarketplaceClient()
        results = client.search("fastapi")
        client.install("my-awesome-template")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_MARKETPLACE_URL,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize marketplace client.

        Args:
            base_url: Marketplace API URL
            api_key: Optional API key for publishing
            cache_dir: Cache directory for downloads
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_dir = cache_dir or Path.home() / ".coding-factory" / "marketplace-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to marketplace."""
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise MarketplaceError(f"Failed to connect to marketplace: {e}")

    def _post(self, endpoint: str, data: Optional[dict] = None, files: Optional[dict] = None) -> dict:
        """Make POST request to marketplace."""
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        try:
            response = self.session.post(url, json=data, files=files, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise MarketplaceError(f"Failed to connect to marketplace: {e}")

    def search(
        self,
        query: str = "",
        language: Optional[str] = None,
        framework: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> MarketplaceSearchResult:
        """Search for templates in the marketplace.

        Args:
            query: Search query (matches name, description, tags)
            language: Filter by language
            framework: Filter by framework
            page: Page number
            per_page: Results per page

        Returns:
            Search results
        """
        params = {
            "q": query,
            "page": page,
            "per_page": per_page,
        }
        if language:
            params["language"] = language
        if framework:
            params["framework"] = framework

        data = self._get("/templates/search", params)

        templates = [
            MarketplaceTemplate(
                id=t["id"],
                name=t["name"],
                description=t.get("description", ""),
                author=t.get("author", "unknown"),
                version=t.get("version", "1.0.0"),
                language=t.get("language", "python"),
                framework=t.get("framework", "custom"),
                downloads=t.get("downloads", 0),
                stars=t.get("stars", 0),
                created_at=t.get("created_at", ""),
                updated_at=t.get("updated_at", ""),
                tags=t.get("tags", []),
                download_url=t.get("download_url", ""),
                checksum=t.get("checksum", ""),
            )
            for t in data.get("templates", [])
        ]

        return MarketplaceSearchResult(
            templates=templates,
            total=data.get("total", len(templates)),
            page=data.get("page", page),
            per_page=data.get("per_page", per_page),
        )

    def get_template_info(self, template_id: str) -> MarketplaceTemplate:
        """Get detailed info about a template.

        Args:
            template_id: Template ID or name

        Returns:
            Template metadata
        """
        data = self._get(f"/templates/{template_id}")

        return MarketplaceTemplate(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            author=data.get("author", "unknown"),
            version=data.get("version", "1.0.0"),
            language=data.get("language", "python"),
            framework=data.get("framework", "custom"),
            downloads=data.get("downloads", 0),
            stars=data.get("stars", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            tags=data.get("tags", []),
            download_url=data.get("download_url", ""),
            checksum=data.get("checksum", ""),
        )

    def list_featured(self, limit: int = 10) -> list[MarketplaceTemplate]:
        """Get featured/popular templates.

        Args:
            limit: Number of templates to return

        Returns:
            List of featured templates
        """
        data = self._get("/templates/featured", params={"limit": limit})
        return [
            MarketplaceTemplate(
                id=t["id"],
                name=t["name"],
                description=t.get("description", ""),
                author=t.get("author", "unknown"),
                version=t.get("version", "1.0.0"),
                language=t.get("language", "python"),
                framework=t.get("framework", "custom"),
                downloads=t.get("downloads", 0),
                stars=t.get("stars", 0),
                created_at=t.get("created_at", ""),
                updated_at=t.get("updated_at", ""),
                tags=t.get("tags", []),
                download_url=t.get("download_url", ""),
                checksum=t.get("checksum", ""),
            )
            for t in data.get("templates", [])
        ]

    def install(self, template_id: str, force: bool = False) -> ProjectTemplate:
        """Download and install a template from the marketplace.

        Args:
            template_id: Template ID or name
            force: Overwrite existing template

        Returns:
            Installed template
        """
        # Get template info
        info = self.get_template_info(template_id)

        # Check if already installed
        templates_dir = get_templates_dir()
        target_dir = templates_dir / info.name

        if target_dir.exists() and not force:
            raise MarketplaceError(
                f"Template '{info.name}' already exists. Use --force to overwrite."
            )

        # Download template package
        download_url = info.download_url
        if not download_url:
            download_url = f"{self.base_url}/templates/{template_id}/download"

        try:
            response = self.session.get(download_url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            raise MarketplaceError(f"Failed to download template: {e}")

        # Verify checksum
        content = response.content
        if info.checksum:
            actual_checksum = hashlib.sha256(content).hexdigest()
            if actual_checksum != info.checksum:
                raise MarketplaceError("Template checksum verification failed")

        # Extract template
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True)

        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                # Extract to temp dir first for safety
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tar.extractall(tmp_dir)

                    # Find the extracted directory
                    extracted = list(Path(tmp_dir).iterdir())
                    if len(extracted) == 1 and extracted[0].is_dir():
                        source = extracted[0]
                    else:
                        source = Path(tmp_dir)

                    # Move to target
                    import shutil
                    for item in source.iterdir():
                        shutil.move(str(item), str(target_dir / item.name))

        except tarfile.TarError as e:
            raise MarketplaceError(f"Failed to extract template: {e}")

        # Load and return template
        load_custom_templates()
        template = load_template_from_dir(target_dir)

        if not template:
            raise MarketplaceError("Failed to load installed template")

        return template

    def package_template(self, template_dir: Path) -> bytes:
        """Package a template for publishing.

        Args:
            template_dir: Path to template directory

        Returns:
            Gzipped tarball bytes
        """
        # Verify template is valid
        template = load_template_from_dir(template_dir)
        if not template:
            raise MarketplaceError("Invalid template directory")

        # Create tarball
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(template_dir, arcname=template.name)

        return buffer.getvalue()

    def publish(
        self,
        template_dir: Path,
        tags: Optional[list[str]] = None,
    ) -> MarketplaceTemplate:
        """Publish a template to the marketplace.

        Requires API key authentication.

        Args:
            template_dir: Path to template directory
            tags: Optional tags for categorization

        Returns:
            Published template info
        """
        if not self.api_key:
            raise MarketplaceError("API key required for publishing")

        # Load and validate template
        template = load_template_from_dir(template_dir)
        if not template:
            raise MarketplaceError("Invalid template directory")

        # Package template
        package = self.package_template(template_dir)
        checksum = hashlib.sha256(package).hexdigest()

        # Upload
        files = {
            "package": (f"{template.name}.tar.gz", package, "application/gzip"),
        }
        data = {
            "name": template.name,
            "description": template.description,
            "language": template.language,
            "framework": template.framework,
            "dependencies": json.dumps(template.dependencies),
            "dev_dependencies": json.dumps(template.dev_dependencies),
            "tags": json.dumps(tags or []),
            "checksum": checksum,
        }

        result = self._post("/templates/publish", data=data, files=files)

        return MarketplaceTemplate(
            id=result["id"],
            name=result["name"],
            description=result.get("description", ""),
            author=result.get("author", ""),
            version=result.get("version", "1.0.0"),
            language=result.get("language", "python"),
            framework=result.get("framework", "custom"),
            downloads=0,
            stars=0,
            created_at=result.get("created_at", datetime.now().isoformat()),
            updated_at=result.get("updated_at", datetime.now().isoformat()),
            tags=tags or [],
            checksum=checksum,
        )


class MarketplaceError(Exception):
    """Marketplace operation error."""
    pass


# =============================================================================
# Offline/Local Registry
# =============================================================================

class LocalRegistry:
    """Local registry of installed marketplace templates.

    Tracks which templates came from the marketplace
    for update checking and management.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize local registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = registry_path or (
            Path.home() / ".coding-factory" / "marketplace-registry.json"
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict = self._load()

    def _load(self) -> dict:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"templates": {}}
        return {"templates": {}}

    def _save(self) -> None:
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def add(self, template: MarketplaceTemplate) -> None:
        """Add a template to the registry.

        Args:
            template: Template metadata
        """
        self._registry["templates"][template.name] = {
            "id": template.id,
            "name": template.name,
            "version": template.version,
            "author": template.author,
            "installed_at": datetime.now().isoformat(),
            "checksum": template.checksum,
        }
        self._save()

    def remove(self, name: str) -> bool:
        """Remove a template from the registry.

        Args:
            name: Template name

        Returns:
            True if removed
        """
        if name in self._registry["templates"]:
            del self._registry["templates"][name]
            self._save()
            return True
        return False

    def get(self, name: str) -> Optional[dict]:
        """Get template info from registry.

        Args:
            name: Template name

        Returns:
            Template info or None
        """
        return self._registry["templates"].get(name)

    def list_installed(self) -> list[dict]:
        """List all installed marketplace templates.

        Returns:
            List of template info dicts
        """
        return list(self._registry["templates"].values())

    def is_marketplace_template(self, name: str) -> bool:
        """Check if a template came from the marketplace.

        Args:
            name: Template name

        Returns:
            True if from marketplace
        """
        return name in self._registry["templates"]
