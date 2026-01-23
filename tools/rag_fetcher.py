#!/usr/bin/env python3
"""RAG Documentation Fetcher.

Fetches and maintains external API documentation for the RAG folder.
Supports multiple sources: llms.txt, OpenAPI specs, GitHub docs, and URLs.

Usage:
    python tools/rag_fetcher.py --all           # Fetch all sources
    python tools/rag_fetcher.py stripe twilio   # Fetch specific sources
    python tools/rag_fetcher.py --list          # List available sources
    python tools/rag_fetcher.py --status        # Show fetch status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    import tomli as tomllib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "rag_sources.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "rag"


@dataclass
class FetchResult:
    """Result of fetching a documentation source."""

    source_name: str
    success: bool
    files_created: list[str] = field(default_factory=list)
    error: str | None = None
    from_cache: bool = False


@dataclass
class SourceConfig:
    """Configuration for a documentation source."""

    name: str
    description: str
    priority: str
    llms_txt: str | None = None
    openapi: str | None = None
    github_docs: str | None = None
    urls: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    safety_critical: list[str] = field(default_factory=list)


class RAGFetcher:
    """Fetches and organizes API documentation for RAG."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        output_dir: Path = DEFAULT_OUTPUT,
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.settings: dict[str, Any] = {}
        self.sources: dict[str, SourceConfig] = {}
        self._load_config()

        # HTTP client with reasonable timeouts
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": self.settings.get(
                    "user_agent", "CodingFactory-RAG-Fetcher/1.0"
                )
            },
        )

    def _load_config(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        content = self.config_path.read_text(encoding="utf-8")
        config = tomllib.loads(content)

        # Extract settings
        self.settings = config.pop("settings", {})
        if "output_dir" in self.settings:
            self.output_dir = PROJECT_ROOT / self.settings["output_dir"]

        # Parse sources
        for name, data in config.items():
            if isinstance(data, dict):
                self.sources[name] = SourceConfig(
                    name=name,
                    description=data.get("description", ""),
                    priority=data.get("priority", "medium"),
                    llms_txt=data.get("llms_txt"),
                    openapi=data.get("openapi"),
                    github_docs=data.get("github_docs"),
                    urls=data.get("urls", []),
                    topics=data.get("topics", []),
                    safety_critical=data.get("safety_critical", []),
                )

        logger.info("Loaded %d sources from config", len(self.sources))

    def _get_meta_path(self, source_name: str) -> Path:
        """Get path to metadata file for a source."""
        return self.output_dir / source_name / "_meta.json"

    def _load_meta(self, source_name: str) -> dict:
        """Load metadata for a source."""
        meta_path = self._get_meta_path(source_name)
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def _save_meta(self, source_name: str, meta: dict) -> None:
        """Save metadata for a source."""
        meta_path = self._get_meta_path(source_name)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    def _is_cache_valid(self, source_name: str) -> bool:
        """Check if cached data is still valid."""
        meta = self._load_meta(source_name)
        if not meta.get("last_fetched"):
            return False

        last_fetched = datetime.fromisoformat(meta["last_fetched"])
        cache_days = self.settings.get("cache_days", 30)
        return datetime.now() - last_fetched < timedelta(days=cache_days)

    def _fetch_url(self, url: str) -> str | None:
        """Fetch content from a URL."""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def _fetch_llms_txt(self, source: SourceConfig) -> list[tuple[str, str]]:
        """Fetch llms.txt documentation.

        Returns list of (filename, content) tuples.
        """
        if not source.llms_txt:
            return []

        content = self._fetch_url(source.llms_txt)
        if not content:
            return []

        # llms.txt is already LLM-optimized, save as-is
        return [("llms.md", f"# {source.name.title()} - LLM Documentation\n\n{content}")]

    def _fetch_openapi(self, source: SourceConfig) -> list[tuple[str, str]]:
        """Fetch and convert OpenAPI spec to markdown.

        Returns list of (filename, content) tuples.
        """
        if not source.openapi:
            return []

        content = self._fetch_url(source.openapi)
        if not content:
            return []

        try:
            # Parse OpenAPI spec
            if source.openapi.endswith(".yaml") or source.openapi.endswith(".yml"):
                import yaml
                spec = yaml.safe_load(content)
            else:
                spec = json.loads(content)

            # Convert to markdown
            md_content = self._openapi_to_markdown(spec, source)
            return [("api_reference.md", md_content)]

        except Exception as e:
            logger.warning("Failed to parse OpenAPI spec for %s: %s", source.name, e)
            return []

    def _openapi_to_markdown(self, spec: dict, source: SourceConfig) -> str:
        """Convert OpenAPI spec to markdown documentation."""
        lines = [
            f"# {source.name.title()} API Reference",
            "",
            f"*Generated from OpenAPI spec*",
            "",
        ]

        # Info section
        info = spec.get("info", {})
        if info.get("description"):
            lines.extend([info["description"], ""])

        # Extract endpoints grouped by tag
        paths = spec.get("paths", {})
        endpoints_by_tag: dict[str, list] = {}

        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    tags = details.get("tags", ["Other"])
                    for tag in tags:
                        if tag not in endpoints_by_tag:
                            endpoints_by_tag[tag] = []
                        endpoints_by_tag[tag].append({
                            "method": method.upper(),
                            "path": path,
                            "summary": details.get("summary", ""),
                            "description": details.get("description", ""),
                            "operation_id": details.get("operationId", ""),
                        })

        # Write endpoints
        for tag, endpoints in sorted(endpoints_by_tag.items()):
            lines.extend([f"## {tag}", ""])

            for ep in endpoints:
                lines.append(f"### `{ep['method']} {ep['path']}`")
                if ep["summary"]:
                    lines.append(f"\n{ep['summary']}")
                if ep["description"] and ep["description"] != ep["summary"]:
                    lines.append(f"\n{ep['description']}")
                lines.append("")

        # Add safety-critical notes if applicable
        if source.safety_critical:
            lines.extend([
                "---",
                "",
                "## Safety-Critical Implementation Notes",
                "",
            ])
            for note in source.safety_critical:
                lines.append(f"- **{note}**: Ensure proper implementation")
            lines.append("")

        return "\n".join(lines)

    def _fetch_github_docs(self, source: SourceConfig) -> list[tuple[str, str]]:
        """Fetch documentation from GitHub repository.

        Returns list of (filename, content) tuples.
        """
        if not source.github_docs:
            return []

        # Parse GitHub URL
        parsed = urlparse(source.github_docs)
        if "github.com" not in parsed.netloc:
            return []

        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            return []

        owner, repo = parts[0], parts[1]

        # Try to fetch README
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
        content = self._fetch_url(readme_url)

        if not content:
            # Try master branch
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
            content = self._fetch_url(readme_url)

        if content:
            return [("readme.md", content)]

        return []

    def _fetch_urls(self, source: SourceConfig) -> list[tuple[str, str]]:
        """Fetch documentation from direct URLs.

        Returns list of (filename, content) tuples.
        """
        results = []

        for url in source.urls:
            content = self._fetch_url(url)
            if not content:
                continue

            # Generate filename from URL
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            if path_parts:
                filename = "_".join(path_parts[-2:]) + ".md"
            else:
                filename = f"doc_{hashlib.md5(url.encode()).hexdigest()[:8]}.md"

            # Clean filename
            filename = re.sub(r"[^\w\-_.]", "_", filename)

            # If it's HTML, try to extract main content
            if "<html" in content.lower():
                content = self._html_to_markdown(content, url)

            results.append((filename, content))

        return results

    def _html_to_markdown(self, html: str, source_url: str) -> str:
        """Convert HTML to markdown (basic extraction)."""
        # Try to use BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Try to find main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive newlines
            text = re.sub(r"\n{3,}", "\n\n", text)

            return f"# Documentation\n\n*Source: {source_url}*\n\n{text}"

        except ImportError:
            # Fallback: basic regex extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return f"# Documentation\n\n*Source: {source_url}*\n\n{text[:10000]}"

    def fetch_source(self, source_name: str, force: bool = False) -> FetchResult:
        """Fetch documentation for a single source.

        Args:
            source_name: Name of the source to fetch
            force: Force fetch even if cache is valid

        Returns:
            FetchResult with status and files created
        """
        if source_name not in self.sources:
            return FetchResult(
                source_name=source_name,
                success=False,
                error=f"Unknown source: {source_name}",
            )

        source = self.sources[source_name]

        # Check cache
        if not force and self._is_cache_valid(source_name):
            logger.info("Using cached data for %s", source_name)
            return FetchResult(
                source_name=source_name,
                success=True,
                from_cache=True,
            )

        logger.info("Fetching documentation for %s...", source_name)

        # Create output directory
        source_dir = self.output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        files_created = []
        all_content: list[tuple[str, str]] = []

        # Fetch from sources in priority order
        # 1. llms.txt (best for LLMs)
        all_content.extend(self._fetch_llms_txt(source))

        # 2. OpenAPI spec
        all_content.extend(self._fetch_openapi(source))

        # 3. GitHub docs
        all_content.extend(self._fetch_github_docs(source))

        # 4. Direct URLs
        all_content.extend(self._fetch_urls(source))

        if not all_content:
            return FetchResult(
                source_name=source_name,
                success=False,
                error="No content fetched from any source",
            )

        # Write files
        for filename, content in all_content:
            file_path = source_dir / filename
            file_path.write_text(content, encoding="utf-8")
            files_created.append(str(file_path.relative_to(self.output_dir)))
            logger.info("  Created: %s", filename)

        # Create index file
        index_content = self._create_index(source, files_created)
        index_path = source_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")
        files_created.append(str(index_path.relative_to(self.output_dir)))

        # Update metadata
        self._save_meta(source_name, {
            "last_fetched": datetime.now().isoformat(),
            "files": files_created,
            "source_urls": {
                "llms_txt": source.llms_txt,
                "openapi": source.openapi,
                "github_docs": source.github_docs,
                "urls": source.urls,
            },
        })

        return FetchResult(
            source_name=source_name,
            success=True,
            files_created=files_created,
        )

    def _create_index(self, source: SourceConfig, files: list[str]) -> str:
        """Create index file for a source."""
        lines = [
            f"# {source.name.title()}",
            "",
            f"*{source.description}*",
            "",
            f"**Priority:** {source.priority}",
            "",
        ]

        if source.topics:
            lines.extend([
                "## Topics",
                "",
                *[f"- {topic}" for topic in source.topics],
                "",
            ])

        if source.safety_critical:
            lines.extend([
                "## Safety-Critical Patterns",
                "",
                "These patterns require careful implementation:",
                "",
                *[f"- **{pattern}**" for pattern in source.safety_critical],
                "",
            ])

        lines.extend([
            "## Documentation Files",
            "",
            *[f"- [{f.split('/')[-1]}]({f.split('/')[-1]})" for f in files if not f.endswith("index.md")],
            "",
            "---",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*",
        ])

        return "\n".join(lines)

    def fetch_all(self, force: bool = False, priority: str | None = None) -> list[FetchResult]:
        """Fetch all configured sources.

        Args:
            force: Force fetch even if cache is valid
            priority: Only fetch sources with this priority

        Returns:
            List of FetchResults
        """
        results = []

        for name, source in self.sources.items():
            if priority and source.priority != priority:
                continue

            result = self.fetch_source(name, force=force)
            results.append(result)

        return results

    def list_sources(self) -> None:
        """Print list of available sources."""
        print("\nAvailable RAG Sources:")
        print("=" * 60)

        by_priority = {"high": [], "medium": [], "low": []}
        for name, source in self.sources.items():
            by_priority.get(source.priority, by_priority["medium"]).append((name, source))

        for priority in ["high", "medium", "low"]:
            if by_priority[priority]:
                print(f"\n[{priority.upper()} PRIORITY]")
                for name, source in sorted(by_priority[priority]):
                    cached = "cached" if self._is_cache_valid(name) else "stale"
                    print(f"  {name:20} - {source.description} ({cached})")

    def show_status(self) -> None:
        """Show status of all sources."""
        print("\nRAG Sources Status:")
        print("=" * 60)

        for name, source in sorted(self.sources.items()):
            meta = self._load_meta(name)

            if meta.get("last_fetched"):
                last = datetime.fromisoformat(meta["last_fetched"])
                age = (datetime.now() - last).days
                status = f"fetched {age} days ago"
                files = len(meta.get("files", []))
                print(f"  {name:20} - {status} ({files} files)")
            else:
                print(f"  {name:20} - not fetched")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and maintain RAG documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all              Fetch all sources
  %(prog)s stripe twilio      Fetch specific sources
  %(prog)s --all --force      Force refresh all
  %(prog)s --priority high    Fetch only high priority
  %(prog)s --list             List available sources
  %(prog)s --status           Show fetch status
        """,
    )

    parser.add_argument("sources", nargs="*", help="Specific sources to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all sources")
    parser.add_argument("--force", action="store_true", help="Force fetch, ignore cache")
    parser.add_argument("--priority", choices=["high", "medium", "low"], help="Filter by priority")
    parser.add_argument("--list", action="store_true", help="List available sources")
    parser.add_argument("--status", action="store_true", help="Show fetch status")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")

    args = parser.parse_args()

    try:
        fetcher = RAGFetcher(config_path=args.config, output_dir=args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        fetcher.list_sources()
        return

    if args.status:
        fetcher.show_status()
        return

    if args.all:
        results = fetcher.fetch_all(force=args.force, priority=args.priority)
    elif args.sources:
        results = [fetcher.fetch_source(s, force=args.force) for s in args.sources]
    else:
        parser.print_help()
        return

    # Print summary
    print("\n" + "=" * 60)
    print("Fetch Summary:")
    print("=" * 60)

    success = sum(1 for r in results if r.success)
    cached = sum(1 for r in results if r.from_cache)
    failed = sum(1 for r in results if not r.success)

    for result in results:
        if result.success:
            if result.from_cache:
                print(f"  [CACHED] {result.source_name}")
            else:
                print(f"  [OK]     {result.source_name} ({len(result.files_created)} files)")
        else:
            print(f"  [FAIL]   {result.source_name}: {result.error}")

    print(f"\nTotal: {success} success ({cached} cached), {failed} failed")


if __name__ == "__main__":
    main()
