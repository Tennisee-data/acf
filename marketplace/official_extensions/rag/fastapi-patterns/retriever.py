"""FastAPI Patterns Retriever - Keyword-based pattern retrieval.

This retriever loads production-ready FastAPI patterns and returns
relevant examples based on keyword matching. No heavy dependencies
required - designed to work with any model size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatternMatch:
    """A matched pattern with relevance score."""

    path: str
    content: str
    category: str
    keywords: list[str]
    score: float


class FastAPIPatternRetriever:
    """Keyword-based FastAPI pattern retriever.

    Loads patterns from the patterns/ directory and returns relevant
    examples based on query keyword matching.
    """

    # Keyword mappings for each category
    CATEGORY_KEYWORDS = {
        "rate-limiting": [
            "rate limit", "rate-limit", "ratelimit", "throttle", "throttling",
            "slowapi", "fastapi-limiter", "429", "too many requests",
            "brute force", "login limit", "request limit"
        ],
        "authentication": [
            "auth", "authentication", "jwt", "token", "oauth", "oauth2",
            "login", "password", "bcrypt", "passlib", "bearer", "api key",
            "session", "cookie", "refresh token"
        ],
        "webhooks": [
            "webhook", "stripe webhook", "github webhook", "payment webhook",
            "signature", "verify signature", "construct_event", "hmac"
        ],
        "database": [
            "database", "db", "sqlalchemy", "postgres", "mysql", "sqlite",
            "session", "connection", "pool", "async database", "alembic",
            "migration", "orm", "query"
        ],
        "file-uploads": [
            "upload", "file upload", "multipart", "image upload",
            "document upload", "s3", "storage", "blob"
        ],
        "caching": [
            "cache", "caching", "redis", "memcache", "memoize",
            "ttl", "expire", "invalidate"
        ],
        "error-handling": [
            "error", "exception", "handler", "middleware", "logging",
            "traceback", "500", "404", "validation error"
        ],
        "testing": [
            "test", "testing", "pytest", "httpx", "testclient",
            "mock", "fixture", "async test"
        ],
    }

    def __init__(self, extension_path: Path | str | None = None):
        """Initialize the retriever.

        Args:
            extension_path: Path to the extension directory.
                           If None, uses the directory containing this file.
        """
        if extension_path:
            self.base_path = Path(extension_path)
        else:
            self.base_path = Path(__file__).parent

        self.patterns_dir = self.base_path / "patterns"
        self.patterns: dict[str, list[dict]] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load all patterns from the patterns directory."""
        if not self.patterns_dir.exists():
            return

        for category_dir in self.patterns_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            self.patterns[category] = []

            for pattern_file in category_dir.glob("*.py"):
                content = pattern_file.read_text(encoding="utf-8")

                # Extract metadata from docstring if present
                keywords = self._extract_keywords(content, category)

                self.patterns[category].append({
                    "path": str(pattern_file.relative_to(self.base_path)),
                    "name": pattern_file.stem,
                    "content": content,
                    "keywords": keywords,
                })

    def _extract_keywords(self, content: str, category: str) -> list[str]:
        """Extract keywords from pattern content.

        Looks for a Keywords: line in the docstring, falls back to
        category keywords.
        """
        # Look for Keywords: line in docstring
        match = re.search(r'Keywords?:\s*(.+?)(?:\n|""")', content, re.IGNORECASE)
        if match:
            keywords = [k.strip().lower() for k in match.group(1).split(",")]
            return keywords

        # Fall back to category keywords
        return self.CATEGORY_KEYWORDS.get(category, [])

    def _calculate_score(self, query: str, pattern: dict) -> float:
        """Calculate relevance score for a pattern.

        Args:
            query: Search query
            pattern: Pattern dict with keywords

        Returns:
            Score from 0.0 to 1.0
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        score = 0.0
        max_score = len(pattern["keywords"]) + 1  # +1 for content match

        # Check keyword matches
        for keyword in pattern["keywords"]:
            if keyword in query_lower:
                score += 1.0
            elif any(word in keyword for word in query_words):
                score += 0.5

        # Check content matches
        content_lower = pattern["content"].lower()
        content_matches = sum(1 for word in query_words if word in content_lower)
        score += min(content_matches / len(query_words), 1.0) if query_words else 0

        return min(score / max_score, 1.0) if max_score > 0 else 0.0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant patterns.

        Args:
            query: Search query (e.g., "rate limiting login")
            top_k: Maximum number of results to return

        Returns:
            List of pattern dicts with relevance scores, sorted by score
        """
        results = []

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                score = self._calculate_score(query, pattern)

                if score > 0.1:  # Minimum threshold
                    results.append({
                        "path": pattern["path"],
                        "name": pattern["name"],
                        "category": category,
                        "content": pattern["content"],
                        "score": score,
                    })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def get_by_category(self, category: str) -> list[dict]:
        """Get all patterns in a category.

        Args:
            category: Category name (e.g., "rate-limiting")

        Returns:
            List of pattern dicts
        """
        return self.patterns.get(category, [])

    def get_categories(self) -> list[str]:
        """Get list of available categories."""
        return list(self.patterns.keys())

    def format_for_prompt(self, patterns: list[dict], max_tokens: int = 2000) -> str:
        """Format patterns for injection into LLM prompt.

        Args:
            patterns: List of pattern dicts from search()
            max_tokens: Approximate max tokens (chars / 4)

        Returns:
            Formatted string for prompt injection
        """
        if not patterns:
            return ""

        lines = [
            "",
            "=" * 60,
            "REFERENCE PATTERNS (production-ready examples)",
            "=" * 60,
            "",
        ]

        char_count = sum(len(line) for line in lines)
        max_chars = max_tokens * 4  # Rough token estimate

        for pattern in patterns:
            pattern_text = [
                f"## {pattern['category'].upper()}: {pattern['name']}",
                f"Score: {pattern['score']:.2f}",
                "",
                "```python",
                pattern["content"],
                "```",
                "",
            ]

            pattern_chars = sum(len(line) for line in pattern_text)

            if char_count + pattern_chars > max_chars:
                break

            lines.extend(pattern_text)
            char_count += pattern_chars

        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)
