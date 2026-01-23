"""Budget-aware RAG retriever.

Retrieves RAG content while respecting model context window limits.
Prioritizes content by relevance and importance.
Features MMR for diversity and structured prompt injection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .token_counter import count_tokens, format_token_count

logger = logging.getLogger(__name__)


class SourcePriority(Enum):
    """Priority levels for RAG sources."""

    CRITICAL = 1   # Safety invariants, must-include
    HIGH = 2       # Directly relevant API docs
    MEDIUM = 3     # Related framework patterns
    LOW = 4        # General reference docs
    OPTIONAL = 5   # Nice-to-have background


# Priority weights for different source types
SOURCE_TYPE_PRIORITIES: dict[str, SourcePriority] = {
    "invariant": SourcePriority.CRITICAL,
    "safety": SourcePriority.CRITICAL,
    "api_docs": SourcePriority.HIGH,
    "framework": SourcePriority.MEDIUM,
    "pattern": SourcePriority.MEDIUM,
    "example": SourcePriority.MEDIUM,
    "reference": SourcePriority.LOW,
    "general": SourcePriority.LOW,
    "background": SourcePriority.OPTIONAL,
}


@dataclass
class SourceMetadata:
    """Versioning and metadata for RAG sources."""

    api_version: str | None = None
    last_updated: str | None = None
    author: str | None = None
    deprecation_warnings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    language: str | None = None


@dataclass
class RAGSource:
    """A RAG content source with metadata."""

    id: str
    content: str
    source_type: str
    source_path: str | None = None
    relevance_score: float = 0.0  # 0.0 to 1.0 from embedding similarity
    priority: SourcePriority = SourcePriority.MEDIUM
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    versioning: SourceMetadata | None = None
    embedding: list[float] | None = None  # For semantic retrieval

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = count_tokens(self.content)
        if self.priority == SourcePriority.MEDIUM:
            self.priority = SOURCE_TYPE_PRIORITIES.get(
                self.source_type, SourcePriority.MEDIUM
            )

    @property
    def effective_score(self) -> float:
        """Combined score considering priority and relevance.

        Lower priority value = higher importance.
        Higher relevance score = more relevant.
        """
        priority_weight = 6 - self.priority.value  # 5 for CRITICAL, 1 for OPTIONAL
        return priority_weight * 0.4 + self.relevance_score * 0.6

    def has_deprecation_warnings(self) -> bool:
        """Check if source has deprecation warnings."""
        return bool(self.versioning and self.versioning.deprecation_warnings)

    def get_deprecation_header(self) -> str:
        """Get deprecation warning header for prompt."""
        if not self.has_deprecation_warnings():
            return ""
        warnings = self.versioning.deprecation_warnings
        return f"⚠️ DEPRECATION WARNING: {'; '.join(warnings)}\n"


@dataclass
class RAGBudgetReport:
    """Report of RAG content selection."""

    model: str
    token_budget: int
    tokens_used: int
    included_sources: list[RAGSource]
    excluded_sources: list[RAGSource]

    @property
    def utilization_percent(self) -> float:
        """Percentage of budget used."""
        return (self.tokens_used / self.token_budget) * 100 if self.token_budget > 0 else 0

    @property
    def sources_included_count(self) -> int:
        return len(self.included_sources)

    @property
    def sources_excluded_count(self) -> int:
        return len(self.excluded_sources)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"RAG Budget Report for {self.model}",
            f"  Budget: {format_token_count(self.token_budget)} tokens",
            f"  Used: {format_token_count(self.tokens_used)} ({self.utilization_percent:.1f}%)",
            f"  Included: {self.sources_included_count} sources",
            f"  Excluded: {self.sources_excluded_count} sources",
        ]

        if self.included_sources:
            lines.append("\n  Included sources:")
            for src in self.included_sources[:5]:
                lines.append(f"    - [{src.priority.name}] {src.id} ({format_token_count(src.tokens)})")
            if len(self.included_sources) > 5:
                lines.append(f"    ... and {len(self.included_sources) - 5} more")

        if self.excluded_sources:
            lines.append("\n  Excluded sources (budget exceeded):")
            for src in self.excluded_sources[:3]:
                lines.append(f"    - [{src.priority.name}] {src.id} ({format_token_count(src.tokens)})")
            if len(self.excluded_sources) > 3:
                lines.append(f"    ... and {len(self.excluded_sources) - 3} more")

        return "\n".join(lines)


class BudgetRetriever:
    """Budget-aware RAG retriever.

    Selects content based on priority and relevance while
    staying within token budget.

    Example:
        retriever = BudgetRetriever()
        retriever.add_source(RAGSource(
            id="stripe_invariants",
            content=invariant_text,
            source_type="invariant",
        ))
        report = retriever.retrieve(
            query="stripe webhook handler",
            token_budget=10000,
            model="qwen3:14b",
        )
    """

    def __init__(self):
        """Initialize retriever."""
        self.sources: list[RAGSource] = []
        self._global_sources: list[RAGSource] = []  # Coding Factory knowledge
        self._project_sources: list[RAGSource] = []  # Project-specific

    def add_source(
        self,
        source: RAGSource,
        tier: str = "project",
    ) -> None:
        """Add a RAG source.

        Args:
            source: The RAG source to add
            tier: "global" (Coding Factory) or "project" (user's code)
        """
        if tier == "global":
            self._global_sources.append(source)
        else:
            self._project_sources.append(source)
        self.sources.append(source)

    def add_invariants(self, invariants_dir: Path) -> int:
        """Load invariants as high-priority RAG sources.

        Args:
            invariants_dir: Path to invariants directory

        Returns:
            Number of invariants loaded
        """
        import json

        count = 0
        if not invariants_dir.exists():
            return count

        for json_file in invariants_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue

            try:
                data = json.loads(json_file.read_text())
                topic = data.get("topic", json_file.stem)

                # Create condensed invariant content
                content_parts = [
                    f"# {topic}",
                    f"\n{data.get('description', '')}",
                    "\n## Must:",
                ]
                for must in data.get("must", []):
                    content_parts.append(f"- {must}")

                content_parts.append("\n## Should:")
                for should in data.get("should", []):
                    content_parts.append(f"- {should}")

                content_parts.append("\n## Anti-patterns:")
                for ap in data.get("anti_patterns", []):
                    msg = ap.get("message", "")
                    hint = ap.get("fix_hint", "")
                    content_parts.append(f"- {msg}")
                    if hint:
                        content_parts.append(f"  Fix: {hint}")

                content = "\n".join(content_parts)

                source = RAGSource(
                    id=f"invariant:{topic}",
                    content=content,
                    source_type="invariant",
                    source_path=str(json_file),
                    priority=SourcePriority.CRITICAL,
                    metadata={
                        "triggers": data.get("triggers", []),
                        "category": data.get("category", ""),
                    },
                )
                self.add_source(source, tier="global")
                count += 1

            except Exception as e:
                logger.warning("Failed to load invariant %s: %s", json_file, e)

        return count

    def add_api_docs(
        self,
        docs_dir: Path,
        provider: str | None = None,
    ) -> int:
        """Load API documentation as high-priority RAG sources.

        Args:
            docs_dir: Path to documentation directory
            provider: Filter by provider (e.g., "stripe", "google")

        Returns:
            Number of docs loaded
        """
        count = 0
        if not docs_dir.exists():
            return count

        for md_file in docs_dir.glob("**/*.md"):
            # Skip if filtering by provider and doesn't match
            if provider and provider.lower() not in str(md_file).lower():
                continue

            try:
                content = md_file.read_text()
                source = RAGSource(
                    id=f"api_docs:{md_file.stem}",
                    content=content,
                    source_type="api_docs",
                    source_path=str(md_file),
                    priority=SourcePriority.HIGH,
                )
                self.add_source(source, tier="global")
                count += 1

            except Exception as e:
                logger.warning("Failed to load doc %s: %s", md_file, e)

        return count

    def _score_relevance(
        self,
        source: RAGSource,
        query: str,
        context: str = "",
    ) -> float:
        """Score source relevance to query.

        Uses trigger matching for invariants and keyword overlap for docs.
        Returns 0.0 for sources that don't match - they won't be included.

        Args:
            source: RAG source
            query: Search query
            context: Additional context (e.g., feature description)

        Returns:
            Relevance score 0.0 to 1.0
        """
        query_lower = query.lower()
        context_lower = context.lower()
        combined_query = f"{query_lower} {context_lower}"

        # For invariants: ONLY include if trigger matches
        triggers = source.metadata.get("triggers", [])
        if triggers:
            for trigger in triggers:
                if trigger.lower() in combined_query:
                    return 1.0  # Perfect match on trigger
            # Has triggers but none matched - don't include
            return 0.0

        # For API docs: require strong keyword match
        source_id_lower = source.id.lower()

        # Check if source ID/path matches query terms
        query_terms = [t for t in query_lower.split() if len(t) > 3]
        for term in query_terms:
            if term in source_id_lower:
                return 0.8  # Good match on source name

        # Fallback: check content overlap (but require significant match)
        content_lower = source.content.lower()[:1000]  # Only check first 1K chars
        query_words = set(t for t in query_lower.split() if len(t) > 3)
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)

        if overlap >= 2:  # Require at least 2 matching words
            return min(0.5, overlap / len(query_words))

        return 0.0  # No match - don't include

    def retrieve(
        self,
        query: str,
        token_budget: int,
        model: str = "",
        context: str = "",
        include_all_critical: bool = True,
    ) -> RAGBudgetReport:
        """Retrieve RAG content within token budget.

        Args:
            query: Search query
            token_budget: Maximum tokens for RAG content
            model: Model name (for reporting)
            context: Additional context for relevance scoring
            include_all_critical: Always include CRITICAL priority sources

        Returns:
            RAGBudgetReport with included/excluded sources
        """
        # Score all sources for relevance
        scored_sources: list[tuple[float, RAGSource]] = []
        for source in self.sources:
            source.relevance_score = self._score_relevance(source, query, context)
            scored_sources.append((source.effective_score, source))

        # Sort by effective score (descending)
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        # Select sources within budget
        included: list[RAGSource] = []
        excluded: list[RAGSource] = []
        tokens_used = 0

        for score, source in scored_sources:
            # Always include CRITICAL sources if flag is set
            if include_all_critical and source.priority == SourcePriority.CRITICAL:
                if source.relevance_score > 0:  # Still must be relevant
                    included.append(source)
                    tokens_used += source.tokens
                    continue

            # Check if fits in budget
            if tokens_used + source.tokens <= token_budget:
                # Only include if actually relevant (score > 0)
                if source.relevance_score > 0:
                    included.append(source)
                    tokens_used += source.tokens
                else:
                    excluded.append(source)
            else:
                excluded.append(source)

        return RAGBudgetReport(
            model=model,
            token_budget=token_budget,
            tokens_used=tokens_used,
            included_sources=included,
            excluded_sources=excluded,
        )

    def get_content(self, report: RAGBudgetReport) -> str:
        """Get combined content from included sources.

        Args:
            report: RAG budget report

        Returns:
            Combined content string
        """
        sections = []
        for source in report.included_sources:
            sections.append(f"<!-- RAG Source: {source.id} -->\n{source.content}")

        return "\n\n---\n\n".join(sections)

    def get_structured_content(self, report: RAGBudgetReport) -> str:
        """Get content formatted as structured prompt sections.

        Organizes content by priority with clear headers:
        - CRITICAL INVARIANTS (must follow)
        - API REFERENCE
        - RECOMMENDED PATTERNS

        Args:
            report: RAG budget report

        Returns:
            Structured content string
        """
        # Group sources by priority
        critical = []
        high = []
        medium = []
        low = []

        for source in report.included_sources:
            deprecation = source.get_deprecation_header()
            content_with_header = f"{deprecation}{source.content}" if deprecation else source.content

            if source.priority == SourcePriority.CRITICAL:
                critical.append((source.id, content_with_header))
            elif source.priority == SourcePriority.HIGH:
                high.append((source.id, content_with_header))
            elif source.priority == SourcePriority.MEDIUM:
                medium.append((source.id, content_with_header))
            else:
                low.append((source.id, content_with_header))

        sections = []

        if critical:
            section_content = "\n\n".join(
                f"### {source_id}\n{content}" for source_id, content in critical
            )
            sections.append(f"""## CRITICAL INVARIANTS (MUST FOLLOW)

These rules MUST be followed. Violations will cause bugs or security issues.

{section_content}""")

        if high:
            section_content = "\n\n".join(
                f"### {source_id}\n{content}" for source_id, content in high
            )
            sections.append(f"""## API REFERENCE

Relevant API documentation for this task.

{section_content}""")

        if medium:
            section_content = "\n\n".join(
                f"### {source_id}\n{content}" for source_id, content in medium
            )
            sections.append(f"""## RECOMMENDED PATTERNS

Best practices and patterns to follow.

{section_content}""")

        if low:
            section_content = "\n\n".join(
                f"### {source_id}\n{content}" for source_id, content in low
            )
            sections.append(f"""## ADDITIONAL REFERENCE

Background information that may be helpful.

{section_content}""")

        return "\n\n---\n\n".join(sections)

    def apply_mmr(
        self,
        candidates: list[RAGSource],
        query_embedding: list[float] | None = None,
        lambda_param: float = 0.7,
        top_k: int | None = None,
    ) -> list[RAGSource]:
        """Apply Maximal Marginal Relevance for diversity.

        Reranks candidates to balance relevance and diversity,
        avoiding redundant content.

        Args:
            candidates: List of candidate sources (should have embeddings)
            query_embedding: Query embedding vector
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            top_k: Max sources to return (None = all)

        Returns:
            Reranked sources
        """
        if not candidates:
            return []

        # Check if we have embeddings for MMR
        has_embeddings = all(s.embedding is not None for s in candidates)
        if not has_embeddings or query_embedding is None:
            # Fall back to relevance score ordering
            logger.debug("MMR skipped: missing embeddings")
            return sorted(candidates, key=lambda x: -x.effective_score)[:top_k]

        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        doc_embeddings = np.array([s.embedding for s in candidates])

        # Normalize for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9
        )

        # Compute query-document similarities
        query_sims = doc_norms @ query_norm

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        n_select = top_k or len(candidates)

        while remaining_indices and len(selected_indices) < n_select:
            if not selected_indices:
                # First selection: highest relevance
                best_idx = max(remaining_indices, key=lambda i: query_sims[i])
            else:
                # MMR: balance relevance and diversity
                best_score = float("-inf")
                best_idx = remaining_indices[0]

                selected_embeddings = doc_norms[selected_indices]

                for idx in remaining_indices:
                    # Relevance to query
                    relevance = query_sims[idx]

                    # Max similarity to already selected (redundancy)
                    doc_vec = doc_norms[idx]
                    similarities = selected_embeddings @ doc_vec
                    max_similarity = np.max(similarities)

                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def clear(self) -> None:
        """Clear all sources."""
        self.sources.clear()
        self._global_sources.clear()
        self._project_sources.clear()
