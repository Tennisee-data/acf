"""Memory retriever for searching and context generation.

Provides the main interface for agents to get historical context
from past runs, including similar features, patterns, and error hints.

Supports hybrid search combining semantic and lexical (BM25) retrieval.
"""

import logging
from dataclasses import dataclass

from rag.embeddings import OllamaEmbeddings
from schemas.memory import (
    ErrorPattern,
    ExtractedPattern,
    HistoricalContext,
    MemorySearchResult,
    MemoryType,
    RunOutcome,
)

from .store import MemoryStore, SearchMode

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for memory retriever."""

    max_similar_features: int = 3
    max_anti_patterns: int = 2  # Failed runs to show as "what not to do"
    max_patterns: int = 5
    max_error_hints: int = 3
    similarity_threshold: float = 0.3
    apply_decay: bool = True
    search_mode: SearchMode = SearchMode.HYBRID


class MemoryRetriever:
    """Retrieve relevant memories for agent context enrichment.

    Main entry point for agents to get historical context:
    - Similar past features
    - Applicable coding patterns
    - Error prevention hints
    - Prior design decisions
    """

    def __init__(
        self,
        store: MemoryStore,
        embeddings: OllamaEmbeddings | None = None,
        embedding_model: str = "nomic-embed-text",
        config: RetrieverConfig | None = None,
    ):
        """Initialize memory retriever.

        Args:
            store: Memory store to search
            embeddings: Embedding generator (creates one if not provided)
            embedding_model: Model name for embeddings
            config: Retriever configuration
        """
        self.store = store
        self.embeddings = embeddings or OllamaEmbeddings(model=embedding_model)
        self.config = config or RetrieverConfig()

    def get_historical_context(
        self,
        feature_description: str,
        stage: str | None = None,
        domains: list[str] | None = None,
    ) -> HistoricalContext:
        """Get comprehensive historical context for a feature.

        This is the main method agents should call to get
        relevant context from past runs.

        Args:
            feature_description: Description of the feature being implemented
            stage: Current pipeline stage (for error hints)
            domains: Feature domains for pattern matching

        Returns:
            HistoricalContext with all relevant information
        """
        import time

        start_time = time.time()

        # Generate embedding for feature description
        query_embedding = self.embeddings.embed(feature_description)

        # Find similar features (successful/partial runs only)
        similar_features = self._find_similar_features(query_embedding, feature_description)

        # Find anti-patterns (failed runs - what NOT to do)
        anti_patterns = self._find_anti_patterns(query_embedding, feature_description)

        # Find prior design decisions (successful/partial runs only)
        prior_decisions = self._find_prior_decisions(query_embedding, feature_description)

        # Get applicable patterns
        applicable_patterns = self._get_applicable_patterns(domains)

        # Get error hints
        potential_errors = self._get_error_hints(stage)

        # Generate summary
        summary = self._generate_summary(
            feature_description,
            similar_features,
            anti_patterns,
            applicable_patterns,
            potential_errors,
            prior_decisions,
        )

        # Calculate timing
        search_time_ms = (time.time() - start_time) * 1000

        context = HistoricalContext(
            similar_features=similar_features,
            anti_patterns=anti_patterns,
            applicable_patterns=applicable_patterns,
            potential_errors=potential_errors,
            prior_decisions=prior_decisions,
            summary=summary,
            total_memories_searched=self.store.metadata.get("count", 0),
            search_time_ms=search_time_ms,
        )

        logger.info(
            "Retrieved historical context: %d similar, %d anti-patterns, %d patterns, %d errors (%.1fms)",
            len(similar_features),
            len(anti_patterns),
            len(applicable_patterns),
            len(potential_errors),
            search_time_ms,
        )

        # Record retrievals for usage tracking
        for result in similar_features + anti_patterns + prior_decisions:
            self.store.record_retrieval(result.entry.id)

        return context

    def _find_similar_features(
        self,
        query_embedding: list[float],
        query_text: str,
    ) -> list[MemorySearchResult]:
        """Find similar past features from successful/partial runs.

        Args:
            query_embedding: Query vector
            query_text: Query text for lexical/hybrid search

        Returns:
            List of similar feature memories (successful runs only)
        """
        return self.store.search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=self.config.max_similar_features,
            threshold=self.config.similarity_threshold,
            memory_type=MemoryType.FEATURE,
            apply_decay=self.config.apply_decay,
            mode=self.config.search_mode,
            outcome_filter=[RunOutcome.SUCCESS, RunOutcome.PARTIAL],
        )

    def _find_anti_patterns(
        self,
        query_embedding: list[float],
        query_text: str,
    ) -> list[MemorySearchResult]:
        """Find similar FAILED runs as anti-patterns (what NOT to do).

        Args:
            query_embedding: Query vector
            query_text: Query text for lexical/hybrid search

        Returns:
            List of failed feature memories to learn from
        """
        return self.store.search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=self.config.max_anti_patterns,
            threshold=self.config.similarity_threshold,
            memory_type=MemoryType.FEATURE,
            apply_decay=self.config.apply_decay,
            mode=self.config.search_mode,
            outcome_filter=[RunOutcome.FAILED],
        )

    def _find_prior_decisions(
        self,
        query_embedding: list[float],
        query_text: str,
    ) -> list[MemorySearchResult]:
        """Find relevant prior design decisions from successful runs.

        Args:
            query_embedding: Query vector
            query_text: Query text for lexical/hybrid search

        Returns:
            List of relevant design decision memories (successful runs only)
        """
        return self.store.search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=self.config.max_similar_features,
            threshold=self.config.similarity_threshold,
            memory_type=MemoryType.DESIGN_DECISION,
            apply_decay=self.config.apply_decay,
            mode=self.config.search_mode,
            outcome_filter=[RunOutcome.SUCCESS, RunOutcome.PARTIAL],
        )

    def _get_applicable_patterns(
        self,
        domains: list[str] | None = None,
    ) -> list[ExtractedPattern]:
        """Get patterns that may apply to current work.

        Args:
            domains: Feature domains for filtering

        Returns:
            List of applicable patterns
        """
        if domains:
            # Get patterns for each domain and deduplicate
            patterns_by_id = {}
            for domain in domains:
                domain_patterns = self.store.get_patterns(
                    domain=domain,
                    min_occurrences=2,  # Only patterns seen multiple times
                )
                for p in domain_patterns:
                    if p.id not in patterns_by_id:
                        patterns_by_id[p.id] = p

            patterns = list(patterns_by_id.values())
        else:
            patterns = self.store.get_patterns(min_occurrences=2)

        # Sort by occurrence count and return top N
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns[: self.config.max_patterns]

    def _get_error_hints(
        self,
        stage: str | None = None,
    ) -> list[ErrorPattern]:
        """Get error patterns to watch out for.

        Args:
            stage: Current stage for filtering

        Returns:
            List of relevant error patterns
        """
        if stage:
            errors = self.store.get_error_patterns(
                stage=stage,
                min_occurrences=1,
            )
        else:
            errors = self.store.get_error_patterns(min_occurrences=2)

        return errors[: self.config.max_error_hints]

    def _generate_summary(
        self,
        feature_description: str,
        similar_features: list[MemorySearchResult],
        anti_patterns: list[MemorySearchResult],
        patterns: list[ExtractedPattern],
        errors: list[ErrorPattern],
        decisions: list[MemorySearchResult],
    ) -> str:
        """Generate an LLM-friendly summary of historical context.

        Args:
            feature_description: Current feature description
            similar_features: Similar past features (successful)
            anti_patterns: Failed attempts to learn from
            patterns: Applicable patterns
            errors: Potential errors
            decisions: Prior decisions

        Returns:
            Summary string for injection into prompts
        """
        sections = []

        # Similar features section (successful runs)
        if similar_features:
            section = "## Similar Past Features (Successful)\n"
            for i, result in enumerate(similar_features, 1):
                outcome = result.entry.outcome.value
                section += (
                    f"{i}. {result.entry.content[:200]}... "
                    f"(Outcome: {outcome}, Score: {result.score:.2f})\n"
                )
            sections.append(section)

        # Anti-patterns section (failed runs - what NOT to do)
        if anti_patterns:
            section = "## Anti-Patterns (What NOT to Do)\n"
            section += "_These similar attempts failed - avoid these approaches:_\n"
            for i, result in enumerate(anti_patterns, 1):
                section += (
                    f"{i}. {result.entry.content[:200]}... "
                    f"(FAILED, Score: {result.score:.2f})\n"
                )
            sections.append(section)

        # Patterns section
        if patterns:
            section = "## Applicable Patterns\n"
            for p in patterns:
                section += (
                    f"- **{p.name}**: {p.description}\n"
                    f"  Use when: {p.when_to_use}\n"
                )
            sections.append(section)

        # Error hints section
        if errors:
            section = "## Watch Out For\n"
            for e in errors:
                section += (
                    f"- **{e.error_signature[:100]}**: {e.prevention_hint or e.fix_description}\n"
                )
            sections.append(section)

        # Prior decisions section
        if decisions:
            section = "## Prior Design Decisions\n"
            for result in decisions:
                section += f"- {result.entry.content[:300]}...\n"
            sections.append(section)

        if not sections:
            return "No relevant historical context found."

        return "\n".join(sections)

    def find_similar(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
        mode: SearchMode | None = None,
    ) -> list[MemorySearchResult]:
        """General similarity search with hybrid retrieval.

        Args:
            query: Search query
            memory_type: Filter by type
            top_k: Number of results
            mode: Search mode (defaults to config setting)

        Returns:
            List of search results
        """
        search_mode = mode or self.config.search_mode
        query_embedding = self.embeddings.embed(query)

        return self.store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            threshold=self.config.similarity_threshold,
            memory_type=memory_type,
            apply_decay=self.config.apply_decay,
            mode=search_mode,
        )

    def get_error_hint(
        self,
        error_message: str,
    ) -> ErrorPattern | None:
        """Find an error pattern matching an error message.

        Args:
            error_message: Error message to match

        Returns:
            Matching error pattern or None
        """
        return self.store.find_error_by_signature(error_message)

    def rate_retrieval(
        self,
        memory_id: str,
        useful: bool,
    ) -> bool:
        """Rate a retrieved memory as useful or not.

        This feedback is used to improve future retrieval ranking.

        Args:
            memory_id: Memory ID to rate
            useful: Whether the retrieval was useful

        Returns:
            True if rating was recorded
        """
        return self.store.record_retrieval(memory_id, useful=useful)
