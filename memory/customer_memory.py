"""Per-customer memory management for web API.

Provides customer-isolated memory stores for the SaaS platform.
Each customer gets their own memory store that learns from their runs.

Storage structure:
    data/memory/{customer_id}/
    ├── embeddings.npy
    ├── memories.json
    ├── patterns.json
    ├── errors.json
    └── metadata.json
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from schemas.memory import (
    ExtractedPattern,
    MemorySearchResult,
    MemoryStats,
    MemoryType,
    RunMemoryEntry,
    RunOutcome,
)

logger = logging.getLogger(__name__)

# Base path for customer memory storage
CUSTOMER_MEMORY_BASE = Path(__file__).parent.parent / "data" / "memory"


class CustomerMemoryManager:
    """Manages per-customer memory stores.

    Provides isolated memory stores for each customer in a multi-tenant
    environment. Memories are stored in customer-specific directories
    and never shared between customers.

    Usage:
        manager = CustomerMemoryManager()

        # Index a completed run
        manager.index_run(customer_id, run_dir)

        # Retrieve relevant memories for a new generation
        memories = manager.retrieve_context(customer_id, prompt)

        # Get customer's learning stats
        stats = manager.get_stats(customer_id)
    """

    def __init__(
        self,
        base_path: Path | None = None,
        decay_half_life_days: int = 90,
    ):
        """Initialize customer memory manager.

        Args:
            base_path: Base path for customer stores (default: data/memory/)
            decay_half_life_days: Memory decay half-life in days
        """
        self.base_path = base_path or CUSTOMER_MEMORY_BASE
        self.decay_half_life_days = decay_half_life_days
        self._stores: dict[str, Any] = {}  # Cache of loaded stores

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_customer_path(self, customer_id: str) -> Path:
        """Get storage path for a customer.

        Args:
            customer_id: Customer UUID

        Returns:
            Path to customer's memory directory
        """
        return self.base_path / customer_id

    def _get_store(self, customer_id: str):
        """Get or create MemoryStore for a customer.

        Args:
            customer_id: Customer UUID

        Returns:
            MemoryStore instance for the customer
        """
        if customer_id not in self._stores:
            from memory.store import MemoryStore

            store_path = self._get_customer_path(customer_id)
            store_path.mkdir(parents=True, exist_ok=True)

            self._stores[customer_id] = MemoryStore(
                store_path=store_path,
                decay_half_life_days=self.decay_half_life_days,
            )

        return self._stores[customer_id]

    def index_run(
        self,
        customer_id: str,
        run_dir: Path,
        outcome: RunOutcome = RunOutcome.SUCCESS,
    ) -> int:
        """Index a completed run into customer's memory.

        Extracts memories from the run artifacts and stores them
        in the customer's isolated memory store.

        Args:
            customer_id: Customer UUID
            run_dir: Path to run artifacts directory
            outcome: Run outcome (success, partial, failed)

        Returns:
            Number of memories indexed
        """
        try:
            from memory.indexer import RunIndexer
            from rag.embeddings import OllamaEmbeddings

            store = self._get_store(customer_id)

            # Try to use embeddings, fall back to lexical-only if not available
            try:
                embeddings = OllamaEmbeddings()
            except Exception as e:
                logger.warning(f"Embeddings not available, using lexical only: {e}")
                embeddings = None

            indexer = RunIndexer(store=store, embeddings=embeddings)
            count = indexer.index_run(run_dir, outcome=outcome)

            if count > 0:
                store.save()
                logger.info(f"Indexed {count} memories for customer {customer_id[:8]}...")

            return count

        except Exception as e:
            logger.error(f"Failed to index run for customer {customer_id}: {e}")
            return 0

    def retrieve_context(
        self,
        customer_id: str,
        query: str,
        limit: int = 5,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve relevant memories for a customer.

        Searches the customer's memory store for relevant past runs
        that can inform the current generation.

        Args:
            customer_id: Customer UUID
            query: Search query (usually the feature description)
            limit: Maximum number of results
            memory_types: Filter by memory types (default: all)

        Returns:
            List of relevant memory search results
        """
        try:
            from memory.store import SearchMode

            store = self._get_store(customer_id)

            # Check if store has any memories
            if not store.memories:
                return []

            # Try to use embeddings for semantic/hybrid search
            query_embedding = None
            search_mode = SearchMode.LEXICAL  # Default to lexical if no embeddings
            try:
                from rag.embeddings import OllamaEmbeddings
                embeddings = OllamaEmbeddings()
                query_embedding = embeddings.embed(query)
                search_mode = SearchMode.HYBRID
            except Exception:
                logger.debug("Embeddings unavailable, using lexical search only")

            # Search directly on store
            # If specific memory_types requested, search for each type
            if memory_types:
                results = []
                for mem_type in memory_types:
                    type_results = store.search(
                        query_embedding=query_embedding,
                        query_text=query,
                        top_k=limit,
                        memory_type=mem_type,
                        mode=search_mode,
                    )
                    results.extend(type_results)
                # Sort by score and limit
                results.sort(key=lambda r: r.score, reverse=True)
                results = results[:limit]
            else:
                results = store.search(
                    query_embedding=query_embedding,
                    query_text=query,
                    top_k=limit,
                    mode=search_mode,
                )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve memories for customer {customer_id}: {e}")
            return []

    def get_relevant_patterns(
        self,
        customer_id: str,
        query: str,
        limit: int = 3,
    ) -> list[ExtractedPattern]:
        """Get learned patterns relevant to a query.

        Returns patterns the system has learned from the customer's
        past runs that are relevant to the current task.

        Args:
            customer_id: Customer UUID
            query: Search query
            limit: Maximum patterns to return

        Returns:
            List of relevant extracted patterns
        """
        try:
            store = self._get_store(customer_id)

            if not store.patterns:
                return []

            # Simple keyword matching for patterns
            query_lower = query.lower()
            scored_patterns = []

            for pattern in store.patterns:
                score = 0
                pattern_text = f"{pattern.name} {pattern.description}".lower()

                # Score based on keyword overlap
                for word in query_lower.split():
                    if len(word) > 3 and word in pattern_text:
                        score += 1

                if score > 0:
                    scored_patterns.append((score, pattern))

            # Sort by score descending
            scored_patterns.sort(key=lambda x: x[0], reverse=True)

            return [p for _, p in scored_patterns[:limit]]

        except Exception as e:
            logger.error(f"Failed to get patterns for customer {customer_id}: {e}")
            return []

    def get_stats(self, customer_id: str) -> dict[str, Any]:
        """Get memory statistics for a customer.

        Args:
            customer_id: Customer UUID

        Returns:
            Dict with memory statistics
        """
        try:
            store = self._get_store(customer_id)

            return {
                "total_memories": len(store.memories),
                "total_patterns": len(store.patterns),
                "total_errors": len(store.errors),
                "runs_indexed": len(store.metadata.get("runs_indexed", set())),
                "last_indexed": store.metadata.get("last_indexed"),
                "created_at": store.metadata.get("created_at"),
            }

        except Exception as e:
            logger.error(f"Failed to get stats for customer {customer_id}: {e}")
            return {
                "total_memories": 0,
                "total_patterns": 0,
                "total_errors": 0,
                "runs_indexed": 0,
                "last_indexed": None,
                "created_at": None,
            }

    def format_context_for_prompt(
        self,
        memories: list[MemorySearchResult],
        patterns: list[ExtractedPattern],
    ) -> str:
        """Format retrieved memories and patterns for prompt injection.

        Creates a formatted string to inject into the generation prompt
        that provides context from past runs.

        Args:
            memories: Retrieved memory results
            patterns: Relevant patterns

        Returns:
            Formatted context string for prompt injection
        """
        if not memories and not patterns:
            return ""

        lines = [
            "",
            "=" * 60,
            "LEARNED CONTEXT FROM YOUR PAST PROJECTS",
            "=" * 60,
            "",
        ]

        if patterns:
            lines.append("## Patterns Learned From Your Projects")
            lines.append("")
            for pattern in patterns:
                lines.append(f"### {pattern.name}")
                lines.append(f"{pattern.description}")
                if pattern.examples:
                    lines.append(f"Example: {pattern.examples[0][:200]}")
                lines.append("")

        if memories:
            lines.append("## Relevant Past Runs")
            lines.append("")
            for mem in memories[:5]:
                entry = mem.entry
                # Get feature_name and summary from metadata (schema stores these there)
                feature_name = entry.metadata.get("feature_name", "Past Project")
                summary = entry.metadata.get("summary", entry.content[:200])
                decisions = entry.metadata.get("decisions", [])

                lines.append(f"### {feature_name}")
                lines.append(f"Summary: {summary[:200]}")
                if decisions:
                    lines.append(f"Key decisions: {', '.join(decisions[:3])}")
                lines.append(f"Outcome: {entry.outcome.value if entry.outcome else 'unknown'}")
                lines.append("")

        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)

    def record_feedback(
        self,
        customer_id: str,
        memory_id: str,
        helpful: bool,
    ) -> bool:
        """Record user feedback on a retrieved memory.

        This helps improve future retrievals by boosting memories
        that users found helpful.

        Args:
            customer_id: Customer UUID
            memory_id: Memory entry ID
            helpful: Whether the memory was helpful

        Returns:
            True if feedback was recorded
        """
        try:
            store = self._get_store(customer_id)

            for memory in store.memories:
                if memory.id == memory_id:
                    memory.times_retrieved += 1
                    # Update usefulness_score as running average
                    feedback_value = 1.0 if helpful else 0.0
                    old_score = memory.usefulness_score
                    retrieval_count = memory.times_retrieved
                    # Running average: (old_avg * (n-1) + new_value) / n
                    memory.usefulness_score = (
                        (old_score * (retrieval_count - 1) + feedback_value) / retrieval_count
                    )
                    store.save()
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    def clear_customer_memory(self, customer_id: str) -> bool:
        """Clear all memories for a customer.

        Use with caution - this permanently deletes all learned data.

        Args:
            customer_id: Customer UUID

        Returns:
            True if cleared successfully
        """
        try:
            import shutil

            customer_path = self._get_customer_path(customer_id)
            if customer_path.exists():
                shutil.rmtree(customer_path)

            # Remove from cache
            if customer_id in self._stores:
                del self._stores[customer_id]

            logger.info(f"Cleared memory for customer {customer_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to clear memory for customer {customer_id}: {e}")
            return False


# Global instance for API usage
_manager: CustomerMemoryManager | None = None


def get_customer_memory_manager() -> CustomerMemoryManager:
    """Get the global customer memory manager instance.

    Returns:
        CustomerMemoryManager singleton
    """
    global _manager
    if _manager is None:
        _manager = CustomerMemoryManager()
    return _manager
