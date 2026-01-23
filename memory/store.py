"""Memory store for past-run learning.

Provides persistent storage for:
- Run memories (embedded for similarity search)
- Extracted patterns (learned from multiple runs)
- Error patterns (with fixes)

Uses NumPy for embeddings and JSON for metadata,
following the same pattern as rag/store.py.

Supports hybrid search combining:
- Semantic similarity (cosine on embeddings)
- Lexical matching (BM25)
"""

import json
import logging
import math
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from schemas.memory import (
    ErrorPattern,
    ExtractedPattern,
    MemorySearchResult,
    MemoryStats,
    MemoryType,
    RunMemoryEntry,
    RunOutcome,
    TriagePattern,
)

from .bm25 import BM25

logger = logging.getLogger(__name__)

# Default global storage location
DEFAULT_STORE_PATH = Path.home() / ".coding-factory" / "memory"

# Outcome weights for scoring - successful runs rank higher
OUTCOME_WEIGHTS: dict[RunOutcome, float] = {
    RunOutcome.SUCCESS: 1.0,    # Full weight for successful runs
    RunOutcome.PARTIAL: 0.7,    # Some issues but shipped
    RunOutcome.FAILED: 0.3,     # Failed runs still retrievable but downweighted
    RunOutcome.CANCELLED: 0.2,  # Minimal weight for cancelled
}


class SearchMode(str, Enum):
    """Search mode for memory retrieval."""

    SEMANTIC = "semantic"  # Cosine similarity on embeddings only
    LEXICAL = "lexical"  # BM25 only
    HYBRID = "hybrid"  # Weighted combination (recommended)


class MemoryStore:
    """Vector store for run memories with hybrid search.

    Stores embeddings and metadata for past-run memories.
    Supports multiple search modes:
    - semantic: Cosine similarity on embeddings
    - lexical: BM25 for exact term matching
    - hybrid: Weighted combination (best practice)

    Storage structure:
        ~/.coding-factory/memory/
        ├── embeddings.npy       # Vector embeddings (NumPy)
        ├── memories.json        # Memory entries with metadata
        ├── patterns.json        # Extracted patterns
        ├── errors.json          # Error patterns
        ├── bm25_index.json      # BM25 inverted index
        └── metadata.json        # Store metadata, version, stats
    """

    VERSION = 2  # Bumped for hybrid search support

    def __init__(
        self,
        store_path: Path | None = None,
        decay_half_life_days: int = 90,
        hybrid_alpha: float = 0.7,
    ):
        """Initialize memory store.

        Args:
            store_path: Path to persist the store (default: ~/.coding-factory/memory)
            decay_half_life_days: Half-life for memory decay in days
            hybrid_alpha: Weight for semantic score in hybrid mode (0-1)
                         Higher = more semantic, lower = more lexical
        """
        self.store_path = store_path or DEFAULT_STORE_PATH
        self.decay_half_life_days = decay_half_life_days
        self.hybrid_alpha = hybrid_alpha

        # In-memory data
        self.embeddings: np.ndarray | None = None
        self.memories: list[RunMemoryEntry] = []
        self.patterns: list[ExtractedPattern] = []
        self.errors: list[ErrorPattern] = []
        self.triage: list[TriagePattern] = []

        # BM25 index for lexical search
        self.bm25 = BM25()

        self.metadata: dict = {
            "version": self.VERSION,
            "count": 0,
            "dimensions": 0,
            "runs_indexed": set(),
            "last_indexed": None,
            "created_at": datetime.now().isoformat(),
        }

        # Load existing store if available
        if self.store_path.exists():
            self.load()

    def _compute_decay(self, created_at: datetime) -> float:
        """Compute time-based decay factor.

        Uses exponential decay with configurable half-life.

        Args:
            created_at: When the memory was created

        Returns:
            Decay factor between 0 and 1
        """
        age_days = (datetime.now() - created_at).days
        if age_days <= 0:
            return 1.0

        # Exponential decay: 0.5^(age / half_life)
        decay = math.pow(0.5, age_days / self.decay_half_life_days)
        return max(0.01, decay)  # Minimum 1% weight

    def _compute_usage_boost(self, memory: RunMemoryEntry) -> float:
        """Compute usage-based boost factor.

        Memories that are frequently retrieved and rated useful
        get a boost to counteract decay.

        Args:
            memory: Memory entry

        Returns:
            Boost factor (1.0 = no boost, higher = more boost)
        """
        # Base boost from retrieval count (log scale)
        retrieval_boost = 1.0 + (0.1 * math.log1p(memory.times_retrieved))

        # Boost from usefulness score
        usefulness_boost = 1.0 + (0.5 * memory.usefulness_score)

        return retrieval_boost * usefulness_boost

    def add(
        self,
        memories: list[RunMemoryEntry],
        embeddings: list[list[float]],
    ) -> int:
        """Add memories with their embeddings.

        Args:
            memories: Memory entries to add
            embeddings: Corresponding embeddings

        Returns:
            Number of memories added
        """
        if len(memories) != len(embeddings):
            raise ValueError("Number of memories must match number of embeddings")

        if not memories:
            return 0

        new_embeddings = np.array(embeddings, dtype=np.float32)

        # Get starting index for BM25
        start_idx = len(self.memories)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.memories = list(memories)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.memories.extend(memories)

        # Add to BM25 index for lexical search
        for i, memory in enumerate(memories):
            doc_id = start_idx + i
            self.bm25.add_document(doc_id, memory.content)

        # Update metadata
        self.metadata["count"] = len(self.memories)
        self.metadata["dimensions"] = new_embeddings.shape[1]
        self.metadata["last_indexed"] = datetime.now().isoformat()

        # Track indexed runs
        for memory in memories:
            if isinstance(self.metadata["runs_indexed"], set):
                self.metadata["runs_indexed"].add(memory.run_id)
            else:
                # Handle case where it was loaded from JSON as list
                runs = set(self.metadata["runs_indexed"])
                runs.add(memory.run_id)
                self.metadata["runs_indexed"] = runs

        logger.info("Added %d memories to store (total: %d)", len(memories), len(self.memories))
        return len(memories)

    def search(
        self,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        threshold: float = 0.0,
        memory_type: MemoryType | None = None,
        run_id: str | None = None,
        apply_decay: bool = True,
        mode: SearchMode = SearchMode.HYBRID,
        outcome_filter: list[RunOutcome] | None = None,
    ) -> list[MemorySearchResult]:
        """Search for similar memories using hybrid retrieval.

        Supports three modes:
        - SEMANTIC: Cosine similarity on embeddings only
        - LEXICAL: BM25 only (requires query_text)
        - HYBRID: Weighted combination (recommended)

        Args:
            query_embedding: Query vector (required for semantic/hybrid)
            query_text: Query text (required for lexical, optional for hybrid)
            top_k: Number of results to return
            threshold: Minimum similarity score (before decay)
            memory_type: Filter by memory type
            run_id: Filter by run ID
            apply_decay: Whether to apply time-based decay
            mode: Search mode (semantic, lexical, or hybrid)
            outcome_filter: Only include memories with these outcomes

        Returns:
            List of search results sorted by score
        """
        if len(self.memories) == 0:
            return []

        # Validate inputs based on mode
        if mode == SearchMode.LEXICAL and not query_text:
            raise ValueError("query_text is required for lexical search")
        if mode == SearchMode.SEMANTIC and query_embedding is None:
            raise ValueError("query_embedding is required for semantic search")
        if mode == SearchMode.HYBRID and query_embedding is None:
            raise ValueError("query_embedding is required for hybrid search")

        # Compute semantic scores
        semantic_scores: dict[int, float] = {}
        if mode in (SearchMode.SEMANTIC, SearchMode.HYBRID) and query_embedding is not None:
            if self.embeddings is not None:
                query = np.array(query_embedding, dtype=np.float32)
                query_norm = query / (np.linalg.norm(query) + 1e-9)
                embeddings_norm = self.embeddings / (
                    np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
                )
                similarities = embeddings_norm @ query_norm
                for idx, score in enumerate(similarities):
                    semantic_scores[idx] = float(score)

        # Compute BM25 scores
        bm25_scores: dict[int, float] = {}
        if mode in (SearchMode.LEXICAL, SearchMode.HYBRID) and query_text:
            # Get all BM25 results (we'll filter later)
            bm25_results = self.bm25.search(query_text, top_k=len(self.memories))
            for doc_id, score in bm25_results:
                bm25_scores[doc_id] = score

        # Normalize scores to [0, 1] range for fair combination
        def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return scores
            min_s = min(scores.values())
            max_s = max(scores.values())
            if max_s - min_s < 1e-9:
                return {k: 1.0 for k in scores}  # All same score
            return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

        # Normalize if doing hybrid
        if mode == SearchMode.HYBRID:
            semantic_scores = normalize_scores(semantic_scores)
            bm25_scores = normalize_scores(bm25_scores)

        # Combine scores based on mode
        final_scores: dict[int, float] = {}
        all_candidates = set(semantic_scores.keys()) | set(bm25_scores.keys())

        for idx in all_candidates:
            if mode == SearchMode.SEMANTIC:
                final_scores[idx] = semantic_scores.get(idx, 0.0)
            elif mode == SearchMode.LEXICAL:
                final_scores[idx] = bm25_scores.get(idx, 0.0)
            else:  # HYBRID
                sem_score = semantic_scores.get(idx, 0.0)
                bm25_score = bm25_scores.get(idx, 0.0)
                # Weighted combination: alpha * semantic + (1 - alpha) * bm25
                final_scores[idx] = (
                    self.hybrid_alpha * sem_score
                    + (1 - self.hybrid_alpha) * bm25_score
                )

        # Build results with filtering
        results = []
        for idx, score in final_scores.items():
            if score < threshold:
                continue

            memory = self.memories[idx]

            # Apply filters
            if memory_type and memory.memory_type != memory_type:
                continue
            if run_id and memory.run_id != run_id:
                continue
            if outcome_filter and memory.outcome not in outcome_filter:
                continue

            # Compute decay and boost
            decay_factor = 1.0
            if apply_decay:
                decay_factor = self._compute_decay(memory.created_at)
                decay_factor *= self._compute_usage_boost(memory)

            # Apply outcome weighting - successful runs rank higher
            outcome_weight = OUTCOME_WEIGHTS.get(memory.outcome, 0.5)

            final_score = score * decay_factor * outcome_weight

            results.append(
                MemorySearchResult(
                    entry=memory,
                    score=final_score,
                    decay_factor=decay_factor,
                )
            )

        # Sort by final score and return top-k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def record_retrieval(
        self,
        memory_id: str,
        useful: bool | None = None,
    ) -> bool:
        """Record that a memory was retrieved.

        Updates retrieval count and optionally usefulness score.

        Args:
            memory_id: ID of the memory
            useful: Whether the retrieval was useful (optional feedback)

        Returns:
            True if memory was found and updated
        """
        for memory in self.memories:
            if memory.id == memory_id:
                memory.times_retrieved += 1
                memory.last_retrieved = datetime.now()

                if useful is not None:
                    # Incremental update of usefulness score
                    # Moving average with more weight on recent feedback
                    old_score = memory.usefulness_score
                    new_value = 1.0 if useful else 0.0
                    memory.usefulness_score = (old_score * 0.7) + (new_value * 0.3)

                logger.debug(
                    "Recorded retrieval for memory %s (count: %d)",
                    memory_id,
                    memory.times_retrieved,
                )
                return True

        return False

    def get_memory(self, memory_id: str) -> RunMemoryEntry | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory entry or None
        """
        for memory in self.memories:
            if memory.id == memory_id:
                return memory
        return None

    def get_memories_by_run(self, run_id: str) -> list[RunMemoryEntry]:
        """Get all memories from a specific run.

        Args:
            run_id: Run ID

        Returns:
            List of memories from that run
        """
        return [m for m in self.memories if m.run_id == run_id]

    def delete_run(self, run_id: str) -> int:
        """Delete all memories from a run.

        Args:
            run_id: Run ID to delete

        Returns:
            Number of memories deleted
        """
        indices_to_keep = [
            i for i, m in enumerate(self.memories) if m.run_id != run_id
        ]

        if len(indices_to_keep) == len(self.memories):
            return 0

        deleted = len(self.memories) - len(indices_to_keep)

        self.memories = [self.memories[i] for i in indices_to_keep]
        if self.embeddings is not None and indices_to_keep:
            self.embeddings = self.embeddings[indices_to_keep]
        elif not indices_to_keep:
            self.embeddings = None

        # Rebuild BM25 index (indices changed after deletion)
        self._rebuild_bm25_index()

        self.metadata["count"] = len(self.memories)

        # Update indexed runs
        if isinstance(self.metadata["runs_indexed"], set):
            self.metadata["runs_indexed"].discard(run_id)
        else:
            runs = set(self.metadata["runs_indexed"])
            runs.discard(run_id)
            self.metadata["runs_indexed"] = runs

        logger.info("Deleted %d memories from run %s", deleted, run_id)
        return deleted

    # Pattern management

    def add_pattern(self, pattern: ExtractedPattern) -> None:
        """Add or update an extracted pattern.

        Args:
            pattern: Pattern to add
        """
        # Check if pattern with same ID exists
        for i, existing in enumerate(self.patterns):
            if existing.id == pattern.id:
                self.patterns[i] = pattern
                logger.info("Updated pattern: %s", pattern.name)
                return

        self.patterns.append(pattern)
        logger.info("Added pattern: %s", pattern.name)

    def get_patterns(
        self,
        pattern_type: str | None = None,
        domain: str | None = None,
        min_occurrences: int = 1,
    ) -> list[ExtractedPattern]:
        """Get extracted patterns with optional filtering.

        Args:
            pattern_type: Filter by pattern type
            domain: Filter by domain
            min_occurrences: Minimum occurrence count

        Returns:
            List of matching patterns
        """
        results = []
        for pattern in self.patterns:
            if pattern.occurrence_count < min_occurrences:
                continue
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            if domain and domain not in pattern.domains:
                continue
            results.append(pattern)

        # Sort by occurrence count
        results.sort(key=lambda p: p.occurrence_count, reverse=True)
        return results

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern.

        Args:
            pattern_id: Pattern ID

        Returns:
            True if deleted
        """
        for i, pattern in enumerate(self.patterns):
            if pattern.id == pattern_id:
                del self.patterns[i]
                return True
        return False

    # Error pattern management

    def add_error_pattern(self, error: ErrorPattern) -> None:
        """Add or update an error pattern.

        Args:
            error: Error pattern to add
        """
        # Check if error with same ID exists
        for i, existing in enumerate(self.errors):
            if existing.id == error.id:
                self.errors[i] = error
                logger.info("Updated error pattern: %s", error.error_signature[:50])
                return

        self.errors.append(error)
        logger.info("Added error pattern: %s", error.error_signature[:50])

    def get_error_patterns(
        self,
        stage: str | None = None,
        min_occurrences: int = 1,
    ) -> list[ErrorPattern]:
        """Get error patterns with optional filtering.

        Args:
            stage: Filter by stage
            min_occurrences: Minimum occurrence count

        Returns:
            List of matching error patterns
        """
        results = []
        for error in self.errors:
            if error.occurrence_count < min_occurrences:
                continue
            if stage and error.stage != stage:
                continue
            results.append(error)

        # Sort by occurrence count
        results.sort(key=lambda e: e.occurrence_count, reverse=True)
        return results

    def find_error_by_signature(self, signature: str) -> ErrorPattern | None:
        """Find an error pattern by signature match.

        Args:
            signature: Error signature to match

        Returns:
            Matching error pattern or None
        """
        signature_lower = signature.lower()
        for error in self.errors:
            if error.error_signature.lower() in signature_lower:
                return error
            if signature_lower in error.error_signature.lower():
                return error
        return None

    def delete_error_pattern(self, error_id: str) -> bool:
        """Delete an error pattern.

        Args:
            error_id: Error pattern ID

        Returns:
            True if deleted
        """
        for i, error in enumerate(self.errors):
            if error.id == error_id:
                del self.errors[i]
                return True
        return False

    # Triage pattern management

    def add_triage_pattern(self, triage: TriagePattern) -> None:
        """Add or update a triage pattern.

        Args:
            triage: Triage pattern to add
        """
        # Check if pattern with same ID exists
        for i, existing in enumerate(self.triage):
            if existing.id == triage.id:
                self.triage[i] = triage
                logger.debug("Updated triage pattern: %s", triage.id)
                return

        self.triage.append(triage)
        logger.debug("Added triage pattern: %s", triage.id)

    def lookup_triage(self, prompt: str) -> TriagePattern | None:
        """Look up triage pattern by prompt similarity.

        Checks for:
        1. Exact pattern match (normalized)
        2. High keyword overlap (>80%)

        Args:
            prompt: Feature description prompt

        Returns:
            Matching triage pattern if found with good confidence
        """
        import hashlib
        import re

        def normalize(text: str) -> str:
            """Normalize prompt for matching."""
            text = text.lower()
            text = re.sub(r'"[^"]*"', '""', text)
            text = re.sub(r"'[^']*'", "''", text)
            text = re.sub(r'\d+', '#', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def extract_keywords(text: str) -> set[str]:
            """Extract characteristic keywords."""
            actions = re.findall(r'\b(add|create|build|make|implement|fix|update|remove|delete|refactor)\b', text.lower())
            targets = re.findall(r'\b(endpoint|api|route|page|component|function|class|service|model|database|auth|login|payment|application|website|app)\b', text.lower())
            return set(actions + targets)

        normalized = normalize(prompt)
        pattern_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]

        # 1. Exact match by ID (hash of normalized prompt)
        for triage in self.triage:
            if triage.id == f"tri-{pattern_hash}":
                # Check confidence threshold
                if triage.confidence >= 0.6 and triage.success_count >= 2:
                    triage.last_used = datetime.now()
                    return triage

        # 2. Keyword similarity match
        prompt_keywords = extract_keywords(prompt)
        if prompt_keywords:
            for triage in self.triage:
                triage_keywords = set(triage.keywords)
                if triage_keywords:
                    overlap = len(prompt_keywords & triage_keywords) / len(prompt_keywords | triage_keywords)
                    if overlap > 0.8 and triage.confidence >= 0.6:
                        triage.last_used = datetime.now()
                        return triage

        return None

    def record_triage_outcome(self, triage_id: str, success: bool, run_id: str | None = None) -> None:
        """Record outcome of a triage decision.

        Args:
            triage_id: Triage pattern ID
            success: Whether the run succeeded with the chosen model
            run_id: Optional run ID to track
        """
        for triage in self.triage:
            if triage.id == triage_id:
                if success:
                    triage.success_count += 1
                else:
                    triage.failure_count += 1
                triage.updated_at = datetime.now()
                if run_id and run_id not in triage.source_runs:
                    triage.source_runs.append(run_id)
                logger.debug(
                    "Recorded %s for triage %s (now: %d/%d)",
                    "success" if success else "failure",
                    triage_id,
                    triage.success_count,
                    triage.failure_count,
                )
                return

    def get_triage_patterns(
        self,
        tier: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[TriagePattern]:
        """Get triage patterns with optional filtering.

        Args:
            tier: Filter by recommended tier
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching triage patterns
        """
        results = []
        for triage in self.triage:
            if triage.confidence < min_confidence:
                continue
            if tier and triage.recommended_tier != tier:
                continue
            results.append(triage)

        # Sort by confidence
        results.sort(key=lambda t: t.confidence, reverse=True)
        return results

    def delete_triage_pattern(self, triage_id: str) -> bool:
        """Delete a triage pattern.

        Args:
            triage_id: Triage pattern ID

        Returns:
            True if deleted
        """
        for i, triage in enumerate(self.triage):
            if triage.id == triage_id:
                del self.triage[i]
                return True
        return False

    # Persistence

    def save(self) -> None:
        """Save store to disk."""
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self.embeddings is not None:
            np.save(self.store_path / "embeddings.npy", self.embeddings)

        # Save memories as JSON
        memories_data = [m.model_dump(mode="json") for m in self.memories]
        with open(self.store_path / "memories.json", "w", encoding="utf-8") as f:
            json.dump(memories_data, f, indent=2, default=str)

        # Save patterns as JSON
        patterns_data = [p.model_dump(mode="json") for p in self.patterns]
        with open(self.store_path / "patterns.json", "w", encoding="utf-8") as f:
            json.dump(patterns_data, f, indent=2, default=str)

        # Save errors as JSON
        errors_data = [e.model_dump(mode="json") for e in self.errors]
        with open(self.store_path / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors_data, f, indent=2, default=str)

        # Save triage patterns as JSON
        triage_data = [t.model_dump(mode="json") for t in self.triage]
        with open(self.store_path / "triage.json", "w", encoding="utf-8") as f:
            json.dump(triage_data, f, indent=2, default=str)

        # Save BM25 index
        bm25_data = self.bm25.to_dict()
        with open(self.store_path / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(bm25_data, f, indent=2)

        # Save metadata (convert set to list for JSON)
        metadata_copy = dict(self.metadata)
        if isinstance(metadata_copy.get("runs_indexed"), set):
            metadata_copy["runs_indexed"] = list(metadata_copy["runs_indexed"])

        with open(self.store_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_copy, f, indent=2)

        logger.info("Saved memory store to %s", self.store_path)

    def load(self) -> None:
        """Load store from disk."""
        # Load metadata
        metadata_path = self.store_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                loaded_metadata = json.load(f)

            # Convert runs_indexed list back to set
            if "runs_indexed" in loaded_metadata:
                loaded_metadata["runs_indexed"] = set(loaded_metadata["runs_indexed"])
            else:
                loaded_metadata["runs_indexed"] = set()

            self.metadata = loaded_metadata

        # Load embeddings
        embeddings_path = self.store_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # Load memories
        memories_path = self.store_path / "memories.json"
        if memories_path.exists():
            with open(memories_path, encoding="utf-8") as f:
                memories_data = json.load(f)
                self.memories = [RunMemoryEntry(**m) for m in memories_data]

        # Load patterns
        patterns_path = self.store_path / "patterns.json"
        if patterns_path.exists():
            with open(patterns_path, encoding="utf-8") as f:
                patterns_data = json.load(f)
                self.patterns = [ExtractedPattern(**p) for p in patterns_data]

        # Load errors
        errors_path = self.store_path / "errors.json"
        if errors_path.exists():
            with open(errors_path, encoding="utf-8") as f:
                errors_data = json.load(f)
                self.errors = [ErrorPattern(**e) for e in errors_data]

        # Load triage patterns
        triage_path = self.store_path / "triage.json"
        if triage_path.exists():
            with open(triage_path, encoding="utf-8") as f:
                triage_data = json.load(f)
                self.triage = [TriagePattern(**t) for t in triage_data]

        # Load BM25 index
        bm25_path = self.store_path / "bm25_index.json"
        if bm25_path.exists():
            with open(bm25_path, encoding="utf-8") as f:
                bm25_data = json.load(f)
                self.bm25 = BM25.from_dict(bm25_data)
        else:
            # Rebuild BM25 index from memories if not found (migration from v1)
            self._rebuild_bm25_index()

        # Auto-migrate old triage memory if present
        self.migrate_old_triage_memory()

        logger.info(
            "Loaded memory store: %d memories, %d patterns, %d errors, %d triage",
            len(self.memories),
            len(self.patterns),
            len(self.errors),
            len(self.triage),
        )

    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from memories.

        Used for migration from v1 stores or after delete operations.
        """
        self.bm25 = BM25()
        for idx, memory in enumerate(self.memories):
            self.bm25.add_document(idx, memory.content)
        logger.debug("Rebuilt BM25 index with %d documents", len(self.memories))

    def migrate_old_triage_memory(self) -> int:
        """Migrate old triage_memory.json to unified store.

        Imports entries from ~/.coding-factory/triage_memory.json if it exists.

        Returns:
            Number of entries migrated
        """
        old_file = Path.home() / ".coding-factory" / "triage_memory.json"
        if not old_file.exists():
            return 0

        try:
            data = json.loads(old_file.read_text())
            migrated = 0

            for old_key, entry in data.items():
                # Generate new ID format
                triage_id = f"tri-{old_key}"

                # Check if already migrated
                exists = any(t.id == triage_id for t in self.triage)
                if exists:
                    continue

                # Create TriagePattern from old format
                pattern = TriagePattern(
                    id=triage_id,
                    pattern=entry.get("pattern", ""),
                    original_prompt=entry.get("original_prompt", ""),
                    size=entry.get("estimate", {}).get("size", "m"),
                    estimated_lines=entry.get("estimate", {}).get("estimated_lines", 100),
                    domains=entry.get("estimate", {}).get("domains", []),
                    recommended_tier=entry.get("estimate", {}).get("recommended_tier", "medium"),
                    keywords=entry.get("extracted_keywords", []),
                    source_runs=[],
                    success_count=entry.get("success_count", 0),
                    failure_count=entry.get("failure_count", 0),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                self.triage.append(pattern)
                migrated += 1

            if migrated > 0:
                self.save()
                logger.info("Migrated %d triage entries from old format", migrated)

                # Optionally rename old file to mark as migrated
                old_file.rename(old_file.with_suffix(".json.migrated"))

            return migrated

        except Exception as e:
            logger.warning("Failed to migrate old triage memory: %s", e)
            return 0

    def clear(self) -> None:
        """Clear all data from the store."""
        self.embeddings = None
        self.memories = []
        self.patterns = []
        self.errors = []
        self.triage = []
        self.bm25 = BM25()  # Reset BM25 index
        self.metadata = {
            "version": self.VERSION,
            "count": 0,
            "dimensions": 0,
            "runs_indexed": set(),
            "last_indexed": None,
            "created_at": datetime.now().isoformat(),
        }
        logger.info("Cleared memory store")

    def is_run_indexed(self, run_id: str) -> bool:
        """Check if a run has been indexed.

        Args:
            run_id: Run ID to check

        Returns:
            True if run is already indexed
        """
        runs = self.metadata.get("runs_indexed", set())
        if isinstance(runs, list):
            runs = set(runs)
        return run_id in runs

    def stats(self) -> MemoryStats:
        """Get store statistics.

        Returns:
            MemoryStats with detailed statistics
        """
        # Count by type
        memories_by_type: dict[str, int] = {}
        memories_by_outcome: dict[str, int] = {}

        for memory in self.memories:
            type_key = memory.memory_type.value
            memories_by_type[type_key] = memories_by_type.get(type_key, 0) + 1

            outcome_key = memory.outcome.value
            memories_by_outcome[outcome_key] = memories_by_outcome.get(outcome_key, 0) + 1

        # Get runs list
        runs = self.metadata.get("runs_indexed", set())
        if isinstance(runs, set):
            runs_list = sorted(runs)
        else:
            runs_list = sorted(runs)

        # Calculate store size
        store_size = 0
        if self.store_path.exists():
            for file in self.store_path.iterdir():
                if file.is_file():
                    store_size += file.stat().st_size

        # Get last indexed timestamp
        last_indexed = None
        if self.metadata.get("last_indexed"):
            try:
                last_indexed = datetime.fromisoformat(self.metadata["last_indexed"])
            except (ValueError, TypeError):
                pass

        return MemoryStats(
            total_memories=len(self.memories),
            total_patterns=len(self.patterns),
            total_errors=len(self.errors),
            total_triage=len(self.triage),
            memories_by_type=memories_by_type,
            memories_by_outcome=memories_by_outcome,
            runs_indexed=len(runs_list),
            run_ids=runs_list,
            store_path=str(self.store_path),
            embedding_dimensions=self.metadata.get("dimensions", 0),
            last_indexed=last_indexed,
            store_size_bytes=store_size,
        )
