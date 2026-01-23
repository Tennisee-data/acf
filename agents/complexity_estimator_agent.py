"""Complexity Estimator Agent - Quick triage of task complexity.

Uses a fast/cheap model to analyze the feature description and estimate:
- Task size (xs/s/m/l/xl)
- Domains involved
- Recommended model tier

Learns from past successful runs to skip LLM calls for known patterns.
Uses the unified memory store for persistence (memory/store.py).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from llm_backend import LLMBackend
from utils.json_repair import parse_llm_json

if TYPE_CHECKING:
    from memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ComplexityEstimate:
    """Result of complexity estimation."""

    size: str  # xs, s, m, l, xl
    estimated_lines: int
    domains: list[str]
    recommended_tier: str  # cheap, medium, premium
    confidence: float  # 0.0 - 1.0
    reasoning: str
    from_memory: bool = False
    memory_key: str | None = None


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt for pattern matching.

    Removes specific details to find similar prompts:
    - Lowercase
    - Remove numbers
    - Remove quoted strings
    - Collapse whitespace
    """
    normalized = prompt.lower()
    # Remove quoted strings
    normalized = re.sub(r'"[^"]*"', '""', normalized)
    normalized = re.sub(r"'[^']*'", "''", normalized)
    # Remove numbers
    normalized = re.sub(r'\d+', '#', normalized)
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _extract_keywords(prompt: str) -> list[str]:
    """Extract characteristic keywords from prompt."""
    # Common action words
    actions = re.findall(
        r'\b(add|create|build|make|implement|fix|update|remove|delete|refactor)\b',
        prompt.lower()
    )
    # Common targets
    targets = re.findall(
        r'\b(endpoint|api|route|page|component|function|class|service|model|'
        r'database|auth|login|payment|application|website|app)\b',
        prompt.lower()
    )
    return list(set(actions + targets))


def _generate_triage_id(prompt: str) -> str:
    """Generate unique triage ID from normalized prompt."""
    normalized = _normalize_prompt(prompt)
    return f"tri-{hashlib.md5(normalized.encode()).hexdigest()[:12]}"


class ComplexityEstimatorAgent:
    """Agent for quick complexity triage.

    Uses a cheap/fast model to analyze task complexity before
    routing to appropriate models for the main pipeline.

    Integrates with the unified memory store (memory/store.py) for
    learning from past successful triage decisions.
    """

    SYSTEM_PROMPT = """You are a task complexity estimator. Analyze the feature request and estimate:

1. size: xs (<10 lines), s (10-50), m (50-200), l (200-500), xl (>500)
2. estimated_lines: rough line count
3. domains: list of areas (api, database, auth, payments, ui, etc.)
4. recommended_tier: cheap (simple CRUD), medium (standard features), premium (complex/critical)
5. reasoning: one sentence explanation

Output ONLY valid JSON:
{"size": "s", "estimated_lines": 30, "domains": ["api"], "recommended_tier": "cheap", "reasoning": "Simple endpoint addition"}"""

    def __init__(self, llm: LLMBackend, memory_store: MemoryStore | None = None):
        """Initialize with a fast/cheap model.

        Args:
            llm: LLM backend for triage calls
            memory_store: Optional unified memory store for learning
        """
        self.llm = llm
        self._memory_store = memory_store
        self._last_triage_id: str | None = None

    @property
    def memory_store(self) -> MemoryStore | None:
        """Get memory store, lazily initializing if needed."""
        if self._memory_store is None:
            try:
                from memory.store import MemoryStore
                self._memory_store = MemoryStore()
            except Exception as e:
                logger.warning("Failed to initialize memory store: %s", e)
        return self._memory_store

    def estimate(self, feature_description: str) -> ComplexityEstimate:
        """Estimate complexity of a feature.

        First checks memory for similar past tasks.
        Falls back to LLM if no good match found.

        Args:
            feature_description: The feature request

        Returns:
            ComplexityEstimate with size, tier recommendation, etc.
        """
        # 1. Check memory first (unified store)
        if self.memory_store:
            cached = self.memory_store.lookup_triage(feature_description)
            if cached:
                logger.info(
                    "Triage memory hit: %s (confidence=%.2f)",
                    cached.id, cached.confidence
                )
                self._last_triage_id = cached.id
                return ComplexityEstimate(
                    size=cached.size,
                    estimated_lines=cached.estimated_lines,
                    domains=cached.domains,
                    recommended_tier=cached.recommended_tier,
                    confidence=cached.confidence,
                    reasoning=f"From memory (matched {cached.success_count} similar tasks)",
                    from_memory=True,
                    memory_key=cached.id,
                )

        # 2. Call LLM for triage
        start_time = time.time()

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": feature_description},
                ],
                temperature=0.1,  # Low temp for consistency
            )

            # Handle response (could be string or dict)
            content = response if isinstance(response, str) else response.get("content", "")

            # Parse JSON
            data = parse_llm_json(content, default={})

            estimate = ComplexityEstimate(
                size=data.get("size", "m"),
                estimated_lines=data.get("estimated_lines", 100),
                domains=data.get("domains", []),
                recommended_tier=data.get("recommended_tier", "medium"),
                confidence=0.8,  # LLM estimates get 0.8 base confidence
                reasoning=data.get("reasoning", "LLM estimate"),
                from_memory=False,
            )

            # Store in unified memory for future
            if self.memory_store:
                triage_id = _generate_triage_id(feature_description)
                from schemas.memory import TriagePattern
                triage_pattern = TriagePattern(
                    id=triage_id,
                    pattern=_normalize_prompt(feature_description),
                    original_prompt=feature_description,
                    size=estimate.size,
                    estimated_lines=estimate.estimated_lines,
                    domains=estimate.domains,
                    recommended_tier=estimate.recommended_tier,
                    keywords=_extract_keywords(feature_description),
                    source_runs=[],
                    success_count=0,
                    failure_count=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                self.memory_store.add_triage_pattern(triage_pattern)
                self.memory_store.save()
                estimate.memory_key = triage_id
                self._last_triage_id = triage_id
                logger.debug("Stored triage decision: %s", triage_id)

            duration = time.time() - start_time
            logger.info(
                "Complexity triage: size=%s, tier=%s (%.1fs)",
                estimate.size, estimate.recommended_tier, duration
            )

            return estimate

        except Exception as e:
            logger.warning("Complexity estimation failed: %s", e)
            # Default to medium on failure
            return ComplexityEstimate(
                size="m",
                estimated_lines=100,
                domains=[],
                recommended_tier="medium",
                confidence=0.5,
                reasoning=f"Default (estimation failed: {e})",
                from_memory=False,
            )

    def record_outcome(self, memory_key: str | None, success: bool, run_id: str | None = None) -> None:
        """Record whether the triage decision was successful.

        Call this after a pipeline run completes to train the memory.

        Args:
            memory_key: Key from the estimate (triage pattern ID)
            success: Whether the run succeeded with the chosen model
            run_id: Optional run ID to track
        """
        if memory_key and self.memory_store:
            self.memory_store.record_triage_outcome(memory_key, success, run_id)
            self.memory_store.save()

    @property
    def last_triage_id(self) -> str | None:
        """Get the ID of the last triage decision."""
        return self._last_triage_id
