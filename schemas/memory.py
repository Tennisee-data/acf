"""Memory schemas for past-run learning.

Defines structured models for:
- Run memories (indexed from completed runs)
- Extracted patterns (learned from multiple runs)
- Error patterns (with fixes)
- Historical context (injected into agents)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunOutcome(str, Enum):
    """Outcome of a pipeline run."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some stages failed but feature shipped
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryType(str, Enum):
    """Type of memory entry."""

    FEATURE = "feature"  # Feature description + spec
    DESIGN_DECISION = "decision"  # Design choice with rationale
    PATTERN = "pattern"  # Extracted coding pattern
    ERROR_FIX = "error_fix"  # Error encountered and how it was fixed
    TECH_DEBT = "tech_debt"  # Technical debt noted
    IMPLEMENTATION = "implementation"  # Implementation approach
    TRIAGE = "triage"  # Complexity triage decision for model routing


class RunMemoryEntry(BaseModel):
    """A single indexed memory from a past run."""

    id: str = Field(..., description="Unique memory entry ID")
    run_id: str = Field(..., description="Source run ID")
    memory_type: MemoryType = Field(..., description="Type of memory")

    # Content for embedding
    content: str = Field(..., description="Text content to embed")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    outcome: RunOutcome = Field(..., description="Run outcome")
    stage: str | None = Field(None, description="Pipeline stage if applicable")

    # Structured data
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional structured data"
    )

    # Relevance tracking
    times_retrieved: int = Field(0, description="How often this was retrieved")
    last_retrieved: datetime | None = Field(None, description="Last retrieval timestamp")
    usefulness_score: float = Field(0.0, description="User-rated usefulness (0-1)")


class ExtractedPattern(BaseModel):
    """A learned pattern from multiple runs."""

    id: str = Field(..., description="Pattern ID")
    name: str = Field(..., description="Pattern name")
    description: str = Field(..., description="What this pattern is")

    # Evidence
    source_runs: list[str] = Field(
        default_factory=list, description="Run IDs where pattern appeared"
    )
    examples: list[str] = Field(
        default_factory=list, description="Code/config examples"
    )

    # Classification
    pattern_type: str = Field(
        ..., description="naming, architecture, testing, config, etc."
    )
    domains: list[str] = Field(
        default_factory=list, description="Applicable domains"
    )

    # Confidence
    occurrence_count: int = Field(1, description="Times pattern observed")
    success_rate: float = Field(1.0, description="Success rate when pattern used")

    # Usage guidance
    when_to_use: str = Field("", description="When to apply this pattern")
    when_to_avoid: str = Field("", description="When not to use this pattern")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ErrorPattern(BaseModel):
    """A learned error pattern with fix."""

    id: str = Field(..., description="Error pattern ID")
    error_signature: str = Field(..., description="Error type/message pattern")

    # Context
    source_runs: list[str] = Field(
        default_factory=list, description="Run IDs where error occurred"
    )
    stage: str = Field(..., description="Stage where error typically occurs")

    # Fix information
    fix_description: str = Field(..., description="How to fix this error")
    fix_code_snippet: str | None = Field(None, description="Code fix if applicable")
    prevention_hint: str = Field("", description="How to prevent this error")

    # Stats
    occurrence_count: int = Field(1, description="Times this error was seen")
    fix_success_rate: float = Field(1.0, description="Success rate of the fix")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TriagePattern(BaseModel):
    """A learned triage pattern for model routing.

    Stores successful complexity estimations to skip LLM triage
    for similar future tasks.
    """

    id: str = Field(..., description="Triage pattern ID")
    pattern: str = Field(..., description="Normalized prompt pattern")
    original_prompt: str = Field(..., description="Original prompt text")

    # Triage decision
    size: str = Field(..., description="Estimated size: xs, s, m, l, xl")
    estimated_lines: int = Field(100, description="Estimated lines of code")
    domains: list[str] = Field(default_factory=list, description="Domains involved")
    recommended_tier: str = Field(..., description="Model tier: cheap, medium, premium")

    # Extracted features for matching
    keywords: list[str] = Field(default_factory=list, description="Characteristic keywords")

    # Evidence and confidence
    source_runs: list[str] = Field(default_factory=list, description="Run IDs where pattern used")
    success_count: int = Field(0, description="Successful runs with this triage")
    failure_count: int = Field(0, description="Failed runs with this triage")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_used: datetime | None = Field(None, description="Last time pattern was matched")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """Calculate confidence based on success rate and sample size."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        # Higher confidence with more samples and higher success rate
        sample_factor = min(1.0, total / 5)  # Max confidence boost at 5 samples
        return self.success_rate * (0.5 + 0.5 * sample_factor)


class MemorySearchResult(BaseModel):
    """Result from memory search."""

    entry: RunMemoryEntry = Field(..., description="The matched memory entry")
    score: float = Field(..., description="Similarity score 0-1")
    decay_factor: float = Field(1.0, description="Time-based decay factor applied")
    relevance_explanation: str | None = Field(
        None, description="Why this is relevant"
    )


class HistoricalContext(BaseModel):
    """Context package from memory to inject into agent prompts."""

    # Similar past features (successful/partial runs)
    similar_features: list[MemorySearchResult] = Field(
        default_factory=list, description="Similar features from past runs"
    )

    # Anti-patterns from failed runs - what NOT to do
    anti_patterns: list[MemorySearchResult] = Field(
        default_factory=list, description="Failed attempts to learn from"
    )

    # Relevant patterns
    applicable_patterns: list[ExtractedPattern] = Field(
        default_factory=list, description="Patterns that may apply"
    )

    # Error prevention hints
    potential_errors: list[ErrorPattern] = Field(
        default_factory=list, description="Errors to watch out for"
    )

    # Design decisions from similar work
    prior_decisions: list[MemorySearchResult] = Field(
        default_factory=list, description="Design decisions from similar features"
    )

    # Summary for prompt
    summary: str = Field(
        "", description="LLM-friendly summary of historical context"
    )

    # Stats
    total_memories_searched: int = Field(0, description="Total memories searched")
    search_time_ms: float = Field(0.0, description="Search time in milliseconds")


class MemoryStats(BaseModel):
    """Statistics about the memory store."""

    total_memories: int = Field(0, description="Total memory entries")
    total_patterns: int = Field(0, description="Total extracted patterns")
    total_errors: int = Field(0, description="Total error patterns")
    total_triage: int = Field(0, description="Total triage patterns")

    # Breakdown by type
    memories_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count by memory type"
    )

    # Breakdown by outcome
    memories_by_outcome: dict[str, int] = Field(
        default_factory=dict, description="Count by run outcome"
    )

    # Runs indexed
    runs_indexed: int = Field(0, description="Number of runs indexed")
    run_ids: list[str] = Field(default_factory=list, description="List of indexed run IDs")

    # Store info
    store_path: str = Field("", description="Path to memory store")
    embedding_dimensions: int = Field(0, description="Embedding vector dimensions")
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")
    store_size_bytes: int = Field(0, description="Total store size in bytes")
