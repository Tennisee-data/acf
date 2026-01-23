"""RAG Budget Agent - Intelligent context window management.

Calculates and allocates RAG content budget based on:
- Model context window size
- Pipeline stage requirements
- Content priority and relevance

Supports two retrieval modes:
- Semantic: Uses LocalEmbeddings + SemanticRetriever (requires sentence-transformers)
- Keyword: Falls back to trigger/keyword matching (no extra dependencies)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.budget_retriever import BudgetRetriever, RAGBudgetReport, RAGSource, SourcePriority
from rag.model_limits import calculate_budget, get_context_window, ModelBudget
from rag.token_counter import count_tokens, estimate_output_tokens, format_token_count

logger = logging.getLogger(__name__)

# Check if semantic retrieval is available
_SEMANTIC_AVAILABLE: bool | None = None


def _check_semantic_available() -> bool:
    """Check if semantic retrieval dependencies are available."""
    global _SEMANTIC_AVAILABLE
    if _SEMANTIC_AVAILABLE is None:
        try:
            from rag.embeddings import LocalEmbeddings
            from rag.semantic_retriever import SemanticRetriever
            # Try to instantiate to verify it works
            _SEMANTIC_AVAILABLE = True
            logger.info("Semantic retrieval available (sentence-transformers installed)")
        except (ImportError, Exception) as e:
            logger.info("Semantic retrieval not available, using keyword matching: %s", e)
            _SEMANTIC_AVAILABLE = False
    return _SEMANTIC_AVAILABLE


@dataclass
class RAGBudgetConfig:
    """Configuration for RAG budget management."""

    # Safety margin as percentage of context window
    safety_margin_percent: float = 5.0

    # Minimum tokens to reserve for output
    min_output_reserve: int = 1000

    # Maximum percentage of context for RAG
    max_rag_percent: float = 60.0

    # Always include invariants for these domains
    critical_domains: list[str] = field(default_factory=lambda: [
        "payments", "security", "auth", "webhook"
    ])


@dataclass
class RAGAllocation:
    """Result of RAG budget allocation."""

    model: str
    stage: str
    context_window: int
    budget_breakdown: ModelBudget
    rag_report: RAGBudgetReport
    rag_content: str

    def summary(self) -> str:
        """Generate allocation summary."""
        lines = [
            f"RAG Allocation for {self.stage} stage ({self.model})",
            f"  Context Window: {format_token_count(self.context_window)}",
            f"  System Prompt: {format_token_count(self.budget_breakdown.system_prompt_tokens)}",
            f"  User Prompt: {format_token_count(self.budget_breakdown.user_prompt_tokens)}",
            f"  Output Reserve: {format_token_count(self.budget_breakdown.expected_output_tokens)}",
            f"  Safety Margin: {format_token_count(self.budget_breakdown.safety_margin_tokens)}",
            f"  Available for RAG: {format_token_count(self.budget_breakdown.available_for_rag)}",
            f"  RAG Used: {format_token_count(self.rag_report.tokens_used)} ({self.rag_report.utilization_percent:.1f}%)",
            f"  Sources: {self.rag_report.sources_included_count} included, {self.rag_report.sources_excluded_count} excluded",
        ]
        return "\n".join(lines)


class RAGBudgetAgent:
    """Agent for intelligent RAG budget management.

    Calculates available token budget for RAG content based on
    model limits and allocates budget to highest-priority content.

    Supports two modes:
    - Semantic: Uses embeddings for similarity search (better quality)
    - Keyword: Uses trigger/keyword matching (no dependencies)

    Example:
        agent = RAGBudgetAgent(invariants_dir=Path("invariants"))
        allocation = agent.allocate(
            model="qwen3:14b",
            stage="implementation",
            query="stripe webhook handler",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        # Use allocation.rag_content in your prompt
    """

    def __init__(
        self,
        invariants_dir: Path | None = None,
        rag_docs_dir: Path | None = None,
        config: RAGBudgetConfig | None = None,
        use_semantic: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        custom_retriever: Any = None,
    ):
        """Initialize RAG budget agent.

        Args:
            invariants_dir: Path to invariants directory
            rag_docs_dir: Path to RAG documentation directory
            config: Budget configuration
            use_semantic: Whether to use semantic retrieval (auto-fallback if unavailable)
            embedding_model: Model for LocalEmbeddings (default: all-MiniLM-L6-v2)
            custom_retriever: Optional custom retriever from marketplace extension
        """
        self.config = config or RAGBudgetConfig()
        self.retriever = BudgetRetriever()
        self._semantic_retriever = None
        self._embeddings = None
        self._custom_retriever = custom_retriever
        self._use_semantic = use_semantic and _check_semantic_available()

        # Use custom retriever from extension if provided
        if custom_retriever is not None:
            self._use_semantic = False  # Custom retriever handles its own logic
            logger.info("Using custom retriever from marketplace extension")
        # Initialize semantic retriever if available and no custom retriever
        elif self._use_semantic:
            self._init_semantic_retriever(embedding_model)

        # Load global sources
        if invariants_dir and invariants_dir.exists():
            count = self._load_invariants(invariants_dir)
            logger.info("Loaded %d invariants (semantic=%s)", count, self._use_semantic)

        if rag_docs_dir and rag_docs_dir.exists():
            count = self.retriever.add_api_docs(rag_docs_dir)
            logger.info("Loaded %d API docs", count)

    def _init_semantic_retriever(self, embedding_model: str) -> None:
        """Initialize semantic retriever with embeddings."""
        try:
            from rag.embeddings import LocalEmbeddings
            from rag.semantic_retriever import SemanticRetriever

            logger.info("Initializing semantic retriever with model: %s", embedding_model)
            self._embeddings = LocalEmbeddings(model=embedding_model)
            self._semantic_retriever = SemanticRetriever(
                embeddings=self._embeddings,
                keyword_boost=0.2,
                priority_boosts={
                    "CRITICAL": 0.3,
                    "HIGH": 0.15,
                    "MEDIUM": 0.0,
                    "LOW": -0.1,
                },
            )
            logger.info("Semantic retriever initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize semantic retriever: %s", e)
            self._use_semantic = False

    def _load_invariants(self, invariants_dir: Path) -> int:
        """Load invariants for both keyword and semantic retrieval.

        Args:
            invariants_dir: Path to invariants directory

        Returns:
            Number of invariants loaded
        """
        from rag.semantic_retriever import Chunk

        count = 0
        chunks_to_add = []

        for json_file in invariants_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue

            try:
                data = json.loads(json_file.read_text())
                topic = data.get("topic", json_file.stem)
                triggers = data.get("triggers", [])

                # Create condensed content
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

                # Add to keyword retriever (fallback)
                source = RAGSource(
                    id=f"invariant:{topic}",
                    content=content,
                    source_type="invariant",
                    source_path=str(json_file),
                    priority=SourcePriority.CRITICAL,
                    metadata={
                        "triggers": triggers,
                        "category": data.get("category", ""),
                    },
                )
                self.retriever.add_source(source, tier="global")

                # Prepare chunk for semantic retriever
                if self._use_semantic:
                    chunk = Chunk(
                        content=content,
                        metadata={
                            "id": f"invariant:{topic}",
                            "priority": "CRITICAL",
                            "triggers": triggers,
                            "source": str(json_file),
                            "category": data.get("category", ""),
                        },
                    )
                    chunks_to_add.append(chunk)

                count += 1

            except Exception as e:
                logger.warning("Failed to load invariant %s: %s", json_file, e)

        # Batch compute embeddings for all chunks
        if self._use_semantic and chunks_to_add and self._semantic_retriever:
            logger.info("Computing embeddings for %d invariants...", len(chunks_to_add))
            self._semantic_retriever.add_chunks(chunks_to_add, compute_embeddings=True)
            logger.info("Embeddings computed successfully")

        return count

    @property
    def is_semantic(self) -> bool:
        """Whether semantic retrieval is active."""
        return self._use_semantic and self._semantic_retriever is not None

    @property
    def is_custom(self) -> bool:
        """Whether a custom extension retriever is active."""
        return self._custom_retriever is not None

    def add_source(self, source: RAGSource, tier: str = "project") -> None:
        """Add a RAG source.

        Args:
            source: RAG source to add
            tier: "global" or "project"
        """
        self.retriever.add_source(source, tier=tier)

    def calculate_budget(
        self,
        model: str,
        stage: str,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> ModelBudget:
        """Calculate available budget for RAG content.

        Args:
            model: Model name
            stage: Pipeline stage
            system_prompt: System prompt text
            user_prompt: User prompt text

        Returns:
            ModelBudget with breakdown
        """
        context_window = get_context_window(model)

        # Calculate safety margin
        safety_margin = max(
            int(context_window * self.config.safety_margin_percent / 100),
            500  # Minimum 500 tokens
        )

        # Estimate output tokens based on stage
        output_tokens = max(
            estimate_output_tokens(stage),
            self.config.min_output_reserve,
        )

        return ModelBudget(
            model=model,
            context_window=context_window,
            system_prompt_tokens=count_tokens(system_prompt),
            user_prompt_tokens=count_tokens(user_prompt),
            expected_output_tokens=output_tokens,
            safety_margin_tokens=safety_margin,
        )

    def allocate(
        self,
        model: str,
        stage: str,
        query: str,
        system_prompt: str = "",
        user_prompt: str = "",
        context: str = "",
    ) -> RAGAllocation:
        """Allocate RAG budget and retrieve content.

        Uses semantic retrieval if available, falls back to keyword matching.

        Args:
            model: Model name
            stage: Pipeline stage
            query: Search query for RAG retrieval
            system_prompt: System prompt text
            user_prompt: User prompt text
            context: Additional context (e.g., feature description)

        Returns:
            RAGAllocation with content and breakdown
        """
        # Calculate budget
        budget = self.calculate_budget(model, stage, system_prompt, user_prompt)

        # Cap RAG budget at max percentage
        context_window = get_context_window(model)
        max_rag_tokens = int(context_window * self.config.max_rag_percent / 100)
        rag_budget = min(budget.available_for_rag, max_rag_tokens)

        # Check if query matches critical domains
        include_critical = any(
            domain in query.lower() or domain in context.lower()
            for domain in self.config.critical_domains
        )

        # Use custom retriever from marketplace extension if available
        if self._custom_retriever is not None:
            report, rag_content = self._custom_allocate(
                query=query,
                context=context,
                token_budget=rag_budget,
                model=model,
            )
        # Use semantic retrieval if available
        elif self._use_semantic and self._semantic_retriever:
            report, rag_content = self._semantic_allocate(
                query=query,
                context=context,
                token_budget=rag_budget,
                model=model,
            )
        else:
            # Fall back to keyword-based retrieval
            report = self.retriever.retrieve(
                query=query,
                token_budget=rag_budget,
                model=model,
                context=context,
                include_all_critical=include_critical,
            )
            rag_content = self.retriever.get_structured_content(report)

        return RAGAllocation(
            model=model,
            stage=stage,
            context_window=context_window,
            budget_breakdown=budget,
            rag_report=report,
            rag_content=rag_content,
        )

    def _semantic_allocate(
        self,
        query: str,
        context: str,
        token_budget: int,
        model: str,
    ) -> tuple[RAGBudgetReport, str]:
        """Perform semantic retrieval within budget.

        Args:
            query: Search query
            context: Additional context
            token_budget: Maximum tokens
            model: Model name for reporting

        Returns:
            Tuple of (RAGBudgetReport, formatted content)
        """
        combined_query = f"{query} {context}".strip()

        # Retrieve with budget constraint
        results, tokens_used = self._semantic_retriever.retrieve_with_budget(
            query=combined_query,
            token_budget=token_budget,
            top_k=50,
            min_score=0.3,
        )

        # Convert results to RAGSource for compatibility
        included_sources = []
        for result in results:
            source = RAGSource(
                id=result.chunk.id,
                content=result.chunk.content,
                source_type="invariant" if result.chunk.priority == "CRITICAL" else "pattern",
                relevance_score=result.similarity,
                priority=SourcePriority[result.chunk.priority] if result.chunk.priority in SourcePriority.__members__ else SourcePriority.MEDIUM,
                tokens=result.chunk.tokens,
                metadata={
                    "triggers": result.chunk.triggers,
                    "boosted": result.boosted,
                    "score": result.score,
                },
            )
            included_sources.append(source)

        # Build report
        report = RAGBudgetReport(
            model=model,
            token_budget=token_budget,
            tokens_used=tokens_used,
            included_sources=included_sources,
            excluded_sources=[],  # Semantic retriever handles this internally
        )

        # Get structured content
        rag_content = self._semantic_retriever.get_content(results, structured=True)

        logger.info(
            "Semantic retrieval: %d sources, %d tokens (%.1f%% of budget)",
            len(results),
            tokens_used,
            (tokens_used / token_budget * 100) if token_budget > 0 else 0,
        )

        return report, rag_content

    def _custom_allocate(
        self,
        query: str,
        context: str,
        token_budget: int,
        model: str,
    ) -> tuple[RAGBudgetReport, str]:
        """Perform retrieval using custom marketplace extension retriever.

        Args:
            query: Search query
            context: Additional context
            token_budget: Maximum tokens
            model: Model name for reporting

        Returns:
            Tuple of (RAGBudgetReport, formatted content)
        """
        combined_query = f"{query} {context}".strip()

        try:
            # Call the custom retriever's retrieve method
            # Extension retrievers should implement: retrieve(query, token_budget) -> (results, content)
            if hasattr(self._custom_retriever, 'retrieve_with_budget'):
                results, rag_content = self._custom_retriever.retrieve_with_budget(
                    query=combined_query,
                    token_budget=token_budget,
                )
            elif hasattr(self._custom_retriever, 'retrieve'):
                results = self._custom_retriever.retrieve(combined_query, top_k=20)
                rag_content = "\n\n".join(getattr(r, 'content', str(r)) for r in results)
            else:
                logger.warning("Custom retriever has no retrieve method")
                results = []
                rag_content = ""

            # Estimate tokens used
            tokens_used = count_tokens(rag_content, model)

            # Build report with minimal info (custom retriever handles details)
            report = RAGBudgetReport(
                model=model,
                token_budget=token_budget,
                tokens_used=tokens_used,
                included_sources=[],  # Custom retriever manages sources
                excluded_sources=[],
            )

            logger.info(
                "Custom retrieval: %d tokens (%.1f%% of budget)",
                tokens_used,
                (tokens_used / token_budget * 100) if token_budget > 0 else 0,
            )

            return report, rag_content

        except Exception as e:
            logger.error("Custom retriever failed: %s", e)
            # Return empty result on error
            return RAGBudgetReport(
                model=model,
                token_budget=token_budget,
                tokens_used=0,
                included_sources=[],
                excluded_sources=[],
            ), ""

    def get_stage_recommendations(self, stage: str) -> dict[str, Any]:
        """Get recommended RAG configuration for a stage.

        Args:
            stage: Pipeline stage name

        Returns:
            Dict with recommendations
        """
        STAGE_CONFIGS = {
            "spec": {
                "output_reserve": 500,
                "priority_sources": ["invariant"],
                "description": "Feature specification - minimal RAG needed",
            },
            "decomposition": {
                "output_reserve": 1000,
                "priority_sources": ["invariant", "pattern"],
                "description": "Task breakdown - some context helpful",
            },
            "context": {
                "output_reserve": 300,
                "priority_sources": [],
                "description": "File listing - no RAG needed",
            },
            "design": {
                "output_reserve": 2000,
                "priority_sources": ["invariant", "api_docs", "pattern"],
                "description": "Architecture design - high RAG value",
            },
            "implementation": {
                "output_reserve": 4000,
                "priority_sources": ["invariant", "api_docs", "example"],
                "description": "Code generation - maximum RAG value",
            },
            "test": {
                "output_reserve": 2000,
                "priority_sources": ["pattern", "example"],
                "description": "Test generation - moderate RAG value",
            },
            "fix": {
                "output_reserve": 2000,
                "priority_sources": ["invariant", "api_docs"],
                "description": "Bug fixing - high RAG value for patterns",
            },
            "verify": {
                "output_reserve": 500,
                "priority_sources": ["invariant"],
                "description": "Validation - minimal RAG needed",
            },
            "docs": {
                "output_reserve": 1500,
                "priority_sources": ["pattern"],
                "description": "Documentation - moderate RAG value",
            },
            "code_review": {
                "output_reserve": 1500,
                "priority_sources": ["invariant", "pattern"],
                "description": "Code review - high RAG value for standards",
            },
        }

        return STAGE_CONFIGS.get(stage, {
            "output_reserve": 2000,
            "priority_sources": ["invariant"],
            "description": "Unknown stage - default configuration",
        })

    def report(self, allocation: RAGAllocation) -> str:
        """Generate detailed report for an allocation.

        Args:
            allocation: RAG allocation result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            allocation.summary(),
            "",
            allocation.rag_report.summary(),
            "=" * 60,
        ]
        return "\n".join(lines)
