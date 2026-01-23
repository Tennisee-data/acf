"""RAG (Retrieval Augmented Generation) module for Coding Factory.

Enables working with large codebases by:
- Retrieving relevant chunks based on queries
- Augmenting LLM prompts with context
- Budget-aware retrieval respecting model context windows

Note: Semantic retrieval with embeddings is available as a premium
extension on the ACF Marketplace.
"""

from .chunker import CodeChunker, CodeChunk
from .store import VectorStore, SearchResult, FileMetadata
from .retriever import CodeRetriever
from .model_limits import (
    MODEL_CONTEXT_WINDOWS,
    ModelBudget,
    get_context_window,
    calculate_budget,
)
from .token_counter import (
    count_tokens,
    count_tokens_cached,
    estimate_output_tokens,
    format_token_count,
    truncate_to_tokens,
)
from .budget_retriever import (
    BudgetRetriever,
    RAGSource,
    RAGBudgetReport,
    SourcePriority,
    SourceMetadata,
)

__all__ = [
    # Chunking
    "CodeChunker",
    "CodeChunk",
    "VectorStore",
    "SearchResult",
    "FileMetadata",
    "CodeRetriever",
    # Model limits
    "MODEL_CONTEXT_WINDOWS",
    "ModelBudget",
    "get_context_window",
    "calculate_budget",
    # Token counting
    "count_tokens",
    "count_tokens_cached",
    "estimate_output_tokens",
    "format_token_count",
    "truncate_to_tokens",
    # Budget retriever
    "BudgetRetriever",
    "RAGSource",
    "RAGBudgetReport",
    "SourcePriority",
    "SourceMetadata",
]
