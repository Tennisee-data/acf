"""Memory system for past-run learning.

This module provides:
- MemoryStore: Persistent storage for run memories with hybrid search
- RunIndexer: Extract and index memories from completed runs
- MemoryRetriever: Search and retrieve relevant memories
- PatternExtractor: Learn patterns from multiple runs
- SearchMode: Search mode enum (semantic, lexical, hybrid)
- BM25: Lexical search implementation
- CustomerMemoryManager: Per-customer memory for multi-tenant web API
"""

from .bm25 import BM25
from .customer_memory import CustomerMemoryManager, get_customer_memory_manager
from .indexer import RunIndexer
from .patterns import PatternExtractor
from .retriever import MemoryRetriever, RetrieverConfig
from .store import MemoryStore, SearchMode

__all__ = [
    "MemoryStore",
    "SearchMode",
    "BM25",
    "RunIndexer",
    "MemoryRetriever",
    "RetrieverConfig",
    "PatternExtractor",
    "CustomerMemoryManager",
    "get_customer_memory_manager",
]
