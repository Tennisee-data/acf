"""Token counting utilities for RAG budget management.

Provides token counting with optional tiktoken support,
falling back to character-based estimation.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
_tiktoken_available = False
_tiktoken_encoder = None

try:
    import tiktoken
    _tiktoken_available = True
    # Use cl100k_base encoding (GPT-4, Claude-compatible)
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    logger.debug("tiktoken not available, using character-based estimation")


# Characters per token ratio (conservative estimate)
# Different content types have different ratios
CHARS_PER_TOKEN = {
    "code": 3.5,      # Code is more compact
    "text": 4.0,      # English text
    "json": 3.0,      # JSON has lots of punctuation
    "markdown": 4.0,  # Similar to text
    "default": 4.0,
}


def count_tokens(text: str, content_type: str = "default") -> int:
    """Count tokens in text.

    Uses tiktoken if available, otherwise falls back to
    character-based estimation.

    Args:
        text: Text to count tokens for
        content_type: Type of content ("code", "text", "json", "markdown")

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    if _tiktoken_available and _tiktoken_encoder:
        try:
            return len(_tiktoken_encoder.encode(text))
        except Exception as e:
            logger.debug("tiktoken encoding failed: %s", e)

    # Fallback to character-based estimation
    ratio = CHARS_PER_TOKEN.get(content_type, CHARS_PER_TOKEN["default"])
    return max(1, int(len(text) / ratio))


def count_tokens_cached(text: str, content_type: str = "default") -> int:
    """Count tokens with caching for repeated texts.

    Uses LRU cache to avoid re-counting the same content.
    Useful for RAG sources that are repeatedly considered.

    Args:
        text: Text to count tokens for
        content_type: Type of content

    Returns:
        Estimated token count
    """
    # Use hash for cache key (text might be too long)
    return _count_tokens_by_hash(hash(text), text, content_type)


@functools.lru_cache(maxsize=1024)
def _count_tokens_by_hash(text_hash: int, text: str, content_type: str) -> int:
    """Internal cached token counter."""
    return count_tokens(text, content_type)


def estimate_output_tokens(task_type: str) -> int:
    """Estimate expected output tokens based on task type.

    Different pipeline stages produce different amounts of output.

    Args:
        task_type: Type of task/stage

    Returns:
        Estimated output tokens to reserve
    """
    OUTPUT_ESTIMATES = {
        # Pipeline stages
        "spec": 500,           # Feature spec is relatively short
        "decomposition": 1000, # Workplan can be detailed
        "context": 300,        # File listing is compact
        "design": 2000,        # Design proposals are verbose
        "implementation": 4000, # Code generation needs room
        "test": 2000,          # Test code
        "fix": 2000,           # Code fixes
        "verify": 500,         # Validation results
        "docs": 1500,          # Documentation
        "code_review": 1500,   # Review feedback

        # Other tasks
        "chat": 1000,          # Conversational responses
        "analysis": 2000,      # Code analysis
        "refactor": 3000,      # Refactoring code
        "default": 2000,
    }

    return OUTPUT_ESTIMATES.get(task_type, OUTPUT_ESTIMATES["default"])


def format_token_count(tokens: int) -> str:
    """Format token count for display.

    Args:
        tokens: Number of tokens

    Returns:
        Formatted string (e.g., "1.5K", "32K")
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    else:
        return str(tokens)


def truncate_to_tokens(text: str, max_tokens: int, content_type: str = "default") -> str:
    """Truncate text to fit within token limit.

    Truncates at word boundaries when possible.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        content_type: Type of content

    Returns:
        Truncated text
    """
    if count_tokens(text, content_type) <= max_tokens:
        return text

    # Binary search for the right length
    ratio = CHARS_PER_TOKEN.get(content_type, CHARS_PER_TOKEN["default"])
    estimated_chars = int(max_tokens * ratio)

    # Start with estimate and adjust
    result = text[:estimated_chars]

    # Adjust up or down
    while count_tokens(result, content_type) > max_tokens and len(result) > 0:
        result = result[:-100]

    while count_tokens(result, content_type) < max_tokens and len(result) < len(text):
        next_chunk = text[len(result):len(result) + 50]
        if count_tokens(result + next_chunk, content_type) <= max_tokens:
            result += next_chunk
        else:
            break

    # Try to truncate at word boundary
    if result and not result[-1].isspace():
        last_space = result.rfind(" ")
        if last_space > len(result) * 0.8:  # Don't lose too much
            result = result[:last_space]

    return result.rstrip()
