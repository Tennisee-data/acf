"""LLM Backend abstraction layer.

Provides a unified interface for different LLM providers.
Supports auto-detection based on available API keys.

Priority order for "auto" mode:
1. Anthropic (if ANTHROPIC_API_KEY set)
2. OpenAI (if OPENAI_API_KEY set)
3. LM Studio (if running on localhost:1234)
4. Ollama (local fallback)
"""

import logging
import os

from .base import LLMBackend
from .ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "LMStudioBackend",
    "get_backend",
    "detect_backend",
]


# LM Studio default configuration
LM_STUDIO_DEFAULT_URL = "http://localhost:1234/v1"


def _get_openai_backend():
    """Lazy import OpenAI backend."""
    from .openai_backend import OpenAIBackend
    return OpenAIBackend


def _get_anthropic_backend():
    """Lazy import Anthropic backend."""
    from .anthropic_backend import AnthropicBackend
    return AnthropicBackend


def _check_lmstudio_running() -> bool:
    """Check if LM Studio server is running on default port.

    Returns:
        True if LM Studio API is reachable at localhost:1234
    """
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 1234))
        sock.close()
        return result == 0
    except Exception:
        return False


def OpenAIBackend(*args, **kwargs):
    """Get OpenAI backend (lazy loaded)."""
    cls = _get_openai_backend()
    return cls(*args, **kwargs)


def AnthropicBackend(*args, **kwargs):
    """Get Anthropic backend (lazy loaded)."""
    cls = _get_anthropic_backend()
    return cls(*args, **kwargs)


def LMStudioBackend(model: str = "local-model", base_url: str | None = None, **kwargs):
    """Get LM Studio backend (uses OpenAI-compatible API).

    LM Studio exposes an OpenAI-compatible API at localhost:1234.
    No API key required - uses a dummy key internally.

    Args:
        model: Model name loaded in LM Studio (default: "local-model")
        base_url: Override LM Studio URL (default: http://localhost:1234/v1)
        **kwargs: Additional backend configuration

    Returns:
        OpenAI backend configured for LM Studio
    """
    cls = _get_openai_backend()
    return cls(
        model=model,
        base_url=base_url or LM_STUDIO_DEFAULT_URL,
        api_key="lm-studio",  # LM Studio ignores API key
        **kwargs,
    )


def detect_backend() -> str:
    """Auto-detect the best available backend based on API keys.

    Priority:
    1. Anthropic (fastest, best for code)
    2. OpenAI (widely used)
    3. LM Studio (if running on localhost:1234)
    4. Ollama (free local fallback)

    Returns:
        Backend name: "anthropic", "openai", "lmstudio", or "ollama"
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Auto-detected: Anthropic API key found")
        return "anthropic"

    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Auto-detected: OpenAI API key found")
        return "openai"

    # Check for LM Studio before Ollama (LM Studio is more common on Mac)
    if _check_lmstudio_running():
        logger.info("Auto-detected: LM Studio running on localhost:1234")
        return "lmstudio"

    logger.info("Auto-detected: No API keys found, using Ollama (local)")
    return "ollama"


def get_backend(kind: str, **kwargs) -> LLMBackend:
    """Factory function to get an LLM backend instance.

    Args:
        kind: Backend type ("auto", "ollama", "openai", "anthropic", "lmstudio")
              "auto" will detect based on available API keys/services
        **kwargs: Backend-specific configuration

    Returns:
        LLMBackend instance

    Raises:
        ValueError: If backend type is unknown
        ImportError: If required package not installed
    """
    # Handle auto-detection
    if kind == "auto":
        kind = detect_backend()
        logger.info(f"Auto-selected backend: {kind}")

    # Backend registry with lazy loading
    if kind == "ollama":
        return OllamaBackend(**kwargs)
    elif kind == "openai":
        cls = _get_openai_backend()
        return cls(**kwargs)
    elif kind == "anthropic":
        cls = _get_anthropic_backend()
        return cls(**kwargs)
    elif kind == "lmstudio":
        return LMStudioBackend(**kwargs)
    else:
        available = "auto, ollama, openai, anthropic, lmstudio"
        raise ValueError(f"Unknown LLM backend: {kind}. Available: {available}")
