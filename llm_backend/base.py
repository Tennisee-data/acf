"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Abstract interface for LLM providers.

    All LLM backends must implement this interface to ensure
    consistent behavior across different providers.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles: "system", "user", "assistant"
            model: Optional model override (uses default if not specified)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            The assistant's response content as a string.

        Raises:
            ConnectionError: If unable to connect to the backend
            TimeoutError: If the request times out
            RuntimeError: If the backend returns an error
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names/identifiers.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and responding.

        Returns:
            True if backend is reachable and ready.
        """
        ...
