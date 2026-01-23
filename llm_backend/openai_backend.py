"""OpenAI backend implementation for GPT models."""

import os
from typing import Any

from .base import LLMBackend


class OpenAIBackend(LLMBackend):
    """OpenAI backend for GPT model inference.

    Requires OPENAI_API_KEY environment variable.
    See: https://platform.openai.com/docs/api-reference
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            model: Default model to use (gpt-4o, gpt-4o-mini, etc.)
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL override (for Azure or proxies)
            timeout: Request timeout in seconds
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )

        self.model = model
        self.timeout = timeout

        # Get API key from param or environment
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model override (uses instance default if not specified)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            The assistant's response content.

        Raises:
            ConnectionError: If unable to connect to OpenAI
            TimeoutError: If the request times out
            RuntimeError: If OpenAI returns an error
        """
        try:
            from openai import APIConnectionError, APITimeoutError, APIError
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        request_kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        # Merge additional kwargs (e.g., top_p, presence_penalty)
        for key in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if key in kwargs:
                request_kwargs[key] = kwargs[key]

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to OpenAI API: {e}") from e
        except APITimeoutError as e:
            raise TimeoutError(f"OpenAI request timed out: {e}") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

        # Extract response content
        if not response.choices:
            raise RuntimeError("OpenAI returned empty response")

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned null content")

        return content

    def list_models(self) -> list[str]:
        """List available OpenAI models.

        Returns:
            List of model IDs.
        """
        try:
            models = self._client.models.list()
            # Filter to chat models only
            chat_models = [
                m.id for m in models.data
                if "gpt" in m.id.lower() or "o1" in m.id.lower()
            ]
            return sorted(chat_models)
        except Exception as e:
            raise ConnectionError(f"Failed to list OpenAI models: {e}") from e

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible.

        Returns:
            True if API key is valid and API is reachable.
        """
        try:
            # Simple models list call to verify connectivity
            self._client.models.list()
            return True
        except Exception:
            return False

    @staticmethod
    def has_api_key() -> bool:
        """Check if OpenAI API key is configured.

        Returns:
            True if OPENAI_API_KEY is set.
        """
        return bool(os.environ.get("OPENAI_API_KEY"))

    def __repr__(self) -> str:
        return f"OpenAIBackend(model={self.model!r})"
