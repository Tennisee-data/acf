"""Anthropic backend implementation for Claude models."""

import os
from typing import Any

from .base import LLMBackend


class AnthropicBackend(LLMBackend):
    """Anthropic backend for Claude model inference.

    Requires ANTHROPIC_API_KEY environment variable.
    See: https://docs.anthropic.com/en/api/getting-started
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout: int = 120,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> None:
        """Initialize Anthropic backend.

        Args:
            model: Default model to use (claude-sonnet-4-20250514, claude-3-5-haiku-20241022, etc.)
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            timeout: Request timeout in seconds
            max_tokens: Default max tokens for responses
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            )

        self.model = model
        self.timeout = timeout
        self.default_max_tokens = max_tokens

        # Get API key from param or environment
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        self._client = Anthropic(
            api_key=self._api_key,
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model override (uses instance default if not specified)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            The assistant's response content.

        Raises:
            ConnectionError: If unable to connect to Anthropic
            TimeoutError: If the request times out
            RuntimeError: If Anthropic returns an error
        """
        try:
            from anthropic import APIConnectionError, APITimeoutError, APIError
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        # Anthropic requires system message to be separate
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Combine multiple system messages if present
                if system_message:
                    system_message += "\n\n" + msg["content"]
                else:
                    system_message = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Ensure messages alternate user/assistant (Anthropic requirement)
        # If first message isn't user, we may need to adjust
        if chat_messages and chat_messages[0]["role"] != "user":
            # Prepend a placeholder user message
            chat_messages.insert(0, {"role": "user", "content": "Please assist me."})

        request_kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature,
        }

        if system_message:
            request_kwargs["system"] = system_message

        # Merge additional kwargs
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                request_kwargs[key] = kwargs[key]

        try:
            response = self._client.messages.create(**request_kwargs)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to Anthropic API: {e}") from e
        except APITimeoutError as e:
            raise TimeoutError(f"Anthropic request timed out: {e}") from e
        except APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e

        # Extract response content
        if not response.content:
            raise RuntimeError("Anthropic returned empty response")

        # Claude returns a list of content blocks
        text_content = []
        for block in response.content:
            if hasattr(block, "text"):
                text_content.append(block.text)

        if not text_content:
            raise RuntimeError("Anthropic returned no text content")

        return "\n".join(text_content)

    def list_models(self) -> list[str]:
        """List available Anthropic models.

        Returns:
            List of model identifiers.
        """
        # Anthropic doesn't have a models list endpoint, return known models
        return [
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ]

    def is_available(self) -> bool:
        """Check if Anthropic API is accessible.

        Returns:
            True if API key is valid and API is reachable.
        """
        try:
            # Send a minimal request to verify connectivity
            self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    @staticmethod
    def has_api_key() -> bool:
        """Check if Anthropic API key is configured.

        Returns:
            True if ANTHROPIC_API_KEY is set.
        """
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model!r})"
