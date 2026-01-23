"""Ollama backend implementation for local LLM inference."""

from typing import Any

import requests

from .base import LLMBackend


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM execution.

    Connects to a local Ollama server for inference.
    See: https://ollama.ai/
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
    ) -> None:
        """Initialize Ollama backend.

        Args:
            model: Default model to use for requests
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model override (uses instance default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens (mapped to num_predict in Ollama)
            **kwargs: Additional Ollama-specific options

        Returns:
            The assistant's response content.

        Raises:
            ConnectionError: If unable to connect to Ollama
            TimeoutError: If the request times out
            RuntimeError: If Ollama returns an error
        """
        url = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # Merge any additional kwargs into options
        payload["options"].update(kwargs.get("options", {}))

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama returned an error: {e}") from e

        data = response.json()

        # Ollama returns {"message": {"role": "assistant", "content": "..."}}
        if "message" not in data or "content" not in data["message"]:
            raise RuntimeError(f"Unexpected Ollama response format: {data}")

        return data["message"]["content"]

    def list_models(self) -> list[str]:
        """List models available in Ollama.

        Returns:
            List of model names.
        """
        url = f"{self.base_url}/api/tags"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to list Ollama models: {e}") from e

        data = response.json()
        models = data.get("models", [])

        return [m["name"] for m in models]

    def is_available(self) -> bool:
        """Check if Ollama server is running and responsive.

        Returns:
            True if Ollama is available.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def __repr__(self) -> str:
        return f"OllamaBackend(model={self.model!r}, base_url={self.base_url!r})"
