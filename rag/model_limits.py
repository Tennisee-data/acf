"""Model context window registry for RAG budget management.

Maps model names to their context window sizes in tokens.
Used to calculate available budget for RAG content.
"""

from __future__ import annotations

from dataclasses import dataclass


# Context window sizes in tokens for known models
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Ollama / Local models
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3.2:3b": 131072,
    "qwen3:7b": 32768,
    "qwen3:14b": 32768,
    "qwen3:30b": 32768,
    "qwen3:72b": 32768,
    "qwen2.5:7b": 131072,
    "qwen2.5:14b": 131072,
    "qwen2.5:32b": 131072,
    "qwen2.5:72b": 131072,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "codellama:34b": 16384,
    "deepseek-coder:6.7b": 16384,
    "deepseek-coder:33b": 16384,
    "mistral:7b": 32768,
    "mixtral:8x7b": 32768,
    "phi3:mini": 128000,
    "phi3:medium": 128000,
    "gemma2:9b": 8192,
    "gemma2:27b": 8192,

    # Anthropic Claude
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,

    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "o1-preview": 128000,
    "o1-mini": 128000,

    # Google
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-pro": 32768,
}

# Default context window for unknown models (conservative)
DEFAULT_CONTEXT_WINDOW = 8192


@dataclass
class ModelBudget:
    """Token budget breakdown for a model."""

    model: str
    context_window: int
    system_prompt_tokens: int
    user_prompt_tokens: int
    expected_output_tokens: int
    safety_margin_tokens: int

    @property
    def available_for_rag(self) -> int:
        """Tokens available for RAG content."""
        used = (
            self.system_prompt_tokens
            + self.user_prompt_tokens
            + self.expected_output_tokens
            + self.safety_margin_tokens
        )
        return max(0, self.context_window - used)

    @property
    def utilization_percent(self) -> float:
        """Percentage of context used by non-RAG content."""
        used = (
            self.system_prompt_tokens
            + self.user_prompt_tokens
            + self.expected_output_tokens
            + self.safety_margin_tokens
        )
        return (used / self.context_window) * 100 if self.context_window > 0 else 100


def get_context_window(model: str) -> int:
    """Get context window size for a model.

    Args:
        model: Model name (e.g., "qwen3:14b", "claude-3-sonnet")

    Returns:
        Context window size in tokens
    """
    # Try exact match first
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Try without version suffix (e.g., "qwen3:14b-q4" -> "qwen3:14b")
    base_model = model.split("-")[0] if "-" in model else model
    if base_model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[base_model]

    # Try model family prefix matching
    for known_model, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(known_model.split(":")[0]):
            return window

    return DEFAULT_CONTEXT_WINDOW


def calculate_budget(
    model: str,
    system_prompt: str = "",
    user_prompt: str = "",
    expected_output_tokens: int = 2000,
    safety_margin: int = 500,
) -> ModelBudget:
    """Calculate token budget for a model.

    Args:
        model: Model name
        system_prompt: System prompt text
        user_prompt: User prompt text
        expected_output_tokens: Reserved tokens for response
        safety_margin: Buffer tokens for safety

    Returns:
        ModelBudget with breakdown
    """
    from .token_counter import count_tokens

    context_window = get_context_window(model)

    return ModelBudget(
        model=model,
        context_window=context_window,
        system_prompt_tokens=count_tokens(system_prompt),
        user_prompt_tokens=count_tokens(user_prompt),
        expected_output_tokens=expected_output_tokens,
        safety_margin_tokens=safety_margin,
    )
