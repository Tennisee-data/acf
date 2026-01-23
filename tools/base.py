"""Base tool interface for agent operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    """Result of a tool execution."""

    status: ToolStatus
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def __bool__(self) -> bool:
        return self.success


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools are deterministic operations that agents can invoke.
    Unlike LLM calls, tools have predictable behavior.
    """

    name: str = "base_tool"
    description: str = "Base tool interface"

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool operation.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with status and output
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
