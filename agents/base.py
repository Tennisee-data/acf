"""Base agent class for all pipeline agents."""

from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import TypeAlias

    Message: TypeAlias = dict[str, str]


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM backends.

    Any LLM client implementing this protocol can be used with agents.
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Assistant response content as string.
        """
        ...

    async def achat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Async version of chat.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Assistant response content as string.
        """
        ...


@dataclass
class UsageMetadata:
    """Token usage and cost metadata from LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None

    def __add__(self, other: UsageMetadata) -> UsageMetadata:
        """Combine usage from multiple calls."""
        return UsageMetadata(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=(self.cost_usd or 0) + (other.cost_usd or 0) if self.cost_usd or other.cost_usd else None,
            model=self.model or other.model,
        )


@dataclass
class AgentInput:
    """Standard input format for agents.

    Attributes:
        context: Current pipeline context (files, spec, config, etc.)
        previous_outputs: Outputs from prior agents, keyed by agent name
        history: Optional conversation history for multi-turn reasoning
    """

    context: dict[str, Any]
    previous_outputs: dict[str, AgentOutput] | None = None
    history: list[dict[str, str]] | None = None

    def get_output(self, agent_name: str) -> AgentOutput | None:
        """Get output from a specific previous agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentOutput or None if not found
        """
        if self.previous_outputs is None:
            return None
        return self.previous_outputs.get(agent_name)

    def safe_context(self) -> dict[str, Any]:
        """Return a shallow copy of context to prevent mutation."""
        return copy.copy(self.context)


@dataclass
class AgentOutput:
    """Standard output format for agents.

    Attributes:
        success: Whether the agent completed successfully
        data: Output data dictionary
        errors: List of error messages if any
        artifacts: List of artifact names produced
        metadata: Execution metadata (timing, tokens, etc.)
        agent_name: Name of the agent that produced this output
    """

    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        artifacts_str = f", artifacts={self.artifacts}" if self.artifacts else ""
        errors_str = f", errors={len(self.errors or [])}" if self.errors else ""
        return f"AgentOutput({status}, data_keys={list(self.data.keys())}{artifacts_str}{errors_str})"


class BaseAgent(ABC):
    """Abstract base class for pipeline agents.

    Each agent follows the pattern:
    - Name and description for identification
    - System prompt (defines role and behavior)
    - Inputs (structured data from previous stages)
    - Deterministic tools (shell, file ops, etc.)
    - Structured output (JSON schema)

    Example:
        class MyAgent(BaseAgent):
            def default_system_prompt(self) -> str:
                return "You are a helpful assistant."

            def run(self, input_data: AgentInput) -> AgentOutput:
                response = self._chat("Hello")
                return AgentOutput(success=True, data={"response": response})

        agent = MyAgent(llm=my_llm, name="my-agent")
        result = agent.run(AgentInput(context={}))
    """

    def __init__(
        self,
        llm: LLMProtocol,
        name: str | None = None,
        system_prompt: str | None = None,
        description: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            llm: LLM backend for inference (must implement LLMProtocol)
            name: Agent identifier (defaults to class name)
            system_prompt: Override default system prompt
            description: Human-readable description of agent's purpose
            logger: Optional logger instance
        """
        self.llm = llm
        self.name = name or self.__class__.__name__
        self.description = description or ""
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.logger = logger or logging.getLogger(f"agent.{self.name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @abstractmethod
    def default_system_prompt(self) -> str:
        """Return the default system prompt for this agent.

        Returns:
            System prompt string.
        """
        ...

    @abstractmethod
    def run(self, input_data: AgentInput) -> AgentOutput:
        """Execute the agent's task.

        Args:
            input_data: Structured input for the agent

        Returns:
            AgentOutput with results and any artifacts.
        """
        ...

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        """Async version of run.

        Default implementation calls sync run(). Override for true async.

        Args:
            input_data: Structured input for the agent

        Returns:
            AgentOutput with results and any artifacts.
        """
        return self.run(input_data)

    def _build_messages(
        self,
        user_content: str | list[dict[str, str]],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build message list for LLM call.

        Args:
            user_content: User message string or list of messages
            history: Optional conversation history

        Returns:
            Complete message list with system prompt
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if history:
            messages.extend(history)

        if isinstance(user_content, str):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.extend(user_content)

        return messages

    def _chat(
        self,
        user_content: str | list[dict[str, str]],
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat message with the system prompt.

        Args:
            user_content: User message string or list of messages
            history: Optional conversation history
            **kwargs: Additional LLM parameters

        Returns:
            Assistant response content.
        """
        messages = self._build_messages(user_content, history)
        self.logger.debug("Sending %d messages to LLM", len(messages))
        return self.llm.chat(messages, **kwargs)

    async def _achat(
        self,
        user_content: str | list[dict[str, str]],
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _chat.

        Args:
            user_content: User message string or list of messages
            history: Optional conversation history
            **kwargs: Additional LLM parameters

        Returns:
            Assistant response content.
        """
        messages = self._build_messages(user_content, history)
        self.logger.debug("Sending %d messages to LLM (async)", len(messages))
        return await self.llm.achat(messages, **kwargs)

    def _create_output(
        self,
        success: bool,
        data: dict[str, Any],
        errors: list[str] | None = None,
        artifacts: list[str] | None = None,
        start_time: float | None = None,
        usage: UsageMetadata | None = None,
    ) -> AgentOutput:
        """Helper to create AgentOutput with metadata.

        Args:
            success: Whether agent succeeded
            data: Output data
            errors: Error messages if any
            artifacts: Artifact names produced
            start_time: Start timestamp for duration calculation
            usage: Token usage metadata

        Returns:
            AgentOutput with agent_name and metadata populated
        """
        metadata: dict[str, Any] = {}

        if start_time is not None:
            metadata["duration_sec"] = round(time.time() - start_time, 3)

        if usage is not None:
            metadata["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage.cost_usd is not None:
                metadata["cost_usd"] = usage.cost_usd
            if usage.model:
                metadata["model"] = usage.model

        return AgentOutput(
            success=success,
            data=data,
            errors=errors,
            artifacts=artifacts,
            metadata=metadata,
            agent_name=self.name,
        )

    def _log_run_start(self, input_data: AgentInput) -> float:
        """Log run start and return start time.

        Args:
            input_data: Agent input

        Returns:
            Start timestamp
        """
        context_keys = list(input_data.context.keys())
        prev_agents = list((input_data.previous_outputs or {}).keys())
        self.logger.info(
            "Starting %s (context: %s, prev: %s)",
            self.name,
            context_keys,
            prev_agents,
        )
        return time.time()

    def _log_run_end(self, output: AgentOutput, start_time: float) -> None:
        """Log run completion.

        Args:
            output: Agent output
            start_time: Start timestamp
        """
        duration = time.time() - start_time
        if output.success:
            self.logger.info(
                "Completed %s in %.2fs (artifacts: %s)",
                self.name,
                duration,
                output.artifacts or [],
            )
        else:
            self.logger.warning(
                "Failed %s in %.2fs: %s",
                self.name,
                duration,
                output.errors,
            )
