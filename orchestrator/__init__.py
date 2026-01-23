"""Orchestrator module for Coding Factory.

State machine-based pipeline orchestration with:
- Explicit stage transitions
- Human approval checkpoints
- State persistence and recovery
- Retry policies
"""

from .state_machine import StateMachine, Transition
from .runner import PipelineRunner
from .checkpoints import CheckpointManager, ApprovalResult

__all__ = [
    "StateMachine",
    "Transition",
    "PipelineRunner",
    "CheckpointManager",
    "ApprovalResult",
]
