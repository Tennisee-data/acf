"""Multi-model routing for cost/quality optimization.

Routes LLM calls to different models based on task complexity,
using workplan data from DecompositionAgent.
"""

from .router import ModelRouter, ModelTier

__all__ = ["ModelRouter", "ModelTier"]
