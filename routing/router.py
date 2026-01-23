"""Model router for task-based LLM selection.

Routes tasks to appropriate model tiers based on:
1. LLM-based complexity estimation (with memory/learning)
2. Pipeline stage (baseline)
3. Task complexity from workplan (size, category)
4. Retry escalation (upgrade on failure)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.complexity_estimator_agent import ComplexityEstimate, ComplexityEstimatorAgent
    from pipeline.config import RoutingConfig
    from schemas.workplan import SubTask

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier for routing decisions."""

    CHEAP = "cheap"      # Fast, low-cost (7B)
    MEDIUM = "medium"    # Balanced (14B)
    PREMIUM = "premium"  # High-quality (30B+)


# Stage-based routing defaults
STAGE_ROUTING: dict[str, ModelTier] = {
    # Low reasoning stages - use cheap
    "spec": ModelTier.CHEAP,           # Structured parsing
    "context": ModelTier.CHEAP,        # File listing, no reasoning
    "test": ModelTier.CHEAP,           # Just run pytest
    "verify": ModelTier.CHEAP,         # Validation only
    "docs": ModelTier.CHEAP,           # Templated output
    # Medium reasoning stages
    "decomposition": ModelTier.MEDIUM,  # Workplan breakdown
    "fix": ModelTier.MEDIUM,            # Error fixing (escalates on retry)
    # High reasoning stages - use premium
    "design": ModelTier.PREMIUM,        # Architecture decisions
    "implementation": ModelTier.PREMIUM,  # Code generation
    "code_review": ModelTier.PREMIUM,   # Judgment required
}

# Task size to minimum tier mapping
SIZE_ROUTING: dict[str, ModelTier] = {
    "xs": ModelTier.CHEAP,    # < 10 lines
    "s": ModelTier.CHEAP,     # 10-50 lines
    "m": ModelTier.MEDIUM,    # 50-200 lines
    "l": ModelTier.PREMIUM,   # 200-500 lines
    "xl": ModelTier.PREMIUM,  # > 500 lines
}

# Tier ordering for upgrades
TIER_ORDER = [ModelTier.CHEAP, ModelTier.MEDIUM, ModelTier.PREMIUM]


class ModelRouter:
    """Routes tasks to appropriate models based on complexity.

    Uses a combination of:
    - LLM-based complexity estimation (with memory/learning)
    - Pipeline stage (baseline routing)
    - Task data from workplan (size, category overrides)
    - Retry escalation (upgrade on failure)

    Example:
        router = ModelRouter(config)
        router.set_complexity_estimate(estimate)  # From ComplexityEstimatorAgent
        tier = router.route("design", task=current_task)
        model = router.get_model(tier)
    """

    def __init__(self, config: RoutingConfig):
        """Initialize router with configuration.

        Args:
            config: Routing configuration with model names and overrides
        """
        self.config = config
        self._premium_domains = set(config.premium_domains)
        self._complexity_estimate: ComplexityEstimate | None = None
        self._feature_description: str | None = None

    def set_complexity_estimate(self, estimate: ComplexityEstimate) -> None:
        """Set complexity estimate from ComplexityEstimatorAgent.

        Args:
            estimate: The complexity estimate from triage
        """
        self._complexity_estimate = estimate
        logger.info(
            "Router using complexity estimate: size=%s, tier=%s, from_memory=%s",
            estimate.size, estimate.recommended_tier, estimate.from_memory
        )

    def set_feature_description(self, description: str) -> None:
        """Set feature description (fallback if no estimate).

        Args:
            description: The feature description/prompt
        """
        self._feature_description = description

    def _get_complexity_tier(self) -> ModelTier:
        """Get complexity tier from estimate or default.

        Returns:
            Model tier based on complexity estimate
        """
        if self._complexity_estimate:
            # Map recommended_tier string to ModelTier
            tier_map = {
                "cheap": ModelTier.CHEAP,
                "medium": ModelTier.MEDIUM,
                "premium": ModelTier.PREMIUM,
            }
            return tier_map.get(
                self._complexity_estimate.recommended_tier,
                ModelTier.MEDIUM
            )

        # Fallback to medium if no estimate
        return ModelTier.MEDIUM

    def get_memory_key(self) -> str | None:
        """Get memory key for recording outcome.

        Returns:
            Memory key if estimate came from triage, None otherwise
        """
        if self._complexity_estimate:
            return self._complexity_estimate.memory_key
        return None

    def route(
        self,
        stage: str,
        task: SubTask | None = None,
        retry_count: int = 0,
    ) -> ModelTier:
        """Determine which model tier to use for a task.

        Routing logic:
        1. Start with stage baseline
        2. Get complexity from LLM-based estimate (with memory)
        3. Use MINIMUM of stage baseline and complexity for simple tasks
        4. Task-based adjustments (if workplan available)
        5. Override to premium for critical domains
        6. Escalate on retry

        Args:
            stage: Pipeline stage name
            task: Optional SubTask from workplan
            retry_count: Number of retries (0 = first attempt)

        Returns:
            Model tier to use (cheap, medium, or premium)
        """
        # 1. Start with stage baseline
        stage_tier = STAGE_ROUTING.get(stage, ModelTier.MEDIUM)

        # Check for stage overrides in config
        if stage in self.config.stage_overrides:
            override = self.config.stage_overrides[stage]
            stage_tier = ModelTier(override)
            logger.debug("Stage %s override to %s", stage, stage_tier.value)

        # 2. Get complexity from LLM-based estimate
        complexity_tier = self._get_complexity_tier()

        # 3. For simple tasks, use the MINIMUM (cheaper model)
        #    For complex tasks, use the MAXIMUM (better model)
        if complexity_tier == ModelTier.CHEAP:
            # Simple task - use cheaper model even for premium stages
            tier = self._min_tier(stage_tier, ModelTier.MEDIUM)
            logger.debug("Simple task (from triage) → capping at medium")
        elif complexity_tier == ModelTier.PREMIUM:
            # Complex task - use stage baseline or higher
            tier = stage_tier
        else:
            # Medium complexity - use minimum of stage and medium
            tier = self._min_tier(stage_tier, ModelTier.MEDIUM)

        # 4. Task-based adjustments (if workplan available)
        if task:
            # Upgrade based on task size
            size_tier = SIZE_ROUTING.get(task.size.value, ModelTier.MEDIUM)
            tier = self._max_tier(tier, size_tier)

            # Premium domain override
            if task.category.value in self._premium_domains:
                tier = ModelTier.PREMIUM
                logger.debug(
                    "Task %s category %s → premium domain",
                    task.id,
                    task.category.value,
                )

        # 5. Escalate on retry
        if retry_count > 0:
            original_tier = tier
            tier = self._upgrade_tier(tier, retry_count)
            if tier != original_tier:
                logger.info(
                    "Retry %d: escalating from %s to %s",
                    retry_count,
                    original_tier.value,
                    tier.value,
                )

        logger.debug(
            "Route: stage=%s, complexity=%s, task=%s, retry=%d → %s",
            stage,
            complexity_tier.value,
            task.id if task else None,
            retry_count,
            tier.value,
        )

        return tier

    def get_model(self, tier: ModelTier) -> str:
        """Get actual model name for a tier.

        Args:
            tier: Model tier

        Returns:
            Model name string for the LLM backend
        """
        return {
            ModelTier.CHEAP: self.config.model_cheap,
            ModelTier.MEDIUM: self.config.model_medium,
            ModelTier.PREMIUM: self.config.model_premium,
        }[tier]

    def get_model_for_stage(
        self,
        stage: str,
        task: SubTask | None = None,
        retry_count: int = 0,
    ) -> str:
        """Convenience method: route and get model name in one call.

        Args:
            stage: Pipeline stage name
            task: Optional SubTask from workplan
            retry_count: Number of retries

        Returns:
            Model name string
        """
        tier = self.route(stage, task, retry_count)
        return self.get_model(tier)

    def _max_tier(self, tier1: ModelTier, tier2: ModelTier) -> ModelTier:
        """Return the higher of two tiers.

        Args:
            tier1: First tier
            tier2: Second tier

        Returns:
            The higher tier (premium > medium > cheap)
        """
        idx1 = TIER_ORDER.index(tier1)
        idx2 = TIER_ORDER.index(tier2)
        return TIER_ORDER[max(idx1, idx2)]

    def _min_tier(self, tier1: ModelTier, tier2: ModelTier) -> ModelTier:
        """Return the lower of two tiers.

        Args:
            tier1: First tier
            tier2: Second tier

        Returns:
            The lower tier (cheap < medium < premium)
        """
        idx1 = TIER_ORDER.index(tier1)
        idx2 = TIER_ORDER.index(tier2)
        return TIER_ORDER[min(idx1, idx2)]

    def _upgrade_tier(self, tier: ModelTier, retry_count: int) -> ModelTier:
        """Upgrade tier based on retry count.

        First retry: cheap → medium
        Second retry+: anything → premium

        Args:
            tier: Current tier
            retry_count: Number of retries

        Returns:
            Upgraded tier (capped at premium)
        """
        idx = TIER_ORDER.index(tier)
        # Upgrade by retry count, capped at premium
        new_idx = min(idx + retry_count, len(TIER_ORDER) - 1)
        return TIER_ORDER[new_idx]

    def explain_routing(
        self,
        stage: str,
        task: SubTask | None = None,
        retry_count: int = 0,
    ) -> dict:
        """Explain routing decision for debugging/CLI.

        Args:
            stage: Pipeline stage
            task: Optional task
            retry_count: Retry count

        Returns:
            Dict with routing explanation
        """
        # Stage baseline
        stage_tier = STAGE_ROUTING.get(stage, ModelTier.MEDIUM)
        reasons = [f"Stage '{stage}' baseline: {stage_tier.value}"]

        final_tier = stage_tier

        # Stage override
        if stage in self.config.stage_overrides:
            override = ModelTier(self.config.stage_overrides[stage])
            reasons.append(f"Config override: {override.value}")
            final_tier = override

        # Task-based
        if task:
            size_tier = SIZE_ROUTING.get(task.size.value, ModelTier.MEDIUM)
            reasons.append(f"Task size '{task.size.value}': {size_tier.value}")
            final_tier = self._max_tier(final_tier, size_tier)

            if task.category.value in self._premium_domains:
                reasons.append(f"Premium domain '{task.category.value}' → premium")
                final_tier = ModelTier.PREMIUM

        # Retry escalation
        if retry_count > 0:
            original = final_tier
            final_tier = self._upgrade_tier(final_tier, retry_count)
            if final_tier != original:
                reasons.append(f"Retry {retry_count}: {original.value} → {final_tier.value}")

        return {
            "stage": stage,
            "task_id": task.id if task else None,
            "task_size": task.size.value if task else None,
            "task_category": task.category.value if task else None,
            "retry_count": retry_count,
            "final_tier": final_tier.value,
            "model": self.get_model(final_tier),
            "reasons": reasons,
        }
