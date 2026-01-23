"""Agents module for Coding Factory.

Provides specialized agents for each pipeline stage:
- SpecAgent: Parse and normalize feature specifications
- DecompositionAgent: Break features into sub-tasks
- ContextAgent: Analyze repository and gather context
- DesignAgent: Propose architecture and design
- APIContractAgent: Define API boundaries before implementation
- ImplementationAgent: Generate code changes
- FixAgent: Fix code errors in an iterative loop
- TestAgent: Run tests and quality checks
- CoverageAgent: Enforce test coverage thresholds
- SecretsScanAgent: Detect and fix hardcoded secrets
- DependencyAuditAgent: Scan for CVEs and outdated packages
- ObservabilityAgent: Inject logging, metrics, and tracing scaffolding
- DocAgent: Generate and sync documentation
- CodeReviewAgent: Senior engineer code review
- PRPackagingAgent: Build rich PR packages with spec-tied changes
- RollbackStrategyAgent: Generate CI/CD rollback and canary deployment strategies
- PolicyAgent: Enforce policy rules and decide approvals
- DockerAgent: Validate code in Docker container
- VerifyAgent: Final verification and summary
- DocumentationRequirementsAgent: Detect when external API docs are needed
- SafetyPatternsAgent: Inject domain-specific implementation invariants

Premium agents available on ACF Marketplace:
- ConfigAgent: Enforce 12-factor config layout ($12)
- InvariantsVerifierAgent: Post-generation code verification ($5)
- RAGOptimizerAgent: Optimize RAG retrieval ($5)
"""

from .api_contract_agent import APIContractAgent
from .doc_requirements_agent import (
    DocumentationRequirementsAgent,
    DocumentationReport,
    DocumentationRequirement,
)
from .safety_patterns_agent import SafetyPatternsAgent, SafetyPattern, SafetyReport
from .tailwind_css_agent import TailwindCSSAgent, TailwindPattern, TailwindReport
from .base import AgentInput, AgentOutput, BaseAgent
from .code_review_agent import CodeReviewAgent
from .context_agent import ContextAgent
from .coverage_agent import CoverageAgent
from .decomposition_agent import DecompositionAgent
from .dependency_audit_agent import DependencyAuditAgent
from .design_agent import DesignAgent
from .doc_agent import DocAgent
from .docker_agent import DockerAgent
from .fix_agent import FixAgent, FixLoopState
from .implementation_agent import ImplementationAgent
from .observability_agent import ObservabilityAgent
from .policy_agent import PolicyAgent
from .pr_packaging_agent import PRPackagingAgent
from .rollback_strategy_agent import RollbackStrategyAgent
from .secrets_scan_agent import SecretsScanAgent
from .spec_agent import SpecAgent
from .test_agent import TestAgent
from .verify_agent import VerifyAgent

__all__ = [
    "BaseAgent",
    "AgentInput",
    "AgentOutput",
    "SpecAgent",
    "DecompositionAgent",
    "ContextAgent",
    "DesignAgent",
    "APIContractAgent",
    "ImplementationAgent",
    "FixAgent",
    "FixLoopState",
    "TestAgent",
    "CoverageAgent",
    "SecretsScanAgent",
    "DependencyAuditAgent",
    "ObservabilityAgent",
    "DocAgent",
    "CodeReviewAgent",
    "PRPackagingAgent",
    "RollbackStrategyAgent",
    "PolicyAgent",
    "DockerAgent",
    "VerifyAgent",
    "DocumentationRequirementsAgent",
    "DocumentationReport",
    "DocumentationRequirement",
    "SafetyPatternsAgent",
    "SafetyPattern",
    "SafetyReport",
    "TailwindCSSAgent",
    "TailwindPattern",
    "TailwindReport",
]
