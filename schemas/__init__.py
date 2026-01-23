"""Schemas module for structured agent I/O.

Provides Pydantic models for:
- Feature specifications
- Workplans (decomposed tasks)
- Context reports
- Design proposals
- API contracts
- Implementation artifacts
- Test reports
- Coverage reports
- Secrets reports
- Dependency audit reports
- Observability configuration
- Config layout reports
- Documentation reports
- Code review reports
- Verification results
- PR packages
- Rollback strategies
- Policy rules and decisions
"""

from .api_contract import (
    APIContract,
    DataType,
    Endpoint,
    HTTPMethod,
    Parameter,
    ParameterLocation,
    PydanticModel,
    Response,
    SchemaField,
    ValidationIssue,
)
from .code_review_report import (
    AutoFix as ReviewAutoFix,
)
from .code_review_report import (
    CodeReviewReport,
    FileReview,
    IssueCategory,
    IssueSeverity,
    ReviewIssue,
    ShipStatus,
    StyleGuide,
)
from .config_layout import (
    ConfigCategory,
    ConfigLayoutReport,
    ConfigType,
    ConfigVariable,
    Environment,
    EnvironmentOverride,
    FeatureFlag,
    GeneratedConfigFile,
    SecretVariable,
)
from .context_report import (
    CodeSnippet,
    ContextReport,
    FileContext,
    RepoStructure,
)
from .coverage_report import (
    CoverageReport,
    CoverageStatus,
    CoverageThreshold,
    DiffCoverage,
    FileCoverage,
    TestGenerationRequest,
    UncoveredBlock,
)
from .dependency_audit_report import (
    AuditStatus,
    DependencyAuditReport,
    DependencySource,
    DeprecatedPackage,
    OutdatedPackage,
    VersionBump,
    Vulnerability,
    VulnerabilitySeverity,
)
from .design_proposal import (
    DesignOption,
    DesignProposal,
    FileChange,
)
from .doc_report import (
    ADRRecord,
    DocReport,
    DocstringStyle,
    DocstringUpdate,
    DocType,
    GeneratedDoc,
    MissingDocstring,
    SyncCheck,
    SyncStatus,
)
from .feature_spec import (
    AcceptanceCriteria,
    Constraint,
    FeatureSpec,
    NonFunctionalRequirement,
)
from .implementation import (
    ChangeSet,
    FileDiff,
    ImplementationNotes,
)
from .observability_config import (
    GrafanaDashboard,
    LoggingConfig,
    MetricsBackend,
    MetricsConfig,
    ObservabilityReport,
    TracingBackend,
    TracingConfig,
)
from .pipeline_state import (
    PipelineState,
    RunStatus,
    StageResult,
)
from .policy import (
    CodeReviewResults,
    DiffStats,
    MatchedRule,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    PolicyRuleSet,
    PolicySeverity,
    RuleCondition,
    SecurityResults,
    TestResults,
)
from .requirements import (
    Requirement,
    RequirementsTracker,
    RequirementStatus,
    RequirementType,
)
from .pr_package import (
    AcceptanceCriteriaItem,
    ChangelogEntry,
    ChangeType,
    PRLabel,
    PRPackage,
    ReportLink,
    SuggestedReviewer,
)
from .rollback_strategy import (
    BlueGreenConfig,
    CanaryConfig,
    DeploymentPattern,
    GeneratedWorkflow,
    HealthCheck,
    PlaybookSection,
    RollbackAction,
    RollbackJob,
    RollbackPlaybook,
    RollbackStep,
    RollbackStrategyReport,
    RollbackTrigger,
    RollingConfig,
)
from .secrets_report import (
    AutoFix,
    DetectedSecret,
    SecretSeverity,
    SecretsReport,
    SecretStatus,
    SecretType,
)
from .test_report import (
    CoverageInfo,
    TestCase,
    TestReport,
    TestResult,
)
from .verification import (
    CriterionCheck,
    HealthCheck,
    VerificationReport,
)
from .workplan import (
    SubTask,
    TaskCategory,
    TaskPriority,
    TaskSize,
    WorkPlan,
)

__all__ = [
    # Feature spec
    "FeatureSpec",
    "AcceptanceCriteria",
    "Constraint",
    "NonFunctionalRequirement",
    # Workplan
    "WorkPlan",
    "SubTask",
    "TaskCategory",
    "TaskPriority",
    "TaskSize",
    # Context
    "ContextReport",
    "FileContext",
    "CodeSnippet",
    "RepoStructure",
    # Design
    "DesignProposal",
    "DesignOption",
    "FileChange",
    # API Contract
    "APIContract",
    "Endpoint",
    "HTTPMethod",
    "Parameter",
    "ParameterLocation",
    "PydanticModel",
    "Response",
    "SchemaField",
    "DataType",
    "ValidationIssue",
    # Implementation
    "ChangeSet",
    "FileDiff",
    "ImplementationNotes",
    # Test
    "TestReport",
    "TestResult",
    "TestCase",
    "CoverageInfo",
    # Coverage
    "CoverageReport",
    "CoverageStatus",
    "CoverageThreshold",
    "DiffCoverage",
    "FileCoverage",
    "TestGenerationRequest",
    "UncoveredBlock",
    # Secrets
    "SecretsReport",
    "DetectedSecret",
    "AutoFix",
    "SecretType",
    "SecretSeverity",
    "SecretStatus",
    # Dependency Audit
    "DependencyAuditReport",
    "Vulnerability",
    "VulnerabilitySeverity",
    "DeprecatedPackage",
    "OutdatedPackage",
    "VersionBump",
    "DependencySource",
    "AuditStatus",
    # Observability
    "ObservabilityReport",
    "LoggingConfig",
    "MetricsConfig",
    "MetricsBackend",
    "TracingConfig",
    "TracingBackend",
    "GrafanaDashboard",
    # Config Layout
    "ConfigLayoutReport",
    "ConfigVariable",
    "ConfigType",
    "ConfigCategory",
    "Environment",
    "EnvironmentOverride",
    "FeatureFlag",
    "SecretVariable",
    "GeneratedConfigFile",
    # Documentation
    "DocReport",
    "DocType",
    "DocstringStyle",
    "DocstringUpdate",
    "GeneratedDoc",
    "ADRRecord",
    "SyncCheck",
    "SyncStatus",
    "MissingDocstring",
    # Code Review
    "CodeReviewReport",
    "ReviewIssue",
    "ReviewAutoFix",
    "FileReview",
    "ShipStatus",
    "IssueSeverity",
    "IssueCategory",
    "StyleGuide",
    # Verification
    "VerificationReport",
    "CriterionCheck",
    "HealthCheck",
    # Pipeline
    "PipelineState",
    "StageResult",
    "RunStatus",
    # PR Package
    "PRPackage",
    "PRLabel",
    "ChangeType",
    "AcceptanceCriteriaItem",
    "ReportLink",
    "ChangelogEntry",
    "SuggestedReviewer",
    # Rollback Strategy
    "RollbackStrategyReport",
    "DeploymentPattern",
    "RollbackTrigger",
    "RollbackAction",
    "HealthCheck",
    "RollbackStep",
    "RollbackJob",
    "CanaryConfig",
    "BlueGreenConfig",
    "RollingConfig",
    "PlaybookSection",
    "RollbackPlaybook",
    "GeneratedWorkflow",
    # Policy
    "PolicyAction",
    "PolicySeverity",
    "RuleCondition",
    "PolicyRule",
    "PolicyRuleSet",
    "DiffStats",
    "TestResults",
    "SecurityResults",
    "CodeReviewResults",
    "PolicyContext",
    "MatchedRule",
    "PolicyDecision",
    # Requirements Tracker
    "Requirement",
    "RequirementsTracker",
    "RequirementStatus",
    "RequirementType",
]
