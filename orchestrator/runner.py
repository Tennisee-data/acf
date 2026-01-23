"""Pipeline runner for orchestrating agent execution."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import (
    AgentInput,
    APIContractAgent,
    CodeReviewAgent,
    ConfigAgent,
    ContextAgent,
    CoverageAgent,
    DependencyAuditAgent,
    DesignAgent,
    DocAgent,
    DockerAgent,
    FixAgent,
    FixLoopState,
    ImplementationAgent,
    ObservabilityAgent,
    SecretsScanAgent,
    SpecAgent,
    TestAgent,
    VerifyAgent,
)
from agents.complexity_estimator_agent import ComplexityEstimatorAgent
from agents.decomposition_agent import DecompositionAgent
from agents.runtime_decision_agent import RuntimeDecisionAgent, decide_runtime
from agents.scaffold_agent import ProjectScaffoldAgent
from llm_backend import LLMBackend, get_backend
from pipeline.config import Config
from plugins import PluginLoader, PluginRegistry
from plugins.registry import LoadedPlugin
from extensions import ExtensionLoader
from extensions.loader import LoadedExtension
from extensions.manifest import HookPoint
from routing import ModelRouter, ModelTier
from schemas.pipeline_state import PipelineState, RunStatus, Stage, StageStatus
from tools.validator import CodeValidator

from .checkpoints import ApprovalResponse, ApprovalResult, CheckpointManager
from .state_machine import StateMachine

# Type alias for stage handlers
StageHandler = Callable[["PipelineRunner", PipelineState], tuple[bool, str | None]]


class PipelineRunner:
    """Orchestrates the feature pipeline execution.

    Uses a state machine to manage pipeline flow with:
    - Stage-by-stage execution
    - Human approval checkpoints
    - State persistence for resume
    - Retry on failure
    """

    def __init__(
        self,
        config: Config,
        console: Console | None = None,
        auto_approve: bool = False,
        ws_callback: Callable[[str, dict], None] | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize pipeline runner.

        Args:
            config: Application configuration
            console: Rich console for output
            auto_approve: Auto-approve all checkpoints
            ws_callback: Callback for WebSocket updates (run_id, message)
            cancellation_check: Callback that returns True if job should be cancelled
        """
        self.config = config
        self.console = console or Console()
        self.artifacts_dir = Path(config.pipeline.artifacts_dir)
        self.ws_callback = ws_callback
        self.cancellation_check = cancellation_check
        self.checkpoint_manager = CheckpointManager(
            console=self.console,
            auto_approve=auto_approve,
        )

        # Initialize LLM backend
        self.llm: LLMBackend | None = None
        self._init_llm_backend()

        # Initialize agents
        self.spec_agent: SpecAgent | None = None
        self._init_agents()

        # Initialize plugin system
        self.plugin_registry: PluginRegistry = PluginRegistry()
        self._init_plugins()

        # Initialize extension system (for ACF Local Edition marketplace)
        self.extension_loader: ExtensionLoader | None = None
        self._init_extensions()

        # Initialize RAG budget agent AFTER extensions (so RAG extensions can be used)
        if not self._lightweight_mode:
            self._init_rag_budget_agent()

        # Stage handlers registry
        self._handlers: dict[Stage, StageHandler] = {
            Stage.INIT: self._handle_init,
            Stage.SPEC: self._handle_spec,
            Stage.DECOMPOSITION: self._handle_decomposition,
            Stage.CONTEXT: self._handle_context,
            Stage.DESIGN: self._handle_design,
            Stage.DESIGN_APPROVAL: self._handle_design_approval,
            Stage.API_CONTRACT: self._handle_api_contract,
            Stage.IMPLEMENTATION: self._handle_implementation,
            Stage.TESTING: self._handle_testing,
            Stage.COVERAGE: self._handle_coverage,
            Stage.SECRETS_SCAN: self._handle_secrets_scan,
            Stage.DEPENDENCY_AUDIT: self._handle_dependency_audit,
            Stage.DOCKER_BUILD: self._handle_docker_build,
            Stage.ROLLBACK_STRATEGY: self._handle_rollback_strategy,
            Stage.OBSERVABILITY: self._handle_observability,
            Stage.CONFIG: self._handle_config,
            Stage.DOCS: self._handle_docs,
            Stage.CODE_REVIEW: self._handle_code_review,
            Stage.POLICY: self._handle_policy,
            Stage.VERIFICATION: self._handle_verification,
            Stage.PR_PACKAGE: self._handle_pr_package,
            Stage.FINAL_APPROVAL: self._handle_final_approval,
            Stage.DEPLOY: self._handle_deploy,
            Stage.DONE: self._handle_done,
        }

        # Feature flags
        self._decompose_enabled = False  # Set via run() parameter
        self._api_contract_enabled = False  # Set via run() parameter
        self._coverage_enabled = False  # Set via run() parameter
        self._coverage_threshold = 80.0  # Default threshold
        self._secrets_scan_enabled = False  # Set via run() parameter
        self._dependency_audit_enabled = False  # Set via run() parameter
        self._rollback_strategy_enabled = False  # Set via run() parameter
        self._observability_enabled = False  # Set via run() parameter
        self._config_enabled = False  # Set via run() parameter
        self._docs_enabled = False  # Set via run() parameter
        self._code_review_enabled = False  # Set via run() parameter
        self._policy_enabled = False  # Set via run() parameter
        self._policy_rules_path: str | None = None  # Custom rules file
        self._pr_package_enabled = False  # Set via run() parameter

        # Current run ID for WebSocket updates
        self._current_run_id: str | None = None

    def _send_ws_update(self, message_type: str, **kwargs: Any) -> None:
        """Send a WebSocket update if callback is configured.

        Args:
            message_type: Type of message (e.g., 'fix_loop', 'stage', 'error')
            **kwargs: Additional message data
        """
        if self.ws_callback and self._current_run_id:
            message = {"type": message_type, "run_id": self._current_run_id, **kwargs}
            try:
                # Handle both sync and async callbacks
                import asyncio
                if asyncio.iscoroutinefunction(self.ws_callback):
                    asyncio.create_task(self.ws_callback(self._current_run_id, message))
                else:
                    self.ws_callback(self._current_run_id, message)
            except Exception:
                pass  # Don't fail the pipeline if WS update fails

    def _init_llm_backend(self) -> None:
        """Initialize the LLM backends based on configuration."""
        # Initialize model router
        self.router: ModelRouter | None = None
        self.model_pool: dict[ModelTier, LLMBackend] = {}

        # Track detected backend for lightweight mode
        from llm_backend import detect_backend
        if self.config.llm.backend == "auto":
            self._detected_backend = detect_backend()
        else:
            self._detected_backend = self.config.llm.backend

        # Check lightweight mode
        self._lightweight_mode = self.config.llm.is_lightweight_mode(self._detected_backend)
        if self._lightweight_mode:
            self.console.print(f"[cyan]Lightweight mode: ON (using {self._detected_backend})[/cyan]")
            self.console.print("[dim]Skipping RAG and verbose prompts for API model[/dim]")

        # General model for specs, summaries, verification
        try:
            self.llm = get_backend(
                self.config.llm.backend,
                model=self.config.llm.model_general,
                base_url=self.config.llm.base_url,
                timeout=self.config.llm.timeout,
            )
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not initialize LLM backend: {e}[/yellow]")
            self.llm = None

        # Code model for design and implementation (use code-specific model if different)
        if self.config.llm.model_code != self.config.llm.model_general:
            try:
                self.llm_code = get_backend(
                    self.config.llm.backend,
                    model=self.config.llm.model_code,
                    base_url=self.config.llm.base_url,
                    timeout=self.config.llm.timeout,
                )
                self.console.print(f"[dim]Using {self.config.llm.model_code} for code generation[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not initialize code LLM, falling back to general: {e}[/yellow]")
                self.llm_code = self.llm
        else:
            self.llm_code = self.llm

        # Initialize model router and pool if routing is enabled
        if self.config.routing.enabled and self.llm is not None:
            self._init_model_routing()

        # RAG budget agent initialized later in __init__ (after extensions are loaded)
        self.rag_budget_agent = None

    def _init_model_routing(self) -> None:
        """Initialize multi-model routing with model pool."""
        self.router = ModelRouter(self.config.routing)

        # Build model pool - cache backends by model name to avoid duplicates
        model_cache: dict[str, LLMBackend] = {}

        for tier in ModelTier:
            model_name = self.router.get_model(tier)

            # Reuse cached backend if same model
            if model_name in model_cache:
                self.model_pool[tier] = model_cache[model_name]
                continue

            try:
                backend = get_backend(
                    self.config.llm.backend,
                    model=model_name,
                    base_url=self.config.llm.base_url,
                    timeout=self.config.llm.timeout,
                )
                self.model_pool[tier] = backend
                model_cache[model_name] = backend
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Could not initialize {tier.value} model ({model_name}): {e}[/yellow]"
                )
                # Fallback to general model
                if self.llm is not None:
                    self.model_pool[tier] = self.llm
                    model_cache[model_name] = self.llm

        if self.model_pool:
            models = [f"{t.value}={self.router.get_model(t)}" for t in ModelTier]
            self.console.print(f"[dim]Model routing enabled: {', '.join(models)}[/dim]")

    def _init_rag_budget_agent(self) -> None:
        """Initialize RAG budget agent for context-aware retrieval.

        Checks for marketplace RAG extensions first, falls back to built-in retriever.
        """
        try:
            from agents.rag_budget_agent import RAGBudgetAgent

            # Check for extension RAG retrievers
            extension_retriever = None
            extension_name = None
            if getattr(self, "extension_loader", None):
                rag_extensions = self.extension_loader.registry.rag
                if rag_extensions:
                    # Use the first enabled RAG extension
                    for name, ext in rag_extensions.items():
                        if ext.enabled and ext.retriever_class:
                            try:
                                # Instantiate the extension retriever
                                extension_retriever = ext.retriever_class()
                                extension_name = name
                                self.console.print(
                                    f"[dim]Using RAG extension: {name} v{ext.manifest.version}[/dim]"
                                )
                                break
                            except Exception as e:
                                self.console.print(
                                    f"[yellow]Warning: Could not load RAG extension {name}: {e}[/yellow]"
                                )

            # Load invariants and RAG docs
            invariants_dir = Path(__file__).parent.parent / "invariants"
            rag_dir = Path(__file__).parent.parent / "rag"

            self.rag_budget_agent = RAGBudgetAgent(
                invariants_dir=invariants_dir,
                rag_docs_dir=rag_dir,
                use_semantic=True,  # Auto-fallback if sentence-transformers not installed
                custom_retriever=extension_retriever,  # Use extension if available
            )

            source_count = len(self.rag_budget_agent.retriever.sources)
            if extension_name:
                mode = f"extension:{extension_name}"
            else:
                mode = "semantic" if self.rag_budget_agent.is_semantic else "keyword"
            self.console.print(f"[dim]RAG initialized: {source_count} sources, {mode} retrieval[/dim]")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not initialize RAG budget agent: {e}[/yellow]")
            self.rag_budget_agent = None

    def _get_rag_content(
        self,
        model: str,
        stage: str,
        query: str,
        context: str = "",
    ) -> str | None:
        """Get budget-aware RAG content for a stage.

        Args:
            model: Model name to calculate budget for
            stage: Pipeline stage (e.g., "implementation")
            query: Search query for RAG retrieval
            context: Additional context (e.g., feature description)

        Returns:
            RAG content string or None if not available
        """
        if not self.rag_budget_agent:
            return None

        try:
            allocation = self.rag_budget_agent.allocate(
                model=model,
                stage=stage,
                query=query,
                context=context,
            )

            if allocation.rag_report.sources_included_count > 0:
                self.console.print(
                    f"[dim]RAG: {allocation.rag_report.sources_included_count} sources, "
                    f"{allocation.rag_report.tokens_used:,} tokens "
                    f"({allocation.rag_report.utilization_percent:.0f}% of budget)[/dim]"
                )
                return allocation.rag_content

        except Exception as e:
            self.console.print(f"[yellow]RAG retrieval failed: {e}[/yellow]")

        return None

    def _run_complexity_triage(self, feature: str) -> None:
        """Run LLM-based complexity triage for smart routing.

        Uses a cheap/fast model to estimate task complexity.
        Results are cached in memory for future similar tasks.

        Args:
            feature: The feature description
        """
        # Use cheap model for triage (fast)
        triage_llm = self.model_pool.get(ModelTier.CHEAP, self.llm)

        if triage_llm is None:
            return

        estimator = ComplexityEstimatorAgent(llm=triage_llm)

        self.console.print("[dim]Estimating task complexity...[/dim]")
        estimate = estimator.estimate(feature)

        # Display triage result
        source = "memory" if estimate.from_memory else "LLM"
        self.console.print(
            f"[dim]Complexity: {estimate.size} → {estimate.recommended_tier} model "
            f"({source}, confidence={estimate.confidence:.0%})[/dim]"
        )

        # Set estimate on router
        if self.router:
            self.router.set_complexity_estimate(estimate)

        # Store estimator for recording outcome later
        self._complexity_estimator = estimator
        self._triage_memory_key = estimate.memory_key

    def _record_triage_outcome(self, success: bool, run_id: str | None = None) -> None:
        """Record whether the triage decision led to success.

        Args:
            success: Whether the pipeline run succeeded
            run_id: Optional run ID to associate with the outcome
        """
        if hasattr(self, "_complexity_estimator") and hasattr(self, "_triage_memory_key"):
            if self._triage_memory_key:
                self._complexity_estimator.record_outcome(
                    self._triage_memory_key, success, run_id
                )

    def _get_llm_for_stage(
        self,
        stage: str,
        task: Any | None = None,
        retry_count: int = 0,
    ) -> LLMBackend:
        """Get appropriate LLM for current stage/task.

        Uses the model router if enabled, otherwise falls back
        to the default model selection (llm_code for code stages).

        Args:
            stage: Pipeline stage name
            task: Optional SubTask from workplan
            retry_count: Number of retries (for escalation)

        Returns:
            LLM backend for the stage
        """
        # Use routing if enabled and available
        if self.router and self.model_pool:
            tier = self.router.route(stage, task, retry_count)
            backend = self.model_pool.get(tier)
            if backend:
                model_name = self.router.get_model(tier)
                self.console.print(f"[dim]→ Using {tier.value} model: {model_name}[/dim]")
                return backend

        # Fallback: use llm_code for code-heavy stages
        code_stages = {"design", "implementation", "fix", "code_review"}
        if stage in code_stages and self.llm_code:
            return self.llm_code

        # Default to general model
        return self.llm

    def _init_agents(self) -> None:
        """Initialize agents with the LLM backend."""
        if self.llm is not None:
            self.spec_agent = SpecAgent(llm=self.llm)
            self.decomposition_agent = DecompositionAgent(llm=self.llm)
            self.api_contract_agent = APIContractAgent(llm=self.llm)
            self.coverage_agent = CoverageAgent(llm=self.llm)
            self.secrets_scan_agent = SecretsScanAgent(llm=self.llm)
            self.dependency_audit_agent = DependencyAuditAgent(llm=self.llm)
            self.observability_agent = ObservabilityAgent(llm=self.llm)
            self.config_agent = ConfigAgent(llm=self.llm)
            self.doc_agent = DocAgent(llm=self.llm)
            self.code_review_agent = CodeReviewAgent(llm=self.llm)
            self.context_agent: ContextAgent | None = None  # Initialized per-run with repo path
            self.design_agent = DesignAgent(llm=self.llm_code)  # Use code model for design
            self.implementation_agent: ImplementationAgent | None = None  # Initialized per-run with repo path
            self.test_agent: TestAgent | None = None  # Initialized per-run with repo path
            self.docker_agent: DockerAgent | None = None  # Initialized per-run with repo path
            self.verify_agent: VerifyAgent | None = None  # Initialized per-run with base URL
        else:
            self.spec_agent = None
            self.decomposition_agent = None
            self.api_contract_agent = None
            self.coverage_agent = None
            self.secrets_scan_agent = None
            self.dependency_audit_agent = None
            self.observability_agent = None
            self.config_agent = None
            self.doc_agent = None
            self.code_review_agent = None
            self.context_agent = None
            self.design_agent = None
            self.implementation_agent = None
            self.test_agent = None
            self.docker_agent = None
            self.verify_agent = None

    def _init_plugins(self) -> None:
        """Initialize plugin system and load plugins."""
        if not self.config.plugins.enabled:
            self.console.print("[dim]Plugin system disabled[/dim]")
            return

        # Build plugin directories list
        plugin_dirs = PluginLoader.get_default_plugin_dirs()

        # Add custom plugins dir if configured
        if self.config.plugins.plugins_dir:
            custom_dir = Path(self.config.plugins.plugins_dir)
            if custom_dir.exists():
                plugin_dirs.insert(0, custom_dir)

        # Create loader with configuration
        loader = PluginLoader(
            plugin_dirs=plugin_dirs,
            enabled_plugins=self.config.plugins.enabled_plugins or None,
            disabled_plugins=self.config.plugins.disabled_plugins,
            plugin_config=self.config.plugins.plugin_config,
        )

        # Load all plugins
        self.plugin_registry = loader.load_all()

        if len(self.plugin_registry) > 0:
            self.console.print(f"[dim]Loaded {len(self.plugin_registry)} plugins[/dim]")
            for plugin in self.plugin_registry.list_all():
                status = "[green]enabled[/green]" if plugin.enabled else "[yellow]disabled[/yellow]"
                self.console.print(f"[dim]  • {plugin.name} ({plugin.hook_point.value}) - {status}[/dim]")

    def _init_extensions(self) -> None:
        """Initialize extension system for ACF Local Edition.

        Extensions are loaded from ~/.coding-factory/extensions/ and provide
        additional agents, profiles, and RAG retrievers from the marketplace.
        """
        # Check if extensions config exists
        extensions_config = getattr(self.config, "extensions", None)
        if extensions_config is None:
            return

        # Get extensions directory
        extensions_dir = Path(
            getattr(extensions_config, "extensions_dir", "")
            or Path.home() / ".coding-factory" / "extensions"
        ).expanduser()

        if not extensions_dir.exists():
            return

        # Get enabled/disabled lists
        enabled = getattr(extensions_config, "agents", None)
        if enabled is not None and isinstance(enabled, list) and len(enabled) == 0:
            enabled = None  # Empty list means all enabled

        # Create and initialize loader
        try:
            self.extension_loader = ExtensionLoader(
                extensions_dir=extensions_dir,
                enabled_extensions=enabled,
            )
            loaded = self.extension_loader.discover()

            if loaded:
                self.console.print(f"[dim]Loaded {len(loaded)} extensions[/dim]")
                for name in loaded:
                    manifest = self.extension_loader.get_manifest(name)
                    if manifest:
                        self.console.print(
                            f"[dim]  • {name} v{manifest.version} ({manifest.type.value})[/dim]"
                        )
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not load extensions: {e}[/yellow]")
            self.extension_loader = None

    def _run_extensions_for_stage(
        self,
        stage: str,
        state: PipelineState,
        before: bool = True,
    ) -> tuple[bool, str | None]:
        """Run extensions registered for a stage hook point.

        Args:
            stage: Stage name (e.g., 'design', 'implementation')
            state: Current pipeline state
            before: If True, run 'before:stage' extensions, else 'after:stage'

        Returns:
            Tuple of (success, error_message)
        """
        if not self.extension_loader:
            return True, None

        position = "before" if before else "after"
        extensions = self.extension_loader.get_hooks_for_stage(stage, position)

        if not extensions:
            return True, None

        run_dir = Path(state.artifacts_dir)

        for ext in extensions:
            if not ext.enabled:
                continue

            manifest = ext.manifest
            self.console.print(
                f"[bold cyan]Extension:[/bold cyan] {manifest.name} ({position}:{stage})"
            )

            try:
                # Get LLM for extension
                llm = self._get_llm_for_stage(stage)

                # Create agent instance
                if ext.agent_class is None:
                    self.console.print(f"[yellow]Extension {manifest.name} has no agent class[/yellow]")
                    continue

                agent = ext.agent_class(llm=llm)

                # Prepare input from pipeline state
                context = self._build_extension_context(state, ext)
                agent_input = AgentInput(context=context)

                # Run the extension agent
                output = agent.run(agent_input)

                # Check for success (extensions may use different output formats)
                success = True
                if hasattr(output, "success"):
                    success = output.success
                elif hasattr(output, "content") and isinstance(output.content, dict):
                    success = output.content.get("success", True)

                if not success:
                    error_msg = "Extension failed"
                    if hasattr(output, "errors") and output.errors:
                        error_msg = "; ".join(output.errors)
                    elif hasattr(output, "content") and isinstance(output.content, dict):
                        error_msg = output.content.get("error", error_msg)

                    self.console.print(f"[yellow]Extension {manifest.name} failed: {error_msg}[/yellow]")
                    # Extensions are optional by default (non-blocking)
                    continue

                # Save extension outputs
                if hasattr(output, "content") and output.content:
                    artifact_path = run_dir / f"ext_{manifest.name}_output.json"
                    artifact_path.write_text(json.dumps(output.content, indent=2, default=str))
                    state.artifacts[f"ext_{manifest.name}"] = str(artifact_path)

                self.console.print(f"[green]  ✓ {manifest.name} completed[/green]")

            except Exception as e:
                self.console.print(f"[yellow]Extension {manifest.name} error: {e}[/yellow]")
                # Extensions are optional - continue with pipeline

        return True, None

    def _build_extension_context(
        self, state: PipelineState, ext: LoadedExtension
    ) -> dict[str, Any]:
        """Build context dict for extension from pipeline state.

        Args:
            state: Pipeline state
            ext: Extension to build context for

        Returns:
            Context dict with pipeline data
        """
        context: dict[str, Any] = {
            "run_id": state.run_id,
            "feature_description": state.feature_description,
            "artifacts_dir": state.artifacts_dir,
            "current_stage": state.current_stage.value if state.current_stage else None,
        }

        run_dir = Path(state.artifacts_dir)

        # Load common artifacts
        common_artifacts = [
            "spec", "context", "design", "implementation",
            "test_report", "code_review",
        ]

        for artifact_name in common_artifacts:
            if artifact_name in state.artifacts:
                artifact_path = Path(state.artifacts[artifact_name])
                if artifact_path.exists() and artifact_path.suffix == ".json":
                    try:
                        with open(artifact_path) as f:
                            context[artifact_name] = json.load(f)
                    except json.JSONDecodeError:
                        context[artifact_name] = artifact_path.read_text()
            else:
                # Try common file names
                for suffix in [".json", ".md", ".yaml"]:
                    artifact_path = run_dir / f"{artifact_name}{suffix}"
                    if artifact_path.exists():
                        if suffix == ".json":
                            try:
                                with open(artifact_path) as f:
                                    context[artifact_name] = json.load(f)
                            except json.JSONDecodeError:
                                context[artifact_name] = artifact_path.read_text()
                        else:
                            context[artifact_name] = artifact_path.read_text()
                        break

        return context

    def _run_plugins_for_stage(
        self,
        stage: str,
        state: PipelineState,
        before: bool = True,
    ) -> tuple[bool, str | None]:
        """Run plugins registered for a stage hook point.

        Args:
            stage: Stage name (e.g., 'design', 'implementation')
            state: Current pipeline state
            before: If True, run 'before:stage' plugins, else 'after:stage'

        Returns:
            Tuple of (success, error_message)
        """
        plugins = self.plugin_registry.get_for_stage(stage, before=before)

        if not plugins:
            return True, None

        hook_type = "before" if before else "after"
        run_dir = Path(state.artifacts_dir)

        for plugin in plugins:
            self.console.print(f"[bold magenta]Plugin:[/bold magenta] {plugin.name} ({hook_type}:{stage})")

            try:
                # Get LLM for plugin if needed
                llm = self._get_llm_for_stage(stage) if plugin.manifest.requires_llm else None

                # Create agent instance
                agent = plugin.create_agent(llm=llm)

                # Prepare input from pipeline state
                context = self._build_plugin_context(state, plugin)
                agent_input = AgentInput(context=context)

                # Run the plugin agent
                output = agent.run(agent_input)

                if not output.success:
                    error_msg = "; ".join(output.errors) if output.errors else "Plugin failed"

                    if plugin.manifest.hook.required:
                        self.console.print(f"[red]Plugin {plugin.name} failed (required): {error_msg}[/red]")
                        return False, f"Required plugin '{plugin.name}' failed: {error_msg}"
                    else:
                        self.console.print(f"[yellow]Plugin {plugin.name} failed: {error_msg}[/yellow]")
                        continue

                # Save plugin outputs
                for output_name in plugin.manifest.outputs:
                    if output_name in output.data:
                        artifact_path = run_dir / f"plugin_{plugin.name}_{output_name}.json"
                        artifact_path.write_text(json.dumps(output.data[output_name], indent=2, default=str))
                        state.artifacts[f"plugin_{plugin.name}_{output_name}"] = str(artifact_path)

                self.console.print(f"[green]  ✓ {plugin.name} completed[/green]")

            except Exception as e:
                if plugin.manifest.hook.required:
                    self.console.print(f"[red]Plugin {plugin.name} error (required): {e}[/red]")
                    return False, f"Required plugin '{plugin.name}' error: {e}"
                else:
                    self.console.print(f"[yellow]Plugin {plugin.name} error: {e}[/yellow]")

        return True, None

    def _build_plugin_context(self, state: PipelineState, plugin: LoadedPlugin) -> dict[str, Any]:
        """Build context dict for plugin from pipeline state.

        Args:
            state: Pipeline state
            plugin: Plugin to build context for

        Returns:
            Context dict with requested artifacts
        """
        context: dict[str, Any] = {
            "run_id": state.run_id,
            "feature_description": state.feature_description,
            "artifacts_dir": state.artifacts_dir,
        }

        run_dir = Path(state.artifacts_dir)

        # Load requested inputs
        for input_name in plugin.manifest.inputs:
            # Check if it's in state.artifacts
            if input_name in state.artifacts:
                artifact_path = Path(state.artifacts[input_name])
                if artifact_path.exists() and artifact_path.suffix == ".json":
                    with open(artifact_path) as f:
                        context[input_name] = json.load(f)
                else:
                    context[input_name] = str(artifact_path)
            else:
                # Try common artifact file names
                for suffix in [".json", ".md", ".yaml", ".txt"]:
                    artifact_path = run_dir / f"{input_name}{suffix}"
                    if artifact_path.exists():
                        if suffix == ".json":
                            with open(artifact_path) as f:
                                context[input_name] = json.load(f)
                        else:
                            context[input_name] = artifact_path.read_text()
                        break

        return context

    def create_run(self, feature: str, repo_path: Path | None = None) -> PipelineState:
        """Create a new pipeline run.

        Args:
            feature: Feature description
            repo_path: Path to target repository

        Returns:
            New PipelineState
        """
        run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        state = PipelineState(
            run_id=run_id,
            feature_description=feature,
            artifacts_dir=str(run_dir),
            config_snapshot={
                "llm_backend": self.config.llm.backend,
                "model_general": self.config.llm.model_general,
                "model_code": self.config.llm.model_code,
            },
        )

        # Run complexity triage for smart model routing
        if self.router and self.llm:
            self._run_complexity_triage(feature)

        # Save initial spec
        spec_path = run_dir / "spec.json"
        spec_data = {
            "run_id": run_id,
            "feature": feature,
            "repo_path": str(repo_path) if repo_path else str(Path.cwd()),
            "started_at": datetime.now().isoformat(),
        }
        spec_path.write_text(json.dumps(spec_data, indent=2))
        state.artifacts["initial_spec"] = str(spec_path)

        return state

    def run(
        self,
        feature: str,
        repo_path: Path | None = None,
        resume_run_id: str | None = None,
        dry_run: bool = False,
        iteration_context: dict | None = None,
        decompose: bool = False,
        api_contract: bool = False,
        coverage: bool = False,
        coverage_threshold: float = 80.0,
        secrets_scan: bool = False,
        dependency_audit: bool = False,
        rollback_strategy: bool = False,
        observability: bool = False,
        config: bool = False,
        docs: bool = False,
        code_review: bool = False,
        policy: bool = False,
        policy_rules: str | None = None,
        pr_package: bool = False,
    ) -> PipelineState:
        """Execute the pipeline.

        Args:
            feature: Feature description
            repo_path: Target repository path
            resume_run_id: Run ID to resume (if resuming)
            dry_run: If True, don't execute actions
            iteration_context: Context for iteration mode (base_run_id, original_feature, improvement_request)
            decompose: If True, decompose feature into sub-tasks before design
            api_contract: If True, generate API contract before implementation
            coverage: If True, enforce test coverage thresholds
            coverage_threshold: Minimum coverage percentage required
            secrets_scan: If True, scan for hardcoded secrets
            dependency_audit: If True, audit dependencies for CVEs and outdated packages
            rollback_strategy: If True, generate CI/CD rollback and canary deployment strategies
            observability: If True, inject logging, metrics, and tracing scaffolding
            config: If True, enforce 12-factor config layout
            docs: If True, generate and sync documentation
            code_review: If True, perform senior engineer code review
            policy: If True, enforce policy rules before verification
            policy_rules: Path to custom policy_rules.yaml file
            pr_package: If True, build rich PR package with spec-tied changes

        Returns:
            Final pipeline state
        """
        # Set feature flags
        self._decompose_enabled = decompose
        self._api_contract_enabled = api_contract
        self._coverage_enabled = coverage
        self._coverage_threshold = coverage_threshold
        self._secrets_scan_enabled = secrets_scan
        self._dependency_audit_enabled = dependency_audit
        self._rollback_strategy_enabled = rollback_strategy
        self._observability_enabled = observability
        self._config_enabled = config
        self._docs_enabled = docs
        self._code_review_enabled = code_review
        self._policy_enabled = policy
        self._policy_rules_path = policy_rules
        self._pr_package_enabled = pr_package
        # Resume or create new run
        if resume_run_id:
            state_machine = self._resume_run(resume_run_id)
            state = state_machine.state
            self.console.print(f"[green]Resuming run {resume_run_id}[/green]")
        else:
            state = self.create_run(feature, repo_path)
            state_machine = StateMachine(state, Path(state.artifacts_dir))

            # Store iteration context if provided
            if iteration_context:
                state.iteration_context = iteration_context
                self.console.print(f"[cyan]Iteration mode: improving {iteration_context.get('base_run_id')}[/cyan]")

            self.console.print(f"[green]Starting new run {state.run_id}[/green]")

        # Set current run ID for WebSocket updates
        self._current_run_id = state.run_id

        if dry_run:
            self.console.print("[yellow]Dry run mode - no actions will be executed[/yellow]")
            return state

        # Start execution
        state.status = RunStatus.RUNNING
        state.started_at = datetime.now()
        state_machine.save_state()

        # Execute stages
        try:
            while not state_machine.is_completed() and not state_machine.is_failed():
                # Check for cancellation before each stage
                if self.cancellation_check and self.cancellation_check():
                    self.console.print("[yellow]Job cancelled by user[/yellow]")
                    logger.info("PIPELINE: Cancelled by user")
                    self._send_ws_update("stage", stage="CANCELLED", status="cancelled")
                    state.status = RunStatus.FAILED
                    state.error = "Cancelled by user"
                    break

                current_stage = state.current_stage

                # Get handler for current stage
                handler = self._handlers.get(current_stage)
                if not handler:
                    state_machine.fail_stage(f"No handler for stage: {current_stage}")
                    break

                # Execute stage
                self._print_stage_start(current_stage)
                logger.info(f"PIPELINE: Starting stage {current_stage.value}")
                # Send progress update to frontend
                self._send_ws_update("stage", stage=current_stage.value, status="running")

                # Run before:stage plugins
                stage_name = current_stage.value.lower()
                plugin_success, plugin_error = self._run_plugins_for_stage(stage_name, state, before=True)
                if not plugin_success:
                    state_machine.fail_stage(plugin_error or "Before-stage plugin failed")
                    break

                # Run before:stage extensions (ACF Local Edition)
                ext_success, ext_error = self._run_extensions_for_stage(stage_name, state, before=True)
                if not ext_success:
                    state_machine.fail_stage(ext_error or "Before-stage extension failed")
                    break

                # Run the stage handler
                print(f"[STAGE] Running: {current_stage.value}", flush=True)
                try:
                    success, error = handler(state)
                    print(f"[STAGE] {current_stage.value} result: success={success}", flush=True)
                except Exception as handler_exc:
                    print(f"[STAGE] {current_stage.value} EXCEPTION: {handler_exc}", flush=True)
                    logger.exception(f"Handler exception in stage {current_stage.value}")
                    success, error = False, str(handler_exc)
                logger.info(f"PIPELINE: Stage {current_stage.value} completed: success={success}, error={error}")
                # Send completion update
                if success:
                    self._send_ws_update("stage", stage=current_stage.value, status="completed")
                else:
                    self._send_ws_update("stage", stage=current_stage.value, status="failed", error=error)

                # Run after:stage plugins (only if stage succeeded)
                if success and state.status != RunStatus.PAUSED:
                    plugin_success, plugin_error = self._run_plugins_for_stage(stage_name, state, before=False)
                    if not plugin_success:
                        state_machine.fail_stage(plugin_error or "After-stage plugin failed")
                        break

                    # Run after:stage extensions (ACF Local Edition)
                    ext_success, ext_error = self._run_extensions_for_stage(stage_name, state, before=False)
                    if not ext_success:
                        state_machine.fail_stage(ext_error or "After-stage extension failed")
                        break

                if success:
                    print(f"[DEBUG] Stage {current_stage.value} succeeded", flush=True)
                    # Check if paused for approval (set by handler)
                    if state.status == RunStatus.PAUSED:
                        self.console.print("[yellow]Pipeline paused for approval[/yellow]")
                        break

                    # Mark current stage as completed/approved before moving on
                    if state_machine.is_approval_stage():
                        state_machine.approve_stage()
                    else:
                        state_machine.complete_stage()

                    # Determine next stage
                    next_stages = state_machine.get_valid_next_stages()
                    print(f"[DEBUG] Next stages available: {[s.value for s in next_stages] if next_stages else 'NONE'}", flush=True)
                    if next_stages:
                        next_stage = self._select_next_stage(state, next_stages)
                        print(f"[DEBUG] Transitioning to: {next_stage.value}", flush=True)
                        state_machine.transition(next_stage)
                    else:
                        print(f"[DEBUG] No next stages - pipeline ending", flush=True)
                else:
                    # Check if just deferred (not a failure)
                    if state.status == RunStatus.PAUSED:
                        self.console.print("[yellow]Pipeline paused - run can be resumed[/yellow]")
                        break
                    state_machine.fail_stage(error or "Stage failed")
                    break

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
            state.status = RunStatus.CANCELLED
            state_machine.save_state()

        except Exception as e:
            self.console.print(f"[red]Pipeline error: {e}[/red]")
            state_machine.fail_stage(str(e))
            # Record triage failure for learning
            self._record_triage_outcome(False, state.run_id)

        # Final status
        if state_machine.is_completed():
            state.status = RunStatus.COMPLETED
            state.completed_at = datetime.now()
            state_machine.save_state()
            self._print_completion_summary(state)
            # Record triage success for learning
            self._record_triage_outcome(True, state.run_id)

        return state

    def _resume_run(self, run_id: str) -> StateMachine:
        """Resume a previous run.

        Args:
            run_id: Run ID to resume

        Returns:
            StateMachine with loaded state
        """
        run_dir = self.artifacts_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run not found: {run_id}")

        return StateMachine.load_state(run_dir)

    def _get_repo_path(self, run_dir: Path) -> Path:
        """Get the correct repo path for post-generation agents.

        Priority:
        1. generated_project/ directory (for generated code)
        2. repo_path from spec.json (for iteration on existing code)
        3. Current working directory (fallback)

        Args:
            run_dir: The pipeline run artifacts directory

        Returns:
            Path to the code to analyze/review
        """
        # First check for generated project
        generated_project = run_dir / "generated_project"
        if generated_project.exists():
            return generated_project

        # Then check spec for original repo path
        spec_file = run_dir / "spec.json"
        if spec_file.exists():
            with open(spec_file) as f:
                spec_data = json.load(f)
                if "repo_path" in spec_data:
                    return Path(spec_data["repo_path"])

        # Fallback to cwd
        return Path.cwd()

    def _convert_text_to_yaml_rules(self, text: str) -> str:
        """Convert plain text policy rules to YAML format.

        Converts bullet points or numbered lists into YAML policy rules.
        Plain text rules are converted to 'require_approval' actions for human review.

        Args:
            text: Plain text rules (bullet points, numbered lists, etc.)

        Returns:
            YAML formatted policy rules
        """
        import re

        lines = text.strip().split('\n')
        rules = []
        rule_count = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Remove bullet points, numbers, dashes
            rule_text = re.sub(r'^[-*•]\s*|\d+[.)]\s*', '', line).strip()
            if not rule_text:
                continue

            rule_count += 1
            rule_id = f"custom_rule_{rule_count}"

            # Create a rule that requires human review for this constraint
            rules.append(f"""  - id: {rule_id}
    description: "{rule_text}"
    when:
      # Custom rule - always triggers for review
      files_changed_gt: -1
    action: require_approval
    severity: info
    message: "Custom policy: {rule_text}"
""")

        yaml_content = f"""version: "1.0"
description: "Custom policy rules from user input"
rules:
{chr(10).join(rules)}"""

        return yaml_content

    def _select_next_stage(self, state: PipelineState, options: list[Stage]) -> Stage:
        """Select the next stage from available options.

        Args:
            state: Current state
            options: Valid next stages

        Returns:
            Selected next stage
        """
        # Default: first option (happy path)
        # Could be enhanced with conditional logic
        return options[0]

    def _print_stage_start(self, stage: Stage) -> None:
        """Print stage start indicator."""
        stage_names = {
            Stage.INIT: "Initializing",
            Stage.SPEC: "Parsing Specification",
            Stage.DECOMPOSITION: "Decomposing Feature",
            Stage.CONTEXT: "Analyzing Codebase",
            Stage.DESIGN: "Creating Design",
            Stage.DESIGN_APPROVAL: "Design Approval",
            Stage.API_CONTRACT: "Generating API Contract",
            Stage.IMPLEMENTATION: "Implementing Changes",
            Stage.TESTING: "Running Tests",
            Stage.COVERAGE: "Checking Coverage",
            Stage.SECRETS_SCAN: "Scanning for Secrets",
            Stage.DEPENDENCY_AUDIT: "Auditing Dependencies",
            Stage.DOCKER_BUILD: "Building Container",
            Stage.OBSERVABILITY: "Injecting Observability",
            Stage.VERIFICATION: "Verifying Behavior",
            Stage.FINAL_APPROVAL: "Final Approval",
            Stage.DEPLOY: "Deploying",
            Stage.DONE: "Complete",
        }
        name = stage_names.get(stage, stage.value)
        self.console.print(f"\n[bold blue]>>> {name}[/bold blue]")

    def _print_completion_summary(self, state: PipelineState) -> None:
        """Print pipeline completion summary."""
        self.console.print()
        self.console.print("[bold green]Pipeline completed successfully![/bold green]")
        self.console.print()
        self.console.print(f"Run ID: {state.run_id}")
        self.console.print(f"Artifacts: {state.artifacts_dir}")
        self.console.print()
        self.console.print("[bold]Stage Summary:[/bold]")
        for stage_name, result in state.stages.items():
            status_color = {
                StageStatus.COMPLETED: "green",
                StageStatus.APPROVED: "green",
                StageStatus.FAILED: "red",
                StageStatus.SKIPPED: "yellow",
            }.get(result.status, "white")
            self.console.print(f"  {stage_name}: [{status_color}]{result.status.value}[/{status_color}]")

    # --- Stage Handlers ---

    def _handle_init(self, state: PipelineState) -> tuple[bool, str | None]:
        """Initialize pipeline run."""
        # Already initialized in create_run
        return True, None

    def _handle_spec(self, state: PipelineState) -> tuple[bool, str | None]:
        """Parse feature specification using SpecAgent."""
        run_dir = Path(state.artifacts_dir)
        spec_file = run_dir / "feature_spec.json"

        # Check if we have the agent available
        if self.spec_agent is None:
            self.console.print("[yellow]SpecAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating minimal spec without LLM...[/dim]")

            # Fallback: create minimal spec without LLM
            spec_data = {
                "id": f"FEAT-{state.run_id}",
                "title": state.feature_description[:50],
                "original_description": state.feature_description,
                "user_story": f"As a user, I want {state.feature_description}",
                "acceptance_criteria": [],
                "domains": [],
                "priority": "medium",
                "assumptions": ["LLM not available - minimal spec generated"],
                "clarifications_needed": [],
            }
            spec_file.write_text(json.dumps(spec_data, indent=2))
            state.artifacts["feature_spec"] = str(spec_file)
            self.console.print(f"[dim]Created: {spec_file}[/dim]")
            return True, None

        # Use SpecAgent to parse the feature description
        self.console.print("[bold]Spec Agent:[/bold] Parsing feature description...")
        self._send_ws_update("substep", stage="SPEC", substep="Analyzing your requirements...")

        input_data = AgentInput(
            context={
                "feature_description": state.feature_description,
                "run_id": state.run_id,
            }
        )

        self._send_ws_update("substep", stage="SPEC", substep="Calling LLM to parse specification...")
        output = self.spec_agent.run(input_data)
        self._send_ws_update("substep", stage="SPEC", substep="Processing parsed requirements...")

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"

            # Check if it's a connection error - fall back to minimal spec
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                self.console.print(f"[yellow]LLM not available: {error_msg}[/yellow]")
                self.console.print("[dim]Falling back to minimal spec...[/dim]")

                spec_data = {
                    "id": f"FEAT-{state.run_id}",
                    "title": state.feature_description[:50],
                    "original_description": state.feature_description,
                    "user_story": f"As a user, I want {state.feature_description}",
                    "acceptance_criteria": [],
                    "domains": [],
                    "priority": "medium",
                    "assumptions": [f"LLM not available ({error_msg}) - minimal spec generated"],
                    "clarifications_needed": ["Full spec parsing requires LLM"],
                }
                spec_file.write_text(json.dumps(spec_data, indent=2))
                state.artifacts["feature_spec"] = str(spec_file)
                self.console.print(f"[dim]Created: {spec_file}[/dim]")
                return True, None

            self.console.print(f"[red]SpecAgent failed: {error_msg}[/red]")
            return False, f"SpecAgent failed: {error_msg}"

        # Save the feature spec (exclude requirements_tracker from spec file)
        spec_data = {k: v for k, v in output.data.items() if k != "requirements_tracker"}
        spec_file.write_text(json.dumps(spec_data, indent=2, default=str))
        state.artifacts["feature_spec"] = str(spec_file)

        # Save the requirements tracker separately
        if "requirements_tracker" in output.data:
            tracker_file = run_dir / "requirements_tracker.json"
            tracker_file.write_text(json.dumps(output.data["requirements_tracker"], indent=2, default=str))
            state.artifacts["requirements_tracker"] = str(tracker_file)
            tracker_data = output.data["requirements_tracker"]
            req_count = len(tracker_data.get("requirements", []))
            self.console.print(f"[green]Requirements Tracked:[/green] {req_count} items")

        # Show summary
        self.console.print(f"[green]Title:[/green] {spec_data.get('title', 'N/A')}")
        self.console.print(f"[green]User Story:[/green] {spec_data.get('user_story', 'N/A')[:100]}...")

        ac_count = len(spec_data.get("acceptance_criteria", []))
        self.console.print(f"[green]Acceptance Criteria:[/green] {ac_count} items")

        domains = spec_data.get("domains", [])
        if domains:
            self.console.print(f"[green]Domains:[/green] {', '.join(domains)}")

        clarifications = spec_data.get("clarifications_needed", [])
        if clarifications:
            self.console.print(f"[yellow]Clarifications needed:[/yellow] {len(clarifications)}")
            for c in clarifications[:3]:
                self.console.print(f"  - {c}")

        self.console.print(f"[dim]Created: {spec_file}[/dim]")

        # Check for documentation requirements (unknown APIs)
        self._check_documentation_requirements(state, spec_data)

        # Check for safety-critical patterns (webhooks, crypto, auth)
        self._check_safety_patterns(state, spec_data)

        # Check for web/frontend patterns (inject Tailwind CSS)
        self._check_tailwind_patterns(state, spec_data)

        return True, None

    def _check_documentation_requirements(
        self,
        state: PipelineState,
        spec_data: dict | None = None,
    ) -> None:
        """Check if feature requires external API documentation.

        Uses allowlist approach: any library not in known-safe list
        is flagged as requiring documentation.
        """
        from agents.doc_requirements_agent import DocumentationRequirementsAgent

        agent = DocumentationRequirementsAgent()

        # Build context for analysis
        spec_content = ""
        if spec_data:
            spec_content = json.dumps(spec_data)

        report = agent.analyze(state.feature_description, spec_content)

        if report.needs_documentation:
            self.console.print("")
            self.console.print(f"[yellow]Documentation Requirements ({report.risk_level} risk):[/yellow]")

            for lib in sorted(report.unknown_libraries):
                req = next((r for r in report.requirements if r.name == lib), None)
                if req and req.suggestion:
                    self.console.print(f"  [yellow]•[/yellow] {lib}: {req.suggestion}")
                else:
                    self.console.print(f"  [yellow]•[/yellow] {lib}: documentation recommended")

            # Check RAG folder for existing docs (in project root)
            rag_path = Path.cwd() / "rag"
            if rag_path.exists():
                coverage = agent.check_rag_coverage(report.unknown_libraries, rag_path)
                covered = [lib for lib, has_docs in coverage.items() if has_docs]
                if covered:
                    self.console.print(f"  [green]✓[/green] Found in RAG: {', '.join(covered)}")

            if report.risk_level == "high":
                self.console.print("")
                self.console.print("[red]WARNING: High-risk API detected (payments/auth).[/red]")
                self.console.print("[red]Generated code MUST be verified against official docs.[/red]")

            # Store in state for later stages
            state.metadata["doc_requirements"] = {
                "unknown_libraries": list(report.unknown_libraries),
                "risk_level": report.risk_level,
            }

    def _check_safety_patterns(
        self,
        state: PipelineState,
        spec_data: dict | None = None,
    ) -> None:
        """Check for safety-critical implementation patterns.

        Detects patterns like webhooks, crypto, auth that require
        specific implementation invariants to be followed.
        """
        from agents.safety_patterns_agent import SafetyPatternsAgent

        agent = SafetyPatternsAgent()

        # Build context for analysis
        spec_content = ""
        if spec_data:
            spec_content = json.dumps(spec_data)

        report = agent.analyze(state.feature_description, spec_content)

        if report.has_safety_concerns:
            self.console.print("")
            self.console.print("[yellow]Safety-Critical Patterns Detected:[/yellow]")

            for pattern in report.patterns_detected:
                premium_flag = " [PREMIUM]" if pattern.force_premium else ""
                self.console.print(f"  [yellow]•[/yellow] {pattern.name}{premium_flag}")
                self.console.print(f"    [dim]Triggers: {', '.join(pattern.triggers_matched)}[/dim]")
                self.console.print(f"    [dim]Invariants: {len(pattern.invariants)} rules will be injected[/dim]")

            if report.requires_premium:
                self.console.print("")
                self.console.print("[cyan]Model will be upgraded to premium for safety-critical code.[/cyan]")

            # Store in state for implementation stage
            state.metadata["safety_patterns"] = {
                "patterns": [p.name for p in report.patterns_detected],
                "requires_premium": report.requires_premium,
                "prompt_injection": report.get_prompt_injection(),
            }

    def _check_tailwind_patterns(
        self,
        state: PipelineState,
        spec_data: dict | None = None,
    ) -> None:
        """Check for web/frontend patterns requiring Tailwind CSS.

        Detects patterns like website, landing page, responsive design
        and injects Tailwind CSS best practices.
        """
        from agents.tailwind_css_agent import TailwindCSSAgent

        agent = TailwindCSSAgent()

        # Check if config loaded successfully
        if agent.load_error:
            self.console.print(f"[yellow]Tailwind CSS patterns unavailable: {agent.load_error}[/yellow]")
            return

        if not agent.patterns:
            self.console.print("[yellow]Tailwind CSS patterns: No patterns loaded from config[/yellow]")
            return

        # Build context for analysis
        spec_content = ""
        if spec_data:
            spec_content = json.dumps(spec_data)

        report = agent.analyze(state.feature_description, spec_content)

        if report.has_web_patterns:
            self.console.print("")
            self.console.print("[blue]Web/Frontend Patterns Detected - Using Tailwind CSS:[/blue]")

            for pattern in report.patterns_detected:
                self.console.print(f"  [blue]•[/blue] {pattern.name}")
                self.console.print(f"    [dim]Triggers: {', '.join(pattern.triggers_matched)}[/dim]")
                self.console.print(f"    [dim]Guidelines: {len(pattern.invariants)} design rules[/dim]")

            self.console.print("")
            self.console.print("[cyan]Tailwind CSS patterns will be injected into implementation.[/cyan]")

            # Store in state for implementation stage
            state.metadata["tailwind_patterns"] = {
                "patterns": [p.name for p in report.patterns_detected],
                "prompt_injection": report.get_prompt_injection(),
            }
        else:
            # Log that check happened but no patterns found
            self.console.print("[dim]Tailwind CSS: No web/frontend patterns detected in prompt[/dim]")

    def _check_implementation_consistency(
        self,
        state: PipelineState,
        design_proposal_data: dict,
        raw_changes: list[dict],
    ) -> bool:
        """Check that implementation matches design proposal.

        Validates:
        1. File paths match design
        2. Code content is semantically relevant
        3. Required imports are present

        Args:
            state: Current pipeline state
            design_proposal_data: Design proposal JSON data
            raw_changes: Implementation file changes

        Returns:
            True if consistency check passes, False otherwise
        """
        from agents.consistency_checker_agent import ConsistencyCheckerAgent

        self.console.print("[bold]Checking implementation consistency...[/bold]")

        # Extract design file paths
        design_file_paths = []
        if design_proposal_data:
            file_changes = design_proposal_data.get("file_changes", [])
            design_file_paths = [fc.get("path", "") for fc in file_changes if fc.get("path")]

        # Get design summary
        design_summary = design_proposal_data.get("summary", "") if design_proposal_data else ""

        # Run consistency check
        checker = ConsistencyCheckerAgent()
        report = checker.check(
            feature_description=state.feature_description,
            design_file_paths=design_file_paths,
            impl_file_changes=raw_changes,
            design_summary=design_summary,
        )

        # Store report in state
        state.metadata["consistency_check"] = {
            "passed": report.passed,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "checks": report.checks_performed,
        }

        if report.passed:
            self.console.print(f"[green]Consistency check passed[/green] ({len(report.checks_performed)} checks)")
            return True

        # Report errors
        self.console.print(f"[red]Consistency check FAILED[/red]")
        for err in report.errors:
            self.console.print(f"  [red][ERROR][/red] {err.category}: {err.message}")
            if err.expected:
                self.console.print(f"    [dim]Expected: {err.expected}[/dim]")
            if err.actual:
                self.console.print(f"    [dim]Actual: {err.actual}[/dim]")
            if err.fix_hint:
                self.console.print(f"    [cyan]Fix: {err.fix_hint}[/cyan]")

        for warn in report.warnings:
            self.console.print(f"  [yellow][WARN][/yellow] {warn.category}: {warn.message}")

        # Save report to file
        run_dir = Path(state.artifacts_dir)
        report_file = run_dir / "consistency_report.md"
        report_file.write_text(f"# Consistency Check Report\n\n{report.summary()}")
        self.console.print(f"[dim]Created: {report_file}[/dim]")

        return False

    def _verify_invariants(
        self,
        state: PipelineState,
        raw_changes: list[dict],
    ) -> None:
        """Verify generated code against domain-specific invariants.

        Runs regex/AST checks to catch common mistakes BEFORE code is committed.
        This is a "semantic linter" that catches issues like:
        - Using request.json() before signature verification
        - Missing constant-time comparison for HMAC
        """
        from agents.invariants_verifier_agent import InvariantsVerifierAgent

        verifier = InvariantsVerifierAgent()

        # Combine all generated code
        all_code = ""
        for change in raw_changes:
            # Note: Implementation agent uses "new_code" field for file content
            content = change.get("new_code", "")
            if content:
                all_code += f"\n# File: {change.get('path', 'unknown')}\n{content}\n"

        if not all_code.strip():
            return

        # Verify against relevant invariants
        result = verifier.verify(all_code, context=state.feature_description)

        if result.violations:
            self.console.print("")
            self.console.print("[yellow]Invariants Verification:[/yellow]")

            for v in result.violations:
                if v.severity == "error":
                    self.console.print(f"  [red][ERROR][/red] {v.pattern_message}")
                else:
                    self.console.print(f"  [yellow][WARN][/yellow] {v.pattern_message}")

                if v.line_number and v.line_content:
                    self.console.print(f"    [dim]Line {v.line_number}: {v.line_content.strip()[:60]}[/dim]")
                if v.fix_hint:
                    self.console.print(f"    [cyan]Fix: {v.fix_hint}[/cyan]")

            # Store in state
            state.metadata["invariant_violations"] = {
                "errors": result.error_count,
                "warnings": result.warning_count,
                "details": [
                    {
                        "topic": v.invariant_topic,
                        "message": v.pattern_message,
                        "severity": v.severity,
                        "line": v.line_number,
                        "fix_hint": v.fix_hint,
                    }
                    for v in result.violations
                ],
            }

            if result.has_errors:
                self.console.print("")
                self.console.print(
                    "[red]WARNING: Generated code has invariant violations![/red]"
                )
                self.console.print(
                    "[red]Review the code carefully before using in production.[/red]"
                )
        else:
            if result.invariants_checked:
                self.console.print(
                    f"[green]Invariants verified:[/green] {', '.join(result.invariants_checked)}"
                )

    def _handle_decomposition(self, state: PipelineState) -> tuple[bool, str | None]:
        """Decompose feature spec into sub-tasks using DecompositionAgent."""
        run_dir = Path(state.artifacts_dir)
        workplan_file = run_dir / "workplan.json"

        # Skip if decomposition is disabled
        if not self._decompose_enabled:
            self.console.print("[dim]Decomposition skipped (use --decompose to enable)[/dim]")
            return True, None

        # Check if we have the agent available
        if self.decomposition_agent is None:
            self.console.print("[yellow]DecompositionAgent not available (LLM backend not initialized)[/yellow]")
            return True, None  # Skip gracefully

        # Load feature spec
        feature_spec_file = run_dir / "feature_spec.json"
        if not feature_spec_file.exists():
            self.console.print("[yellow]No feature_spec.json found, skipping decomposition[/yellow]")
            return True, None

        with open(feature_spec_file) as f:
            feature_spec_data = json.load(f)

        self.console.print("[bold]Decomposition Agent:[/bold] Breaking feature into sub-tasks...")

        input_data = AgentInput(
            context={"feature_spec": feature_spec_data}
        )

        output = self.decomposition_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Decomposition failed: {error_msg}[/yellow]")
            self.console.print("[dim]Continuing without workplan...[/dim]")
            return True, None  # Don't fail the pipeline, just skip

        # Save workplan
        workplan_file.write_text(json.dumps(output.data, indent=2, default=str))
        state.artifacts["workplan"] = str(workplan_file)

        # Log summary
        tasks = output.data.get("tasks", [])
        self.console.print(f"[green]Decomposed into {len(tasks)} sub-tasks:[/green]")
        for task in tasks[:5]:  # Show first 5
            priority = task.get("priority", "p1")
            size = task.get("size", "m")
            self.console.print(f"  [{priority}][{size}] {task.get('title', 'Untitled')}")
        if len(tasks) > 5:
            self.console.print(f"  ... and {len(tasks) - 5} more")

        # Cost/complexity check - warn if too many tasks
        TASK_THRESHOLD = 10  # More than 10 tasks suggests feature is too complex
        COST_PER_TASK_USD = 0.15  # Rough estimate: ~$0.15 per task
        estimated_cost = len(tasks) * COST_PER_TASK_USD

        if len(tasks) > TASK_THRESHOLD:
            warning = (
                f"⚠️  HIGH COMPLEXITY: {len(tasks)} tasks decomposed\n"
                f"   Estimated cost: ${estimated_cost:.2f}\n"
                f"   Consider simplifying your prompt or breaking into smaller features."
            )
            self.console.print(f"[yellow]{warning}[/yellow]")
            state.metadata["complexity_warning"] = {
                "task_count": len(tasks),
                "estimated_cost_usd": estimated_cost,
                "message": warning,
            }
            # Save warning to file for API to pick up
            warning_file = run_dir / "complexity_warning.json"
            warning_file.write_text(json.dumps(state.metadata["complexity_warning"], indent=2))

        self.console.print(f"[dim]Created: {workplan_file}[/dim]")
        return True, None

    def _handle_context(self, state: PipelineState) -> tuple[bool, str | None]:
        """Analyze codebase context using ContextAgent."""
        run_dir = Path(state.artifacts_dir)

        # Determine repo path from initial spec
        repo_path = Path.cwd()  # Default to current directory
        spec_file = run_dir / "spec.json"
        if spec_file.exists():
            with open(spec_file) as f:
                spec_data = json.load(f)
                if "repo_path" in spec_data:
                    repo_path = Path(spec_data["repo_path"])

        # Check if we have LLM available
        if self.llm is None:
            self.console.print("[yellow]ContextAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating minimal context report...[/dim]")

            context_file = run_dir / "context_report.md"
            context_file.write_text(f"""# Context Report

## Feature: {state.feature_description}

## Repository Analysis

*LLM not available - minimal context generated*

## Repository Path

{repo_path}

## Note

Full context analysis requires LLM connection.
""")
            state.artifacts["context_report"] = str(context_file)
            self.console.print(f"[dim]Created: {context_file}[/dim]")
            return True, None

        # Check if RAG is available for this repository
        # Skip RAG in lightweight mode (API models don't need it)
        retriever = None
        if not self._lightweight_mode:
            try:
                from rag import CodeRetriever
                retriever = CodeRetriever(
                    repo_path=repo_path,
                    ollama_url=self.config.llm.base_url,
                )
                if retriever.is_indexed():
                    self.console.print("[dim]Using RAG for semantic code search[/dim]")
                else:
                    retriever = None  # No index, skip RAG
            except Exception:
                retriever = None  # RAG not available
        else:
            self.console.print("[dim]Skipping RAG (lightweight mode)[/dim]")

        # Check if memory system is available
        # Skip memory in lightweight mode (API models have sufficient training)
        memory_retriever = None
        if self.config.memory.enabled and not self._lightweight_mode:
            try:
                from memory import MemoryStore, MemoryRetriever
                from rag.embeddings import OllamaEmbeddings

                # Determine store path
                if self.config.memory.store_location == "local":
                    store_path = Path.cwd() / ".coding-factory-memory"
                else:
                    store_path = Path.home() / ".coding-factory" / "memory"

                if store_path.exists():
                    store = MemoryStore(
                        store_path=store_path,
                        decay_half_life_days=self.config.memory.decay_half_life_days,
                    )
                    embeddings = OllamaEmbeddings(model=self.config.memory.embedding_model)
                    memory_retriever = MemoryRetriever(store=store, embeddings=embeddings)
                    self.console.print("[dim]Using memory for historical context[/dim]")
            except Exception:
                memory_retriever = None  # Memory not available

        # Initialize ContextAgent with repo path and optional RAG/memory retrievers
        self.context_agent = ContextAgent(
            llm=self.llm,
            repo_path=repo_path,
            retriever=retriever,
            memory_retriever=memory_retriever,
        )

        self.console.print("[bold]Context Agent:[/bold] Analyzing repository...")
        self.console.print(f"[dim]Repository path: {repo_path}[/dim]")

        # Load feature spec if available
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        # Prepare input for ContextAgent
        context_data = {
            "feature_spec": feature_spec_data,
            "feature_description": state.feature_description,
            "run_id": state.run_id,
        }

        # Add iteration context if this is an iteration run
        if state.iteration_context:
            context_data["iteration_context"] = state.iteration_context
            context_data["is_iteration"] = True
            self.console.print(f"[cyan]Iteration mode: analyzing existing project[/cyan]")

        input_data = AgentInput(context=context_data)

        # Run the agent
        self._send_ws_update("substep", stage="CONTEXT", substep="Scanning codebase structure...")
        self._send_ws_update("substep", stage="CONTEXT", substep="Calling LLM to analyze context...")
        output = self.context_agent.run(input_data)
        self._send_ws_update("substep", stage="CONTEXT", substep="Building context report...")

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"

            # Check if it's a connection error - fall back to minimal report
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                self.console.print(f"[yellow]LLM not available: {error_msg}[/yellow]")
                self.console.print("[dim]Falling back to minimal context report...[/dim]")

                context_file = run_dir / "context_report.md"
                context_file.write_text(f"""# Context Report

## Feature: {state.feature_description}

## Repository Analysis

*LLM connection failed - minimal context generated*

## Error

{error_msg}

## Repository Path

{repo_path}
""")
                state.artifacts["context_report"] = str(context_file)
                self.console.print(f"[dim]Created: {context_file}[/dim]")
                return True, None

            self.console.print(f"[red]ContextAgent failed: {error_msg}[/red]")
            return False, f"ContextAgent failed: {error_msg}"

        # Save the context report as JSON
        context_json_file = run_dir / "context_report.json"
        context_json_file.write_text(json.dumps(output.data, indent=2, default=str))
        state.artifacts["context_report_json"] = str(context_json_file)

        # Generate markdown report using the agent's helper method
        from schemas.context_report import ContextReport
        try:
            context_report = ContextReport(**output.data)
            markdown_report = self.context_agent.generate_markdown_report(context_report)
        except Exception:
            # Fallback to simple markdown
            markdown_report = f"""# Context Report

## Feature: {state.feature_description}

## Analysis

{json.dumps(output.data, indent=2, default=str)}
"""

        context_md_file = run_dir / "context_report.md"
        context_md_file.write_text(markdown_report)
        state.artifacts["context_report"] = str(context_md_file)

        # Show summary
        data = output.data
        repo_struct = data.get("repo_structure", {})
        self.console.print(f"[green]Framework:[/green] {repo_struct.get('framework', 'Unknown')}")
        self.console.print(f"[green]Language:[/green] {repo_struct.get('language', 'Unknown')}")

        relevant_count = len(data.get("relevant_files", []))
        self.console.print(f"[green]Relevant Files:[/green] {relevant_count} found")

        modify_count = len(data.get("files_to_modify", []))
        create_count = len(data.get("files_to_create", []))
        if modify_count or create_count:
            self.console.print(f"[green]Suggested Changes:[/green] {modify_count} to modify, {create_count} to create")

        risks = data.get("integration_risks", [])
        if risks:
            self.console.print(f"[yellow]Integration Risks:[/yellow] {len(risks)}")
            for risk in risks[:2]:
                self.console.print(f"  - {risk}")

        mental_model = data.get("mental_model", "")
        if mental_model:
            self.console.print(f"[dim]{mental_model[:200]}...[/dim]")

        self.console.print(f"[dim]Created: {context_md_file}[/dim]")
        self.console.print(f"[dim]Created: {context_json_file}[/dim]")

        # Determine runtime environment (docker/venv/local)
        self._decide_runtime(state, repo_path, data.get("repo_structure", {}).get("framework", ""))

        return True, None

    def _decide_runtime(
        self,
        state: PipelineState,
        project_dir: Path,
        framework: str,
    ) -> None:
        """Decide optimal runtime environment for the project.

        Uses RuntimeDecisionAgent to analyze project and decide between
        docker, venv, or local execution.

        Args:
            state: Current pipeline state
            project_dir: Path to project directory
            framework: Detected framework (if any)
        """
        # Check config override
        runtime_mode = self.config.runtime.mode

        if runtime_mode != "auto":
            # Use configured mode directly
            state.artifacts["runtime"] = runtime_mode
            state.artifacts["runtime_reason"] = f"Configured: {runtime_mode}"
            self.console.print(f"[dim]Runtime: {runtime_mode} (from config)[/dim]")
            return

        # Auto-detect using RuntimeDecisionAgent
        self.console.print("[dim]Analyzing project for optimal runtime...[/dim]")

        try:
            decision = decide_runtime(
                project_dir=project_dir,
                feature_description=state.feature_description,
                framework=framework,
                llm=self.llm,
            )

            state.artifacts["runtime"] = decision.runtime
            state.artifacts["runtime_reason"] = decision.reason
            state.artifacts["runtime_confidence"] = str(decision.confidence)

            # Color based on runtime type
            color = {
                "docker": "cyan",
                "venv": "yellow",
                "local": "green",
            }.get(decision.runtime, "white")

            self.console.print(
                f"[{color}]Runtime Decision:[/{color}] {decision.runtime} "
                f"(confidence: {decision.confidence:.0%})"
            )
            self.console.print(f"[dim]  {decision.reason}[/dim]")

        except Exception as e:
            # Default to docker on error (safest option)
            state.artifacts["runtime"] = "docker"
            state.artifacts["runtime_reason"] = f"Default (error: {e})"
            self.console.print(f"[yellow]Runtime: docker (default due to error: {e})[/yellow]")

    def _handle_design(self, state: PipelineState) -> tuple[bool, str | None]:
        """Generate design proposal using DesignAgent."""
        run_dir = Path(state.artifacts_dir)

        # Check if we have LLM available
        if self.design_agent is None:
            self.console.print("[yellow]DesignAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating minimal design proposal...[/dim]")

            design_file = run_dir / "design_proposal.md"
            design_file.write_text(f"""# Design Proposal

## Feature: {state.feature_description}

## Summary

*LLM not available - minimal design generated*

## Note

Full design proposal requires LLM connection.
""")
            state.artifacts["design_proposal"] = str(design_file)
            self.console.print(f"[dim]Created: {design_file}[/dim]")
            return True, None

        self.console.print("[bold]Design Agent:[/bold] Creating design proposal...")

        # Load feature spec if available
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        # Load context report if available
        context_report_data = {}
        context_report_file = run_dir / "context_report.json"
        if context_report_file.exists():
            with open(context_report_file) as f:
                context_report_data = json.load(f)

        # Load workplan if available (from DecompositionAgent)
        workplan_data = {}
        workplan_file = run_dir / "workplan.json"
        current_task = None
        if workplan_file.exists():
            with open(workplan_file) as f:
                workplan_data = json.load(f)
            tasks = workplan_data.get("tasks", [])
            self.console.print(f"[dim]Using workplan with {len(tasks)} sub-tasks[/dim]")
            # Get first task for routing (representative of overall complexity)
            if tasks:
                from schemas.workplan import SubTask
                current_task = SubTask(**tasks[0])

        # Get routed LLM based on stage and task
        routed_llm = self._get_llm_for_stage("design", current_task)
        if routed_llm and self.design_agent:
            self.design_agent.llm = routed_llm

        # Prepare input for DesignAgent
        context_data = {
            "feature_spec": feature_spec_data,
            "context_report": context_report_data,
            "workplan": workplan_data,  # Pass workplan to design agent
            "feature_description": state.feature_description,
            "feature_id": feature_spec_data.get("id", f"FEAT-{state.run_id}"),
        }

        # Add iteration context if this is an iteration run
        if state.iteration_context:
            context_data["iteration_context"] = state.iteration_context
            context_data["is_iteration"] = True
            self.console.print("[cyan]Designing improvements for existing project[/cyan]")

        input_data = AgentInput(context=context_data)

        # Run the agent
        output = self.design_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"

            # Check if it's a connection error - fall back to minimal report
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                self.console.print(f"[yellow]LLM not available: {error_msg}[/yellow]")
                self.console.print("[dim]Falling back to minimal design proposal...[/dim]")

                design_file = run_dir / "design_proposal.md"
                design_file.write_text(f"""# Design Proposal

## Feature: {state.feature_description}

## Summary

*LLM connection failed - minimal design generated*

## Error

{error_msg}
""")
                state.artifacts["design_proposal"] = str(design_file)
                self.console.print(f"[dim]Created: {design_file}[/dim]")
                return True, None

            self.console.print(f"[red]DesignAgent failed: {error_msg}[/red]")
            return False, f"DesignAgent failed: {error_msg}"

        # Save the design proposal as JSON
        design_json_file = run_dir / "design_proposal.json"
        design_json_file.write_text(json.dumps(output.data, indent=2, default=str))
        state.artifacts["design_proposal_json"] = str(design_json_file)

        # Generate markdown report
        from schemas.design_proposal import DesignProposal
        try:
            design_proposal = DesignProposal(**output.data)
            markdown_report = self.design_agent.generate_markdown_report(design_proposal)
        except Exception:
            # Fallback to simple markdown
            markdown_report = f"""# Design Proposal

## Feature: {state.feature_description}

## Summary

{output.data.get('summary', 'No summary provided')}

## Chosen Approach

{output.data.get('chosen_approach', 'Not specified')}

{output.data.get('rationale', '')}

## Details

{json.dumps(output.data, indent=2, default=str)}
"""

        design_md_file = run_dir / "design_proposal.md"
        design_md_file.write_text(markdown_report)
        state.artifacts["design_proposal"] = str(design_md_file)

        # Show summary
        data = output.data
        self.console.print(f"[green]Approach:[/green] {data.get('chosen_approach', 'Not specified')}")

        file_changes = data.get("file_changes", [])
        if file_changes:
            creates = sum(1 for fc in file_changes if fc.get("change_type") == "create")
            modifies = sum(1 for fc in file_changes if fc.get("change_type") == "modify")
            self.console.print(f"[green]File Changes:[/green] {creates} new, {modifies} modified")

        new_deps = data.get("new_dependencies", [])
        if new_deps:
            dep_names = ", ".join([d.get("name", "unknown") for d in new_deps[:3]])
            self.console.print(f"[green]New Dependencies:[/green] {dep_names}")

        risks = data.get("risks", [])
        if risks:
            self.console.print(f"[yellow]Risks:[/yellow] {len(risks)} identified")

        summary = data.get("summary", "")
        if summary:
            # Show first 150 chars of summary
            preview = summary[:150] + "..." if len(summary) > 150 else summary
            self.console.print(f"[dim]{preview}[/dim]")

        self.console.print(f"[dim]Created: {design_md_file}[/dim]")
        self.console.print(f"[dim]Created: {design_json_file}[/dim]")
        return True, None

    def _handle_design_approval(self, state: PipelineState) -> tuple[bool, str | None]:
        """Handle design approval checkpoint."""
        checkpoint = self.checkpoint_manager.get_design_checkpoint(state)
        response = self.checkpoint_manager.request_approval(
            checkpoint,
            state,
            Path(state.artifacts_dir),
        )

        if response.result == ApprovalResult.APPROVED:
            return True, None
        elif response.result == ApprovalResult.DEFERRED:
            state.status = RunStatus.PAUSED
            return False, "Deferred by user"
        else:
            return False, f"Rejected: {response.notes}"

    def _handle_api_contract(self, state: PipelineState) -> tuple[bool, str | None]:
        """Generate API contract using APIContractAgent."""
        run_dir = Path(state.artifacts_dir)
        contract_file = run_dir / "api_contract.json"
        openapi_file = run_dir / "openapi.yaml"
        schemas_file = run_dir / "schemas.py"

        # Skip if API contract generation is disabled
        if not self._api_contract_enabled:
            self.console.print("[dim]API contract skipped (use --api-contract to enable)[/dim]")
            return True, None

        # Check if we have the agent available
        if self.api_contract_agent is None:
            self.console.print("[yellow]APIContractAgent not available (LLM backend not initialized)[/yellow]")
            return True, None  # Skip gracefully

        # Load feature spec
        feature_spec_file = run_dir / "feature_spec.json"
        if not feature_spec_file.exists():
            self.console.print("[yellow]No feature_spec.json found, skipping API contract[/yellow]")
            return True, None

        with open(feature_spec_file) as f:
            feature_spec_data = json.load(f)

        # Load design proposal
        design_file = run_dir / "design_proposal.json"
        design_data = {}
        if design_file.exists():
            with open(design_file) as f:
                design_data = json.load(f)

        # Load workplan if available
        workplan_file = run_dir / "workplan.json"
        workplan_data = {}
        if workplan_file.exists():
            with open(workplan_file) as f:
                workplan_data = json.load(f)

        self.console.print("[bold]API Contract Agent:[/bold] Defining API boundaries...")

        input_data = AgentInput(
            context={
                "feature_spec": feature_spec_data,
                "design_proposal": design_data,
                "workplan": workplan_data,
            }
        )

        output = self.api_contract_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]API contract generation failed: {error_msg}[/yellow]")
            self.console.print("[dim]Continuing without API contract...[/dim]")
            return True, None  # Don't fail the pipeline, just skip

        # Save contract JSON
        contract_file.write_text(json.dumps(output.data, indent=2, default=str))
        state.artifacts["api_contract"] = str(contract_file)

        # Generate and save OpenAPI spec
        from schemas.api_contract import APIContract
        contract = APIContract(**output.data)
        openapi_yaml = self.api_contract_agent.generate_openapi_yaml(contract)
        openapi_file.write_text(openapi_yaml)
        state.artifacts["openapi"] = str(openapi_file)

        # Generate and save Pydantic schemas
        schemas_py = self.api_contract_agent.generate_schemas_py(contract)
        schemas_file.write_text(schemas_py)
        state.artifacts["schemas"] = str(schemas_file)

        # Log summary
        endpoints = output.data.get("endpoints", [])
        models = output.data.get("models", [])
        issues = output.data.get("validation_issues", [])

        self.console.print(f"[green]API Contract generated:[/green]")
        self.console.print(f"  Endpoints: {len(endpoints)}")
        self.console.print(f"  Models: {len(models)}")
        if issues:
            errors = [i for i in issues if i.get("severity") == "error"]
            warnings = [i for i in issues if i.get("severity") == "warning"]
            if errors:
                self.console.print(f"  [red]Errors: {len(errors)}[/red]")
            if warnings:
                self.console.print(f"  [yellow]Warnings: {len(warnings)}[/yellow]")

        return True, None

    def _handle_implementation(self, state: PipelineState) -> tuple[bool, str | None]:
        """Generate implementation using ImplementationAgent."""
        print(f"[DEBUG] _handle_implementation called! run_id={state.run_id}", flush=True)
        run_dir = Path(state.artifacts_dir)
        print(f"[DEBUG] run_dir={run_dir}, exists={run_dir.exists()}", flush=True)

        # Determine repo path from initial spec
        repo_path = Path.cwd()
        spec_file = run_dir / "spec.json"
        if spec_file.exists():
            with open(spec_file) as f:
                spec_data = json.load(f)
                if "repo_path" in spec_data:
                    repo_path = Path(spec_data["repo_path"])

        # Check if we have LLM available
        if self.llm is None:
            self.console.print("[yellow]ImplementationAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating placeholder implementation...[/dim]")

            diff_file = run_dir / "diff.patch"
            diff_file.write_text("# Placeholder diff - LLM not available\n")
            state.artifacts["diff"] = str(diff_file)

            notes_file = run_dir / "implementation_notes.md"
            notes_file.write_text(f"""# Implementation Notes

## Feature: {state.feature_description}

## Summary

*LLM not available - no implementation generated*
""")
            state.artifacts["implementation_notes"] = str(notes_file)
            self.console.print(f"[dim]Created: {diff_file}[/dim]")
            return True, None

        self.console.print("[bold]Implementation Agent:[/bold] Generating code changes...")

        # Load feature spec
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        # Load context report
        context_report_data = {}
        context_report_file = run_dir / "context_report.json"
        if context_report_file.exists():
            with open(context_report_file) as f:
                context_report_data = json.load(f)

        # Load design proposal
        design_proposal_data = {}
        design_proposal_file = run_dir / "design_proposal.json"
        if design_proposal_file.exists():
            with open(design_proposal_file) as f:
                design_proposal_data = json.load(f)

        # Load workplan if available (from DecompositionAgent)
        workplan_data = {}
        current_task = None
        workplan_file = run_dir / "workplan.json"
        if workplan_file.exists():
            with open(workplan_file) as f:
                workplan_data = json.load(f)
            # Get first pending task for routing
            tasks = workplan_data.get("tasks", [])
            if tasks:
                from schemas.workplan import SubTask
                # Find first non-completed task, or use first task
                for task_data in tasks:
                    if task_data.get("status") != "completed":
                        current_task = SubTask(**task_data)
                        break
                if not current_task and tasks:
                    current_task = SubTask(**tasks[0])

        # Check if safety patterns require premium model
        safety_patterns = state.metadata.get("safety_patterns", {})
        force_premium = safety_patterns.get("requires_premium", False)

        # Get routed LLM based on stage and task
        if force_premium and self.router:
            # Force premium tier for safety-critical code
            routed_llm = self.model_pool.get(
                ModelTier.PREMIUM,
                self._get_llm_for_stage("implementation", current_task)
            )
            self.console.print("[cyan]Using premium model for safety-critical implementation[/cyan]")
        else:
            routed_llm = self._get_llm_for_stage("implementation", current_task)

        # Initialize ImplementationAgent with repo path and routed model
        logger.info(f"PIPELINE: Implementation using LLM: {type(routed_llm).__name__}, model={getattr(routed_llm, 'model', 'unknown')}")
        self._send_ws_update("substep", stage="IMPLEMENTATION", substep="Initializing code generator...")
        self.implementation_agent = ImplementationAgent(llm=routed_llm, repo_path=repo_path)

        # Prepare input for ImplementationAgent
        context_data = {
            "feature_spec": feature_spec_data,
            "context_report": context_report_data,
            "design_proposal": design_proposal_data,
            "workplan": workplan_data,  # Pass workplan to implementation agent
            "feature_description": state.feature_description,
            "feature_id": feature_spec_data.get("id", f"FEAT-{state.run_id}"),
        }

        # Get RAG content with budget management (skip in lightweight mode)
        # Note: Post-generation verification still runs regardless of mode
        if not self._lightweight_mode:
            # Get model name for budget calculation
            model_name = self.config.llm.model_code
            if routed_llm and hasattr(routed_llm, 'model'):
                model_name = routed_llm.model

            # Use RAG budget agent for context-aware retrieval
            rag_content = self._get_rag_content(
                model=model_name,
                stage="implementation",
                query=state.feature_description,
                context=state.feature_description,
            )
            if rag_content:
                context_data["safety_invariants"] = rag_content

            # Fallback to legacy safety_patterns if RAG agent not available
            if not rag_content:
                safety_patterns = state.metadata.get("safety_patterns", {})
                if safety_patterns.get("prompt_injection"):
                    context_data["safety_invariants"] = safety_patterns["prompt_injection"]
                    self.console.print(f"[cyan]Injecting {len(safety_patterns.get('patterns', []))} safety pattern(s)[/cyan]")

            # Inject Tailwind CSS patterns for web/frontend features
            tailwind_patterns = state.metadata.get("tailwind_patterns", {})
            if tailwind_patterns.get("prompt_injection"):
                # Append to safety_invariants or create new
                existing = context_data.get("safety_invariants", "")
                context_data["safety_invariants"] = existing + "\n" + tailwind_patterns["prompt_injection"]
                self.console.print(f"[blue]Injecting {len(tailwind_patterns.get('patterns', []))} Tailwind CSS pattern(s)[/blue]")
        else:
            safety_patterns = state.metadata.get("safety_patterns", {})
            if safety_patterns.get("patterns"):
                self.console.print(f"[dim]Skipping RAG/invariant injection (lightweight mode) - will verify after generation[/dim]")

        # Add iteration context if this is an iteration run
        if state.iteration_context:
            context_data["iteration_context"] = state.iteration_context
            context_data["is_iteration"] = True
            self.console.print(f"[cyan]Implementing improvements to existing project[/cyan]")

        input_data = AgentInput(context=context_data)

        # Run the agent
        self._send_ws_update("substep", stage="IMPLEMENTATION", substep="Calling LLM (this may take 1-2 minutes)...")
        output = self.implementation_agent.run(input_data)
        self._send_ws_update("substep", stage="IMPLEMENTATION", substep="Processing LLM response...")

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"

            # Check if it's a connection error - fall back to placeholder
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                self.console.print(f"[yellow]LLM not available: {error_msg}[/yellow]")
                self.console.print("[dim]Falling back to placeholder implementation...[/dim]")

                diff_file = run_dir / "diff.patch"
                diff_file.write_text("# Placeholder diff - LLM connection failed\n")
                state.artifacts["diff"] = str(diff_file)

                notes_file = run_dir / "implementation_notes.md"
                notes_file.write_text(f"""# Implementation Notes

## Feature: {state.feature_description}

## Summary

*LLM connection failed - no implementation generated*

## Error

{error_msg}
""")
                state.artifacts["implementation_notes"] = str(notes_file)
                self.console.print(f"[dim]Created: {diff_file}[/dim]")
                return True, None

            self.console.print(f"[red]ImplementationAgent failed: {error_msg}[/red]")
            return False, f"ImplementationAgent failed: {error_msg}"

        # Extract change set and notes from output
        self._send_ws_update("substep", stage="IMPLEMENTATION", substep="Extracting generated files...")
        change_set_data = output.data.get("change_set", {})
        impl_notes_data = output.data.get("implementation_notes", {})

        # Save the diff patch
        combined_patch = change_set_data.get("combined_patch", "# No changes generated\n")
        diff_file = run_dir / "diff.patch"
        diff_file.write_text(combined_patch)
        state.artifacts["diff"] = str(diff_file)

        # Save change set as JSON
        change_set_json_file = run_dir / "change_set.json"
        change_set_json_file.write_text(json.dumps(change_set_data, indent=2, default=str))
        state.artifacts["change_set_json"] = str(change_set_json_file)

        # Generate implementation notes markdown
        from schemas.implementation import ImplementationNotes
        try:
            impl_notes = ImplementationNotes(**impl_notes_data)
            notes_md = self.implementation_agent.generate_notes_markdown(impl_notes)
        except Exception:
            # Fallback to simple markdown
            notes_md = f"""# Implementation Notes

## Feature: {state.feature_description}

## Summary

{impl_notes_data.get('summary', 'Implementation completed')}

## Details

{json.dumps(impl_notes_data, indent=2, default=str)}
"""

        notes_file = run_dir / "implementation_notes.md"
        notes_file.write_text(notes_md)
        state.artifacts["implementation_notes"] = str(notes_file)

        # Save implementation notes as JSON
        impl_notes_json_file = run_dir / "implementation_notes.json"
        impl_notes_json_file.write_text(json.dumps(impl_notes_data, indent=2, default=str))
        state.artifacts["implementation_notes_json"] = str(impl_notes_json_file)

        # Show summary
        files_changed = change_set_data.get("files_changed", 0)
        insertions = change_set_data.get("insertions", 0)
        deletions = change_set_data.get("deletions", 0)

        self.console.print(f"[green]Files Changed:[/green] {files_changed}")
        self.console.print(f"[green]Changes:[/green] +{insertions} -{deletions}")

        new_funcs = len(impl_notes_data.get("new_functions", []))
        new_classes = len(impl_notes_data.get("new_classes", []))
        if new_funcs or new_classes:
            self.console.print(f"[green]New:[/green] {new_funcs} functions, {new_classes} classes")

        deps = impl_notes_data.get("dependencies_added", [])
        if deps:
            self.console.print(f"[green]Dependencies:[/green] {', '.join(deps[:3])}")

        tech_debt = len(impl_notes_data.get("tech_debt_items", []))
        if tech_debt:
            self.console.print(f"[yellow]Tech Debt:[/yellow] {tech_debt} items noted")

        todos = len(impl_notes_data.get("todos", []))
        if todos:
            self.console.print(f"[yellow]TODOs:[/yellow] {todos} items")

        summary = impl_notes_data.get("summary", "")
        if summary:
            preview = summary[:150] + "..." if len(summary) > 150 else summary
            self.console.print(f"[dim]{preview}[/dim]")

        self.console.print(f"[dim]Created: {diff_file}[/dim]")
        self.console.print(f"[dim]Created: {notes_file}[/dim]")

        # Extract generated code to project directory
        raw_changes = output.data.get("raw_changes", [])
        logger.info(f"PIPELINE: Implementation raw_changes count: {len(raw_changes)}")
        if raw_changes:
            for i, change in enumerate(raw_changes[:3]):  # Log first 3
                logger.info(f"PIPELINE: raw_change[{i}]: path={change.get('path')}, code_len={len(change.get('new_code', ''))}")
        if raw_changes or state.iteration_context:
            project_dir = self._extract_project_files(run_dir, raw_changes, state)
            state.artifacts["project_dir"] = str(project_dir)
            self.console.print(f"[green]Project files extracted to:[/green] {project_dir}")

        # Run consistency check - validates implementation matches design
        if raw_changes:
            consistency_ok = self._check_implementation_consistency(
                state=state,
                design_proposal_data=design_proposal_data,
                raw_changes=raw_changes,
            )
            if not consistency_ok:
                return False, "Implementation failed consistency check - code doesn't match design"

        # Run fix loop if enabled
        if self.config.pipeline.fix_loop.enabled and raw_changes:
            fix_result = self._run_fix_loop(state, raw_changes)
            if fix_result:
                # Update artifacts with fixed code
                self.console.print("[green]Fix loop completed successfully[/green]")

        # Verify generated code against invariants
        if raw_changes:
            self._verify_invariants(state, raw_changes)

        # Generate deployment-ready scaffold files
        if raw_changes or (run_dir / "generated_project").exists():
            project_dir = run_dir / "generated_project"
            if project_dir.exists():
                self._run_scaffold_agent(state, project_dir)

        return True, None

    def _run_fix_loop(
        self,
        state: PipelineState,
        raw_changes: list[dict],
    ) -> bool:
        """Run iterative fix loop to validate and correct code.

        Validates both:
        1. Syntax/type errors (via CodeValidator)
        2. Invariant violations (via InvariantsVerifierAgent)

        Safeguards against infinite loops:
        - Max iterations (from config)
        - Same errors repeating
        - Same invariant violations repeating

        Args:
            state: Current pipeline state
            raw_changes: List of file changes from implementation

        Returns:
            True if code is valid (or was fixed), False if unfixable
        """
        run_dir = Path(state.artifacts_dir)
        fix_config = self.config.pipeline.fix_loop

        # Extract code files from raw_changes
        code_files: dict[str, str] = {}
        for change in raw_changes:
            path = change.get("path", "")
            code = change.get("new_code", "")
            if path and code:
                code_files[path] = code

        if not code_files:
            self.console.print("[dim]No code files to validate[/dim]")
            return True

        # Get correct repo path for validation (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        # Initialize validator and fix loop state
        validator = CodeValidator(repo_path=repo_path)
        loop_state = FixLoopState(
            max_iterations=fix_config.max_iterations,
            timeout_seconds=fix_config.timeout_seconds,
        )

        # Initialize invariants verifier
        from agents.invariants_verifier_agent import InvariantsVerifierAgent
        invariants_verifier = InvariantsVerifierAgent()
        previous_violations: set[str] = set()  # Track for no-progress detection

        # Initial validation (syntax + invariants)
        self.console.print("[bold]Validating generated code...[/bold]")
        self._send_ws_update("fix_loop", status="validating", message="Validating generated code...")
        validation = validator.validate_files(code_files)

        # Run invariant verification
        all_code = "\n".join(f"# File: {p}\n{c}" for p, c in code_files.items())
        invariant_result = invariants_verifier.verify(all_code, context=state.feature_description)
        invariant_errors = self._format_invariant_violations(invariant_result)

        if validation.valid and not invariant_result.has_errors:
            self.console.print("[green]Code validation passed[/green]")
            if invariant_result.violations:
                # Only warnings, show them
                for v in invariant_result.violations:
                    self.console.print(f"  [yellow][WARN][/yellow] {v.pattern_message}")
            self._send_ws_update("fix_loop", status="passed", message="Code validation passed")
            return True

        # Report syntax errors
        total_errors = len(validation.errors) + len(invariant_errors)
        self.console.print(f"[yellow]Found {total_errors} issues ({len(validation.errors)} syntax, {len(invariant_errors)} invariant)[/yellow]")
        self._send_ws_update(
            "fix_loop",
            status="errors_found",
            error_count=total_errors,
            syntax_errors=len(validation.errors),
            invariant_errors=len(invariant_errors),
            errors=validation.errors[:3] + invariant_errors[:2],
            message=f"Found {total_errors} issues"
        )
        for error in validation.errors[:3]:
            self.console.print(f"[dim]  [syntax] {error}[/dim]")
        if len(validation.errors) > 3:
            self.console.print(f"[dim]  ... and {len(validation.errors) - 3} more syntax errors[/dim]")

        # Report invariant violations
        for error in invariant_errors[:3]:
            self.console.print(f"[red]  [invariant] {error}[/red]")
        if len(invariant_errors) > 3:
            self.console.print(f"[dim]  ... and {len(invariant_errors) - 3} more invariant violations[/dim]")

        # Track invariant violations for no-progress detection
        current_violations = {v.pattern_message for v in invariant_result.violations if v.severity == "error"}
        if current_violations and current_violations == previous_violations:
            self.console.print("[yellow]Same invariant violations as last iteration - may not be fixable[/yellow]")
        previous_violations = current_violations

        # Check if we have LLM for fixing
        if self.llm is None:
            self.console.print("[yellow]Cannot run fix loop: LLM not available[/yellow]")
            self._send_ws_update("fix_loop", status="skipped", message="LLM not available")
            return False

        # Get initial LLM for fix stage (will escalate on retries)
        initial_llm = self._get_llm_for_stage("fix", retry_count=0)
        fix_agent = FixAgent(llm=initial_llm, repo_path=repo_path)

        # Fix loop
        while True:
            should_continue, reason = loop_state.should_continue(validation.errors)

            if not should_continue:
                if "All errors fixed" in reason:
                    self.console.print(f"[green]{reason}[/green]")
                    self._send_ws_update("fix_loop", status="success", message=reason)
                    return True
                else:
                    self.console.print(f"[yellow]Fix loop stopped: {reason}[/yellow]")
                    self._send_ws_update(
                        "fix_loop",
                        status="stopped",
                        reason=reason,
                        iterations=loop_state.iteration,
                        remaining_errors=len(validation.errors),
                        message=f"Fix loop stopped: {reason}"
                    )
                    # Save the fix loop report
                    self._save_fix_loop_report(run_dir, loop_state, validation.errors)
                    return False

            # Escalate model on retry (cheap → medium → premium)
            if loop_state.iteration > 0:
                escalated_llm = self._get_llm_for_stage("fix", retry_count=loop_state.iteration)
                fix_agent.llm = escalated_llm

            self.console.print(f"[bold]Fix attempt {loop_state.iteration}/{fix_config.max_iterations}[/bold]")
            self._send_ws_update(
                "fix_loop",
                status="fixing",
                iteration=loop_state.iteration,
                max_iterations=fix_config.max_iterations,
                error_count=len(validation.errors),
                message=f"Fix attempt {loop_state.iteration}/{fix_config.max_iterations}"
            )

            # Call FixAgent with both syntax and invariant errors
            combined_errors = validation.errors + [
                f"[INVARIANT] {e}" for e in invariant_errors
            ]
            fix_input = AgentInput(
                context={
                    "code_files": code_files,
                    "errors": combined_errors,
                    "error_output": validation.error_output,
                    "invariant_violations": [
                        {
                            "message": v.pattern_message,
                            "fix_hint": v.fix_hint,
                            "line": v.line_number,
                            "severity": v.severity,
                        }
                        for v in invariant_result.violations
                        if v.severity == "error"
                    ],
                }
            )

            fix_output = fix_agent.run(fix_input)

            if not fix_output.success:
                self.console.print("[yellow]FixAgent failed to generate fixes[/yellow]")
                self._send_ws_update("fix_loop", status="fix_failed", message="FixAgent failed to generate fixes")
                continue

            # Apply fixes
            fixes = fix_output.data.get("fixes", [])
            if not fixes:
                self.console.print("[yellow]No fixes generated[/yellow]")
                self._send_ws_update("fix_loop", status="no_fixes", message="No fixes generated")
                continue

            self.console.print(f"[dim]Applying {len(fixes)} fixes...[/dim]")
            self._send_ws_update(
                "fix_loop",
                status="applying",
                fix_count=len(fixes),
                message=f"Applying {len(fixes)} fixes..."
            )

            for fix in fixes:
                path = fix.get("path", "")
                fixed_code = fix.get("fixed_code", "")
                if path and fixed_code:
                    code_files[path] = fixed_code
                    error_fixed = fix.get("error_fixed", "unknown")
                    self.console.print(f"[dim]  Fixed: {path} - {error_fixed[:50]}[/dim]")

            # Re-validate syntax
            validation = validator.validate_files(code_files)

            # Re-verify invariants
            all_code = "\n".join(f"# File: {p}\n{c}" for p, c in code_files.items())
            invariant_result = invariants_verifier.verify(all_code, context=state.feature_description)
            invariant_errors = self._format_invariant_violations(invariant_result)

            # Check for no-progress on invariants
            current_violations = {v.pattern_message for v in invariant_result.violations if v.severity == "error"}
            same_violations = current_violations and current_violations == previous_violations
            previous_violations = current_violations

            if validation.valid and not invariant_result.has_errors:
                self.console.print("[green]All errors fixed![/green]")
                self._send_ws_update(
                    "fix_loop",
                    status="success",
                    iterations=loop_state.iteration,
                    message="All errors fixed!"
                )
                # Update the artifacts with fixed code
                self._update_implementation_with_fixes(run_dir, code_files, loop_state)
                return True

            # Check if invariant violations are stuck (same as previous iteration)
            if same_violations and validation.valid:
                self.console.print(f"[yellow]Syntax fixed but invariant violations unchanged - stopping[/yellow]")
                self._send_ws_update(
                    "fix_loop",
                    status="invariants_stuck",
                    iterations=loop_state.iteration,
                    invariant_errors=len(invariant_errors),
                    message="Invariant violations not fixable by LLM"
                )
                # Still save the syntax-fixed code
                self._update_implementation_with_fixes(run_dir, code_files, loop_state)
                return False

            total_remaining = len(validation.errors) + len(invariant_errors)
            self.console.print(f"[yellow]Still {total_remaining} issues remaining ({len(validation.errors)} syntax, {len(invariant_errors)} invariant)[/yellow]")
            self._send_ws_update(
                "fix_loop",
                status="revalidating",
                error_count=total_remaining,
                syntax_errors=len(validation.errors),
                invariant_errors=len(invariant_errors),
                message=f"Still {total_remaining} issues remaining"
            )

        return False

    def _format_invariant_violations(self, result) -> list[str]:
        """Format invariant violations as error strings for the fix loop.

        Args:
            result: VerificationResult from InvariantsVerifierAgent

        Returns:
            List of error strings (only errors, not warnings)
        """
        errors = []
        for v in result.violations:
            if v.severity == "error":
                msg = v.pattern_message
                if v.fix_hint:
                    msg += f" (Fix: {v.fix_hint})"
                errors.append(msg)
        return errors

    def _extract_project_files(
        self,
        run_dir: Path,
        raw_changes: list[dict],
        state: PipelineState | None = None,
    ) -> Path:
        """Extract generated code files to a project directory.

        For iteration runs, copies base project first then applies changes.

        Args:
            run_dir: Run artifacts directory
            raw_changes: List of file changes with path and new_code
            state: Pipeline state (to check for iteration context)

        Returns:
            Path to the generated project directory
        """
        import shutil

        project_dir = run_dir / "generated_project"
        project_dir.mkdir(exist_ok=True)
        logger.info(f"EXTRACT: Created project_dir: {project_dir}")
        logger.info(f"EXTRACT: raw_changes count: {len(raw_changes)}")

        files_created = 0
        files_copied = 0

        # For iteration runs, copy base project files first
        if state and state.iteration_context:
            base_run_id = state.iteration_context.get("base_run_id")
            if base_run_id:
                base_project = self.artifacts_dir / base_run_id / "generated_project"
                if base_project.exists():
                    self.console.print(f"[cyan]Copying base project from {base_run_id}...[/cyan]")
                    # Copy all files from base project
                    for item in base_project.rglob("*"):
                        if item.is_file():
                            # Skip hidden files and __pycache__
                            if any(part.startswith('.') or part == '__pycache__' for part in item.parts):
                                continue
                            rel_path = item.relative_to(base_project)
                            dest_path = project_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_path)
                            files_copied += 1
                    self.console.print(f"[dim]  Copied {files_copied} files from base project[/dim]")

        # Apply raw_changes (new or modified files)
        for change in raw_changes:
            file_path = change.get("path", "")
            code = change.get("new_code", "")

            if not file_path or not code:
                continue

            # Strip leading slashes to prevent absolute path issues
            file_path = file_path.lstrip("/")

            # Create the full path within project directory
            full_path = project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file (overwrites if exists from base)
            full_path.write_text(code)
            self.console.print(f"[dim]  Created: {file_path}[/dim]")
            files_created += 1

        # If no files created from raw_changes, try parsing diff.patch
        if files_created == 0:
            diff_file = run_dir / "diff.patch"
            if diff_file.exists():
                self.console.print("[dim]Applying changes from diff.patch...[/dim]")
                files_created = self._apply_patch_to_project(diff_file, project_dir)

        logger.info(f"EXTRACT: Completed - files_created={files_created}, files_copied={files_copied}")
        logger.info(f"EXTRACT: project_dir exists={project_dir.exists()}, contents={list(project_dir.iterdir()) if project_dir.exists() else []}")
        return project_dir

    def _extract_from_patch(self, patch_file: Path, project_dir: Path) -> int:
        """Extract files from a unified diff patch.

        Args:
            patch_file: Path to the patch file
            project_dir: Directory to write extracted files

        Returns:
            Number of files extracted
        """
        import re

        patch_content = patch_file.read_text()
        files_created = 0

        # Split by file headers
        file_pattern = re.compile(r'\+\+\+ b/(.+?)(?=\n|$)')
        current_file = None
        current_lines = []

        for line in patch_content.split('\n'):
            # Check for new file header
            match = file_pattern.match(line)
            if match or line.startswith('+++ b/'):
                # Save previous file if any
                if current_file and current_lines:
                    self._write_extracted_file(project_dir, current_file, current_lines)
                    files_created += 1

                # Start new file
                if match:
                    current_file = match.group(1)
                else:
                    current_file = line[6:]  # Remove '+++ b/'
                current_lines = []
            elif line.startswith('+') and not line.startswith('+++'):
                # This is an added line (remove the + prefix)
                current_lines.append(line[1:])

        # Save last file
        if current_file and current_lines:
            self._write_extracted_file(project_dir, current_file, current_lines)
            files_created += 1

        return files_created

    def _write_extracted_file(self, project_dir: Path, file_path: str, lines: list[str]) -> None:
        """Write extracted file content to project directory."""
        # Strip leading slashes to prevent absolute path issues
        file_path = file_path.lstrip("/")
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text('\n'.join(lines))
        self.console.print(f"[dim]  Extracted: {file_path}[/dim]")

    def _apply_patch_to_project(self, patch_file: Path, project_dir: Path) -> int:
        """Apply unified diff patch to project directory.

        For iterations, properly merges changes with existing files.

        Args:
            patch_file: Path to the patch file
            project_dir: Directory containing project files

        Returns:
            Number of files modified/created
        """
        import re
        import subprocess

        patch_content = patch_file.read_text()
        files_modified = 0

        # Try using system patch command first (most reliable)
        try:
            result = subprocess.run(
                ["patch", "-p1", "--forward", "--no-backup-if-mismatch"],
                cwd=project_dir,
                input=patch_content,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Count files from output
                for line in result.stdout.split('\n'):
                    if 'patching file' in line:
                        files_modified += 1
                        file_name = line.split("patching file")[-1].strip().strip("'")
                        self.console.print(f"[dim]  Patched: {file_name}[/dim]")
                return files_modified
        except FileNotFoundError:
            pass  # patch command not available, fall through

        # Fallback: manual patch application
        # Parse the diff and apply changes
        file_diffs = self._parse_unified_diff(patch_content)

        for file_path, hunks in file_diffs.items():
            full_path = project_dir / file_path

            if full_path.exists():
                # Modify existing file
                original_lines = full_path.read_text().split('\n')
                new_lines = self._apply_hunks(original_lines, hunks)
                full_path.write_text('\n'.join(new_lines))
                self.console.print(f"[dim]  Modified: {file_path}[/dim]")
            else:
                # New file - extract added lines
                new_lines = []
                for hunk in hunks:
                    for line in hunk['lines']:
                        if line.startswith('+') and not line.startswith('+++'):
                            new_lines.append(line[1:])
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text('\n'.join(new_lines))
                self.console.print(f"[dim]  Created: {file_path}[/dim]")

            files_modified += 1

        return files_modified

    def _parse_unified_diff(self, patch_content: str) -> dict[str, list[dict]]:
        """Parse unified diff into structured format.

        Returns:
            Dict mapping file paths to list of hunks
        """
        import re

        files: dict[str, list[dict]] = {}
        current_file = None
        current_hunk = None

        for line in patch_content.split('\n'):
            # New file header
            if line.startswith('+++ b/'):
                current_file = line[6:]
                files[current_file] = []
            elif line.startswith('+++ '):
                current_file = line[4:].lstrip('b/')
                files[current_file] = []
            # Hunk header
            elif line.startswith('@@'):
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match and current_file:
                    current_hunk = {
                        'old_start': int(match.group(1)),
                        'old_count': int(match.group(2) or 1),
                        'new_start': int(match.group(3)),
                        'new_count': int(match.group(4) or 1),
                        'lines': []
                    }
                    files[current_file].append(current_hunk)
            # Diff content
            elif current_hunk is not None and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                current_hunk['lines'].append(line)

        return files

    def _apply_hunks(self, original_lines: list[str], hunks: list[dict]) -> list[str]:
        """Apply diff hunks to original file content.

        Args:
            original_lines: Original file lines
            hunks: List of diff hunks to apply

        Returns:
            New file lines after applying hunks
        """
        result = original_lines.copy()
        offset = 0  # Track line number offset from insertions/deletions

        for hunk in hunks:
            start = hunk['old_start'] - 1 + offset
            hunk_lines = hunk['lines']

            # Remove old lines and add new ones
            new_content = []
            for line in hunk_lines:
                if line.startswith('+') and not line.startswith('+++'):
                    new_content.append(line[1:])
                elif line.startswith(' '):
                    new_content.append(line[1:])
                # Lines starting with '-' are removed (not added to new_content)

            # Calculate how many lines to remove
            lines_to_remove = sum(1 for l in hunk_lines if l.startswith('-') or l.startswith(' '))

            # Replace the section
            result = result[:start] + new_content + result[start + lines_to_remove:]

            # Update offset
            lines_added = sum(1 for l in hunk_lines if l.startswith('+') and not l.startswith('+++'))
            lines_removed = sum(1 for l in hunk_lines if l.startswith('-') and not l.startswith('---'))
            offset += lines_added - lines_removed

        return result

    def _run_scaffold_agent(self, state: PipelineState, project_dir: Path) -> None:
        """Run scaffold agent to generate deployment-ready files.

        Generates:
        - requirements.txt
        - .env.example
        - Dockerfile (if runtime=docker)
        - docker-compose.yml (if runtime=docker)
        - README.md
        - run.sh

        Args:
            state: Current pipeline state
            project_dir: Path to generated project directory
        """
        self.console.print()
        self.console.print("[bold cyan]Generating deployment files...[/bold cyan]")

        # Get runtime decision from state
        runtime = state.artifacts.get("runtime", "docker")
        self.console.print(f"[dim]Runtime mode: {runtime}[/dim]")

        try:
            scaffold_agent = ProjectScaffoldAgent()

            input_data = AgentInput(
                context={
                    "project_dir": str(project_dir),
                    "feature_name": state.feature_description[:50] if state.feature_description else "Project",
                    "runtime": runtime,  # Pass runtime decision
                }
            )

            output = scaffold_agent.run(input_data)

            if output.success:
                files = output.data.get("files_generated", [])
                self.console.print(f"[green]Generated {len(files)} deployment files:[/green]")
                for f in files:
                    self.console.print(f"[dim]  - {f}[/dim]")

                # Store scaffold info in state
                state.artifacts["scaffold_files"] = files
                state.artifacts["framework"] = output.data.get("framework", "python")

                # Initialize git repository
                self._init_git_repo(state, project_dir)

                self.console.print()
                self.console.print("[bold green]Project is now deployment-ready![/bold green]")
                self.console.print(f"[dim]Run: cd {project_dir} && ./run.sh[/dim]")
            else:
                self.console.print(f"[yellow]Scaffold generation had issues: {output.errors}[/yellow]")

        except Exception as e:
            self.console.print(f"[yellow]Could not generate scaffold: {e}[/yellow]")

    def _init_git_repo(
        self,
        state: PipelineState,
        project_dir: Path,
        remote_url: str | None = None,
    ) -> None:
        """Initialize git repository for the generated project.

        For new projects: initializes repo and commits on main branch.
        For iterations: creates iteration branch and commits changes.
        If auto_push enabled: pushes to remote.

        Args:
            state: Current pipeline state
            project_dir: Path to generated project directory
            remote_url: Optional remote URL override
        """
        from tools.git_manager import GitManager, commit_iteration, init_project_repo

        try:
            is_iteration = state.iteration_context is not None
            git = GitManager(project_dir)

            if is_iteration:
                # Iteration: create branch and commit
                improvement = state.iteration_context.get("improvement_request", "")
                base_run_id = state.iteration_context.get("base_run_id")

                result = commit_iteration(
                    project_dir=project_dir,
                    run_id=state.run_id,
                    improvement_description=improvement,
                    base_run_id=base_run_id,
                )

                if result.success:
                    self.console.print(f"[green]Git:[/green] {result.message}")
                    branch = git.get_current_branch()
                    state.artifacts["git_branch"] = branch
                    self.console.print(f"[dim]  Branch: {branch}[/dim]")
                else:
                    self.console.print(f"[yellow]Git: {result.message}[/yellow]")
                    return
            else:
                # New project: initialize repo
                result = init_project_repo(
                    project_dir=project_dir,
                    run_id=state.run_id,
                    feature_description=state.feature_description,
                )

                if result.success:
                    self.console.print(f"[green]Git:[/green] {result.message}")
                    state.artifacts["git_branch"] = "main"
                    state.artifacts["git_initialized"] = True
                else:
                    self.console.print(f"[yellow]Git: {result.message}[/yellow]")
                    return

            # Auto-push if enabled
            self._auto_push_if_enabled(git, state, remote_url)

        except Exception as e:
            self.console.print(f"[yellow]Git initialization skipped: {e}[/yellow]")

    def _auto_push_if_enabled(
        self,
        git: "GitManager",
        state: PipelineState,
        remote_url: str | None = None,
    ) -> None:
        """Push to remote if auto_push is enabled.

        Args:
            git: GitManager instance
            state: Current pipeline state
            remote_url: Optional remote URL override
        """
        from tools.git_manager import GitManager

        # Get remote URL from config or parameter
        url = remote_url or self.config.git.remote_url

        if not self.config.git.auto_push and not url:
            return  # Auto-push not enabled and no URL provided

        if not url:
            self.console.print("[dim]Git: auto_push enabled but no remote_url configured[/dim]")
            return

        # Add or update remote
        result = git.add_remote(url, "origin")
        if not result.success:
            self.console.print(f"[yellow]Git remote: {result.message}[/yellow]")
            return

        self.console.print(f"[dim]Git: Remote set to {url}[/dim]")

        # Sync and push (handles rebase if remote has changes)
        branch = git.get_current_branch()
        self.console.print(f"[dim]Git: Pushing {branch} to origin...[/dim]")

        push_result = git.sync_and_push("origin", branch)

        if push_result.success:
            self.console.print(f"[green]Git:[/green] {push_result.message}")
            state.artifacts["git_remote"] = url
            state.artifacts["git_pushed"] = True
        else:
            self.console.print(f"[yellow]Git push: {push_result.message}[/yellow]")
            if "conflict" in push_result.message.lower():
                self.console.print("[yellow]  Manual conflict resolution required[/yellow]")
                self.console.print(f"[dim]  cd {git.project_dir} && git status[/dim]")

    def _save_fix_loop_report(
        self,
        run_dir: Path,
        loop_state: FixLoopState,
        remaining_errors: list[str],
    ) -> None:
        """Save a report of the fix loop attempts."""
        report = {
            "iterations": loop_state.iteration,
            "max_iterations": loop_state.max_iterations,
            "error_counts": loop_state.error_counts,
            "remaining_errors": remaining_errors,
            "status": "incomplete",
        }

        report_file = run_dir / "fix_loop_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        self.console.print(f"[dim]Created: {report_file}[/dim]")

    def _update_implementation_with_fixes(
        self,
        run_dir: Path,
        fixed_code: dict[str, str],
        loop_state: FixLoopState,
    ) -> None:
        """Update implementation artifacts with fixed code."""
        # Save fixed code files
        fixed_dir = run_dir / "fixed_code"
        fixed_dir.mkdir(exist_ok=True)

        for path, code in fixed_code.items():
            # Create subdirectories if needed
            file_path = fixed_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code)

        # Update the diff.patch with fixed code
        from agents import ImplementationAgent

        combined_patch_parts = []
        for path, code in fixed_code.items():
            lines = [
                f"--- /dev/null",
                f"+++ b/{path}",
                f"@@ -0,0 +1,{len(code.splitlines())} @@",
            ]
            for line in code.splitlines():
                lines.append(f"+{line}")
            combined_patch_parts.append("\n".join(lines))

        diff_file = run_dir / "diff.patch"
        diff_file.write_text("\n".join(combined_patch_parts))

        # Save fix loop summary
        summary = {
            "iterations": loop_state.iteration,
            "status": "success",
            "files_fixed": list(fixed_code.keys()),
        }
        summary_file = run_dir / "fix_loop_report.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        self.console.print(f"[dim]Created: {fixed_dir}[/dim]")

    def _handle_testing(self, state: PipelineState) -> tuple[bool, str | None]:
        """Run tests and quality checks using TestAgent."""
        run_dir = Path(state.artifacts_dir)

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        # Check if we have LLM available
        if self.llm is None:
            self.console.print("[yellow]TestAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating placeholder test report...[/dim]")

            report_file = run_dir / "test_report.md"
            report_file.write_text(f"""# Test Report

## Feature: {state.feature_description}

## Results

*LLM not available - no analysis performed*

## Summary

- Total: 0
- Passed: 0
- Failed: 0
""")
            state.artifacts["test_report"] = str(report_file)
            self.console.print(f"[dim]Created: {report_file}[/dim]")
            return True, None

        # Initialize TestAgent with repo path
        self.test_agent = TestAgent(llm=self.llm, repo_path=repo_path)

        self.console.print("[bold]Test Agent:[/bold] Running tests and quality checks...")

        # Load feature spec for feature_id
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        # Prepare input for TestAgent
        input_data = AgentInput(
            context={
                "feature_id": feature_spec_data.get("id", f"FEAT-{state.run_id}"),
                "feature_description": state.feature_description,
            }
        )

        # Run the agent
        output = self.test_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[red]TestAgent failed: {error_msg}[/red]")
            return False, f"TestAgent failed: {error_msg}"

        # Save test report as JSON
        report_json_file = run_dir / "test_report.json"
        report_json_file.write_text(json.dumps(output.data, indent=2))
        state.artifacts["test_report_json"] = str(report_json_file)

        # Generate markdown report
        from schemas.test_report import TestReport
        try:
            test_report = TestReport(**output.data)
            markdown_report = self.test_agent.generate_markdown_report(test_report)
        except Exception:
            # Fallback to simple markdown
            test_results = output.data.get("test_results", {})
            markdown_report = f"""# Test Report

## Feature: {state.feature_description}

## Results

- Total: {test_results.get('total', 0)}
- Passed: {test_results.get('passed', 0)}
- Failed: {test_results.get('failed', 0)}
- Skipped: {test_results.get('skipped', 0)}
- Errors: {test_results.get('errors', 0)}

## Quality Score

{output.data.get('quality_score', 'N/A')}/100

## Ready for Deploy

{output.data.get('ready_for_deploy', False)}
"""

        report_md_file = run_dir / "test_report.md"
        report_md_file.write_text(markdown_report)
        state.artifacts["test_report"] = str(report_md_file)

        # Show summary
        data = output.data
        test_results = data.get("test_results", {})

        total = test_results.get("total", 0)
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        errors = test_results.get("errors", 0)

        if total > 0:
            self.console.print(f"[green]Tests:[/green] {passed}/{total} passed")
        else:
            self.console.print("[yellow]Tests:[/yellow] No tests found")

        if failed > 0:
            self.console.print(f"[red]Failed:[/red] {failed} test(s)")
        if errors > 0:
            self.console.print(f"[red]Errors:[/red] {errors} test(s)")

        lint_count = len(data.get("lint_issues", []))
        if lint_count > 0:
            self.console.print(f"[yellow]Lint Issues:[/yellow] {lint_count}")

        type_count = len(data.get("type_issues", []))
        if type_count > 0:
            self.console.print(f"[yellow]Type Errors:[/yellow] {type_count}")

        quality_score = data.get("quality_score")
        if quality_score is not None:
            color = "green" if quality_score >= 80 else "yellow" if quality_score >= 60 else "red"
            self.console.print(f"[{color}]Quality Score:[/{color}] {quality_score:.1f}/100")

        ready = data.get("ready_for_deploy", False)
        if ready:
            self.console.print("[green]Ready for Deploy:[/green] Yes")
        else:
            self.console.print("[yellow]Ready for Deploy:[/yellow] No")
            blocking = data.get("blocking_issues", [])
            for issue in blocking[:3]:
                self.console.print(f"  - {issue}")

        self.console.print(f"[dim]Created: {report_md_file}[/dim]")
        self.console.print(f"[dim]Created: {report_json_file}[/dim]")
        return True, None

    def _handle_coverage(self, state: PipelineState) -> tuple[bool, str | None]:
        """Check test coverage using CoverageAgent."""
        run_dir = Path(state.artifacts_dir)
        coverage_file = run_dir / "coverage_report.json"

        # Skip if coverage is disabled
        if not self._coverage_enabled:
            self.console.print("[dim]Coverage check skipped (use --coverage to enable)[/dim]")
            return True, None

        # Check if we have the agent available
        if self.coverage_agent is None:
            self.console.print("[yellow]CoverageAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        # Get list of changed files from implementation
        changed_files = []
        diff_file = run_dir / "diff.patch"
        if diff_file.exists():
            # Parse diff to get changed files
            diff_content = diff_file.read_text()
            import re
            for match in re.finditer(r"^\+\+\+ [ab]/(.+)$", diff_content, re.MULTILINE):
                changed_files.append(match.group(1))

        # Load previous coverage report if this is an iteration
        previous_report = None
        iteration = 1
        if coverage_file.exists():
            with open(coverage_file) as f:
                previous_report = json.load(f)
                iteration = previous_report.get("iteration", 1) + 1

        self.console.print(f"[bold]Coverage Agent:[/bold] Checking test coverage (iteration {iteration})...")

        # Update threshold from config
        from schemas.coverage_report import CoverageThreshold
        self.coverage_agent.threshold = CoverageThreshold(
            overall_min=self._coverage_threshold,
            diff_min=self._coverage_threshold,
        )

        input_data = AgentInput(
            context={
                "repo_path": str(repo_path),
                "changed_files": changed_files,
                "iteration": iteration,
                "previous_report": previous_report,
            }
        )

        output = self.coverage_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Coverage check failed: {error_msg}[/yellow]")
            self.console.print("[dim]Continuing without coverage enforcement...[/dim]")
            return True, None  # Don't fail the pipeline

        # Save coverage report
        coverage_file.write_text(json.dumps(output.data, indent=2))
        state.artifacts["coverage_report"] = str(coverage_file)

        # Display summary
        status = output.data.get("status", "unknown")
        overall = output.data.get("overall_coverage", 0)
        summary = output.data.get("summary", "")

        if status == "passing":
            self.console.print(f"[green]{summary}[/green]")
        elif status == "warning":
            self.console.print(f"[yellow]{summary}[/yellow]")
        else:
            self.console.print(f"[red]{summary}[/red]")

        # Show recommendations if any
        recommendations = output.data.get("recommendations", [])
        if recommendations:
            self.console.print("[dim]Recommendations:[/dim]")
            for rec in recommendations[:3]:
                self.console.print(f"  - {rec}")

        # Check if we should iterate (generate more tests)
        from schemas.coverage_report import CoverageReport, CoverageStatus
        report = CoverageReport(**output.data)

        if self.coverage_agent.should_iterate(report):
            test_requests = output.data.get("test_requests", [])
            if test_requests:
                self.console.print(
                    f"[yellow]Coverage below {self._coverage_threshold}%, "
                    f"{len(test_requests)} files need more tests[/yellow]"
                )
                # Could trigger TestGeneratorAgent here in future
                # For now, just log the recommendation

        return True, None

    def _handle_secrets_scan(self, state: PipelineState) -> tuple[bool, str | None]:
        """Scan for hardcoded secrets using SecretsScanAgent."""
        run_dir = Path(state.artifacts_dir)
        secrets_file = run_dir / "secrets_report.json"

        # Skip if secrets scan is disabled
        if not self._secrets_scan_enabled:
            self.console.print("[dim]Secrets scan skipped (use --secrets-scan to enable)[/dim]")
            return True, None

        # Check if we have the agent available
        if self.secrets_scan_agent is None:
            self.console.print("[yellow]SecretsScanAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        self.console.print(f"[bold]Secrets Scan Agent:[/bold] Scanning {repo_path} for hardcoded secrets...")

        input_data = AgentInput(
            context={
                "repo_path": str(repo_path),
                "scan_phase": "pre_docker",
                "auto_fix": True,
            }
        )

        output = self.secrets_scan_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Secrets scan failed: {error_msg}[/yellow]")
            self.console.print("[dim]Continuing without secrets scan...[/dim]")
            return True, None  # Don't fail the pipeline

        # Save secrets report
        secrets_file.write_text(json.dumps(output.data, indent=2))
        state.artifacts["secrets_report"] = str(secrets_file)

        # Display summary
        total = output.data.get("total_detected", 0)
        auto_fixed = output.data.get("auto_fixed_count", 0)
        manual = output.data.get("manual_required_count", 0)
        blocked = output.data.get("blocked", False)
        summary = output.data.get("summary", "")

        if total == 0:
            self.console.print("[green]No secrets detected[/green]")
        elif blocked:
            self.console.print(f"[red]{summary}[/red]")
            block_reason = output.data.get("block_reason", "Unresolved secrets")
            self.console.print(f"[red]BLOCKED: {block_reason}[/red]")
            return False, f"Secrets scan blocked: {block_reason}"
        else:
            if auto_fixed > 0:
                self.console.print(f"[yellow]{summary}[/yellow]")
                self.console.print(f"[green]{auto_fixed} secret(s) auto-fixed[/green]")
            if manual > 0:
                self.console.print(f"[yellow]{manual} secret(s) need manual review[/yellow]")

        # Show what was updated
        if output.data.get("env_example_updated"):
            self.console.print("[dim].env.example updated with new variables[/dim]")
        if output.data.get("readme_updated"):
            self.console.print("[dim]README updated with configuration instructions[/dim]")

        return True, None

    def _handle_dependency_audit(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Audit dependencies for CVEs and outdated packages."""
        run_dir = Path(state.artifacts_dir)
        audit_file = run_dir / "dependency_audit_report.json"

        # Skip if dependency audit is disabled
        if not self._dependency_audit_enabled:
            self.console.print(
                "[dim]Dependency audit skipped (use --dependency-audit to enable)[/dim]"
            )
            return True, None

        # Check if we have the agent available
        if self.dependency_audit_agent is None:
            self.console.print("[yellow]DependencyAuditAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        self.console.print(
            f"[bold]Dependency Audit Agent:[/bold] Scanning {repo_path} for vulnerabilities..."
        )

        input_data = AgentInput(
            repo_path=str(repo_path),
            feature_description=state.feature_description,
            context={
                "auto_patch": True,
                "run_tests_after_patch": True,
                "block_on_critical": True,
            },
        )

        output = self.dependency_audit_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[red]Dependency audit failed: {error_msg}[/red]")
            return False, f"Dependency audit failed: {error_msg}"

        # Save audit report
        audit_file.write_text(json.dumps(output.result, indent=2))
        state.artifacts["dependency_audit_report"] = str(audit_file)

        # Display summary
        result = output.result
        status = result.get("status", "unknown")
        critical = result.get("critical_count", 0)
        high = result.get("high_count", 0)
        medium = result.get("medium_count", 0)
        deprecated = result.get("deprecated_count", 0)
        outdated = result.get("outdated_count", 0)
        bumps_applied = result.get("bumps_applied", 0)
        blocked = result.get("blocked", False)

        # Show vulnerability summary
        if critical == 0 and high == 0 and medium == 0:
            self.console.print("[green]No vulnerabilities detected[/green]")
        else:
            self.console.print(
                f"[yellow]Vulnerabilities: {critical} critical, "
                f"{high} high, {medium} medium[/yellow]"
            )

        # Show deprecated/outdated counts
        if deprecated > 0:
            self.console.print(f"[yellow]{deprecated} deprecated package(s)[/yellow]")
        if outdated > 0:
            self.console.print(f"[dim]{outdated} outdated package(s)[/dim]")

        # Show patches applied
        if bumps_applied > 0:
            self.console.print(
                f"[green]{bumps_applied} version bump(s) applied[/green]"
            )

            # Show test result if tests were run
            test_result = result.get("test_result")
            if test_result:
                if test_result.get("passed"):
                    self.console.print("[green]Post-patch tests passed[/green]")
                else:
                    self.console.print("[yellow]Post-patch tests failed[/yellow]")
                    if result.get("rollback_applied"):
                        self.console.print("[yellow]Changes rolled back[/yellow]")

        # Show LLM recommendations
        recommendations = result.get("llm_recommendations", [])
        if recommendations:
            self.console.print("\n[bold]Recommendations:[/bold]")
            for rec in recommendations[:3]:
                self.console.print(f"  • {rec}")

        # Check if pipeline should be blocked
        if blocked:
            block_reason = result.get("block_reason", "Critical vulnerabilities found")
            self.console.print(f"\n[red]BLOCKED: {block_reason}[/red]")
            return False, f"Dependency audit blocked: {block_reason}"

        return True, None

    def _handle_docker_build(self, state: PipelineState) -> tuple[bool, str | None]:
        """Build and validate code in Docker container using DockerAgent."""
        run_dir = Path(state.artifacts_dir)

        # Check runtime decision - skip Docker if not needed
        runtime = state.artifacts.get("runtime", "docker")
        if runtime != "docker":
            self.console.print(f"[dim]Docker build skipped: runtime is '{runtime}'[/dim]")
            reason = state.artifacts.get("runtime_reason", "Non-Docker runtime selected")

            # Create skip report
            report_file = run_dir / "docker_validation.md"
            report_file.write_text(f"""# Docker Validation Report

## Feature: {state.feature_description}

## Status

**Skipped** - Runtime decision: {runtime}

Reason: {reason}

Docker is not required for this project type.
""")
            state.artifacts["docker_validation"] = str(report_file)
            return True, None

        # Determine repo path
        repo_path = Path.cwd()
        spec_file = run_dir / "spec.json"
        if spec_file.exists():
            with open(spec_file) as f:
                spec_data = json.load(f)
                if "repo_path" in spec_data:
                    repo_path = Path(spec_data["repo_path"])

        # Check if Dockerfile exists - first in generated_project, then in repo
        generated_project = run_dir / "generated_project"
        dockerfile_path = None

        # Priority 1: Generated project (created by scaffold agent)
        if (generated_project / "Dockerfile").exists():
            dockerfile_path = generated_project / "Dockerfile"
            # Use generated_project as the build context
            repo_path = generated_project
            self.console.print(f"[dim]Using generated Dockerfile from {generated_project}[/dim]")
        # Priority 2: Original repository
        elif (repo_path / "Dockerfile").exists():
            dockerfile_path = repo_path / "Dockerfile"
        else:
            self.console.print("[yellow]Docker validation skipped: No Dockerfile found[/yellow]")
            self.console.print("[dim]Add a Dockerfile to enable container validation[/dim]")

            # Create a skip report
            report_file = run_dir / "docker_validation.md"
            report_file.write_text(f"""# Docker Validation Report

## Feature: {state.feature_description}

## Status

**Skipped** - No Dockerfile found in repository or generated project.

To enable Docker validation, add a Dockerfile to your project root.
""")
            state.artifacts["docker_validation"] = str(report_file)
            return True, None  # Not a failure, just skipped

        # Check if we have LLM available
        if self.llm is None:
            self.console.print("[yellow]DockerAgent not available (LLM backend not initialized)[/yellow]")
            # Still try to run Docker validation without LLM analysis
            pass

        # Initialize DockerAgent
        self.docker_agent = DockerAgent(llm=self.llm, repo_path=repo_path)

        self.console.print("[bold]Docker Agent:[/bold] Validating code in container...")

        # Load feature spec for feature_id
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        feature_id = feature_spec_data.get("id", f"FEAT-{state.run_id}")

        # Check for .env file to pass secrets at runtime
        env_file = None
        env_path = repo_path / ".env"
        if env_path.exists():
            env_file = str(env_path)

        # Prepare input for DockerAgent
        # Secrets (DATABASE_URL, API_KEY, etc.) are passed via env_file
        # They are injected at runtime, never baked into the image
        # This is the same pattern Render.com uses
        input_data = AgentInput(
            context={
                "feature_id": feature_id,
                "image_tag": f"coding-factory:{state.run_id}",
                "container_name": f"cf-{state.run_id}",
                "port": 8000,
                "health_endpoint": "/health",
                "run_tests": True,
                "env_file": env_file,  # Runtime secrets from .env
                # API probes from acceptance criteria
                "feature_spec_path": str(feature_spec_file) if feature_spec_file.exists() else None,
                "run_api_probes": True,
                # Integration tests
                "run_integration_tests": True,
                "integration_test_path": "tests/integration",
            }
        )

        # Run the agent
        output = self.docker_agent.run(input_data)

        # Save validation report as JSON
        report_json_file = run_dir / "docker_validation.json"
        report_json_file.write_text(json.dumps(output.data, indent=2))
        state.artifacts["docker_validation_json"] = str(report_json_file)

        # Generate markdown report
        markdown_report = self.docker_agent.generate_report(output.data)
        report_md_file = run_dir / "docker_validation.md"
        report_md_file.write_text(markdown_report)
        state.artifacts["docker_validation"] = str(report_md_file)

        # Show summary
        data = output.data
        stages = data.get("stages", {})

        # Build status
        build_stage = stages.get("build", {})
        if build_stage.get("success"):
            self.console.print(f"[green]Build:[/green] Image built successfully")
        elif build_stage:
            self.console.print(f"[red]Build:[/red] Failed - {build_stage.get('error', 'Unknown error')[:50]}")

        # Run status
        run_stage = stages.get("run", {})
        if run_stage.get("success"):
            self.console.print(f"[green]Container:[/green] Started ({data.get('container_id', 'N/A')})")
        elif run_stage:
            self.console.print(f"[red]Container:[/red] Failed to start")

        # Ready status
        ready_stage = stages.get("ready", {})
        if ready_stage.get("success"):
            attempts = ready_stage.get("attempts", 1)
            self.console.print(f"[green]Ready:[/green] Service ready after {attempts} attempt(s)")
        elif ready_stage:
            self.console.print(f"[red]Ready:[/red] Service did not become ready")

        # Health status
        health_stage = stages.get("health", {})
        if health_stage.get("success"):
            self.console.print(f"[green]Health:[/green] Health check passed")
        elif health_stage:
            self.console.print(f"[red]Health:[/red] Health check failed")

        # API probes status
        probe_stage = stages.get("api_probes", {})
        if probe_stage:
            total = probe_stage.get("total", 0)
            passed = probe_stage.get("passed", 0)
            failed = probe_stage.get("failed", 0)
            if total == 0:
                self.console.print(f"[yellow]API Probes:[/yellow] No testable acceptance criteria")
            elif probe_stage.get("success"):
                self.console.print(f"[green]API Probes:[/green] {passed}/{total} criteria passed")
            else:
                self.console.print(f"[red]API Probes:[/red] {failed}/{total} criteria failed")

        # Integration tests status
        int_stage = stages.get("integration_tests", {})
        if int_stage:
            if int_stage.get("skipped"):
                self.console.print(f"[yellow]Integration:[/yellow] Skipped ({int_stage.get('reason', 'N/A')})")
            elif int_stage.get("success"):
                passed = int_stage.get("passed", 0)
                self.console.print(f"[green]Integration:[/green] {passed} tests passed")
            else:
                self.console.print(f"[red]Integration:[/red] Tests failed")

        # Test status
        test_stage = stages.get("tests", {})
        if test_stage.get("skipped"):
            self.console.print(f"[yellow]Tests:[/yellow] Skipped ({test_stage.get('reason', 'N/A')})")
        elif test_stage.get("success"):
            passed = test_stage.get("passed", 0)
            failed = test_stage.get("failed", 0)
            self.console.print(f"[green]Tests:[/green] {passed} passed, {failed} failed")
        elif test_stage:
            self.console.print(f"[red]Tests:[/red] Failed in container")

        # Overall result
        if output.success:
            self.console.print("[green]Docker Validation:[/green] PASSED ✓")
        else:
            self.console.print("[red]Docker Validation:[/red] FAILED ✗")
            for error in data.get("errors", [])[:3]:
                self.console.print(f"  - {error}")

        self.console.print(f"[dim]Created: {report_md_file}[/dim]")
        self.console.print(f"[dim]Created: {report_json_file}[/dim]")

        # Docker validation failure shouldn't block pipeline, but report it
        if not output.success:
            self.console.print("[yellow]Note: Docker validation failed but pipeline continues[/yellow]")

        return True, None

    def _handle_rollback_strategy(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Generate CI/CD rollback and canary deployment strategies."""
        if not self._rollback_strategy_enabled:
            self.console.print(
                "[dim]Rollback strategy skipped (use --rollback-strategy to enable)[/dim]"
            )
            return True, None

        run_dir = Path(state.artifacts_dir)
        project_dir = run_dir / "generated_project"

        if not project_dir.exists():
            self.console.print(
                "[yellow]No generated_project found, skipping rollback strategy[/yellow]"
            )
            return True, None

        self.console.print(
            "[bold]Rollback Strategy Agent:[/bold] Generating deployment strategies..."
        )

        from agents.rollback_strategy_agent import RollbackStrategyAgent
        from schemas.rollback_strategy import DeploymentPattern

        # Initialize agent
        rollback_agent = RollbackStrategyAgent(llm=self.llm)

        # Detect deployment pattern from feature description
        feature_lower = state.feature_description.lower()
        if "canary" in feature_lower:
            pattern = DeploymentPattern.CANARY
        elif "blue" in feature_lower and "green" in feature_lower:
            pattern = DeploymentPattern.BLUE_GREEN
        else:
            pattern = DeploymentPattern.ROLLING

        # Detect if project has database
        has_database = any(
            project_dir.glob("**/migrations/**/*.py")
        ) or any(
            project_dir.glob("**/alembic/**/*.py")
        )

        # Detect if using Kubernetes
        kubernetes = (
            (project_dir / "k8s").exists()
            or (project_dir / "kubernetes").exists()
            or any(project_dir.glob("**/*.yaml"))
        )

        # Extract service name from feature or use default
        service_name = "app"
        if "service" in feature_lower:
            # Try to extract service name
            words = state.feature_description.split()
            for i, word in enumerate(words):
                if word.lower() == "service" and i > 0:
                    service_name = words[i - 1].lower().replace(",", "")
                    break

        try:
            report = rollback_agent.run(
                repo_path=project_dir,
                service_name=service_name,
                deployment_pattern=pattern,
                has_database=has_database,
                kubernetes=kubernetes,
            )

            # Write artifacts to project
            created_files = rollback_agent.write_artifacts(report, project_dir)

            # Save report
            report_file = run_dir / "rollback_strategy_report.json"
            report_file.write_text(report.model_dump_json(indent=2))
            state.artifacts["rollback_strategy_report"] = str(report_file)

            # Show summary
            self.console.print(
                f"[green]Deployment Pattern:[/green] {report.deployment_pattern.value}"
            )
            self.console.print(
                f"[green]Workflows Generated:[/green] {len(report.workflows)}"
            )
            for wf in report.workflows:
                self.console.print(f"  - {wf.filename}: {wf.description}")

            self.console.print(
                f"[green]Rollback Jobs:[/green] {len(report.rollback_jobs)}"
            )

            if report.playbook:
                self.console.print("[green]Playbook:[/green] docs/ops.md")

            if report.recommendations:
                self.console.print(
                    f"[green]Recommendations:[/green] {len(report.recommendations)}"
                )
                for rec in report.recommendations[:3]:
                    self.console.print(f"  • {rec}")

            self.console.print(f"[dim]Created: {report_file}[/dim]")
            for f in created_files:
                self.console.print(f"[dim]Created: {f}[/dim]")

        except Exception as e:
            self.console.print(f"[red]Rollback strategy generation failed: {e}[/red]")
            return False, f"Rollback strategy generation failed: {e}"

        return True, None

    def _handle_observability(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Inject observability scaffolding (logging, metrics, tracing)."""
        run_dir = Path(state.artifacts_dir)
        observability_file = run_dir / "observability_report.json"

        # Skip if observability is disabled
        if not self._observability_enabled:
            self.console.print(
                "[dim]Observability skipped (use --observability to enable)[/dim]"
            )
            return True, None

        # Check if we have the agent available
        if self.observability_agent is None:
            self.console.print("[yellow]ObservabilityAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        self.console.print(
            f"[bold]Observability Agent:[/bold] Injecting observability into {repo_path}..."
        )

        input_data = AgentInput(
            repo_path=str(repo_path),
            feature_description=state.feature_description,
            context={
                "log_level": "INFO",
                "metrics_enabled": True,
                "metrics_backend": "prometheus",
                "tracing_enabled": True,
                "service_name": repo_path.name,
            },
        )

        output = self.observability_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Observability setup warning: {error_msg}[/yellow]")
            # Don't fail the pipeline, just warn
            return True, None

        # Save report
        observability_file.write_text(json.dumps(output.result, indent=2))
        state.artifacts["observability_report"] = str(observability_file)

        # Display summary
        result = output.result
        framework = result.get("framework", "unknown")
        files_generated = result.get("files_generated", [])
        dependencies = result.get("dependencies_added", [])
        dashboard_file = result.get("dashboard_file")

        self.console.print(f"[green]Framework detected: {framework}[/green]")
        self.console.print(f"[green]{len(files_generated)} file(s) generated:[/green]")

        for f in files_generated:
            self.console.print(f"  • {f.get('path')}: {f.get('description')}")

        if dependencies:
            self.console.print(f"\n[dim]Dependencies to add: {', '.join(dependencies)}[/dim]")

        if dashboard_file:
            self.console.print(f"[dim]Grafana dashboard: {dashboard_file}[/dim]")

        # Show logging config
        logging_config = result.get("logging_config", {})
        if logging_config:
            fmt = logging_config.get("format", "json")
            level = logging_config.get("level", "INFO")
            self.console.print(f"\n[bold]Logging:[/bold] {fmt} format, {level} level")

        # Show metrics config
        metrics_config = result.get("metrics_config", {})
        if metrics_config.get("enabled"):
            backend = metrics_config.get("backend", "prometheus")
            endpoint = metrics_config.get("endpoint", "/metrics")
            self.console.print(f"[bold]Metrics:[/bold] {backend} at {endpoint}")

        return True, None

    def _handle_config(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Enforce 12-factor config layout."""
        run_dir = Path(state.artifacts_dir)
        config_file = run_dir / "config_layout_report.json"

        # Skip if config is disabled
        if not self._config_enabled:
            self.console.print(
                "[dim]Config layout skipped (use --config to enable)[/dim]"
            )
            return True, None

        # Check if we have the agent available
        if self.config_agent is None:
            self.console.print("[yellow]ConfigAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        self.console.print(
            f"[bold]Config Agent:[/bold] Enforcing 12-factor config in {repo_path}..."
        )

        input_data = AgentInput(
            repo_path=str(repo_path),
            feature_description=state.feature_description,
            context={
                "environments": ["development", "staging", "production"],
                "generate_settings": True,
                "generate_env_example": True,
                "generate_config_yaml": True,
                "update_readme": True,
            },
        )

        output = self.config_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Config setup warning: {error_msg}[/yellow]")
            # Don't fail the pipeline, just warn
            return True, None

        # Save report
        config_file.write_text(json.dumps(output.result, indent=2))
        state.artifacts["config_layout_report"] = str(config_file)

        # Display summary
        result = output.result
        total_vars = result.get("total_variables", 0)
        total_secrets = result.get("total_secrets", 0)
        total_flags = result.get("total_feature_flags", 0)
        twelve_factor = result.get("twelve_factor_compliant", False)
        files_generated = result.get("files_generated", [])

        self.console.print(f"[green]Configuration variables: {total_vars}[/green]")
        self.console.print(f"[green]Secrets detected: {total_secrets}[/green]")
        self.console.print(f"[green]Feature flags: {total_flags}[/green]")

        if twelve_factor:
            self.console.print("[green]✓ 12-factor compliant[/green]")
        else:
            self.console.print("[yellow]⚠ Not fully 12-factor compliant[/yellow]")
            notes = result.get("compliance_notes", [])
            for note in notes[:3]:  # Show first 3 notes
                self.console.print(f"  • {note}")

        if files_generated:
            self.console.print(f"\n[dim]{len(files_generated)} file(s) generated:[/dim]")
            for f in files_generated:
                path = f.get("path", "unknown")
                desc = f.get("description", "")
                self.console.print(f"  • {path}: {desc}")

        if result.get("readme_section_added"):
            self.console.print("[dim]README.md updated with configuration docs[/dim]")

        return True, None

    def _handle_docs(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Generate and sync project documentation."""
        run_dir = Path(state.artifacts_dir)
        doc_file = run_dir / "doc_report.json"

        # Skip if docs is disabled
        if not self._docs_enabled:
            self.console.print(
                "[dim]Documentation generation skipped (use --docs to enable)[/dim]"
            )
            return True, None

        # Check if we have the agent available
        if self.doc_agent is None:
            self.console.print("[yellow]DocAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Get correct repo path (generated_project or original repo)
        repo_path = self._get_repo_path(run_dir)

        self.console.print(
            f"[bold]Doc Agent:[/bold] Generating documentation for {repo_path}..."
        )

        input_data = AgentInput(
            repo_path=str(repo_path),
            feature_description=state.feature_description,
            context={
                "generate_docs_folder": True,
                "add_docstrings": True,
                "create_adrs": True,
                "sync_docs": True,
            },
        )

        output = self.doc_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Documentation warning: {error_msg}[/yellow]")
            # Don't fail the pipeline, just warn
            return True, None

        # Save report
        doc_file.write_text(json.dumps(output.result, indent=2))
        state.artifacts["doc_report"] = str(doc_file)

        # Display summary
        result = output.result
        docs_generated = result.get("total_docs_generated", 0)
        docs_updated = result.get("total_docs_updated", 0)
        docstrings_added = result.get("total_docstrings_added", 0)
        coverage = result.get("coverage_after", 0.0)
        adrs_created = result.get("adrs_created", [])
        all_in_sync = result.get("all_in_sync", True)

        self.console.print(f"[green]Documentation files: {docs_generated} new, {docs_updated} updated[/green]")

        if docstrings_added > 0:
            self.console.print(f"[green]Docstrings added: {docstrings_added}[/green]")

        if coverage > 0:
            self.console.print(f"[dim]Docstring coverage: {coverage}%[/dim]")

        if adrs_created:
            self.console.print(f"[green]ADRs created: {len(adrs_created)}[/green]")
            for adr in adrs_created:
                self.console.print(f"  • ADR-{adr.get('number', '?'):03d}: {adr.get('title', 'Untitled')}")

        if all_in_sync:
            self.console.print("[green]✓ All documentation in sync[/green]")
        else:
            sync_checks = result.get("sync_checks", [])
            out_of_sync = [c for c in sync_checks if c.get("status") == "out_of_sync"]
            if out_of_sync:
                self.console.print(f"[yellow]⚠ {len(out_of_sync)} sync issue(s) detected[/yellow]")

        # Show generated files
        docs_list = result.get("docs_generated", [])
        if docs_list:
            self.console.print(f"\n[dim]Files generated:[/dim]")
            for doc in docs_list[:5]:  # Show first 5
                self.console.print(f"  • {doc.get('path')}: {doc.get('description')}")

        return True, None

    def _handle_code_review(
        self, state: PipelineState
    ) -> tuple[bool, str | None]:
        """Perform senior engineer code review."""
        run_dir = Path(state.artifacts_dir)
        review_file = run_dir / "code_review_report.json"

        # Skip if code review is disabled
        if not self._code_review_enabled:
            self.console.print(
                "[dim]Code review skipped (use --code-review to enable)[/dim]"
            )
            return True, None

        # Check if we have the agent available
        if self.code_review_agent is None:
            self.console.print("[yellow]CodeReviewAgent not available[/yellow]")
            return True, None  # Skip gracefully

        # Determine repo path - use generated_project from this run, not cwd
        generated_project = run_dir / "generated_project"
        if generated_project.exists():
            repo_path = generated_project
        else:
            # Fallback: check if we have a spec with repo_path
            spec_file = run_dir / "spec.json"
            if spec_file.exists():
                with open(spec_file) as f:
                    spec_data = json.load(f)
                    repo_path = Path(spec_data.get("repo_path", "."))
            else:
                repo_path = Path.cwd()

        self.console.print(
            f"[bold]Code Review Agent:[/bold] Reviewing code in {repo_path}..."
        )

        input_data = AgentInput(
            repo_path=str(repo_path),
            feature_description=state.feature_description,
            context={
                "llm_review": True,
                "auto_fix": False,  # Don't auto-fix by default
            },
        )

        output = self.code_review_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"
            self.console.print(f"[yellow]Code review warning: {error_msg}[/yellow]")
            return True, None

        # Save report
        review_file.write_text(json.dumps(output.result, indent=2))
        state.artifacts["code_review_report"] = str(review_file)

        # Display summary
        result = output.result
        ship_status = result.get("ship_status", "ship")
        ship_reason = result.get("ship_status_reason", "")
        total_issues = result.get("total_issues", 0)
        critical = result.get("critical_count", 0)
        major = result.get("major_count", 0)
        minor = result.get("minor_count", 0)
        nits = result.get("nit_count", 0)
        files_reviewed = result.get("files_reviewed", 0)

        # Show ship status with color
        status_colors = {
            "ship": "green",
            "ship_with_nits": "yellow",
            "dont_ship": "red",
        }
        color = status_colors.get(ship_status, "white")
        status_display = ship_status.upper().replace("_", " ")

        self.console.print(f"[bold {color}]{status_display}[/bold {color}]: {ship_reason}")
        self.console.print(f"[dim]Reviewed {files_reviewed} files[/dim]")
        self.console.print(
            f"[dim]Issues: {critical} critical, {major} major, "
            f"{minor} minor, {nits} nits[/dim]"
        )

        # Show top concerns if any
        top_concerns = result.get("top_concerns", [])
        if top_concerns:
            self.console.print("\n[bold]Top concerns:[/bold]")
            for concern in top_concerns[:3]:
                self.console.print(f"  • {concern[:100]}")

        # Show auto-fixes if applied
        auto_fixes = result.get("auto_fixes_applied", 0)
        if auto_fixes > 0:
            self.console.print(f"\n[green]Applied {auto_fixes} auto-fix(es)[/green]")

        # Block pipeline if don't ship
        if ship_status == "dont_ship":
            self.console.print(
                "\n[red]Code review blocked deployment. Please address critical issues.[/red]"
            )
            # Don't fail pipeline, but warn strongly
            # Actual blocking should be done at FINAL_APPROVAL

        return True, None

    def _handle_policy(self, state: PipelineState) -> tuple[bool, str | None]:
        """Enforce policy rules before verification using PolicyAgent."""
        if not self._policy_enabled:
            self.console.print("[dim]Policy enforcement skipped (use --policy to enable)[/dim]")
            return True, None

        run_dir = Path(state.artifacts_dir)
        project_dir = run_dir / "generated_project"

        self.console.print("[bold]Policy Agent:[/bold] Evaluating policy rules...")

        from agents.policy_agent import PolicyAgent
        from schemas.policy import (
            CodeReviewResults,
            DiffStats,
            PolicyAction,
            PolicyContext,
            SecurityResults,
            TestResults,
        )

        # Initialize agent with rules path or content
        rules_path = None
        if self._policy_rules_path:
            potential_path = Path(self._policy_rules_path)
            if potential_path.exists():
                # It's a file path
                rules_path = potential_path
            else:
                # It's raw content - write to temp file in run dir
                temp_rules_file = run_dir / "custom_policy_rules.yaml"
                # Check if content is YAML or plain text
                content = self._policy_rules_path
                if not content.strip().startswith("version:") and not content.strip().startswith("rules:"):
                    # Plain text rules - convert to YAML format
                    yaml_rules = self._convert_text_to_yaml_rules(content)
                    temp_rules_file.write_text(yaml_rules)
                else:
                    # Already YAML
                    temp_rules_file.write_text(content)
                rules_path = temp_rules_file
                self.console.print(f"[dim]Using custom policy rules from user input[/dim]")
        elif Path("policy_rules.yaml").exists():
            rules_path = Path("policy_rules.yaml")

        policy_agent = PolicyAgent(llm=self.llm, rules_path=rules_path)
        policy_agent.load_rules()

        # Build context from pipeline artifacts
        # Get changed files
        files_changed = []
        if project_dir.exists():
            # Get all Python files as proxy for "changed" in new project
            files_changed = [
                str(f.relative_to(project_dir))
                for f in project_dir.rglob("*.py")
                if "__pycache__" not in str(f)
            ]

        # Get diff stats
        diff_stats = DiffStats(
            files_changed=len(files_changed),
            lines_added=0,
            lines_deleted=0,
        )

        # Load test results
        test_results = TestResults()
        test_report_file = run_dir / "test_report.json"
        if test_report_file.exists():
            try:
                with open(test_report_file) as f:
                    data = json.load(f)
                test_results = TestResults(
                    passed=data.get("passed", 0) > 0 and data.get("failed", 0) == 0,
                    total=data.get("total", 0),
                    failed=data.get("failed", 0),
                    skipped=data.get("skipped", 0),
                    coverage=data.get("coverage", 0) / 100.0
                    if data.get("coverage", 0) > 1
                    else data.get("coverage", 0),
                )
            except Exception:
                pass

        # Load coverage
        coverage_file = run_dir / "coverage_report.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    data = json.load(f)
                cov = data.get("overall_coverage", 0)
                test_results.coverage = cov / 100.0 if cov > 1 else cov
            except Exception:
                pass

        # Load security results
        security_results = SecurityResults()
        secrets_file = run_dir / "secrets_report.json"
        if secrets_file.exists():
            try:
                with open(secrets_file) as f:
                    data = json.load(f)
                secrets = data.get("detected_secrets", [])
                security_results.has_secrets = len(secrets) > 0
                security_results.secrets_count = len(secrets)
            except Exception:
                pass

        dep_audit_file = run_dir / "dependency_audit_report.json"
        if dep_audit_file.exists():
            try:
                with open(dep_audit_file) as f:
                    data = json.load(f)
                vulns = data.get("vulnerabilities", [])
                security_results.has_vulnerabilities = len(vulns) > 0
                security_results.vulnerability_count = len(vulns)
            except Exception:
                pass

        # Load code review results
        code_review = CodeReviewResults()
        review_file = run_dir / "code_review_report.json"
        if review_file.exists():
            try:
                with open(review_file) as f:
                    data = json.load(f)
                code_review.ship_status = data.get("ship_status", "unknown")
                issues = data.get("issues", [])
                code_review.total_issues = len(issues)
                code_review.critical_issues = sum(
                    1 for i in issues if i.get("severity") == "critical"
                )
                code_review.major_issues = sum(
                    1 for i in issues if i.get("severity") == "major"
                )
            except Exception:
                pass

        # Build context
        context = PolicyContext(
            run_id=state.run_id,
            actor="coding_factory",
            branch="feature",
            base_branch="main",
            files_changed=files_changed,
            diff_stats=diff_stats,
            test_results=test_results,
            security_results=security_results,
            code_review=code_review,
            feature_description=state.feature_description,
        )

        # Evaluate policies
        decision = policy_agent.run(context)

        # Save decision
        decision_file = run_dir / "policy_decision.json"
        decision_file.write_text(decision.model_dump_json(indent=2))
        state.artifacts["policy_decision"] = str(decision_file)

        # Show results
        status_color = {
            PolicyAction.ALLOW: "green",
            PolicyAction.REQUIRE_APPROVAL: "yellow",
            PolicyAction.BLOCK: "red",
        }[decision.status]

        self.console.print(
            f"[{status_color}]Policy Decision:[/{status_color}] {decision.status.value.upper()}"
        )

        self.console.print(f"[dim]{decision.summary}[/dim]")

        if decision.matched_rules:
            self.console.print(f"\n[bold]Matched Rules ({len(decision.matched_rules)}):[/bold]")
            for rule in decision.matched_rules[:5]:
                action_icon = {"allow": "✓", "require_approval": "⚠", "block": "✗"}[
                    rule.action.value
                ]
                self.console.print(f"  {action_icon} {rule.rule_id}: {rule.description}")

        if decision.required_role:
            self.console.print(f"\n[yellow]Required reviewer:[/yellow] {decision.required_role}")

        self.console.print(f"\n[dim]Created: {decision_file}[/dim]")

        # Handle decision
        if decision.status == PolicyAction.BLOCK:
            self.console.print(
                "\n[red bold]Pipeline blocked by policy rules.[/red bold]"
            )
            self.console.print("[red]Fix the issues above before proceeding.[/red]")
            return False, f"Policy blocked: {'; '.join(decision.reasons[:3])}"

        if decision.status == PolicyAction.REQUIRE_APPROVAL:
            self.console.print(
                "\n[yellow]Policy requires human approval before proceeding.[/yellow]"
            )
            # Don't block, but flag for approval at FINAL_APPROVAL
            state.user_decisions["policy_approval_required"] = "true"
            state.user_decisions["policy_required_role"] = decision.required_role or "reviewer"

        return True, None

    def _handle_verification(self, state: PipelineState) -> tuple[bool, str | None]:
        """Verify behavior and generate summary using VerifyAgent."""
        run_dir = Path(state.artifacts_dir)

        # Check if we have LLM available
        if self.llm is None:
            self.console.print("[yellow]VerifyAgent not available (LLM backend not initialized)[/yellow]")
            self.console.print("[dim]Creating placeholder verification report...[/dim]")

            report_file = run_dir / "verification_report.md"
            report_file.write_text(f"""# Verification Report

## Feature: {state.feature_description}

## Summary

*LLM not available - no verification performed*

## Recommendation

needs_review - Manual verification required.
""")
            state.artifacts["verification_report"] = str(report_file)
            self.console.print(f"[dim]Created: {report_file}[/dim]")
            return True, None

        # Initialize VerifyAgent (no base URL for now - container not running)
        self.verify_agent = VerifyAgent(llm=self.llm, base_url=None)

        self.console.print("[bold]Verify Agent:[/bold] Generating verification report...")

        # Load all previous artifacts
        feature_spec_data = {}
        feature_spec_file = run_dir / "feature_spec.json"
        if feature_spec_file.exists():
            with open(feature_spec_file) as f:
                feature_spec_data = json.load(f)

        change_set_data = {}
        change_set_file = run_dir / "change_set.json"
        if change_set_file.exists():
            with open(change_set_file) as f:
                change_set_data = json.load(f)

        impl_notes_data = {}
        impl_notes_file = run_dir / "implementation_notes.json"
        if impl_notes_file.exists():
            with open(impl_notes_file) as f:
                impl_notes_data = json.load(f)

        test_report_data = {}
        test_report_file = run_dir / "test_report.json"
        if test_report_file.exists():
            with open(test_report_file) as f:
                test_report_data = json.load(f)

        # Load requirements tracker
        requirements_tracker_data = {}
        requirements_tracker_file = run_dir / "requirements_tracker.json"
        if requirements_tracker_file.exists():
            with open(requirements_tracker_file) as f:
                requirements_tracker_data = json.load(f)

        # Find project directory for verification
        project_dir = run_dir / "generated_project"
        if not project_dir.exists():
            project_dir = run_dir

        # Prepare input for VerifyAgent
        input_data = AgentInput(
            context={
                "feature_id": feature_spec_data.get("id", f"FEAT-{state.run_id}"),
                "feature_description": state.feature_description,
                "feature_spec": feature_spec_data,
                "change_set": change_set_data,
                "implementation_notes": impl_notes_data,
                "test_report": test_report_data,
                "requirements_tracker": requirements_tracker_data,
                "project_dir": str(project_dir),
            }
        )

        # Run the agent
        output = self.verify_agent.run(input_data)

        if not output.success:
            error_msg = "; ".join(output.errors) if output.errors else "Unknown error"

            # Check if it's a connection error - fall back to placeholder
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                self.console.print(f"[yellow]LLM not available: {error_msg}[/yellow]")
                self.console.print("[dim]Falling back to placeholder verification report...[/dim]")

                report_file = run_dir / "verification_report.md"
                report_file.write_text(f"""# Verification Report

## Feature: {state.feature_description}

## Summary

*LLM connection failed - verification incomplete*

## Recommendation

needs_review - Manual verification required.
""")
                state.artifacts["verification_report"] = str(report_file)
                self.console.print(f"[dim]Created: {report_file}[/dim]")
                return True, None

            self.console.print(f"[red]VerifyAgent failed: {error_msg}[/red]")
            return False, f"VerifyAgent failed: {error_msg}"

        # Save verification report as JSON
        report_json_file = run_dir / "verification_report.json"
        report_json_file.write_text(json.dumps(output.data, indent=2))
        state.artifacts["verification_report_json"] = str(report_json_file)

        # Generate markdown report
        from schemas.verification import VerificationReport
        try:
            verification_report = VerificationReport(**output.data)
            markdown_report = self.verify_agent.generate_markdown_report(verification_report)
        except Exception:
            # Fallback to simple markdown
            markdown_report = f"""# Verification Report

## Feature: {state.feature_description}

## Recommendation

**{output.data.get('recommendation', 'needs_review').upper()}**

{output.data.get('recommendation_rationale', '')}

## Technical Summary

{output.data.get('technical_summary', 'N/A')}

## Behavioral Summary

{output.data.get('behavioral_summary', 'N/A')}
"""

        report_md_file = run_dir / "verification_report.md"
        report_md_file.write_text(markdown_report)
        state.artifacts["verification_report"] = str(report_md_file)

        # Show summary
        data = output.data
        recommendation = data.get("recommendation", "needs_review")

        rec_color = {
            "approve": "green",
            "reject": "red",
            "needs_review": "yellow",
        }.get(recommendation, "white")

        self.console.print(f"[{rec_color}]Recommendation:[/{rec_color}] {recommendation.upper()}")

        rationale = data.get("recommendation_rationale", "")
        if rationale:
            preview = rationale[:100] + "..." if len(rationale) > 100 else rationale
            self.console.print(f"[dim]{preview}[/dim]")

        criteria_summary = data.get("criteria_summary", {})
        if criteria_summary:
            passed = criteria_summary.get("pass", 0)
            failed = criteria_summary.get("fail", 0)
            self.console.print(f"[green]Criteria:[/green] {passed} passed, {failed} failed")

        all_met = data.get("all_criteria_met", False)
        if all_met:
            self.console.print("[green]All acceptance criteria met[/green]")

        # Show requirements verification status
        req_verification = data.get("requirements_verification", {})
        if req_verification:
            total = req_verification.get("total", 0)
            verified = req_verification.get("verified", 0)
            failed = req_verification.get("failed", 0)
            if total > 0:
                self.console.print(f"[bold]Requirements:[/bold] {verified}/{total} verified, {failed} failed")
                unmet = req_verification.get("unmet_requirements", [])
                if unmet:
                    self.console.print("[red]Unmet Requirements:[/red]")
                    for req in unmet[:5]:
                        self.console.print(f"  - {req['id']}: {req['description'][:60]}...")
                        self.console.print(f"    [dim]Reason: {req.get('reason', 'Unknown')}[/dim]")

        risks = data.get("residual_risks", [])
        if risks:
            self.console.print(f"[yellow]Residual Risks:[/yellow] {len(risks)}")
            for risk in risks[:2]:
                self.console.print(f"  - {risk}")

        questions = data.get("open_questions", [])
        if questions:
            self.console.print(f"[yellow]Open Questions:[/yellow] {len(questions)}")

        if data.get("pr_description"):
            self.console.print("[green]PR Description:[/green] Generated")

        if data.get("release_notes"):
            self.console.print("[green]Release Notes:[/green] Generated")

        self.console.print(f"[dim]Created: {report_md_file}[/dim]")
        self.console.print(f"[dim]Created: {report_json_file}[/dim]")

        # Check for unmet requirements and run fix loop if needed
        req_verification = data.get("requirements_verification", {})
        unmet_requirements = req_verification.get("unmet_requirements", [])

        if unmet_requirements:
            self.console.print("")
            self.console.print(f"[yellow]Found {len(unmet_requirements)} unmet requirement(s) - attempting to fix...[/yellow]")

            fix_success = self._run_requirements_fix_loop(
                state=state,
                unmet_requirements=unmet_requirements,
                max_attempts=3,
            )

            if not fix_success:
                self.console.print("[red]Requirements fix loop failed - some requirements remain unmet[/red]")
                # Don't fail the build, but the recommendation will be "reject"
            else:
                self.console.print("[green]All requirements now verified[/green]")

        return True, None

    def _run_requirements_fix_loop(
        self,
        state: PipelineState,
        unmet_requirements: list[dict],
        max_attempts: int = 3,
    ) -> bool:
        """Run fix loop to address unmet requirements.

        Generates targeted fix instructions from unmet requirements,
        calls ImplementationAgent to fix specific files, and re-verifies.

        Args:
            state: Current pipeline state
            unmet_requirements: List of unmet requirement dicts
            max_attempts: Maximum fix attempts

        Returns:
            True if all requirements now pass, False otherwise
        """
        run_dir = Path(state.artifacts_dir)
        project_dir = run_dir / "generated_project"

        if not project_dir.exists():
            project_dir = run_dir

        for attempt in range(max_attempts):
            self.console.print(f"[bold]Requirements Fix Loop:[/bold] Attempt {attempt + 1}/{max_attempts}")
            self._send_ws_update(
                "requirements_fix",
                status="fixing",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                unmet_count=len(unmet_requirements),
            )

            # Generate fix instructions from unmet requirements
            fix_instructions = self._generate_requirements_fix_instructions(unmet_requirements)

            if not fix_instructions:
                self.console.print("[yellow]Could not generate fix instructions[/yellow]")
                return False

            self.console.print(f"[dim]Fix instructions: {fix_instructions[:200]}...[/dim]")

            # Apply fixes using ImplementationAgent
            success = self._apply_requirements_fixes(
                state=state,
                project_dir=project_dir,
                fix_instructions=fix_instructions,
            )

            if not success:
                self.console.print(f"[yellow]Fix attempt {attempt + 1} failed to apply changes[/yellow]")
                continue

            # Re-verify requirements
            self.console.print("[dim]Re-verifying requirements...[/dim]")
            new_unmet = self._reverify_requirements(state, project_dir)

            if not new_unmet:
                self.console.print("[green]All requirements now verified![/green]")
                self._send_ws_update(
                    "requirements_fix",
                    status="success",
                    attempt=attempt + 1,
                    message="All requirements verified",
                )
                return True

            # Check if we made progress
            old_ids = {r["id"] for r in unmet_requirements}
            new_ids = {r["id"] for r in new_unmet}

            if new_ids == old_ids:
                self.console.print("[yellow]No progress made - same requirements still unmet[/yellow]")
                # Continue trying but warn
            elif len(new_unmet) < len(unmet_requirements):
                fixed_count = len(unmet_requirements) - len(new_unmet)
                self.console.print(f"[green]Fixed {fixed_count} requirement(s), {len(new_unmet)} remaining[/green]")

            unmet_requirements = new_unmet

        self.console.print(f"[red]Requirements fix loop exhausted after {max_attempts} attempts[/red]")
        self._send_ws_update(
            "requirements_fix",
            status="failed",
            attempt=max_attempts,
            unmet_count=len(unmet_requirements),
            message=f"Could not fix {len(unmet_requirements)} requirements",
        )
        return False

    def _generate_requirements_fix_instructions(self, unmet_requirements: list[dict]) -> str:
        """Generate fix instructions from unmet requirements.

        Args:
            unmet_requirements: List of unmet requirement dicts

        Returns:
            Fix instructions string for ImplementationAgent
        """
        instructions = ["Fix the following unmet requirements:\n"]

        for req in unmet_requirements:
            req_id = req.get("id", "REQ-???")
            description = req.get("description", "Unknown requirement")
            reason = req.get("reason", "Not specified")
            req_type = req.get("type", "unknown")

            instructions.append(f"## {req_id}: {description}")
            instructions.append(f"   Type: {req_type}")
            instructions.append(f"   Failure reason: {reason}")

            # Generate specific fix guidance based on failure reason
            if "still found in" in reason.lower():
                # Pattern found that shouldn't be there
                pattern_match = reason.split("'")[1] if "'" in reason else ""
                files = reason.split("in:")[-1].strip() if "in:" in reason else ""
                instructions.append(f"   Action: Remove or replace '{pattern_match}' in files: {files}")
            elif "not found" in reason.lower():
                # Something missing
                instructions.append(f"   Action: Ensure the requirement is implemented")
            elif "no readme" in reason.lower():
                instructions.append(f"   Action: Create or update README.md with proper documentation")

            instructions.append("")

        instructions.append("\nIMPORTANT: Make minimal, targeted changes to fix only these specific issues.")
        instructions.append("Do not refactor or change unrelated code.")

        return "\n".join(instructions)

    def _apply_requirements_fixes(
        self,
        state: PipelineState,
        project_dir: Path,
        fix_instructions: str,
    ) -> bool:
        """Apply fixes using ImplementationAgent.

        Args:
            state: Pipeline state
            project_dir: Project directory
            fix_instructions: Instructions for fixes

        Returns:
            True if fixes were applied successfully
        """
        if self.llm is None:
            self.console.print("[yellow]LLM not available for fixes[/yellow]")
            return False

        try:
            # Use ImplementationAgent to apply fixes
            impl_agent = ImplementationAgent(llm=self.llm)

            input_data = AgentInput(
                context={
                    "feature_description": fix_instructions,
                    "repo_path": str(project_dir),
                    "mode": "fix",  # Indicate this is a fix, not new implementation
                }
            )

            output = impl_agent.run(input_data)

            if not output.success:
                self.console.print(f"[yellow]ImplementationAgent failed: {output.errors}[/yellow]")
                return False

            # Apply the changes to files
            changes = output.data.get("changes", [])
            if not changes:
                self.console.print("[yellow]No changes generated[/yellow]")
                return False

            for change in changes:
                file_path = project_dir / change.get("path", "")
                new_code = change.get("new_code", "")

                if file_path and new_code:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(new_code)
                    self.console.print(f"[dim]Updated: {file_path.name}[/dim]")

            return True

        except Exception as e:
            self.console.print(f"[red]Error applying fixes: {e}[/red]")
            return False

    def _reverify_requirements(
        self,
        state: PipelineState,
        project_dir: Path,
    ) -> list[dict]:
        """Re-verify requirements after fixes.

        Args:
            state: Pipeline state
            project_dir: Project directory

        Returns:
            List of still-unmet requirements (empty if all pass)
        """
        run_dir = Path(state.artifacts_dir)

        # Load requirements tracker
        tracker_file = run_dir / "requirements_tracker.json"
        if not tracker_file.exists():
            return []

        try:
            with open(tracker_file) as f:
                tracker_data = json.load(f)

            from schemas.requirements import RequirementsTracker
            tracker = RequirementsTracker(**tracker_data)

            # Re-run verification
            if self.llm is None:
                return []

            verify_agent = VerifyAgent(llm=self.llm, base_url=None)

            # Just verify requirements, not the full verification
            verification = verify_agent._verify_requirements(
                tracker=tracker,
                project_dir=str(project_dir),
                feature_spec=None,
            )

            # Update tracker file with new status
            tracker_file.write_text(json.dumps(tracker.model_dump(), indent=2))

            return verification.get("unmet_requirements", [])

        except Exception as e:
            self.console.print(f"[yellow]Error re-verifying: {e}[/yellow]")
            return []

    def _handle_pr_package(self, state: PipelineState) -> tuple[bool, str | None]:
        """Build PR package with spec-tied changes using PRPackagingAgent."""
        if not self._pr_package_enabled:
            self.console.print("[dim]PR package stage skipped (not enabled)[/dim]")
            return True, None

        run_dir = Path(state.artifacts_dir)
        project_dir = run_dir / "generated_project"

        if not project_dir.exists():
            self.console.print("[yellow]No generated_project found, skipping PR package[/yellow]")
            return True, None

        self.console.print("[bold]PR Packaging Agent:[/bold] Building PR package...")

        from agents.pr_packaging_agent import PRPackagingAgent

        # Initialize agent (no LLM needed for basic packaging)
        pr_agent = PRPackagingAgent(llm=self.llm)

        # Load spec if available
        spec = None
        spec_file = run_dir / "feature_spec.json"
        if spec_file.exists():
            try:
                with open(spec_file) as f:
                    from schemas.feature_spec import FeatureSpec
                    spec = FeatureSpec(**json.load(f))
            except Exception as e:
                self.console.print(f"[yellow]Could not load spec: {e}[/yellow]")

        # Build PR package
        try:
            package = pr_agent.run(
                repo_path=project_dir,
                spec=spec,
                run_id=state.run_id,
                artifacts_dir=Path(self.config.pipeline.artifacts_dir),
                source_branch="feature",
                target_branch="main",
            )

            # Save package as JSON
            package_file = run_dir / "pr_package.json"
            package_file.write_text(package.model_dump_json(indent=2))
            state.artifacts["pr_package"] = str(package_file)

            # Save rendered body as markdown
            body_file = run_dir / "pr_body.md"
            body_file.write_text(package.rendered_body)
            state.artifacts["pr_body"] = str(body_file)

            # Show summary
            self.console.print(f"[green]Title:[/green] {package.title}")

            criteria_met = sum(1 for c in package.acceptance_criteria if c.met)
            criteria_total = len(package.acceptance_criteria)
            if criteria_total > 0:
                self.console.print(f"[green]Criteria:[/green] {criteria_met}/{criteria_total} met")

            if package.report_links:
                self.console.print(f"[green]Reports:[/green] {len(package.report_links)} linked")

            if package.suggested_labels:
                labels = ", ".join(l.value for l in package.suggested_labels)
                self.console.print(f"[green]Labels:[/green] {labels}")

            if package.changelog_entries:
                self.console.print(f"[green]Changelog:[/green] {len(package.changelog_entries)} entries")

            self.console.print(f"[dim]Created: {package_file}[/dim]")
            self.console.print(f"[dim]Created: {body_file}[/dim]")

        except Exception as e:
            self.console.print(f"[red]PR packaging failed: {e}[/red]")
            return False, f"PR packaging failed: {e}"

        return True, None

    def _handle_final_approval(self, state: PipelineState) -> tuple[bool, str | None]:
        """Handle final approval checkpoint."""
        checkpoint = self.checkpoint_manager.get_final_checkpoint(state)
        response = self.checkpoint_manager.request_approval(
            checkpoint,
            state,
            Path(state.artifacts_dir),
        )

        if response.result == ApprovalResult.APPROVED:
            return True, None
        elif response.result == ApprovalResult.DEFERRED:
            state.status = RunStatus.PAUSED
            return False, "Deferred by user"
        else:
            return False, f"Rejected: {response.notes}"

    def _handle_deploy(self, state: PipelineState) -> tuple[bool, str | None]:
        """Deploy to production."""
        # TODO: Implement production deploy
        self.console.print("[dim]Deploy Agent: Deploying...[/dim]")
        self.console.print("[dim]Skipped: Production deploy not yet implemented[/dim]")
        return True, None

    def _handle_done(self, state: PipelineState) -> tuple[bool, str | None]:
        """Handle completion.

        Also indexes the run into memory if auto_index is enabled.
        """
        # Auto-index run into memory if enabled
        if self.config.memory.enabled and self.config.memory.auto_index:
            try:
                from memory import MemoryStore, RunIndexer
                from rag.embeddings import OllamaEmbeddings

                # Determine store path
                if self.config.memory.store_location == "local":
                    store_path = Path.cwd() / ".coding-factory-memory"
                else:
                    store_path = Path.home() / ".coding-factory" / "memory"

                store = MemoryStore(
                    store_path=store_path,
                    decay_half_life_days=self.config.memory.decay_half_life_days,
                )
                embeddings = OllamaEmbeddings(model=self.config.memory.embedding_model)
                indexer = RunIndexer(store=store, embeddings=embeddings)

                run_dir = Path(state.artifacts_dir)
                count = indexer.index_run(run_dir)

                if count > 0:
                    store.save()
                    self.console.print(f"[dim]Indexed {count} memories from run[/dim]")
            except Exception as e:
                # Don't fail the pipeline if memory indexing fails
                self.console.print(f"[dim]Memory indexing skipped: {e}[/dim]")

        return True, None

    def list_runs(self) -> list[dict[str, Any]]:
        """List all pipeline runs.

        Returns:
            List of run summaries
        """
        runs = []
        if not self.artifacts_dir.exists():
            return runs

        for run_dir in sorted(self.artifacts_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            state_file = run_dir / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                runs.append({
                    "run_id": state_data.get("run_id"),
                    "feature": state_data.get("feature_description", "")[:50],
                    "status": state_data.get("status"),
                    "current_stage": state_data.get("current_stage"),
                    "created_at": state_data.get("created_at"),
                })
            else:
                # Legacy run without state.json
                spec_file = run_dir / "spec.json"
                if spec_file.exists():
                    with open(spec_file) as f:
                        spec_data = json.load(f)
                    runs.append({
                        "run_id": spec_data.get("run_id", run_dir.name),
                        "feature": spec_data.get("feature", "")[:50],
                        "status": "unknown",
                        "current_stage": "unknown",
                        "created_at": spec_data.get("started_at"),
                    })

        return runs
