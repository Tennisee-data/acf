"""ACF Local Edition CLI.

Main command-line interface for running the local pipeline.
"""

import sys
from pathlib import Path
from typing import Optional
import uuid

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from cli.agentcodefactory.output import (
    PipelineDisplay,
    console,
    print_success,
    print_error,
    print_info,
    print_warning,
)

app = typer.Typer(
    name="acf",
    help="ACF Local Edition - AI code generation with local LLMs",
    no_args_is_help=True,
)

# Config sub-app
config_app = typer.Typer(
    name="config",
    help="Manage configuration settings.",
)
app.add_typer(config_app, name="config")

# Register extension sub-apps
from cli.commands.extensions import extensions_app
from cli.commands.marketplace import marketplace_app
from cli.commands.auth import auth_app

app.add_typer(extensions_app, name="extensions")
app.add_typer(marketplace_app, name="marketplace")
app.add_typer(auth_app, name="auth")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Natural language description of what to generate"),
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Target repository path (default: current directory)",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="LLM backend: auto|ollama|openai|anthropic|lmstudio",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model name",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "-y",
        help="Skip approval checkpoints",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing",
    ),
    decompose: bool = typer.Option(
        False,
        "--decompose",
        help="Decompose feature into sub-tasks",
    ),
    api_contract: bool = typer.Option(
        False,
        "--api-contract",
        help="Generate API contract before implementation",
    ),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        help="Enforce test coverage thresholds",
    ),
    coverage_threshold: float = typer.Option(
        80.0,
        "--coverage-threshold",
        help="Minimum coverage percentage",
    ),
    secrets_scan: bool = typer.Option(
        False,
        "--secrets-scan",
        help="Scan for hardcoded secrets",
    ),
    dependency_audit: bool = typer.Option(
        False,
        "--dependency-audit",
        help="Audit dependencies for CVEs",
    ),
    code_review: bool = typer.Option(
        False,
        "--code-review",
        help="Perform senior engineer code review",
    ),
    docs: bool = typer.Option(
        False,
        "--docs",
        help="Generate documentation",
    ),
    observability: bool = typer.Option(
        False,
        "--observability",
        help="Inject logging and metrics scaffolding",
    ),
    config_12factor: bool = typer.Option(
        False,
        "--config",
        help="Enforce 12-factor config layout",
    ),
    policy: bool = typer.Option(
        False,
        "--policy",
        help="Enforce policy rules",
    ),
    policy_rules: Optional[str] = typer.Option(
        None,
        "--policy-rules",
        help="Path to custom policy_rules.yaml",
    ),
    pr_package: bool = typer.Option(
        False,
        "--pr-package",
        help="Build rich PR package",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume a previous run by run ID",
    ),
) -> None:
    """Generate code from a natural language prompt.

    Examples:
        acf generate "Add a hello world endpoint"
        acf generate "Add user authentication" --backend ollama
        acf generate "Fix the login bug" --auto-approve
        acf generate "Add REST API" --api-contract --docs
    """
    from pipeline.config import get_config, load_config, find_config_file
    from orchestrator.runner import PipelineRunner

    # Load config
    config = get_config()

    # Apply overrides
    if backend:
        config.llm.backend = backend
    if model:
        config.llm.model_code = model
        config.llm.model_general = model

    # Target repository
    repo_path = repo or Path.cwd()
    if not repo_path.exists():
        print_error(f"Repository path does not exist: {repo_path}")
        raise typer.Exit(1)

    # Create pipeline runner
    try:
        runner = PipelineRunner(
            config=config,
            console=console,
            auto_approve=auto_approve,
        )
    except Exception as e:
        print_error(f"Failed to initialize pipeline: {e}")
        raise typer.Exit(1)

    # Generate run ID for display
    run_id = resume or str(uuid.uuid4())

    # Run pipeline with progress display
    print_info(f"Starting pipeline for: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print_info(f"Target: {repo_path}")

    try:
        with PipelineDisplay(run_id) as display:
            state = runner.run(
                feature=prompt,
                repo_path=repo_path,
                resume_run_id=resume,
                dry_run=dry_run,
                decompose=decompose,
                api_contract=api_contract,
                coverage=coverage,
                coverage_threshold=coverage_threshold,
                secrets_scan=secrets_scan,
                dependency_audit=dependency_audit,
                code_review=code_review,
                docs=docs,
                observability=observability,
                config=config_12factor,
                policy=policy,
                policy_rules=policy_rules,
                pr_package=pr_package,
            )

            # Update display based on state
            if state.status.value == "completed":
                file_count = len(state.files) if hasattr(state, 'files') and state.files else 0
                display.complete(file_count)
            elif state.status.value == "failed":
                display.fail(state.error or "Pipeline failed")

        # Final result
        if state.status.value == "completed":
            print_success(f"Pipeline completed successfully!")
            print_info(f"Run ID: {state.run_id}")
            if hasattr(state, 'files') and state.files:
                print_info(f"Generated {len(state.files)} files")
        else:
            print_error(f"Pipeline failed: {state.error or 'Unknown error'}")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        print_warning("Pipeline interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"Pipeline error: {e}")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show environment status.

    Displays detected LLM backend, configuration, and installed extensions.
    """
    from llm_backend import detect_backend
    from pipeline.config import get_config, find_config_file

    console.print(Panel("[bold]ACF Local Edition Status[/bold]", border_style="cyan"))

    # Config file
    config_path = find_config_file()
    if config_path:
        print_info(f"Config file: {config_path}")
    else:
        print_warning("No config.toml found (using defaults)")

    config = get_config()

    # Backend detection
    console.print("\n[bold]LLM Backend[/bold]")
    detected = detect_backend()
    configured = config.llm.backend

    if configured == "auto":
        print_info(f"Backend: auto (detected: {detected})")
    else:
        print_info(f"Backend: {configured}")

    print_info(f"Model (code): {config.llm.model_code}")
    print_info(f"Model (general): {config.llm.model_general}")

    # Test backend connectivity
    console.print("\n[bold]Backend Connectivity[/bold]")
    try:
        from llm_backend import get_backend

        backend = get_backend(
            config.llm.backend,
            model=config.llm.model_general,
            base_url=config.llm.base_url,
        )
        # Try a simple test
        print_success(f"Connected to {detected} backend")
    except Exception as e:
        print_error(f"Backend connection failed: {e}")

    # Extensions
    console.print("\n[bold]Extensions[/bold]")
    try:
        from extensions import ExtensionLoader

        loader = ExtensionLoader()
        installed = loader.discover()
        if installed:
            print_info(f"{len(installed)} extensions loaded")
            for ext in installed[:5]:  # Show first 5
                console.print(f"  - {ext.manifest.name} ({ext.manifest.type.value})")
            if len(installed) > 5:
                console.print(f"  ... and {len(installed) - 5} more")
        else:
            print_info("No extensions installed")
    except Exception:
        print_info("Extensions system not initialized")

    # Routing
    if config.routing.enabled:
        console.print("\n[bold]Model Routing[/bold]")
        print_info(f"Cheap: {config.routing.model_cheap}")
        print_info(f"Medium: {config.routing.model_medium}")
        print_info(f"Premium: {config.routing.model_premium}")


@config_app.command("show")
def config_show(
    section: Optional[str] = typer.Argument(
        None,
        help="Config section to show (llm, pipeline, git, etc.)",
    ),
) -> None:
    """Show current configuration.

    Examples:
        acf config show           # Show all config
        acf config show llm       # Show LLM section only
    """
    from pipeline.config import get_config, find_config_file

    config_path = find_config_file()
    if config_path:
        print_info(f"Config file: {config_path}")
    else:
        print_warning("No config.toml found (using defaults)")

    config = get_config()

    if section:
        # Show specific section
        section_lower = section.lower()
        section_map = {
            "llm": config.llm,
            "pipeline": config.pipeline,
            "git": config.git,
            "runtime": config.runtime,
            "routing": config.routing,
            "extensions": config.extensions,
            "memory": config.memory,
            "plugins": config.plugins,
            "local_storage": config.local_storage,
        }

        if section_lower not in section_map:
            print_error(f"Unknown section: {section}")
            print_info(f"Available: {', '.join(section_map.keys())}")
            raise typer.Exit(1)

        section_config = section_map[section_lower]
        console.print(f"\n[bold][{section_lower}][/bold]")

        table = Table(show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in vars(section_config).items():
            if not key.startswith("_"):
                table.add_row(key, str(value))

        console.print(table)
    else:
        # Show all sections
        sections = [
            ("llm", config.llm),
            ("pipeline", config.pipeline),
            ("git", config.git),
            ("runtime", config.runtime),
            ("routing", config.routing),
            ("extensions", config.extensions),
        ]

        for name, section_config in sections:
            console.print(f"\n[bold][{name}][/bold]")
            for key, value in vars(section_config).items():
                if not key.startswith("_"):
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    console.print(f"  {key} = {value_str}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., llm.backend, routing.enabled)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value.

    Examples:
        acf config set llm.backend ollama
        acf config set llm.model_code qwen2.5-coder:32b
        acf config set routing.enabled false
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    import tomli_w

    from pipeline.config import find_config_file

    config_path = find_config_file()
    if not config_path:
        print_error("No config.toml found. Run 'acf config init' first.")
        raise typer.Exit(1)

    # Parse the key
    parts = key.split(".")
    if len(parts) != 2:
        print_error("Key must be in format 'section.key' (e.g., llm.backend)")
        raise typer.Exit(1)

    section, setting = parts

    # Load current config
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)

    # Update value
    if section not in config_data:
        config_data[section] = {}

    # Parse value type
    parsed_value: str | int | float | bool | list = value
    if value.lower() == "true":
        parsed_value = True
    elif value.lower() == "false":
        parsed_value = False
    elif value.isdigit():
        parsed_value = int(value)
    elif value.replace(".", "").isdigit() and value.count(".") == 1:
        parsed_value = float(value)
    elif value.startswith("[") and value.endswith("]"):
        # Simple list parsing
        inner = value[1:-1].strip()
        if inner:
            parsed_value = [v.strip().strip('"').strip("'") for v in inner.split(",")]
        else:
            parsed_value = []

    config_data[section][setting] = parsed_value

    # Write back
    with open(config_path, "wb") as f:
        tomli_w.dump(config_data, f)

    print_success(f"Set {key} = {parsed_value}")

    # Reload config
    from pipeline.config import reload_config

    reload_config()


@config_app.command("init")
def config_init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config.toml",
    ),
) -> None:
    """Create a default config.toml file.

    Example:
        acf config init
        acf config init --force
    """
    config_path = Path.cwd() / "config.toml"

    if config_path.exists() and not force:
        print_warning(f"Config file already exists: {config_path}")
        print_info("Use --force to overwrite")
        raise typer.Exit(1)

    default_config = '''# ACF Local Edition Configuration
# Auto-generated by 'acf config init'

[llm]
# Backend: "auto" | "ollama" | "openai" | "anthropic" | "lmstudio"
backend = "auto"

# Models (adjust based on your hardware)
model_general = "qwen2.5-coder:14b"
model_code = "qwen2.5-coder:14b"

# Timeout for LLM requests (seconds)
timeout = 600

[pipeline]
artifacts_dir = "artifacts"
log_level = "INFO"

[pipeline.fix_loop]
enabled = true
max_iterations = 5

[git]
auto_commit = true
auto_push = false

[runtime]
mode = "local"

[routing]
enabled = true
model_cheap = "qwen2.5-coder:7b"
model_medium = "qwen2.5-coder:14b"
model_premium = "qwen2.5-coder:32b"
premium_domains = ["payments", "security", "auth", "database"]

[plugins]
enabled = true

[extensions]
extensions_dir = "~/.coding-factory/extensions"
marketplace_url = "https://marketplace.agentcodefactory.com/api/v1"
agents = []
profiles = []
rag_kits = []

[memory]
enabled = true
store_location = "global"
search_mode = "hybrid"

[local_storage]
auto_commit = true
commit_prefix = "[ACF]"
auto_tag = false
'''

    config_path.write_text(default_config)
    print_success(f"Created config file: {config_path}")


@app.command()
def version() -> None:
    """Show ACF version."""
    from cli.agentcodefactory import __version__

    console.print(f"ACF Local Edition v{__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
