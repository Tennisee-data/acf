"""CLI entrypoint for Coding Factory."""

import warnings
from pathlib import Path
from typing import Optional

# Suppress Pydantic serialization warnings (mixed types in Any fields)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from llm_backend import get_backend
from orchestrator import PipelineRunner
from pipeline import __version__
from pipeline.config import get_config
from scaffolding import ProjectGenerator, TEMPLATES

# Import extension CLI commands (ACF Local Edition)
try:
    from cli.commands.marketplace import marketplace_app
    from cli.commands.extensions import extensions_app
    from cli.commands.auth import auth_app
    EXTENSIONS_CLI_AVAILABLE = True
except ImportError:
    EXTENSIONS_CLI_AVAILABLE = False

app = typer.Typer(
    name="acf",
    help="AI-powered feature pipeline with local LLM support.",
    add_completion=False,
)
console = Console()

# Register extension CLI sub-apps
if EXTENSIONS_CLI_AVAILABLE:
    app.add_typer(marketplace_app, name="marketplace")
    app.add_typer(extensions_app, name="extensions")
    app.add_typer(auth_app, name="auth")


@app.command()
def run(
    feature: str = typer.Argument(..., help="Feature description to implement"),
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to target repository (default: current directory)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for generated project (default: ./{run_id}/)",
    ),
    profile: str = typer.Option(
        "dev",
        "--profile",
        "-p",
        help="Configuration profile to use",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "-y",
        help="Auto-approve all checkpoints (non-interactive)",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume a previous run by ID",
    ),
    decompose: bool = typer.Option(
        False,
        "--decompose",
        "-d",
        help="Decompose feature into sub-tasks before design",
    ),
    api_contract: bool = typer.Option(
        False,
        "--api-contract",
        "-c",
        help="Generate API contract (OpenAPI + Pydantic) before implementation",
    ),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        help="Enforce test coverage thresholds after testing",
    ),
    coverage_threshold: float = typer.Option(
        80.0,
        "--coverage-threshold",
        help="Minimum coverage percentage required (default: 80)",
    ),
    secrets_scan: bool = typer.Option(
        False,
        "--secrets-scan",
        help="Scan for hardcoded secrets and auto-fix with environment variables",
    ),
    dependency_audit: bool = typer.Option(
        False,
        "--dependency-audit",
        help="Audit dependencies for CVEs, deprecated packages, and outdated versions",
    ),
    rollback_strategy: bool = typer.Option(
        False,
        "--rollback-strategy",
        help="Generate CI/CD rollback jobs and canary deployment templates",
    ),
    observability: bool = typer.Option(
        False,
        "--observability",
        help="Inject logging, metrics, and tracing scaffolding for production",
    ),
    config_layout: bool = typer.Option(
        False,
        "--config",
        help="Enforce 12-factor config layout (settings.py, .env.example, README docs)",
    ),
    docs: bool = typer.Option(
        False,
        "--docs",
        help="Generate and sync documentation (docs/, docstrings, ADRs)",
    ),
    code_review: bool = typer.Option(
        False,
        "--code-review",
        help="Perform senior engineer code review (ship/ship-with-nits/don't-ship)",
    ),
    policy: bool = typer.Option(
        False,
        "--policy",
        help="Enforce policy rules (allow/require_approval/block) before verification",
    ),
    policy_rules: str = typer.Option(
        None,
        "--policy-rules",
        help="Path to custom policy_rules.yaml file",
    ),
    pr_package: bool = typer.Option(
        False,
        "--pr-package",
        help="Build rich PR package with spec-tied changes, report links, changelog",
    ),
    jira: Optional[str] = typer.Option(
        None,
        "--jira",
        help="JIRA issue key to fetch (e.g., PROJ-123). Overrides feature argument.",
    ),
    issue: Optional[str] = typer.Option(
        None,
        "--issue",
        help="Issue URL to fetch (GitHub, JIRA). Overrides feature argument.",
    ),
) -> None:
    """Run the feature pipeline for a given description.

    Examples:
        acf run "Add login rate-limit"
        acf run "Add caching" --auto-approve
        acf run "Add OAuth" --decompose
        acf run "Build REST API" --api-contract
        acf run "Implement feature" --coverage --coverage-threshold 90
        acf run "Build API" --secrets-scan
        acf run "Add feature" --dependency-audit
        acf run "Deploy service" --rollback-strategy
        acf run "Build service" --observability
        acf run "Build app" --config
        acf run "Build project" --docs
        acf run "Review code" --code-review
        acf run "Deploy feature" --policy
        acf run "Deploy feature" --policy --policy-rules custom_rules.yaml
        acf run "Build feature" --pr-package
        acf run "Fix bug" --resume 2026-01-04-143052
        acf run - --jira PROJ-123 --repo ./project
        acf run - --issue https://github.com/org/repo/issues/42
    """
    config = get_config()
    repo_path = repo or Path.cwd()

    # Handle issue tracker integration
    actual_feature = feature
    issue_data = None

    if jira or issue:
        from integrations import resolve_issue

        try:
            issue_ref = jira or issue
            rprint(f"[dim]Fetching issue: {issue_ref}...[/dim]")
            issue_data = resolve_issue(issue_ref)
            actual_feature = issue_data.to_feature_description()
            rprint(f"[green]Issue fetched:[/green] {issue_data.title}")
            rprint(f"[dim]Source: {issue_data.url}[/dim]")
        except ValueError as e:
            rprint(f"[red]Error fetching issue: {e}[/red]")
            raise typer.Exit(1)
        except ConnectionError as e:
            rprint(f"[red]Connection error: {e}[/red]")
            raise typer.Exit(1)

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint(f"[dim]Profile: {profile}[/dim]")
    rprint()
    rprint(f"[green]Feature:[/green] {actual_feature[:100]}{'...' if len(actual_feature) > 100 else ''}")
    rprint(f"[green]Repository:[/green] {repo_path}")
    rprint()

    # Create runner
    runner = PipelineRunner(
        config=config,
        console=console,
        auto_approve=auto_approve,
    )

    # Execute pipeline
    try:
        state = runner.run(
            feature=actual_feature,
            repo_path=repo_path,
            output_dir=output,
            resume_run_id=resume,
            dry_run=dry_run,
            decompose=decompose,
            api_contract=api_contract,
            coverage=coverage,
            coverage_threshold=coverage_threshold,
            secrets_scan=secrets_scan,
            dependency_audit=dependency_audit,
            rollback_strategy=rollback_strategy,
            observability=observability,
            config=config_layout,
            docs=docs,
            code_review=code_review,
            policy=policy,
            policy_rules=policy_rules,
            pr_package=pr_package,
        )

        # Show final status
        if state.status.value == "completed":
            rprint()
            # Check for verification warnings
            verification_warnings = state.metadata.get("verification_warnings", {})
            if verification_warnings:
                rprint("[bold yellow]Run completed with warnings![/bold yellow]")
                rprint("[dim]Some issues were overridden by user approval. Consider running another iteration.[/dim]")
            else:
                rprint("[bold green]Run completed successfully![/bold green]")
        elif state.status.value == "paused":
            rprint()
            rprint("[yellow]Run paused. Resume with:[/yellow]")
            # Use short description for resume command
            short_feature = actual_feature[:50] + "..." if len(actual_feature) > 50 else actual_feature
            rprint(f"  acf run \"{short_feature}\" --resume {state.run_id}")
        elif state.status.value == "failed":
            rprint()
            rprint(f"[red]Run failed: {state.last_error}[/red]")
            raise typer.Exit(1)
        elif state.status.value == "cancelled":
            rprint()
            rprint("[yellow]Run cancelled[/yellow]")

    except ValueError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted[/yellow]")
        raise typer.Exit(130)


@app.command()
def new(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option(
        "fastapi",
        "--template",
        "-t",
        help="Project template (fastapi, cli, minimal)",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Project description",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: current directory)",
    ),
    no_git: bool = typer.Option(
        False,
        "--no-git",
        help="Skip git initialization",
    ),
    install: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Install dependencies after creation",
    ),
) -> None:
    """Create a new project from template.

    Examples:
        acf new my-api
        acf new my-cli --template cli
        acf new my-app --template fastapi --description "My REST API"
        acf new my-project -o ~/projects --install
    """
    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()

    # Validate template
    if template not in TEMPLATES:
        rprint(f"[red]Unknown template: {template}[/red]")
        rprint(f"[dim]Available templates: {', '.join(TEMPLATES.keys())}[/dim]")
        raise typer.Exit(1)

    try:
        generator = ProjectGenerator(
            name=name,
            template=template,
            description=description,
            output_dir=output_dir,
        )
        generator.generate(init_git=not no_git, install_deps=install)

    except FileExistsError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def templates(
    local_only: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Show only custom templates",
    ),
    builtin_only: bool = typer.Option(
        False,
        "--builtin",
        "-b",
        help="Show only built-in templates",
    ),
) -> None:
    """List available project templates."""
    from scaffolding import list_templates, BUILTIN_TEMPLATES

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()

    all_templates = list_templates()

    # Filter if requested
    if local_only:
        all_templates = [t for t in all_templates if not t["builtin"]]
        rprint("[bold]Custom Templates:[/bold]")
    elif builtin_only:
        all_templates = [t for t in all_templates if t["builtin"]]
        rprint("[bold]Built-in Templates:[/bold]")
    else:
        rprint("[bold]Available Templates:[/bold]")

    rprint()

    if not all_templates:
        rprint("[dim]No templates found.[/dim]")
        return

    for tmpl in all_templates:
        source = "[dim](built-in)[/dim]" if tmpl["builtin"] else "[green](custom)[/green]"
        rprint(f"  [cyan]{tmpl['name']}[/cyan] {source}")
        rprint(f"    {tmpl['description']}")
        rprint(f"    [dim]Language: {tmpl['language']} | Framework: {tmpl['framework']}[/dim]")
        rprint()


@app.command()
def runs() -> None:
    """List all pipeline runs."""
    config = get_config()
    runner = PipelineRunner(config=config, console=console)

    run_list = runner.list_runs()

    if not run_list:
        rprint("[dim]No runs found.[/dim]")
        return

    table = Table(title="Pipeline Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Feature", style="white")
    table.add_column("Status", style="green")
    table.add_column("Stage", style="yellow")

    for r in run_list:
        status = r.get("status", "unknown")
        status_style = {
            "completed": "green",
            "running": "blue",
            "paused": "yellow",
            "failed": "red",
            "cancelled": "dim",
        }.get(status, "white")

        table.add_row(
            r.get("run_id", ""),
            r.get("feature", "")[:40] + ("..." if len(r.get("feature", "")) > 40 else ""),
            f"[{status_style}]{status}[/{status_style}]",
            r.get("current_stage", ""),
        )

    console.print(table)


def _find_run_artifacts(run_id: str, config) -> Path | None:
    """Find artifacts directory for a run, checking both new and legacy structure.

    Args:
        run_id: Run ID to find
        config: Configuration object

    Returns:
        Path to artifacts directory, or None if not found
    """
    # 1. Check legacy artifacts/ directory
    legacy_dir = Path(config.pipeline.artifacts_dir) / run_id
    if legacy_dir.exists():
        return legacy_dir

    # 2. Check new structure: scan for .acf/runs/{run_id}/ in current directory
    cwd = Path.cwd()
    for proj_dir in cwd.iterdir():
        if not proj_dir.is_dir():
            continue

        acf_run_dir = proj_dir / ".acf" / "runs" / run_id
        if acf_run_dir.exists():
            return acf_run_dir

    # 3. Check if run_id is a project directory with .acf/runs/
    potential_project = cwd / run_id
    if potential_project.exists():
        acf_runs = potential_project / ".acf" / "runs"
        if acf_runs.exists():
            # Return the first (and should be only) run in that project
            for run_dir in acf_runs.iterdir():
                if run_dir.is_dir():
                    return run_dir

    return None


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to show details for"),
) -> None:
    """Show details of a specific run."""
    config = get_config()
    artifacts_dir = _find_run_artifacts(run_id, config)

    if not artifacts_dir:
        rprint(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    state_file = artifacts_dir / "state.json"
    if state_file.exists():
        import json

        with open(state_file) as f:
            state_data = json.load(f)

        rprint(f"[bold]Run: {run_id}[/bold]")
        rprint()
        rprint(f"[green]Feature:[/green] {state_data.get('feature_description', '')}")
        rprint(f"[green]Status:[/green] {state_data.get('status', 'unknown')}")
        rprint(f"[green]Current Stage:[/green] {state_data.get('current_stage', 'unknown')}")

        # Show project directory for new structure
        project_dir = state_data.get("project_dir")
        if project_dir:
            rprint(f"[green]Project:[/green] {project_dir}")

        rprint()

        # Show stages
        stages = state_data.get("stages", {})
        if stages:
            rprint("[bold]Stages:[/bold]")
            for stage_name, result in stages.items():
                status = result.get("status", "unknown")
                status_color = {
                    "completed": "green",
                    "approved": "green",
                    "running": "blue",
                    "failed": "red",
                    "skipped": "dim",
                    "awaiting_approval": "yellow",
                }.get(status, "white")
                rprint(f"  {stage_name}: [{status_color}]{status}[/{status_color}]")

        # Show artifacts
        rprint()
        rprint("[bold]Artifacts:[/bold]")
        for artifact in artifacts_dir.iterdir():
            if artifact.is_file() and artifact.name != "state.json":
                rprint(f"  {artifact.name}")
    else:
        rprint(f"[yellow]No state file found for run {run_id}[/yellow]")
        rprint()
        rprint("[bold]Artifacts:[/bold]")
        for artifact in artifacts_dir.iterdir():
            if artifact.is_file():
                rprint(f"  {artifact.name}")


@app.command()
def extract(
    run_id: str = typer.Argument(..., help="Run ID to extract code from"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: project_dir or artifacts/RUN_ID/generated_project)",
    ),
) -> None:
    """Extract generated code files from a run.

    This extracts the generated source code from the diff.patch file
    into actual runnable files.

    Examples:
        acf extract 2026-01-05-151822
        acf extract 2026-01-05-151822 -o ~/my-project
    """
    import re
    import json as json_module

    config = get_config()
    artifacts_dir = _find_run_artifacts(run_id, config)

    if not artifacts_dir:
        rprint(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    diff_file = artifacts_dir / "diff.patch"
    if not diff_file.exists():
        rprint(f"[red]No diff.patch found in run {run_id}[/red]")
        raise typer.Exit(1)

    # Determine output directory
    if output:
        project_dir = output
    else:
        # Try to get project_dir from state (new structure)
        state_file = artifacts_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state_data = json_module.load(f)
            proj_dir = state_data.get("project_dir")
            if proj_dir:
                project_dir = Path(proj_dir)
            else:
                project_dir = artifacts_dir / "generated_project"
        else:
            project_dir = artifacts_dir / "generated_project"

    project_dir.mkdir(parents=True, exist_ok=True)

    rprint(f"[bold blue]Extracting code from {run_id}[/bold blue]")
    rprint(f"[dim]Output: {project_dir}[/dim]")
    rprint()

    # Parse and extract files from diff.patch
    patch_content = diff_file.read_text()
    files_created = 0
    current_file = None
    current_lines = []

    for line in patch_content.split('\n'):
        if line.startswith('+++ b/'):
            # Save previous file if any
            if current_file and current_lines:
                file_path = project_dir / current_file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text('\n'.join(current_lines))
                rprint(f"  [green]Created:[/green] {current_file}")
                files_created += 1

            # Start new file
            current_file = line[6:]  # Remove '+++ b/'
            current_lines = []
        elif line.startswith('+') and not line.startswith('+++'):
            # This is an added line (remove the + prefix)
            current_lines.append(line[1:])

    # Save last file
    if current_file and current_lines:
        file_path = project_dir / current_file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text('\n'.join(current_lines))
        rprint(f"  [green]Created:[/green] {current_file}")
        files_created += 1

    rprint()
    if files_created > 0:
        rprint(f"[bold green]Extracted {files_created} files to {project_dir}[/bold green]")
    else:
        rprint("[yellow]No files found to extract[/yellow]")


@app.command()
def scaffold(
    run_id: str = typer.Argument(..., help="Run ID to generate scaffold for"),
) -> None:
    """Generate deployment-ready files for a run.

    Creates requirements.txt, Dockerfile, docker-compose.yml,
    README.md, .env.example, and run.sh in the generated_project folder.

    Examples:
        acf scaffold 2026-01-05-151822
    """
    from agents.scaffold_agent import ProjectScaffoldAgent
    from agents import AgentInput

    config = get_config()
    artifacts_dir = Path(config.pipeline.artifacts_dir) / run_id
    project_dir = artifacts_dir / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        rprint(f"[dim]Run 'acf extract {run_id}' first[/dim]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Generating scaffold for {run_id}[/bold blue]")
    rprint(f"[dim]Project: {project_dir}[/dim]")
    rprint()

    # Get feature name from state
    state_file = artifacts_dir / "state.json"
    feature_name = "Generated Project"
    if state_file.exists():
        import json
        with open(state_file) as f:
            state = json.load(f)
            feature_name = state.get("feature_description", "")[:50]

    scaffold_agent = ProjectScaffoldAgent()
    input_data = AgentInput(
        context={
            "project_dir": str(project_dir),
            "feature_name": feature_name,
        }
    )

    output = scaffold_agent.run(input_data)

    if output.success:
        files = output.data.get("files_generated", [])
        rprint(f"[green]Generated {len(files)} files:[/green]")
        for f in files:
            rprint(f"  [cyan]{f}[/cyan]")

        # Show security warnings if any
        warnings = output.data.get("security_warnings", [])
        if warnings:
            rprint()
            rprint("[yellow bold]Security Warnings:[/yellow bold]")
            for w in warnings:
                if "CRITICAL" in w:
                    rprint(f"  [red]{w}[/red]")
                elif "WARNING" in w:
                    rprint(f"  [yellow]{w}[/yellow]")
                else:
                    rprint(f"  [dim]{w}[/dim]")

        rprint()
        rprint(f"[bold green]Project is deployment-ready![/bold green]")
        rprint()
        rprint("[bold]Quick start:[/bold]")
        rprint(f"  cd {project_dir}")
        rprint("  chmod +x run.sh && ./run.sh")
        rprint()
        rprint("[bold]Or with Docker:[/bold]")
        rprint(f"  cd {project_dir}")
        rprint("  docker-compose up --build")
    else:
        rprint(f"[red]Scaffold generation failed: {output.errors}[/red]")
        raise typer.Exit(1)


@app.command("security-scan")
def security_scan(
    path: Path = typer.Argument(..., help="Path to project directory to scan"),
    full: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Run full scan including bandit and dependency checks",
    ),
) -> None:
    """Scan project files for security issues.

    Basic scan checks:
    - Dockerfile security (root user, secrets in layers)
    - docker-compose.yml (privileged mode, dangerous mounts)
    - Environment files (hardcoded credentials)

    Full scan (--full) also runs:
    - Bandit Python security scanner
    - Dependency vulnerability check (pip-audit)

    Examples:
        acf security-scan ./my-project
        acf security-scan artifacts/2026-01-05/generated_project --full
    """
    from tools.security import scan_generated_files, full_security_scan

    if not path.exists():
        rprint(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Security scan: {path}[/bold blue]")
    if full:
        rprint("[dim]Mode: Full (Docker + Bandit + Dependencies)[/dim]")
    rprint()

    if full:
        results = full_security_scan(path)

        # Docker issues
        if results["docker_issues"]:
            rprint("[bold]Docker/Compose Issues:[/bold]")
            for filename, warnings in results["docker_issues"].items():
                rprint(f"  [cyan]{filename}[/cyan]")
                for w in warnings:
                    if "CRITICAL" in w:
                        rprint(f"    [red]✗ {w}[/red]")
                    elif "WARNING" in w:
                        rprint(f"    [yellow]⚠ {w}[/yellow]")
                    else:
                        rprint(f"    [dim]ℹ {w}[/dim]")
            rprint()

        # Bandit results
        bandit = results["bandit_scan"]
        if bandit.get("success"):
            if bandit.get("issues"):
                rprint("[bold]Bandit Security Issues:[/bold]")
                for issue in bandit["issues"]:
                    severity = issue["severity"]
                    color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(severity, "white")
                    rprint(f"  [{color}]{severity}[/{color}]: {issue['message']}")
                    rprint(f"    [dim]{issue['file']}:{issue['line']}[/dim]")
                rprint()
        else:
            rprint(f"[dim]Bandit: {bandit.get('error', 'Not available')}[/dim]")
            rprint()

        # Dependency warnings
        if results["dependency_warnings"]:
            rprint("[bold]Dependency Issues:[/bold]")
            for dep in results["dependency_warnings"]:
                if dep["type"] == "vulnerability":
                    rprint(f"  [red]✗ {dep['package']}[/red]: {dep['message']}")
                else:
                    rprint(f"  [yellow]⚠ {dep['package']}[/yellow]: {dep['message']}")
            rprint()

        # Summary
        summary = results["summary"]
        total_issues = sum(summary.values())

        if total_issues == 0:
            rprint("[bold green]✓ No security issues found[/bold green]")
        else:
            rprint("[bold]Summary:[/bold]")
            if summary["critical"] > 0:
                rprint(f"  [red]Critical: {summary['critical']}[/red]")
            if summary["high"] > 0:
                rprint(f"  [red]High: {summary['high']}[/red]")
            if summary["medium"] > 0:
                rprint(f"  [yellow]Medium: {summary['medium']}[/yellow]")
            if summary["low"] > 0:
                rprint(f"  [dim]Low: {summary['low']}[/dim]")

            if summary["critical"] > 0 or summary["high"] > 0:
                raise typer.Exit(1)

    else:
        # Basic scan (docker files only)
        results = scan_generated_files(path)

        if not results:
            rprint("[bold green]✓ No security issues found[/bold green]")
            rprint()
            rprint("[dim]Tip: Use --full for comprehensive scan with bandit[/dim]")
            return

        total_critical = 0
        total_warnings = 0

        for filename, warnings in results.items():
            rprint(f"[cyan]{filename}[/cyan]")
            for w in warnings:
                if "CRITICAL" in w:
                    rprint(f"  [red]✗ {w}[/red]")
                    total_critical += 1
                elif "WARNING" in w:
                    rprint(f"  [yellow]⚠ {w}[/yellow]")
                    total_warnings += 1
                else:
                    rprint(f"  [dim]ℹ {w}[/dim]")
            rprint()

        rprint()
        if total_critical > 0:
            rprint(f"[red bold]{total_critical} critical issues found![/red bold]")
            raise typer.Exit(1)
        elif total_warnings > 0:
            rprint(f"[yellow]{total_warnings} warnings (review recommended)[/yellow]")

        rprint()
        rprint("[dim]Tip: Use --full for comprehensive scan with bandit[/dim]")


@app.command("check-health")
def check_health(
    path: Path = typer.Argument(
        ".",
        help="Path to project directory (default: current directory)",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Apply auto-fixes for deprecations and code style",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        "-v",
        help="Run tests after applying fixes to verify nothing broke",
    ),
    pr: bool = typer.Option(
        False,
        "--pr",
        help="Create a PR with the fixes (requires git repo and gh CLI)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Self-healing health check for code deprecations and updates.

    Scans for:
    - Deprecated Python stdlib usage (datetime.utcnow, typing.List, etc.)
    - Old syntax patterns (% formatting, old except syntax)
    - Outdated dependencies
    - Known vulnerabilities (via pip-audit)

    Auto-fixes (with --fix):
    - Modernizes syntax using pyupgrade (Python 3.11+)
    - Fixes imports and unused code with ruff
    - Updates deprecated patterns where safe

    Examples:
        acf check-health
        acf check-health --fix
        acf check-health --fix --verify
        acf check-health ./my-project --fix --pr
    """
    from tools.health_check import check_health as run_health_check, create_health_pr

    if not path.exists():
        rprint(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Health Check: {path.resolve()}[/bold blue]")
    if fix:
        rprint("[dim]Mode: Fix (applying auto-corrections)[/dim]")
    else:
        rprint("[dim]Mode: Scan only (use --fix to apply corrections)[/dim]")
    rprint()

    # Run health check
    report = run_health_check(path, fix=fix, verify=verify)

    if json_output:
        import json
        rprint(json.dumps(report.to_dict(), indent=2))
        return

    # Display deprecation issues
    if report.issues:
        rprint("[bold]Code Deprecation Issues:[/bold]")
        for issue in report.issues:
            severity_color = {
                "error": "red",
                "warning": "yellow",
                "info": "dim",
            }.get(issue.severity, "white")

            fix_indicator = "[green]✓ fixable[/green]" if issue.auto_fixable else ""
            rprint(
                f"  [{severity_color}]{issue.severity.upper()}[/{severity_color}]: "
                f"{issue.message} {fix_indicator}"
            )
            rprint(f"    [dim]{issue.file}:{issue.line}[/dim]")
            rprint(f"    [dim]→ {issue.replacement}[/dim]")
        rprint()

    # Display outdated dependencies
    if report.outdated_deps:
        rprint("[bold]Outdated Dependencies:[/bold]")
        for dep in report.outdated_deps:
            rprint(
                f"  [yellow]{dep['name']}[/yellow]: "
                f"{dep['current']} → {dep['latest']}"
            )
        rprint()

    # Display vulnerabilities
    if report.vulnerability_deps:
        rprint("[bold red]Security Vulnerabilities:[/bold red]")
        for vuln in report.vulnerability_deps:
            rprint(f"  [red]✗ {vuln['package']} ({vuln['version']})[/red]")
            rprint(f"    {vuln['vuln_id']}: {vuln['description'][:80]}...")
            if vuln.get("fix_versions"):
                rprint(f"    [green]Fix: upgrade to {vuln['fix_versions'][0]}[/green]")
        rprint()

    # Display fixes applied
    if report.fixes_applied:
        rprint("[bold green]Fixes Applied:[/bold green]")
        for fix_item in report.fixes_applied:
            rprint(f"  [green]✓[/green] {fix_item}")
        rprint()

    # Display test results
    if report.tests_passed is not None:
        if report.tests_passed:
            rprint("[bold green]✓ Tests passed after fixes[/bold green]")
        else:
            rprint("[bold red]✗ Tests failed after fixes![/bold red]")
        rprint()

    # Summary
    summary = report.summary
    rprint("[bold]Summary:[/bold]")
    rprint(f"  Total issues: {summary.get('total_issues', 0)}")
    rprint(f"  Auto-fixable: {summary.get('auto_fixable', 0)}")
    rprint(f"  Outdated packages: {summary.get('outdated_packages', 0)}")
    rprint(f"  Vulnerabilities: {summary.get('vulnerabilities', 0)}")

    if fix:
        rprint(f"  Fixes applied: {summary.get('fixes_applied', 0)}")

    # Create PR if requested
    if pr and report.fixes_applied:
        rprint()
        rprint("[bold]Creating PR...[/bold]")
        pr_result = create_health_pr(path, report)
        if pr_result["success"]:
            rprint(f"[green]✓ PR created: {pr_result['pr_url']}[/green]")
        else:
            rprint(f"[red]✗ PR creation failed: {pr_result['error']}[/red]")

    # Exit with error if vulnerabilities found
    if report.vulnerability_deps:
        raise typer.Exit(1)

    # Suggest next steps
    if not fix and (report.issues or report.outdated_deps):
        rprint()
        rprint("[dim]Tip: Run with --fix to auto-correct issues[/dim]")


@app.command("api-probe")
def api_probe(
    run_id: str = typer.Argument(..., help="Run ID with feature_spec.json"),
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        "-u",
        help="Base URL of the running API",
    ),
    auth_token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="Auth token for authenticated endpoints",
    ),
) -> None:
    """Run API probes from acceptance criteria against a running service.

    Reads acceptance criteria from the feature_spec.json and generates
    probes to test if the API meets the requirements.

    Examples:
        acf api-probe 2026-01-05-151822
        acf api-probe 2026-01-05-151822 --url http://localhost:5000
    """
    from tools.api_probe import run_acceptance_probes, generate_probe_report_markdown

    config = get_config()
    feature_spec_path = Path(config.pipeline.artifacts_dir) / run_id / "feature_spec.json"

    if not feature_spec_path.exists():
        rprint(f"[red]Feature spec not found: {feature_spec_path}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]API Probe: {run_id}[/bold blue]")
    rprint(f"[dim]Target: {url}[/dim]")
    rprint()

    # Run probes
    report = run_acceptance_probes(
        feature_spec_path=feature_spec_path,
        base_url=url,
        auth_token=auth_token,
    )

    # Display results
    if report.total == 0:
        rprint("[yellow]No testable acceptance criteria found[/yellow]")
        return

    for result in report.results:
        if result.success:
            rprint(f"[green]:white_check_mark: {result.criterion_id}[/green] - {result.message}")
        else:
            rprint(f"[red]:x: {result.criterion_id}[/red] - {result.message}")
        if result.duration_ms:
            rprint(f"    [dim]({result.duration_ms:.0f}ms)[/dim]")

    rprint()
    rprint(f"[bold]Results:[/bold] {report.passed}/{report.total} passed")

    if report.success:
        rprint("[bold green]All acceptance criteria met![/bold green]")
    else:
        rprint(f"[bold red]{report.failed} criteria failed[/bold red]")
        raise typer.Exit(1)


@app.command("runtime-check")
def runtime_check(
    path: Path = typer.Argument(
        None,
        help="Path to project directory (default: current directory)",
    ),
    feature: str = typer.Option(
        "",
        "--feature",
        "-f",
        help="Feature description for context",
    ),
) -> None:
    """Check recommended runtime environment for a project.

    Analyzes project characteristics and recommends:
    - docker: For web apps, services, complex dependencies
    - venv: For CLI tools, moderate complexity
    - local: For simple scripts, libraries

    Examples:
        acf runtime-check ./my-project
        acf runtime-check . --feature "Deploy to production"
    """
    from agents.runtime_decision_agent import decide_runtime

    project_dir = path or Path.cwd()

    if not project_dir.exists():
        rprint(f"[red]Path not found: {project_dir}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Runtime Decision: {project_dir}[/bold blue]")
    if feature:
        rprint(f"[dim]Feature context: {feature}[/dim]")
    rprint()

    # Detect framework first
    framework = ""
    for py_file in project_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            if "from fastapi" in content or "import fastapi" in content:
                framework = "fastapi"
                break
            elif "from flask" in content or "import flask" in content:
                framework = "flask"
                break
            elif "from django" in content or "import django" in content:
                framework = "django"
                break
        except Exception:
            continue

    # Get runtime decision
    decision = decide_runtime(
        project_dir=project_dir,
        feature_description=feature,
        framework=framework,
    )

    # Display result with appropriate color
    color = {
        "docker": "cyan",
        "venv": "yellow",
        "local": "green",
    }.get(decision.runtime, "white")

    rprint(f"[{color} bold]Recommended: {decision.runtime.upper()}[/{color} bold]")
    rprint(f"[dim]Confidence: {decision.confidence:.0%}[/dim]")
    rprint()
    rprint(f"[bold]Reason:[/bold] {decision.reason}")
    rprint()

    # Show signals
    signals = decision.signals
    rprint("[bold]Analysis:[/bold]")
    if signals.get("framework"):
        rprint(f"  Framework: {signals['framework']} ({signals.get('framework_type', 'unknown')} type)")
    if signals.get("line_count"):
        rprint(f"  Code size: {signals['line_count']} lines across {signals.get('file_count', 0)} files")
    if signals.get("services_detected"):
        rprint(f"  Services: {', '.join(signals['services_detected'])}")
    if signals.get("has_dockerfile"):
        rprint("  Existing Dockerfile: Yes")
    if signals.get("docker_intent_keywords"):
        rprint(f"  Docker keywords: {', '.join(signals['docker_intent_keywords'])}")
    if signals.get("local_intent_keywords"):
        rprint(f"  Local keywords: {', '.join(signals['local_intent_keywords'])}")

    rprint()
    rprint("[dim]To override, set runtime.mode in config.toml[/dim]")


@app.command("generate-tests")
def generate_tests(
    run_id: str = typer.Argument(..., help="Run ID to generate tests for"),
) -> None:
    """Generate pytest test stubs for a generated project.

    Analyzes the code and creates:
    - Test files for each source file
    - API endpoint tests for Flask/FastAPI
    - conftest.py with common fixtures

    Examples:
        acf generate-tests 2026-01-05-151822
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Generating tests for: {run_id}[/bold blue]")
    rprint()

    # Detect framework
    framework = "python"
    for py_file in project_dir.rglob("*.py"):
        content = py_file.read_text()
        if "from fastapi" in content or "import fastapi" in content:
            framework = "fastapi"
            break
        elif "from flask" in content or "import flask" in content:
            framework = "flask"
            break

    rprint(f"[dim]Framework detected: {framework}[/dim]")

    from agents.test_generator_agent import TestGeneratorAgent
    from agents.base import AgentInput

    agent = TestGeneratorAgent()
    output = agent.run(AgentInput(context={
        "project_dir": str(project_dir),
        "framework": framework,
    }))

    if output.success:
        files = output.data.get("files_generated", [])
        rprint(f"[green]Generated {len(files)} test files:[/green]")
        for f in files:
            rprint(f"  [cyan]{f}[/cyan]")

        rprint()
        rprint(f"[dim]Functions found: {output.data.get('functions_found', 0)}[/dim]")
        rprint(f"[dim]Classes found: {output.data.get('classes_found', 0)}[/dim]")
        rprint(f"[dim]Endpoints found: {output.data.get('endpoints_found', 0)}[/dim]")

        rprint()
        rprint("[bold]Run tests with:[/bold]")
        rprint(f"  cd {project_dir} && pytest tests/ -v")
    else:
        rprint(f"[red]Test generation failed: {output.errors}[/red]")
        raise typer.Exit(1)


@app.command()
def check() -> None:
    """Check if the LLM backend (Ollama) is available."""
    config = get_config()

    rprint("[bold]Checking LLM backend...[/bold]")
    rprint(f"[dim]Backend: {config.llm.backend}[/dim]")
    rprint(f"[dim]URL: {config.llm.base_url}[/dim]")
    rprint()

    try:
        backend = get_backend(
            config.llm.backend,
            model=config.llm.model_general,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )

        if backend.is_available():
            rprint("[green]Ollama is running and accessible.[/green]")
            rprint()

            # List available models
            models = backend.list_models()
            if models:
                table = Table(title="Available Models")
                table.add_column("Model", style="cyan")

                for model in models:
                    table.add_row(model)

                console.print(table)
            else:
                rprint("[yellow]No models found. Pull a model with:[/yellow]")
                rprint("  ollama pull llama3.1:8b")
        else:
            rprint("[red]Ollama is not responding.[/red]")
            rprint()
            rprint("[yellow]Make sure Ollama is running:[/yellow]")
            rprint("  ollama serve")
            raise typer.Exit(1)

    except ConnectionError as e:
        rprint(f"[red]Connection error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    config = get_config()

    rprint(f"[bold blue]Coding Factory[/bold blue] v{__version__}")
    rprint()
    rprint(f"[dim]LLM Backend:[/dim] {config.llm.backend}")
    rprint(f"[dim]General Model:[/dim] {config.llm.model_general}")
    rprint(f"[dim]Code Model:[/dim] {config.llm.model_code}")


@app.command()
def config_show() -> None:
    """Show current configuration."""
    cfg = get_config()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # LLM settings
    table.add_row("llm.backend", cfg.llm.backend)
    table.add_row("llm.base_url", cfg.llm.base_url)
    table.add_row("llm.model_general", cfg.llm.model_general)
    table.add_row("llm.model_code", cfg.llm.model_code)
    table.add_row("llm.timeout", str(cfg.llm.timeout))

    # Pipeline settings
    table.add_row("pipeline.artifacts_dir", cfg.pipeline.artifacts_dir)
    table.add_row("pipeline.log_level", cfg.pipeline.log_level)

    # Deploy settings
    table.add_row("deploy.strategy", cfg.deploy.strategy)
    if cfg.deploy.registry:
        table.add_row("deploy.registry", cfg.deploy.registry)

    console.print(table)


# =============================================================================
# Deployment Commands
# =============================================================================


@app.command()
def deploy(
    run_id: str = typer.Argument(..., help="Run ID to deploy"),
    version: str = typer.Option(
        None,
        "--version",
        "-v",
        help="Version tag (default: run_id)",
    ),
    strategy: str = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Deployment strategy (overrides config)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deployed without deploying",
    ),
) -> None:
    """Deploy a generated project to production.

    Supports multiple deployment strategies:
    - docker-push: Push image to registry
    - render: Deploy to Render.com
    - fly: Deploy to Fly.io
    - k8s: Deploy to Kubernetes
    - ssh: Deploy via SSH/rsync
    - custom: Run custom deploy script

    Examples:
        acf deploy 2026-01-05-151822
        acf deploy 2026-01-05-151822 --version v1.0.0
        acf deploy 2026-01-05-151822 --strategy fly
    """
    from tools.deploy import DeploymentManager, DeployConfig, generate_deploy_script

    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    version = version or run_id
    deploy_strategy = strategy or config.deploy.strategy

    rprint(f"[bold blue]Deploy: {run_id}[/bold blue]")
    rprint(f"[dim]Strategy: {deploy_strategy}[/dim]")
    rprint(f"[dim]Version: {version}[/dim]")
    rprint()

    if dry_run:
        rprint("[yellow]Dry run - showing deploy script:[/yellow]")
        rprint()
        script = generate_deploy_script(
            project_dir,
            strategy=deploy_strategy,
            config={
                "registry": config.deploy.registry,
                "app_name": config.deploy.fly_app_name,
            },
        )
        rprint(script)
        return

    # Create deploy config from CLI config
    deploy_config = DeployConfig(
        strategy=deploy_strategy,
        registry=config.deploy.registry,
        image_name=config.deploy.image_name or project_dir.name,
        render_service_id=config.deploy.render_service_id,
        fly_app_name=config.deploy.fly_app_name,
        k8s_namespace=config.deploy.k8s_namespace,
        k8s_deployment=config.deploy.k8s_deployment,
        k8s_context=config.deploy.k8s_context,
        ssh_host=config.deploy.ssh_host,
        ssh_user=config.deploy.ssh_user,
        ssh_key_path=config.deploy.ssh_key_path,
        ssh_deploy_path=config.deploy.ssh_deploy_path,
        custom_script=config.deploy.custom_script,
        health_check_url=config.deploy.health_check_url,
        health_check_timeout=config.deploy.health_check_timeout,
        rollback_on_failure=config.deploy.rollback_on_failure,
    )

    manager = DeploymentManager(deploy_config)

    rprint("[bold]Deploying...[/bold]")
    result = manager.deploy(project_dir, version)

    if result.success:
        rprint(f"[green]Deploy successful![/green]")
        rprint(f"  Target: {result.target}")
        rprint(f"  Version: {result.version}")
        if result.url:
            rprint(f"  URL: {result.url}")
        rprint(f"  Duration: {result.duration_seconds:.1f}s")
    else:
        rprint(f"[red]Deploy failed![/red]")
        rprint(f"  Error: {result.message}")
        if result.logs:
            rprint()
            rprint("[dim]Logs:[/dim]")
            rprint(result.logs[:1000])
        raise typer.Exit(1)


@app.command()
def rollback(
    run_id: str = typer.Argument(..., help="Run ID to rollback"),
    version: str = typer.Option(
        None,
        "--to",
        "-t",
        help="Version to rollback to (default: previous)",
    ),
    strategy: str = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Rollback strategy (overrides config)",
    ),
) -> None:
    """Rollback a deployment to a previous version.

    Supports rollback for:
    - docker: Retag previous image as latest
    - k8s: kubectl rollout undo
    - fly: flyctl releases rollback
    - git: Checkout previous tag
    - custom: Run rollback.sh script

    Examples:
        acf rollback 2026-01-05-151822
        acf rollback 2026-01-05-151822 --to v0.9.0
    """
    from tools.rollback import RollbackManager, generate_rollback_script

    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"
    history_file = Path(config.pipeline.artifacts_dir) / run_id / "deploy_history.json"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    rollback_strategy = strategy or config.deploy.strategy.replace("-push", "")

    rprint(f"[bold blue]Rollback: {run_id}[/bold blue]")
    rprint(f"[dim]Strategy: {rollback_strategy}[/dim]")
    if version:
        rprint(f"[dim]Target version: {version}[/dim]")
    else:
        rprint("[dim]Target: previous version[/dim]")
    rprint()

    manager = RollbackManager(history_file=history_file)

    # Get rollback config
    rollback_config = {
        "registry": config.deploy.registry,
        "image_name": config.deploy.image_name or project_dir.name,
        "namespace": config.deploy.k8s_namespace,
        "deployment": config.deploy.k8s_deployment,
        "context": config.deploy.k8s_context,
        "app_name": config.deploy.fly_app_name,
    }

    rprint("[bold]Rolling back...[/bold]")
    result = manager.rollback(
        project_dir=project_dir,
        target_version=version,
        strategy=rollback_strategy,
        config=rollback_config,
    )

    if result.success:
        rprint(f"[green]Rollback successful![/green]")
        rprint(f"  From: {result.from_version}")
        rprint(f"  To: {result.to_version}")
        rprint(f"  Duration: {result.duration_seconds:.1f}s")
    else:
        rprint(f"[red]Rollback failed![/red]")
        rprint(f"  Error: {result.message}")
        if result.logs:
            rprint()
            rprint("[dim]Logs:[/dim]")
            rprint(result.logs[:1000])
        raise typer.Exit(1)


@app.command("deploy-history")
def deploy_history(
    run_id: str = typer.Argument(..., help="Run ID to show history for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of records to show"),
) -> None:
    """Show deployment history for a run.

    Examples:
        acf deploy-history 2026-01-05-151822
    """
    from tools.rollback import RollbackManager

    config = get_config()
    history_file = Path(config.pipeline.artifacts_dir) / run_id / "deploy_history.json"

    if not history_file.exists():
        rprint(f"[yellow]No deployment history found for {run_id}[/yellow]")
        return

    manager = RollbackManager(history_file=history_file)
    history = manager.get_history(limit=limit)

    if not history:
        rprint("[yellow]No deployments recorded[/yellow]")
        return

    table = Table(title=f"Deployment History: {run_id}")
    table.add_column("Version", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Target", style="blue")
    table.add_column("Status")

    for record in history:
        status = "[green]Success[/green]" if record.success else "[red]Failed[/red]"
        timestamp = record.timestamp[:19]  # Remove microseconds
        table.add_row(record.version, timestamp, record.target, status)

    console.print(table)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
) -> None:
    """Start the web dashboard for monitoring runs.

    The dashboard provides:
    - List of all pipeline runs
    - Run details and stage status
    - Artifact viewing
    - Deploy and rollback actions

    Requires: pip install fastapi uvicorn

    Examples:
        acf dashboard
        acf dashboard --port 3000
    """
    try:
        from dashboard import run_dashboard
    except ImportError:
        rprint("[red]Dashboard requires FastAPI and uvicorn.[/red]")
        rprint("Install with: [cyan]pip install fastapi uvicorn[/cyan]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Coding Factory Dashboard[/bold blue]")
    rprint(f"[green]Starting server at http://{host}:{port}[/green]")
    rprint("[dim]Press Ctrl+C to stop[/dim]")
    rprint()

    run_dashboard(host=host, port=port)


# =============================================================================
# RAG Commands
# =============================================================================


@app.command()
def index(
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository path (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-index even if index exists",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        "-i",
        help="Only index changed files (faster updates)",
    ),
    model: str = typer.Option(
        "nomic-embed-text",
        "--model",
        "-m",
        help="Embedding model to use",
    ),
) -> None:
    """Index repository for RAG-powered search.

    Creates vector embeddings of your codebase for semantic search.
    The index is stored in .acf-index/ directory.

    Use --incremental for fast updates that only re-index changed files.

    Examples:
        acf index
        acf index --force
        acf index --incremental
        acf index -r /path/to/repo
    """
    from rag import CodeRetriever

    config = get_config()
    repo_path = repo or Path.cwd()

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Repository:[/green] {repo_path}")
    rprint(f"[green]Embedding Model:[/green] {model}")
    if incremental:
        rprint(f"[green]Mode:[/green] Incremental (changed files only)")
    rprint()

    # Check for PDF files and warn about limitations
    pdf_files = list(repo_path.rglob("*.pdf"))
    if pdf_files:
        rprint("[yellow]Note:[/yellow] PDF files detected. PDF indexing extracts TEXT ONLY.")
        rprint("[yellow]      Images, tables, charts, and diagrams are NOT supported.[/yellow]")
        rprint()

    try:
        retriever = CodeRetriever(
            repo_path=repo_path,
            embedding_model=model,
            ollama_url=config.llm.base_url,
        )
        stats = retriever.index(force=force, incremental=incremental)

        rprint()
        rprint("[bold]Index Statistics:[/bold]")
        rprint(f"  Chunks: {stats.get('total_chunks', 0)}")
        rprint(f"  Files: {stats.get('total_files', 0)}")
        rprint(f"  Tracked Files: {stats.get('tracked_files', 0)}")
        rprint(f"  Dimensions: {stats.get('dimensions', 0)}")
        if stats.get('last_indexed'):
            rprint(f"  Last Indexed: {stats.get('last_indexed')}")

        # Show incremental stats if available
        if incremental:
            rprint()
            rprint("[bold]Update Summary:[/bold]")
            rprint(f"  Files Updated: {stats.get('files_updated', 0)}")
            rprint(f"  Files Deleted: {stats.get('files_deleted', 0)}")

    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository path (default: current directory)",
    ),
    top_k: int = typer.Option(
        5,
        "--top",
        "-k",
        help="Number of results to show",
    ),
    show_content: bool = typer.Option(
        False,
        "--content",
        "-c",
        help="Show full chunk content",
    ),
) -> None:
    """Search indexed codebase using natural language.

    Examples:
        acf search "authentication logic"
        acf search "database connection" --top 10
        acf search "error handling" --content
    """
    from rag import CodeRetriever

    config = get_config()
    repo_path = repo or Path.cwd()

    retriever = CodeRetriever(
        repo_path=repo_path,
        ollama_url=config.llm.base_url,
    )

    if not retriever.is_indexed():
        rprint("[yellow]Repository not indexed. Run 'acf index' first.[/yellow]")
        raise typer.Exit(1)

    rprint(f"[dim]Searching for: {query}[/dim]")
    rprint()

    results = retriever.search(query, top_k=top_k)

    if not results:
        rprint("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        score_pct = result.score * 100

        rprint(f"[bold cyan]{i}.[/bold cyan] {chunk.file_path}")
        rprint(f"   [dim]Lines {chunk.start_line}-{chunk.end_line} | Score: {score_pct:.1f}%[/dim]")

        if chunk.name:
            rprint(f"   [green]{chunk.chunk_type}:[/green] {chunk.name}")

        if show_content:
            rprint()
            # Show first 10 lines of content
            lines = chunk.content.split("\n")[:10]
            for line in lines:
                rprint(f"   [dim]{line}[/dim]")
            if len(chunk.content.split("\n")) > 10:
                rprint(f"   [dim]... ({len(chunk.content.split(chr(10))) - 10} more lines)[/dim]")

        rprint()


@app.command()
def index_stats(
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository path (default: current directory)",
    ),
) -> None:
    """Show RAG index statistics."""
    from rag import CodeRetriever

    config = get_config()
    repo_path = repo or Path.cwd()

    retriever = CodeRetriever(
        repo_path=repo_path,
        ollama_url=config.llm.base_url,
    )

    if not retriever.is_indexed():
        rprint("[yellow]Repository not indexed.[/yellow]")
        raise typer.Exit(1)

    stats = retriever.stats()

    rprint("[bold]Index Statistics:[/bold]")
    rprint()
    rprint(f"  [cyan]Total Chunks:[/cyan] {stats.get('total_chunks', 0)}")
    rprint(f"  [cyan]Total Files:[/cyan] {stats.get('total_files', 0)}")
    rprint(f"  [cyan]Tracked Files:[/cyan] {stats.get('tracked_files', 0)}")
    rprint(f"  [cyan]Dimensions:[/cyan] {stats.get('dimensions', 0)}")
    rprint(f"  [cyan]Index Version:[/cyan] {stats.get('version', 1)}")
    if stats.get('last_indexed'):
        rprint(f"  [cyan]Last Indexed:[/cyan] {stats.get('last_indexed')}")

    # Show repositories if multi-repo
    repos = stats.get("repos", [])
    if repos:
        rprint()
        rprint("[bold]Indexed Repositories:[/bold]")
        for r in repos:
            rprint(f"  {r}")

    files = stats.get("files", [])
    if files:
        rprint()
        rprint("[bold]Indexed Files (sample):[/bold]")
        for f in files[:20]:
            rprint(f"  {f}")
        if len(files) > 20:
            rprint(f"  ... and {len(files) - 20} more")


@app.command("index-add-repo")
def index_add_repo(
    repo_path: Path = typer.Argument(..., help="Path to repository to add"),
    repo_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Repository identifier (default: directory name)",
    ),
    base_repo: Optional[Path] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base repository containing the index (default: current directory)",
    ),
    model: str = typer.Option(
        "nomic-embed-text",
        "--model",
        "-m",
        help="Embedding model to use",
    ),
) -> None:
    """Add another repository to the index (multi-repo support).

    Allows indexing multiple repositories into a single searchable index.

    Examples:
        acf index-add-repo /path/to/other-repo
        acf index-add-repo /path/to/lib --id my-lib
        acf index-add-repo /path/to/dep --base /path/to/main-repo
    """
    from rag import CodeRetriever

    config = get_config()
    base_path = base_repo or Path.cwd()

    if not (base_path / ".acf-index").exists():
        rprint("[yellow]Base repository not indexed. Run 'acf index' first.[/yellow]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Base Repository:[/green] {base_path}")
    rprint(f"[green]Adding Repository:[/green] {repo_path}")
    if repo_id:
        rprint(f"[green]Repository ID:[/green] {repo_id}")
    rprint()

    try:
        retriever = CodeRetriever(
            repo_path=base_path,
            embedding_model=model,
            ollama_url=config.llm.base_url,
        )
        stats = retriever.add_repo(repo_path, repo_id=repo_id)

        rprint()
        rprint("[bold]Index Statistics:[/bold]")
        rprint(f"  Total Chunks: {stats.get('total_chunks', 0)}")
        rprint(f"  Total Files: {stats.get('total_files', 0)}")
        rprint(f"  Repositories: {len(stats.get('repos', []))}")

    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("index-remove-repo")
def index_remove_repo(
    repo_id: str = typer.Argument(..., help="Repository ID to remove"),
    base_repo: Optional[Path] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base repository containing the index (default: current directory)",
    ),
) -> None:
    """Remove a repository from the index.

    Examples:
        acf index-remove-repo my-lib
        acf index-remove-repo old-dep --base /path/to/main-repo
    """
    from rag import CodeRetriever

    config = get_config()
    base_path = base_repo or Path.cwd()

    if not (base_path / ".acf-index").exists():
        rprint("[yellow]No index found.[/yellow]")
        raise typer.Exit(1)

    try:
        retriever = CodeRetriever(
            repo_path=base_path,
            ollama_url=config.llm.base_url,
        )
        deleted = retriever.remove_repo(repo_id)

        if deleted > 0:
            rprint(f"[green]Removed {deleted} chunks from '{repo_id}'[/green]")
        else:
            rprint(f"[yellow]Repository '{repo_id}' not found in index[/yellow]")

    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("index-repos")
def index_repos(
    repo: Optional[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository path (default: current directory)",
    ),
) -> None:
    """List all repositories in the index.

    Examples:
        acf index-repos
        acf index-repos --repo /path/to/project
    """
    from rag import CodeRetriever

    config = get_config()
    repo_path = repo or Path.cwd()

    retriever = CodeRetriever(
        repo_path=repo_path,
        ollama_url=config.llm.base_url,
    )

    if not retriever.is_indexed():
        rprint("[yellow]Repository not indexed.[/yellow]")
        raise typer.Exit(1)

    repos = retriever.list_repos()

    if not repos:
        rprint("[yellow]No repositories registered in index.[/yellow]")
        return

    rprint("[bold]Indexed Repositories:[/bold]")
    rprint()
    for repo_info in repos:
        rprint(f"  [cyan]{repo_info['repo_id']}[/cyan]")
        rprint(f"    Path: {repo_info.get('path', 'N/A')}")
        rprint(f"    Files: {repo_info.get('file_count', 0)}")
        rprint(f"    Indexed: {repo_info.get('indexed_at', 'N/A')}")
        rprint()


@app.command("rag-optimize")
def rag_optimize(
    docs_dir: Path = typer.Argument(..., help="Directory containing documents to optimize"),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for optimized index (default: docs_dir/.rag-optimized)",
    ),
    skip_images: bool = typer.Option(
        False,
        "--skip-images",
        help="Skip image processing (OCR and vision)",
    ),
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Skip LLM-based extraction (summaries, Q&A, entities)",
    ),
    index_after: bool = typer.Option(
        False,
        "--index",
        "-i",
        help="Run RAG indexing on optimized output after processing",
    ),
) -> None:
    """Optimize documents for better RAG retrieval.

    Preprocesses documents to create a multi-layer index:
    - Smart chunks (split by headers, paragraphs)
    - Summaries (document and section level)
    - Q&A pairs (extracted questions and answers)
    - Entities (key terms and definitions)
    - Image descriptions (OCR + vision)

    This dramatically improves retrieval quality by providing
    multiple pathways for different query types.

    Examples:
        acf rag-optimize ./docs
        acf rag-optimize ./docs --output ./optimized
        acf rag-optimize ./docs --skip-images
        acf rag-optimize ./docs --index
    """
    from agents.rag_optimizer_agent import RAGOptimizerAgent

    if not docs_dir.exists():
        rprint(f"[red]Directory not found: {docs_dir}[/red]")
        raise typer.Exit(1)

    # Default output directory
    if output_dir is None:
        output_dir = docs_dir / ".rag-optimized"

    rprint(f"[bold blue]RAG Optimizer[/bold blue]")
    rprint(f"[dim]Input:  {docs_dir.resolve()}[/dim]")
    rprint(f"[dim]Output: {output_dir.resolve()}[/dim]")
    rprint()

    # Initialize LLM if not skipping
    llm = None
    if not skip_llm:
        try:
            from llm_backend import get_backend
            config = get_config()
            llm = get_backend(
                config.llm.backend,
                base_url=config.llm.base_url,
                model=config.llm.model_general,
            )
            rprint("[dim]LLM: Enabled (summaries, Q&A, entities)[/dim]")
        except Exception as e:
            rprint(f"[yellow]Warning: LLM not available ({e}), skipping extraction[/yellow]")
            skip_llm = True

    if skip_llm:
        rprint("[dim]LLM: Disabled (chunking only)[/dim]")
    if skip_images:
        rprint("[dim]Images: Skipped[/dim]")

    rprint()
    rprint("[bold]Processing documents...[/bold]")

    # Run optimization
    agent = RAGOptimizerAgent(llm=llm)
    index = agent.optimize(
        docs_dir=docs_dir,
        output_dir=output_dir,
        skip_images=skip_images,
        skip_llm=skip_llm,
    )

    # Display results
    stats = index.to_dict()["stats"]
    rprint()
    rprint("[bold green]Optimization complete![/bold green]")
    rprint()
    rprint("[bold]Generated:[/bold]")
    rprint(f"  Text chunks:    {stats['total_chunks']}")
    rprint(f"  Summaries:      {stats['total_summaries']}")
    rprint(f"  Q&A pairs:      {stats['total_qa_pairs']}")
    rprint(f"  Entities:       {stats['total_entities']}")
    rprint(f"  Image descs:    {stats['total_images']}")
    rprint()
    rprint(f"[dim]Total items for indexing: {len(index.all_chunks())}[/dim]")
    rprint(f"[dim]Output saved to: {output_dir}[/dim]")

    # Run indexing if requested
    if index_after:
        rprint()
        rprint("[bold]Running RAG indexing on optimized content...[/bold]")
        from rag import CodeRetriever
        config = get_config()

        chunks_dir = output_dir / "chunks"
        if chunks_dir.exists():
            retriever = CodeRetriever(
                repo_path=chunks_dir,
                ollama_url=config.llm.base_url,
            )
            retriever.index(force=True)
            rprint("[green]✓ Indexing complete[/green]")
        else:
            rprint("[yellow]No chunks directory found, skipping indexing[/yellow]")


# =============================================================================
# Web Interface
# =============================================================================


@app.command()
def web(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
) -> None:
    """Start the web interface.

    Examples:
        acf web
        acf web --port 8080
        acf web --reload
    """
    import uvicorn

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Starting web interface...[/green]")
    rprint(f"[dim]URL: http://{host}:{port}[/dim]")
    rprint()

    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def iterate(
    run_id: str = typer.Argument(..., help="Run ID of existing project to improve"),
    improvement: str = typer.Argument(..., help="Description of improvements to make"),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "-y",
        help="Auto-approve all checkpoints",
    ),
) -> None:
    """Iterate on an existing generated project.

    Takes a previous run and applies improvements/modifications.

    Examples:
        acf iterate 2026-01-05-151822 "Add product categories"
        acf iterate 2026-01-05-151822 "Improve CSS with animations"
        acf iterate 2026-01-05-151822 "Add search functionality" -y
    """
    import json as json_module

    config = get_config()
    artifacts_dir = _find_run_artifacts(run_id, config)

    if not artifacts_dir:
        rprint(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    # Load previous state to get context and project_dir
    state_file = artifacts_dir / "state.json"
    original_feature = "Unknown"
    project_dir = None

    if state_file.exists():
        with open(state_file) as f:
            state = json_module.load(f)
            original_feature = state.get("feature_description", "")
            # New structure: project_dir is in state
            proj_dir = state.get("project_dir")
            if proj_dir and Path(proj_dir).exists():
                project_dir = Path(proj_dir)

    # Legacy fallback: generated_project inside artifacts
    if not project_dir:
        legacy_dir = artifacts_dir / "generated_project"
        if legacy_dir.exists():
            project_dir = legacy_dir

    if not project_dir or not project_dir.exists():
        rprint(f"[red]No project found for run {run_id}[/red]")
        rprint(f"[dim]Run 'acf extract {run_id}' first if needed[/dim]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Coding Factory - Iteration Mode[/bold blue]")
    rprint()
    rprint(f"[green]Base project:[/green] {run_id}")
    rprint(f"[dim]Original feature: {original_feature[:60]}...[/dim]" if len(original_feature) > 60 else f"[dim]Original: {original_feature}[/dim]")
    rprint()
    rprint(f"[green]Improvement:[/green] {improvement}")
    rprint()

    # Create runner with iteration context
    runner = PipelineRunner(
        config=config,
        console=console,
        auto_approve=auto_approve,
    )

    # Run pipeline with existing project as repo
    try:
        # The improvement becomes the new "feature" but with iteration context
        iteration_feature = f"[ITERATION of {run_id}] {improvement}"

        state = runner.run(
            feature=iteration_feature,
            repo_path=project_dir,  # Use existing project as repo
            dry_run=False,
            iteration_context={
                "base_run_id": run_id,
                "original_feature": original_feature,
                "improvement_request": improvement,
            },
        )

        if state.status.value == "completed":
            rprint()
            rprint("[bold green]Iteration completed![/bold green]")
            rprint()
            rprint(f"[dim]New run ID: {state.run_id}[/dim]")
            # Show project location (new structure vs legacy)
            if hasattr(state, 'project_dir') and state.project_dir:
                rprint(f"[dim]Project: {state.project_dir}[/dim]")
            else:
                rprint(f"[dim]Project: {Path(config.pipeline.artifacts_dir) / state.run_id / 'generated_project'}[/dim]")
        elif state.status.value == "paused":
            rprint()
            rprint("[yellow]Iteration paused. Resume with:[/yellow]")
            rprint(f"  acf run \"{iteration_feature}\" --resume {state.run_id}")
        elif state.status.value == "failed":
            rprint()
            rprint(f"[red]Iteration failed: {state.last_error}[/red]")
            raise typer.Exit(1)

    except ValueError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted[/yellow]")
        raise typer.Exit(130)


@app.command()
def history(
    run_id: str = typer.Argument(..., help="Run ID to show git history for"),
) -> None:
    """Show git history for a generated project.

    Displays commit history and branches for version control.

    Examples:
        acf history 2026-01-05-151822
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    from tools.git_manager import GitManager

    git = GitManager(project_dir)

    if not git.is_repo():
        rprint(f"[yellow]No git repository found in {run_id}/generated_project[/yellow]")
        rprint("[dim]Git integration was added in a later version.[/dim]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Git History: {run_id}[/bold blue]")
    rprint()

    # Show current branch
    current_branch = git.get_current_branch()
    rprint(f"[green]Current branch:[/green] {current_branch}")
    rprint()

    # Show all branches
    try:
        result = git._run_git("branch", "-a")
        branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
        if branches:
            rprint("[bold]Branches:[/bold]")
            for branch in branches:
                if branch.startswith("*"):
                    rprint(f"  [green]{branch}[/green]")
                else:
                    rprint(f"  {branch}")
            rprint()
    except Exception:
        pass

    # Show commit history
    commits = git.get_log(max_count=20)
    if commits:
        table = Table(title="Commit History")
        table.add_column("Hash", style="cyan", width=10)
        table.add_column("Message", style="white")
        table.add_column("Date", style="dim")

        for commit in commits:
            table.add_row(
                commit["hash"],
                commit["message"][:50] + ("..." if len(commit["message"]) > 50 else ""),
                commit["date"][:10],
            )

        console.print(table)
    else:
        rprint("[dim]No commits found.[/dim]")

    rprint()
    rprint(f"[dim]Project path: {project_dir}[/dim]")


@app.command()
def branches(
    run_id: str = typer.Argument(..., help="Run ID to list branches for"),
) -> None:
    """List all branches for a generated project.

    Shows available iteration branches.

    Examples:
        acf branches 2026-01-05-151822
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    from tools.git_manager import GitManager

    git = GitManager(project_dir)

    if not git.is_repo():
        rprint(f"[yellow]No git repository found[/yellow]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Branches: {run_id}[/bold blue]")
    rprint()

    try:
        result = git._run_git("branch", "-a", "--format=%(refname:short) %(objectname:short) %(subject)")
        lines = result.stdout.strip().split("\n")

        table = Table()
        table.add_column("Branch", style="cyan")
        table.add_column("Hash", style="dim")
        table.add_column("Last Commit", style="white")

        for line in lines:
            if line.strip():
                parts = line.split(" ", 2)
                branch = parts[0] if len(parts) > 0 else ""
                hash_val = parts[1] if len(parts) > 1 else ""
                message = parts[2] if len(parts) > 2 else ""

                # Highlight current branch
                current = git.get_current_branch()
                if branch == current:
                    branch = f"* {branch}"

                table.add_row(branch, hash_val, message[:40])

        console.print(table)

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")


@app.command()
def push(
    run_id: str = typer.Argument(..., help="Run ID to push to remote"),
    remote: str = typer.Option(
        None,
        "--remote",
        "-r",
        help="Remote URL (e.g., git@github.com:user/repo.git)",
    ),
    branch: Optional[str] = typer.Option(
        None,
        "--branch",
        "-b",
        help="Branch to push (defaults to current)",
    ),
) -> None:
    """Push a generated project to a remote repository.

    Handles fetch, rebase (if needed), and push.

    Examples:
        acf push 2026-01-05-151822 -r git@github.com:user/my-app.git
        acf push 2026-01-05-151822  # Uses config remote_url
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    from tools.git_manager import GitManager

    git = GitManager(project_dir)

    if not git.is_repo():
        rprint(f"[red]No git repository found. Run 'acf scaffold {run_id}' first.[/red]")
        raise typer.Exit(1)

    # Get remote URL
    url = remote or config.git.remote_url
    if not url:
        rprint("[red]No remote URL provided.[/red]")
        rprint("[dim]Use --remote or set git.remote_url in config.toml[/dim]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Pushing: {run_id}[/bold blue]")
    rprint()

    # Add/update remote
    result = git.add_remote(url, "origin")
    rprint(f"[dim]Remote: {url}[/dim]")

    # Get branch
    target_branch = branch or git.get_current_branch()
    rprint(f"[dim]Branch: {target_branch}[/dim]")
    rprint()

    # Sync and push
    rprint("[dim]Syncing with remote...[/dim]")
    push_result = git.sync_and_push("origin", target_branch)

    if push_result.success:
        rprint(f"[green]✓ {push_result.message}[/green]")
    else:
        rprint(f"[red]✗ {push_result.message}[/red]")
        if "conflict" in push_result.message.lower():
            rprint()
            rprint("[yellow]Conflict detected. Resolve manually:[/yellow]")
            rprint(f"  cd {project_dir}")
            rprint("  git status")
            rprint("  # Fix conflicts, then:")
            rprint("  git add . && git rebase --continue")
            rprint("  git push origin " + target_branch)
        raise typer.Exit(1)


@app.command()
def tag(
    run_id: str = typer.Argument(..., help="Run ID to tag"),
    message: str = typer.Option(
        None,
        "--message",
        "-m",
        help="Tag message (defaults to feature description)",
    ),
    bump: str = typer.Option(
        "patch",
        "--bump",
        "-b",
        help="Version bump type: major, minor, or patch",
    ),
    push_tag: bool = typer.Option(
        False,
        "--push",
        "-p",
        help="Push tag to remote after creating",
    ),
) -> None:
    """Create a release tag for a generated project.

    Auto-increments version based on existing tags (v0.1.0 → v0.1.1).

    Examples:
        acf tag 2026-01-05-151822
        acf tag 2026-01-05-151822 -m "First release" --push
        acf tag 2026-01-05-151822 --bump minor
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    from tools.git_manager import GitManager

    git = GitManager(project_dir)

    if not git.is_repo():
        rprint(f"[red]No git repository found[/red]")
        raise typer.Exit(1)

    # Get feature description for default message
    if not message:
        state_file = Path(config.pipeline.artifacts_dir) / run_id / "state.json"
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
                message = state.get("feature_description", "Release")[:100]
        else:
            message = "Release"

    rprint(f"[bold blue]Creating Release Tag: {run_id}[/bold blue]")
    rprint()

    # Show current tags
    tags = git.get_tags()
    if tags:
        rprint(f"[dim]Existing tags: {', '.join(tags[:5])}[/dim]")

    next_version = git.get_next_version(bump)
    rprint(f"[dim]Next version: {next_version} ({bump} bump)[/dim]")
    rprint()

    # Create tag
    result = git.create_release_tag(
        message=message,
        bump=bump,
        push=push_tag,
        remote="origin",
    )

    if result.success:
        rprint(f"[green]✓ {result.message}[/green]")
        rprint(f"[dim]Message: {message}[/dim]")
        if not push_tag:
            rprint()
            rprint("[dim]To push this tag:[/dim]")
            rprint(f"  acf push {run_id}")
            rprint(f"  cd {project_dir} && git push origin {result.output}")
    else:
        rprint(f"[red]✗ {result.message}[/red]")
        raise typer.Exit(1)


@app.command()
def pr(
    run_id: str = typer.Argument(..., help="Run ID to create PR for"),
    title: str = typer.Option(
        None,
        "--title",
        "-t",
        help="PR title (defaults to feature/improvement description)",
    ),
    body: str = typer.Option(
        "",
        "--body",
        "-b",
        help="PR description",
    ),
    base: str = typer.Option(
        "main",
        "--base",
        help="Target branch for the PR",
    ),
    draft: bool = typer.Option(
        False,
        "--draft",
        "-d",
        help="Create as draft PR",
    ),
) -> None:
    """Create a GitHub Pull Request for a generated project.

    Requires gh CLI installed and authenticated (gh auth login).

    Examples:
        acf pr 2026-01-05-151822
        acf pr 2026-01-05-151822 -t "Add dark mode feature"
        acf pr 2026-01-05-151822 --draft
    """
    config = get_config()
    project_dir = Path(config.pipeline.artifacts_dir) / run_id / "generated_project"

    if not project_dir.exists():
        rprint(f"[red]No generated_project found for run {run_id}[/red]")
        raise typer.Exit(1)

    from tools.git_manager import GitManager

    git = GitManager(project_dir)

    if not git.is_repo():
        rprint(f"[red]No git repository found[/red]")
        raise typer.Exit(1)

    # Check if on main branch (can't PR from main to main)
    current_branch = git.get_current_branch()
    if current_branch == base:
        rprint(f"[red]Cannot create PR: current branch '{current_branch}' is same as base '{base}'[/red]")
        rprint("[dim]PRs are typically created from iteration branches.[/dim]")
        raise typer.Exit(1)

    # Get title from state if not provided
    if not title:
        state_file = Path(config.pipeline.artifacts_dir) / run_id / "state.json"
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
                # Check for iteration context
                iteration = state.get("iteration_context", {})
                if iteration:
                    title = f"Iteration: {iteration.get('improvement_request', 'Improvements')[:60]}"
                else:
                    title = state.get("feature_description", "Feature update")[:60]
        else:
            title = f"Changes from {run_id}"

    # Build body if not provided
    if not body:
        state_file = Path(config.pipeline.artifacts_dir) / run_id / "state.json"
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
                body_parts = []
                body_parts.append(f"## Summary\n{state.get('feature_description', '')}\n")

                iteration = state.get("iteration_context", {})
                if iteration:
                    body_parts.append(f"**Improvement:** {iteration.get('improvement_request', '')}\n")
                    body_parts.append(f"**Base run:** {iteration.get('base_run_id', '')}\n")

                body_parts.append(f"\n---\n*Generated by Coding Factory*\n*Run ID: {run_id}*")
                body = "\n".join(body_parts)

    rprint(f"[bold blue]Creating Pull Request: {run_id}[/bold blue]")
    rprint()
    rprint(f"[dim]Branch: {current_branch} → {base}[/dim]")
    rprint(f"[dim]Title: {title}[/dim]")
    if draft:
        rprint(f"[dim]Status: Draft[/dim]")
    rprint()

    # Create PR
    result = git.create_pull_request(
        title=title,
        body=body,
        base=base,
        head=current_branch,
        draft=draft,
    )

    if result.success:
        rprint(f"[green]✓ {result.message}[/green]")
        rprint()
        rprint(f"[bold]PR URL:[/bold] {result.output}")
    else:
        rprint(f"[red]✗ {result.message}[/red]")
        if "not installed" in result.message.lower():
            rprint()
            rprint("[yellow]Install gh CLI:[/yellow]")
            rprint("  brew install gh      # macOS")
            rprint("  apt install gh       # Ubuntu")
            rprint("  https://cli.github.com")
        elif "not authenticated" in result.message.lower():
            rprint()
            rprint("[yellow]Authenticate with GitHub:[/yellow]")
            rprint("  gh auth login")
        raise typer.Exit(1)


# =============================================================================
# Template Management
# =============================================================================


@app.command("template-create")
def template_create(
    name: str = typer.Argument(..., help="Template name"),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Source project directory (default: current directory)",
    ),
    description: str = typer.Option(
        "",
        "--description",
        "-d",
        help="Template description",
    ),
    language: str = typer.Option(
        "python",
        "--language",
        "-l",
        help="Programming language",
    ),
    framework: str = typer.Option(
        "custom",
        "--framework",
        "-f",
        help="Framework used",
    ),
) -> None:
    """Create a custom template from an existing project.

    Saves the project structure as a reusable template.

    Examples:
        acf template-create my-api
        acf template-create my-cli -s /path/to/project -d "My CLI template"
    """
    from scaffolding import save_template, get_templates_dir

    source_dir = source or Path.cwd()

    if not source_dir.exists():
        rprint(f"[red]Source directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Creating template:[/green] {name}")
    rprint(f"[green]Source:[/green] {source_dir}")
    rprint()

    try:
        template_dir = save_template(
            name=name,
            source_dir=source_dir,
            description=description,
            language=language,
            framework=framework,
        )
        rprint(f"[green]✓ Template created successfully![/green]")
        rprint(f"[dim]Location: {template_dir}[/dim]")
        rprint()
        rprint(f"Use with: [cyan]acf new my-project --template {name}[/cyan]")

    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("template-show")
def template_show(
    name: str = typer.Argument(..., help="Template name"),
) -> None:
    """Show details of a template.

    Examples:
        acf template-show fastapi
        acf template-show my-custom-template
    """
    from scaffolding import get_template, BUILTIN_TEMPLATES

    template = get_template(name)

    if not template:
        rprint(f"[red]Template not found: {name}[/red]")
        raise typer.Exit(1)

    source = "built-in" if name in BUILTIN_TEMPLATES else "custom"

    rprint(f"[bold blue]Template: {name}[/bold blue]")
    rprint()
    rprint(f"[cyan]Description:[/cyan] {template.description}")
    rprint(f"[cyan]Type:[/cyan] {source}")
    rprint(f"[cyan]Language:[/cyan] {template.language}")
    rprint(f"[cyan]Framework:[/cyan] {template.framework}")
    rprint()

    if template.dependencies:
        rprint("[cyan]Dependencies:[/cyan]")
        for dep in template.dependencies:
            rprint(f"  - {dep}")
        rprint()

    if template.dev_dependencies:
        rprint("[cyan]Dev Dependencies:[/cyan]")
        for dep in template.dev_dependencies:
            rprint(f"  - {dep}")
        rprint()

    rprint("[cyan]Files:[/cyan]")
    for file_path in sorted(template.files.keys()):
        rprint(f"  {file_path}")


@app.command("template-delete")
def template_delete(
    name: str = typer.Argument(..., help="Template name to delete"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Delete a custom template.

    Cannot delete built-in templates.

    Examples:
        acf template-delete my-old-template
        acf template-delete unused-template --force
    """
    from scaffolding import delete_template, BUILTIN_TEMPLATES

    if name in BUILTIN_TEMPLATES:
        rprint(f"[red]Cannot delete built-in template: {name}[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete template '{name}'?")
        if not confirm:
            rprint("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    if delete_template(name):
        rprint(f"[green]✓ Template '{name}' deleted[/green]")
    else:
        rprint(f"[red]Template not found: {name}[/red]")
        raise typer.Exit(1)


@app.command("template-dir")
def template_dir() -> None:
    """Show the custom templates directory.

    Templates placed here are automatically discovered.
    """
    from scaffolding import get_templates_dir

    templates_dir = get_templates_dir()
    rprint(f"[bold]Custom templates directory:[/bold]")
    rprint(f"  {templates_dir}")
    rprint()
    rprint("[dim]Place template folders here with a template.toml manifest.[/dim]")


# =============================================================================
# Template Marketplace
# =============================================================================


@app.command("market-search")
def market_search(
    query: str = typer.Argument("", help="Search query"),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Filter by language",
    ),
    framework: Optional[str] = typer.Option(
        None,
        "--framework",
        "-f",
        help="Filter by framework",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of results",
    ),
) -> None:
    """Search the template marketplace.

    Examples:
        acf market-search fastapi
        acf market-search --language python
        acf market-search api --framework flask
    """
    from scaffolding import MarketplaceClient, MarketplaceError

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[dim]Searching marketplace for: {query or '(all)'}[/dim]")
    rprint()

    try:
        client = MarketplaceClient()
        results = client.search(
            query=query,
            language=language,
            framework=framework,
            per_page=limit,
        )

        if not results.templates:
            rprint("[yellow]No templates found.[/yellow]")
            return

        rprint(f"[bold]Found {results.total} templates:[/bold]")
        rprint()

        for tmpl in results.templates:
            stars = f"★{tmpl.stars}" if tmpl.stars else ""
            downloads = f"↓{tmpl.downloads}" if tmpl.downloads else ""
            stats = f"[dim]{stars} {downloads}[/dim]" if stars or downloads else ""

            rprint(f"  [cyan]{tmpl.name}[/cyan] by {tmpl.author} {stats}")
            rprint(f"    {tmpl.description}")
            rprint(f"    [dim]{tmpl.language} | {tmpl.framework}[/dim]")
            rprint()

    except MarketplaceError as e:
        rprint(f"[yellow]Marketplace unavailable: {e}[/yellow]")
        rprint("[dim]The marketplace service may not be running.[/dim]")


@app.command("market-install")
def market_install(
    template_id: str = typer.Argument(..., help="Template ID or name to install"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing template",
    ),
) -> None:
    """Install a template from the marketplace.

    Examples:
        acf market-install awesome-fastapi
        acf market-install user/template-name --force
    """
    from scaffolding import MarketplaceClient, MarketplaceError, LocalRegistry

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Installing template:[/green] {template_id}")
    rprint()

    try:
        client = MarketplaceClient()
        template = client.install(template_id, force=force)

        # Track in local registry
        registry = LocalRegistry()
        info = client.get_template_info(template_id)
        registry.add(info)

        rprint(f"[green]✓ Template '{template.name}' installed successfully![/green]")
        rprint()
        rprint(f"Use with: [cyan]acf new my-project --template {template.name}[/cyan]")

    except MarketplaceError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("market-info")
def market_info(
    template_id: str = typer.Argument(..., help="Template ID or name"),
) -> None:
    """Show details of a marketplace template.

    Examples:
        acf market-info awesome-fastapi
    """
    from scaffolding import MarketplaceClient, MarketplaceError

    try:
        client = MarketplaceClient()
        tmpl = client.get_template_info(template_id)

        rprint(f"[bold blue]Template: {tmpl.name}[/bold blue]")
        rprint()
        rprint(f"[cyan]Author:[/cyan] {tmpl.author}")
        rprint(f"[cyan]Version:[/cyan] {tmpl.version}")
        rprint(f"[cyan]Description:[/cyan] {tmpl.description}")
        rprint(f"[cyan]Language:[/cyan] {tmpl.language}")
        rprint(f"[cyan]Framework:[/cyan] {tmpl.framework}")
        rprint(f"[cyan]Downloads:[/cyan] {tmpl.downloads}")
        rprint(f"[cyan]Stars:[/cyan] {tmpl.stars}")
        if tmpl.tags:
            rprint(f"[cyan]Tags:[/cyan] {', '.join(tmpl.tags)}")
        rprint(f"[cyan]Created:[/cyan] {tmpl.created_at}")
        rprint(f"[cyan]Updated:[/cyan] {tmpl.updated_at}")

    except MarketplaceError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("market-publish")
def market_publish(
    template_dir: Path = typer.Argument(..., help="Template directory to publish"),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help="Comma-separated tags",
    ),
) -> None:
    """Publish a template to the marketplace.

    Requires CODING_FACTORY_API_KEY environment variable.

    Examples:
        acf market-publish ~/.acf/templates/my-template
        acf market-publish ./my-template --tags "fastapi,api,rest"
    """
    import os
    from scaffolding import MarketplaceClient, MarketplaceError

    api_key = os.environ.get("CODING_FACTORY_API_KEY")
    if not api_key:
        rprint("[red]Error:[/red] CODING_FACTORY_API_KEY environment variable required")
        rprint("[dim]Get your API key from the marketplace website.[/dim]")
        raise typer.Exit(1)

    if not template_dir.exists():
        rprint(f"[red]Template directory not found: {template_dir}[/red]")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()
    rprint(f"[green]Publishing template:[/green] {template_dir}")
    rprint()

    try:
        client = MarketplaceClient(api_key=api_key)
        result = client.publish(template_dir, tags=tag_list)

        rprint(f"[green]✓ Template published successfully![/green]")
        rprint(f"[dim]ID: {result.id}[/dim]")
        rprint(f"[dim]Name: {result.name}[/dim]")

    except MarketplaceError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("market-featured")
def market_featured(
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of templates to show",
    ),
) -> None:
    """Show featured/popular templates from the marketplace.

    Examples:
        acf market-featured
        acf market-featured --limit 10
    """
    from scaffolding import MarketplaceClient, MarketplaceError

    rprint(f"[bold blue]Coding Factory v{__version__}[/bold blue]")
    rprint()

    try:
        client = MarketplaceClient()
        templates = client.list_featured(limit=limit)

        if not templates:
            rprint("[yellow]No featured templates available.[/yellow]")
            return

        rprint("[bold]Featured Templates:[/bold]")
        rprint()

        for tmpl in templates:
            stars = f"★{tmpl.stars}" if tmpl.stars else ""
            downloads = f"↓{tmpl.downloads}" if tmpl.downloads else ""

            rprint(f"  [cyan]{tmpl.name}[/cyan] by {tmpl.author}")
            rprint(f"    {tmpl.description}")
            rprint(f"    [dim]{tmpl.language} | {tmpl.framework} | {stars} {downloads}[/dim]")
            rprint()

    except MarketplaceError as e:
        rprint(f"[yellow]Marketplace unavailable: {e}[/yellow]")
        rprint("[dim]The marketplace service may not be running.[/dim]")


# --- Memory Commands ---


@app.command("memory-index")
def memory_index(
    run_id: Optional[str] = typer.Option(
        None,
        "--run",
        "-r",
        help="Index a specific run by ID",
    ),
    all_runs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Index all completed runs",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-index even if already indexed",
    ),
) -> None:
    """Index completed runs into the memory store.

    Extracts and indexes memories from pipeline runs for
    similarity search and pattern learning.

    Examples:
        acf memory-index --all
        acf memory-index --run 2026-01-04-132634
        acf memory-index --all --force
    """
    from memory import MemoryStore, RunIndexer
    from rag.embeddings import OllamaEmbeddings

    config = get_config()

    if not config.memory.enabled:
        rprint("[yellow]Memory system is disabled in config[/yellow]")
        raise typer.Exit(1)

    if not run_id and not all_runs:
        rprint("[red]Specify --run <id> or --all[/red]")
        raise typer.Exit(1)

    # Determine store path
    if config.memory.store_location == "local":
        store_path = Path.cwd() / ".acf-memory"
    else:
        store_path = Path.home() / ".acf" / "memory"

    rprint(f"[bold blue]Memory Indexing[/bold blue]")
    rprint(f"[dim]Store: {store_path}[/dim]")
    rprint()

    # Initialize components
    store = MemoryStore(
        store_path=store_path,
        decay_half_life_days=config.memory.decay_half_life_days,
    )
    embeddings = OllamaEmbeddings(model=config.memory.embedding_model)
    indexer = RunIndexer(store=store, embeddings=embeddings)

    artifacts_dir = Path(config.pipeline.artifacts_dir)

    if run_id:
        run_dir = artifacts_dir / run_id
        if not run_dir.exists():
            rprint(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        count = indexer.index_run(run_dir, force=force)
        rprint(f"[green]Indexed {count} memories from run {run_id}[/green]")
    else:
        results = indexer.index_all_runs(artifacts_dir, force=force)
        total = sum(results.values())
        indexed_count = len([r for r in results.values() if r > 0])
        rprint(f"[green]Indexed {indexed_count} runs ({total} total memories)[/green]")

    # Save store
    store.save()
    rprint(f"[dim]Store saved to {store_path}[/dim]")


@app.command("memory-search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    memory_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by type: feature, decision, pattern, error_fix, tech_debt",
    ),
    top_k: int = typer.Option(
        5,
        "--top",
        "-k",
        help="Number of results",
    ),
    mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Search mode: semantic, lexical, or hybrid (default: from config)",
    ),
) -> None:
    """Search the memory store with hybrid retrieval.

    Supports three modes:
    - semantic: Cosine similarity on embeddings
    - lexical: BM25 for exact term matching (function names, APIs, keywords)
    - hybrid: Weighted combination (recommended, default)

    Examples:
        acf memory-search "rate limiting"
        acf memory-search "authentication" --type decision
        acf memory-search "StreamingResponse" --mode lexical
        acf memory-search "error handling" --mode hybrid
    """
    from memory import MemoryStore, MemoryRetriever, SearchMode
    from memory.retriever import RetrieverConfig
    from rag.embeddings import OllamaEmbeddings
    from schemas.memory import MemoryType

    config = get_config()

    if not config.memory.enabled:
        rprint("[yellow]Memory system is disabled in config[/yellow]")
        raise typer.Exit(1)

    # Determine store path
    if config.memory.store_location == "local":
        store_path = Path.cwd() / ".acf-memory"
    else:
        store_path = Path.home() / ".acf" / "memory"

    if not store_path.exists():
        rprint("[yellow]No memory store found. Run 'memory-index --all' first.[/yellow]")
        raise typer.Exit(1)

    # Parse search mode
    search_mode = None
    if mode:
        try:
            search_mode = SearchMode(mode)
        except ValueError:
            valid = ", ".join([m.value for m in SearchMode])
            rprint(f"[red]Invalid mode: {mode}. Valid: {valid}[/red]")
            raise typer.Exit(1)
    else:
        # Use config default
        search_mode = SearchMode(config.memory.search_mode)

    # Initialize components
    store = MemoryStore(
        store_path=store_path,
        hybrid_alpha=config.memory.hybrid_alpha,
    )
    embeddings = OllamaEmbeddings(model=config.memory.embedding_model)
    retriever_config = RetrieverConfig(search_mode=search_mode)
    retriever = MemoryRetriever(
        store=store,
        embeddings=embeddings,
        config=retriever_config,
    )

    # Parse memory type
    mem_type = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            valid = ", ".join([t.value for t in MemoryType])
            rprint(f"[red]Invalid type: {memory_type}. Valid: {valid}[/red]")
            raise typer.Exit(1)

    rprint(f"[bold blue]Memory Search: {query}[/bold blue]")
    rprint(f"[dim]Mode: {search_mode.value}[/dim]")
    rprint()

    # Search
    results = retriever.find_similar(query, memory_type=mem_type, top_k=top_k)

    if not results:
        rprint("[yellow]No matching memories found[/yellow]")
        return

    for i, result in enumerate(results, 1):
        entry = result.entry
        rprint(f"[cyan]{i}. [{entry.memory_type.value}][/cyan] [dim]{entry.run_id}[/dim]")
        rprint(f"   Score: {result.score:.3f} (decay: {result.decay_factor:.2f})")
        rprint(f"   {entry.content[:200]}...")
        rprint()


@app.command("memory-patterns")
def memory_patterns(
    extract: bool = typer.Option(
        False,
        "--extract",
        "-e",
        help="Extract patterns from indexed memories using LLM",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        help="Export patterns to markdown file",
    ),
) -> None:
    """View or extract learned patterns.

    Examples:
        acf memory-patterns
        acf memory-patterns --extract
        acf memory-patterns --export PATTERNS.md
    """
    from memory import MemoryStore, PatternExtractor

    config = get_config()

    if not config.memory.enabled:
        rprint("[yellow]Memory system is disabled in config[/yellow]")
        raise typer.Exit(1)

    # Determine store path
    if config.memory.store_location == "local":
        store_path = Path.cwd() / ".acf-memory"
    else:
        store_path = Path.home() / ".acf" / "memory"

    if not store_path.exists():
        rprint("[yellow]No memory store found. Run 'memory-index --all' first.[/yellow]")
        raise typer.Exit(1)

    store = MemoryStore(store_path=store_path)

    if extract:
        rprint("[bold blue]Extracting Patterns[/bold blue]")
        rprint()

        llm = get_backend()
        extractor = PatternExtractor(store=store, llm=llm)

        patterns = extractor.extract_patterns()
        error_patterns = extractor.extract_error_patterns()

        rprint(f"[green]Extracted {len(patterns)} patterns and {len(error_patterns)} error patterns[/green]")
        store.save()

    if export:
        from memory import PatternExtractor as PE
        llm = get_backend()
        extractor = PE(store=store, llm=llm)
        extractor.export_patterns_markdown(export)
        rprint(f"[green]Exported patterns to {export}[/green]")
        return

    # Display existing patterns
    rprint("[bold blue]Learned Patterns[/bold blue]")
    rprint()

    patterns = store.get_patterns()
    if patterns:
        rprint("[bold]Patterns:[/bold]")
        for p in patterns:
            rprint(f"  [cyan]{p.name}[/cyan] ({p.pattern_type})")
            rprint(f"    {p.description[:100]}...")
            rprint(f"    [dim]Seen {p.occurrence_count} times[/dim]")
            rprint()
    else:
        rprint("[yellow]No patterns found. Run --extract first.[/yellow]")
        rprint()

    errors = store.get_error_patterns()
    if errors:
        rprint("[bold]Error Patterns:[/bold]")
        for e in errors:
            rprint(f"  [red]{e.error_signature[:60]}...[/red]")
            rprint(f"    Stage: {e.stage}")
            rprint(f"    Fix: {e.fix_description[:100]}...")
            rprint()


@app.command("memory-stats")
def memory_stats() -> None:
    """Show memory store statistics.

    Examples:
        acf memory-stats
    """
    from memory import MemoryStore

    config = get_config()

    if not config.memory.enabled:
        rprint("[yellow]Memory system is disabled in config[/yellow]")
        raise typer.Exit(1)

    # Determine store path
    if config.memory.store_location == "local":
        store_path = Path.cwd() / ".acf-memory"
    else:
        store_path = Path.home() / ".acf" / "memory"

    if not store_path.exists():
        rprint("[yellow]No memory store found. Run 'memory-index --all' first.[/yellow]")
        raise typer.Exit(1)

    store = MemoryStore(store_path=store_path)
    stats = store.stats()

    rprint("[bold blue]Memory Store Statistics[/bold blue]")
    rprint()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Store Path", stats.store_path)
    table.add_row("Total Memories", str(stats.total_memories))
    table.add_row("Total Patterns", str(stats.total_patterns))
    table.add_row("Total Error Patterns", str(stats.total_errors))
    table.add_row("Runs Indexed", str(stats.runs_indexed))
    table.add_row("Embedding Dimensions", str(stats.embedding_dimensions))
    table.add_row("Store Size", f"{stats.store_size_bytes / 1024:.1f} KB")

    if stats.last_indexed:
        table.add_row("Last Indexed", stats.last_indexed.strftime("%Y-%m-%d %H:%M"))

    console.print(table)

    if stats.memories_by_type:
        rprint()
        rprint("[bold]Memories by Type:[/bold]")
        for mem_type, count in sorted(stats.memories_by_type.items()):
            rprint(f"  {mem_type}: {count}")

    if stats.memories_by_outcome:
        rprint()
        rprint("[bold]Memories by Outcome:[/bold]")
        for outcome, count in sorted(stats.memories_by_outcome.items()):
            rprint(f"  {outcome}: {count}")


@app.command("routing-check")
def routing_check(
    stage: Optional[str] = typer.Option(
        None,
        "--stage",
        "-s",
        help="Show routing for a specific stage (e.g., design, implementation)",
    ),
    task_size: Optional[str] = typer.Option(
        None,
        "--size",
        help="Simulate task size (xs, s, m, l, xl)",
    ),
    task_category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Simulate task category (e.g., payments, auth, api)",
    ),
    retry: int = typer.Option(
        0,
        "--retry",
        "-r",
        help="Simulate retry count for escalation",
    ),
) -> None:
    """Show model routing configuration and simulate routing decisions.

    Examples:
        # Show all routing configuration
        acf routing-check

        # Check routing for a specific stage
        acf routing-check --stage design

        # Simulate routing for a task
        acf routing-check --stage implementation --size l --category payments

        # See retry escalation
        acf routing-check --stage fix --retry 2
    """
    from routing import ModelRouter, ModelTier
    from routing.router import STAGE_ROUTING, SIZE_ROUTING

    config = get_config()

    if not config.routing.enabled:
        rprint("[yellow]Model routing is disabled in config[/yellow]")
        rprint("[dim]Enable with: routing.enabled = true in config.toml[/dim]")
        raise typer.Exit(1)

    router = ModelRouter(config.routing)

    # Show routing configuration
    rprint("[bold blue]Model Routing Configuration[/bold blue]")
    rprint()

    # Model pool
    table = Table(title="Model Pool", show_header=True, header_style="bold")
    table.add_column("Tier")
    table.add_column("Model")
    table.add_column("Use Case")

    table.add_row("cheap", config.routing.model_cheap, "Simple tasks, parsing, docs")
    table.add_row("medium", config.routing.model_medium, "Moderate reasoning, decomposition")
    table.add_row("premium", config.routing.model_premium, "Complex tasks, code, security")

    console.print(table)
    rprint()

    # Stage routing
    stage_table = Table(title="Stage Routing Defaults", show_header=True, header_style="bold")
    stage_table.add_column("Stage")
    stage_table.add_column("Default Tier")

    for stage_name, tier in STAGE_ROUTING.items():
        override = config.routing.stage_overrides.get(stage_name)
        if override:
            stage_table.add_row(stage_name, f"{tier.value} → {override} (override)")
        else:
            stage_table.add_row(stage_name, tier.value)

    console.print(stage_table)
    rprint()

    # Premium domains
    rprint("[bold]Premium Domains:[/bold]")
    rprint(f"  {', '.join(config.routing.premium_domains)}")
    rprint()

    # If stage specified, show routing decision
    if stage:
        rprint(f"[bold cyan]Routing Simulation for '{stage}'[/bold cyan]")

        # Build simulated task if size/category provided
        task = None
        if task_size or task_category:
            from schemas.workplan import SubTask, TaskCategory, TaskSize

            task = SubTask(
                id="SIMULATED",
                title="Simulated task",
                description="Simulated task for routing check",
                category=TaskCategory(task_category) if task_category else TaskCategory.BACKEND,
                size=TaskSize(task_size) if task_size else TaskSize.M,
            )
            rprint(f"[dim]Task: size={task.size.value}, category={task.category.value}[/dim]")

        # Get routing explanation
        explanation = router.explain_routing(stage, task, retry)

        rprint()
        for reason in explanation["reasons"]:
            rprint(f"  • {reason}")

        rprint()
        rprint(f"[bold green]Final: {explanation['final_tier']} → {explanation['model']}[/bold green]")


@app.command("plugins")
def plugins_list() -> None:
    """List installed plugins.

    Examples:
        acf plugins
    """
    from plugins import PluginLoader

    config = get_config()

    if not config.plugins.enabled:
        rprint("[yellow]Plugin system is disabled in config[/yellow]")
        raise typer.Exit(1)

    # Build plugin directories
    plugin_dirs = PluginLoader.get_default_plugin_dirs()
    if config.plugins.plugins_dir:
        custom_dir = Path(config.plugins.plugins_dir)
        if custom_dir.exists():
            plugin_dirs.insert(0, custom_dir)

    # Load plugins
    loader = PluginLoader(
        plugin_dirs=plugin_dirs,
        enabled_plugins=config.plugins.enabled_plugins or None,
        disabled_plugins=config.plugins.disabled_plugins,
    )
    registry = loader.load_all()

    if len(registry) == 0:
        rprint("[dim]No plugins installed[/dim]")
        rprint()
        rprint("Plugin directories searched:")
        for d in plugin_dirs:
            exists = "[green]✓[/green]" if d.exists() else "[dim]✗[/dim]"
            rprint(f"  {exists} {d}")
        rprint()
        rprint("Use 'acf plugin-create <name>' to create a new plugin")
        return

    rprint("[bold blue]Installed Plugins[/bold blue]")
    rprint()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Hook Point")
    table.add_column("Status")
    table.add_column("Description")

    for plugin in registry.list_all():
        status = "[green]enabled[/green]" if plugin.enabled else "[yellow]disabled[/yellow]"
        table.add_row(
            plugin.name,
            plugin.manifest.version,
            plugin.hook_point.value,
            status,
            plugin.manifest.description[:40] + "..." if len(plugin.manifest.description) > 40 else plugin.manifest.description,
        )

    console.print(table)


@app.command("plugin-install")
def plugin_install(
    path: Path = typer.Argument(..., help="Path to plugin directory"),
    global_install: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Install to global plugins directory (~/.acf/plugins)",
    ),
) -> None:
    """Install a plugin from a local directory.

    Examples:
        # Install to local .plugins/
        acf plugin-install ./my-plugin

        # Install globally
        acf plugin-install ./my-plugin --global
    """
    import shutil

    from plugins import PluginLoader

    if not path.exists():
        rprint(f"[red]Plugin path does not exist: {path}[/red]")
        raise typer.Exit(1)

    manifest_path = path / "plugin.yaml"
    if not manifest_path.exists():
        rprint(f"[red]No plugin.yaml found in {path}[/red]")
        raise typer.Exit(1)

    # Load manifest to get plugin name
    loader = PluginLoader()
    try:
        manifest = loader.load_manifest(path)
    except Exception as e:
        rprint(f"[red]Invalid plugin manifest: {e}[/red]")
        raise typer.Exit(1)

    # Determine target directory
    if global_install:
        target_base = Path.home() / ".acf" / "plugins"
    else:
        target_base = Path.cwd() / ".plugins"

    target_base.mkdir(parents=True, exist_ok=True)
    target_dir = target_base / manifest.name

    if target_dir.exists():
        rprint(f"[yellow]Plugin '{manifest.name}' already exists at {target_dir}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)
        shutil.rmtree(target_dir)

    # Copy plugin
    shutil.copytree(path, target_dir)

    rprint(f"[green]Installed plugin '{manifest.name}' to {target_dir}[/green]")


@app.command("plugin-create")
def plugin_create(
    name: str = typer.Argument(..., help="Plugin name"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: current directory)",
    ),
    hook: str = typer.Option(
        "after:code_review",
        "--hook",
        "-h",
        help="Hook point (e.g., before:design, after:implementation)",
    ),
) -> None:
    """Create a new plugin from template.

    Examples:
        # Create plugin in current directory
        acf plugin-create my-validator

        # Create with specific hook point
        acf plugin-create compliance-checker --hook after:code_review

        # Create in specific directory
        acf plugin-create my-plugin -o ./plugins
    """
    output_dir = output or Path.cwd()
    plugin_dir = output_dir / name

    if plugin_dir.exists():
        rprint(f"[red]Directory already exists: {plugin_dir}[/red]")
        raise typer.Exit(1)

    plugin_dir.mkdir(parents=True)

    # Create plugin.yaml
    manifest_content = f'''name: {name}
version: 1.0.0
description: Custom plugin for {name}
author: ""

agent:
  class: agent.{name.replace("-", "_").title().replace("_", "")}Agent

hook:
  point: {hook}
  priority: 100
  required: false
  skip_on_failure: true

inputs:
  - feature_spec
  - code_review_report

outputs:
  - {name.replace("-", "_")}_report

config:
  # Add custom configuration fields here
  # example_option:
  #   type: str
  #   default: "value"
  #   description: "Example configuration option"

requires_llm: true
enabled_by_default: true
'''
    (plugin_dir / "plugin.yaml").write_text(manifest_content)

    # Create agent.py
    class_name = name.replace("-", "_").title().replace("_", "") + "Agent"
    report_key = name.replace("-", "_") + "_report"
    agent_content = f'''"""Custom agent for {name} plugin."""

from agents import AgentInput, AgentOutput
from agents.base import BaseAgent


class {class_name}(BaseAgent):
    """Custom agent that runs at {hook}.

    Common patterns:
    - Validator: Check code/artifacts, return issues with severity
    - Generator: Use LLM to create new content
    - Analyzer: Scan inputs, produce structured report
    - Transformer: Convert one format to another
    """

    def _run(self, input_data: AgentInput) -> AgentOutput:
        """Execute the agent logic.

        Args:
            input_data: Input containing pipeline context and artifacts
                - context["run_id"]: Current run ID
                - context["feature_description"]: Feature being implemented
                - context["artifacts_dir"]: Path to artifacts folder
                - context["feature_spec"]: Parsed feature spec (if available)
                - context["code_review_report"]: Code review results (if available)
                - ... other artifacts based on plugin.yaml inputs

        Returns:
            AgentOutput with success=True/False and data dict
        """
        context = input_data.context
        issues = []

        # ──────────────────────────────────────────────────────────────
        # 1. ACCESS PIPELINE DATA
        # ──────────────────────────────────────────────────────────────
        feature_desc = context.get("feature_description", "")
        feature_spec = context.get("feature_spec", {{}})
        code_review = context.get("code_review_report", {{}})

        # ──────────────────────────────────────────────────────────────
        # 2. IMPLEMENT YOUR LOGIC
        # ──────────────────────────────────────────────────────────────

        # Example: Validation pattern
        # if some_condition_fails:
        #     issues.append({{
        #         "severity": "warning",  # or "error", "info"
        #         "message": "Description of the issue",
        #         "location": "file.py:42",
        #     }})

        # Example: Call LLM (self.llm is available if requires_llm: true)
        # if self.llm:
        #     response = self.llm.generate(
        #         prompt="Analyze this code...",
        #         system_prompt="You are a code reviewer.",
        #     )

        # Example: Read files from artifacts
        # from pathlib import Path
        # artifacts_dir = Path(context.get("artifacts_dir", "."))
        # design_file = artifacts_dir / "design_proposal.json"
        # if design_file.exists():
        #     import json
        #     design = json.loads(design_file.read_text())

        # ──────────────────────────────────────────────────────────────
        # 3. BUILD RESULT
        # ──────────────────────────────────────────────────────────────
        has_errors = any(i.get("severity") == "error" for i in issues)

        result = {{
            "status": "failed" if has_errors else "passed",
            "issues": issues,
            "summary": f"{{len(issues)}} issues found" if issues else "All checks passed",
        }}

        return AgentOutput(
            success=not has_errors,  # False blocks pipeline if hook.required=true
            data={{"{report_key}": result}},
            errors=[i["message"] for i in issues if i.get("severity") == "error"],
        )
'''
    (plugin_dir / "agent.py").write_text(agent_content)

    # Create README
    readme_content = f'''# {name}

Custom plugin for Coding Factory.

## Installation

```bash
# Install locally
acf plugin-install ./{name}

# Or install globally
acf plugin-install ./{name} --global
```

## Configuration

Add to your `config.toml`:

```toml
[plugins.plugin_config.{name}]
# Add configuration here
```

## Hook Point

This plugin runs at `{hook}`.

## Inputs

- `feature_spec` - Feature specification
- `code_review_report` - Code review results

## Outputs

- `{name.replace("-", "_")}_report` - Plugin analysis report
'''
    (plugin_dir / "README.md").write_text(readme_content)

    rprint(f"[green]Created plugin '{name}' at {plugin_dir}[/green]")
    rprint()
    rprint("Files created:")
    rprint(f"  • {plugin_dir}/plugin.yaml")
    rprint(f"  • {plugin_dir}/agent.py")
    rprint(f"  • {plugin_dir}/README.md")
    rprint()
    rprint("Next steps:")
    rprint(f"  1. Edit {plugin_dir}/agent.py to implement your logic")
    rprint(f"  2. Install: acf plugin-install {plugin_dir}")
    rprint("  3. Run pipeline to test your plugin")


if __name__ == "__main__":
    app()
