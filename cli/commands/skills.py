"""Skills CLI commands for ACF.

Run standalone code transformations on files and directories.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

skills_app = typer.Typer(
    name="skill",
    help="Run standalone code transformation skills.",
)


def get_loader():
    """Get extension loader with discovered extensions."""
    from extensions import ExtensionLoader

    loader = ExtensionLoader()
    loader.discover()
    return loader


@skills_app.command("run")
def run_skill(
    name: str = typer.Argument(..., help="Skill name"),
    target: Path = typer.Argument(..., help="Target file or directory"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview changes without applying",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="JSON config string for the skill",
    ),
) -> None:
    """Run a skill on target files.

    Examples:
        acf skill run add-error-handling ./src/main.py
        acf skill run add-error-handling ./src --dry-run
        acf skill run add-type-hints ./src -c '{"strict": true}'
    """
    import json

    loader = get_loader()
    manifest = loader.get_manifest(name)

    if not manifest:
        console.print(f"[red]Skill '{name}' not found[/red]")
        console.print("[dim]List available skills: acf skill list[/dim]")
        raise typer.Exit(1)

    from extensions.manifest import ExtensionType

    if manifest.type != ExtensionType.SKILL:
        console.print(f"[red]'{name}' is a {manifest.type.value}, not a skill[/red]")
        raise typer.Exit(1)

    if not target.exists():
        console.print(f"[red]Target does not exist: {target}[/red]")
        raise typer.Exit(1)

    # Parse config
    skill_config = {}
    if config:
        try:
            skill_config = json.loads(config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON config: {e}[/red]")
            raise typer.Exit(1)

    # Check if this is a chained skill
    if manifest.chain:
        console.print(f"[yellow]'{name}' is a chained skill. Use 'acf skill chain' instead.[/yellow]")
        raise typer.Exit(1)

    # Get skill class
    skill_class = loader.get_skill(name)
    if skill_class is None:
        console.print(f"[red]Could not load skill class for '{name}'[/red]")
        raise typer.Exit(1)

    # Run the skill
    from skills.runner import SkillRunner

    runner = SkillRunner()
    file_patterns = manifest.file_patterns or None

    if dry_run:
        console.print(f"[bold]Preview: {name}[/bold] on {target}")
        console.print()

    output = runner.run(
        skill_class=skill_class,
        target=target,
        config=skill_config,
        dry_run=dry_run,
        file_patterns=file_patterns,
    )

    # Display results
    _display_output(output, dry_run)


@skills_app.command("chain")
def chain_skill(
    name: str = typer.Argument(..., help="Chained skill name"),
    target: Path = typer.Argument(..., help="Target file or directory"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview changes without applying",
    ),
) -> None:
    """Run a chained skill (multi-step transformation).

    Examples:
        acf skill chain production-ready ./src
        acf skill chain production-ready ./src --dry-run
    """
    loader = get_loader()
    manifest = loader.get_manifest(name)

    if not manifest:
        console.print(f"[red]Skill '{name}' not found[/red]")
        raise typer.Exit(1)

    from extensions.manifest import ExtensionType

    if manifest.type != ExtensionType.SKILL:
        console.print(f"[red]'{name}' is a {manifest.type.value}, not a skill[/red]")
        raise typer.Exit(1)

    if not manifest.chain:
        console.print(f"[yellow]'{name}' is not a chained skill. Use 'acf skill run' instead.[/yellow]")
        raise typer.Exit(1)

    if not target.exists():
        console.print(f"[red]Target does not exist: {target}[/red]")
        raise typer.Exit(1)

    # Show chain steps
    console.print(f"[bold]Chain: {name}[/bold] ({len(manifest.chain)} steps)")
    for i, step in enumerate(manifest.chain, 1):
        step_config = step.get("config", {})
        config_str = f" [dim]({step_config})[/dim]" if step_config else ""
        console.print(f"  {i}. {step.get('skill', '?')}{config_str}")
    console.print()

    # Run the chain
    from skills.chain_runner import ChainRunner

    runner = ChainRunner(loader=loader)
    output = runner.run(
        chain=manifest.chain,
        target=target,
        dry_run=dry_run,
    )

    _display_output(output, dry_run)


@skills_app.command("list")
def list_skills() -> None:
    """List all installed skills.

    Example:
        acf skill list
    """
    from extensions.manifest import ExtensionType

    loader = get_loader()
    skills = loader.list_extensions(ExtensionType.SKILL)

    if not skills:
        console.print("[yellow]No skills installed[/yellow]")
        console.print("[dim]Install skills with: acf marketplace install <name>[/dim]")
        return

    table = Table(title="Installed Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Input", style="green")
    table.add_column("Output", style="blue")
    table.add_column("Patterns")
    table.add_column("Chain", justify="center")
    table.add_column("Dry Run", justify="center")
    table.add_column("Author")

    for skill in sorted(skills, key=lambda x: x.name):
        chain_str = str(len(skill.chain)) if skill.chain else "-"
        table.add_row(
            skill.name,
            skill.version,
            skill.input_type or "-",
            skill.output_type or "-",
            ", ".join(skill.file_patterns) if skill.file_patterns else "*",
            chain_str,
            "Y" if skill.supports_dry_run else "-",
            skill.author,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(skills)} skills[/dim]")


@skills_app.command("show")
def show_skill(
    name: str = typer.Argument(..., help="Skill name"),
) -> None:
    """Show details of an installed skill.

    Example:
        acf skill show add-error-handling
    """
    loader = get_loader()
    manifest = loader.get_manifest(name)

    if not manifest:
        console.print(f"[red]Skill '{name}' not found[/red]")
        raise typer.Exit(1)

    from extensions.manifest import ExtensionType

    if manifest.type != ExtensionType.SKILL:
        console.print(f"[red]'{name}' is a {manifest.type.value}, not a skill[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{manifest.name}[/bold cyan] v{manifest.version}")
    console.print(f"[dim]by {manifest.author}[/dim]\n")
    console.print(f"{manifest.description}\n")

    console.print("[bold]Configuration[/bold]")
    console.print(f"  License: {manifest.license.value}")
    if manifest.input_type:
        console.print(f"  Input Type: {manifest.input_type}")
    if manifest.output_type:
        console.print(f"  Output Type: {manifest.output_type}")
    if manifest.file_patterns:
        console.print(f"  File Patterns: {', '.join(manifest.file_patterns)}")
    console.print(f"  Supports Dry Run: {'Yes' if manifest.supports_dry_run else 'No'}")

    if manifest.skill_class:
        console.print(f"  Skill Class: {manifest.skill_class}")

    if manifest.chain:
        console.print(f"\n[bold]Chain Steps ({len(manifest.chain)})[/bold]")
        for i, step in enumerate(manifest.chain, 1):
            step_config = step.get("config", {})
            config_str = f"  config: {step_config}" if step_config else ""
            console.print(f"  {i}. {step.get('skill', '?')}{config_str}")

    if manifest.price_usd > 0:
        console.print(f"\n[bold]Price[/bold]: ${manifest.price_usd:.2f}")

    if manifest.keywords:
        console.print(f"\n[bold]Keywords[/bold]: {', '.join(manifest.keywords)}")


def _display_output(output, dry_run: bool) -> None:
    """Display skill output to console."""
    if output.success:
        prefix = "[Preview]" if dry_run else "[Applied]"
        console.print(f"[green]{prefix} {output.summary}[/green]")
    else:
        console.print(f"[red]Failed: {output.summary}[/red]")

    if output.changes:
        console.print(f"\n[bold]Changes ({len(output.changes)} files):[/bold]")
        for change in output.changes:
            icon = {"modified": "M", "created": "+", "deleted": "-"}.get(
                change.change_type, "?"
            )
            style = {"modified": "yellow", "created": "green", "deleted": "red"}.get(
                change.change_type, "white"
            )
            console.print(f"  [{style}]{icon}[/{style}] {change.path}")

    if output.errors:
        console.print(f"\n[bold red]Errors:[/bold red]")
        for err in output.errors:
            console.print(f"  - {err}")
