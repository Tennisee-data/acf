"""Extensions CLI commands for ACF Local Edition.

Manage locally installed extensions.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

extensions_app = typer.Typer(
    name="extensions",
    help="Manage locally installed extensions.",
)


def get_loader():
    """Get extension loader."""
    from extensions import ExtensionLoader

    return ExtensionLoader()


def get_installer():
    """Get extension installer."""
    from extensions import ExtensionInstaller

    return ExtensionInstaller()


@extensions_app.command("list")
def list_extensions(
    ext_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by type (agent, profile, rag)",
    ),
) -> None:
    """List all installed extensions.

    Examples:
        acf extensions list
        acf extensions list --type agent
    """
    from extensions.manifest import ExtensionType

    installer = get_installer()
    installed = installer.list_installed()

    # Filter by type if specified
    if ext_type:
        try:
            filter_type = ExtensionType(ext_type)
            installed = [e for e in installed if e.type == filter_type]
        except ValueError:
            console.print(f"[red]Invalid type: {ext_type}. Use: agent, profile, or rag[/red]")
            raise typer.Exit(1)

    if not installed:
        console.print("[yellow]No extensions installed[/yellow]")
        console.print("[dim]Install extensions with: acf marketplace install <name>[/dim]")
        return

    table = Table(title="Installed Extensions")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Version")
    table.add_column("Context", justify="right")
    table.add_column("Min Model", style="yellow")
    table.add_column("Hook Point")
    table.add_column("Author")

    for ext in sorted(installed, key=lambda x: x.name):
        hook = ext.hook_point.value if ext.hook_point else "-"
        # Format context tokens and model tier
        ctx = ext.context_tokens_formatted if hasattr(ext, 'context_tokens_formatted') else "-"
        tier = ext.min_model_tier.value if hasattr(ext, 'min_model_tier') else "any"
        table.add_row(
            ext.name,
            ext.type.value,
            ext.version,
            ctx,
            tier,
            hook,
            ext.author,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(installed)} extensions[/dim]")


@extensions_app.command("show")
def show(
    name: str = typer.Argument(..., help="Extension name"),
) -> None:
    """Show details of an installed extension.

    Example:
        acf extensions show secrets-scan
    """
    installer = get_installer()
    manifest = installer.get_installed(name)

    if not manifest:
        console.print(f"[red]Extension '{name}' is not installed[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{manifest.name}[/bold cyan] v{manifest.version}")
    console.print(f"[dim]by {manifest.author}[/dim]\n")

    console.print(f"{manifest.description}\n")

    console.print("[bold]Configuration[/bold]")
    console.print(f"  Type: {manifest.type.value}")
    console.print(f"  License: {manifest.license.value}")
    console.print(f"  Priority: {manifest.priority}")

    if manifest.hook_point:
        console.print(f"  Hook Point: {manifest.hook_point.value}")

    # Token budget info
    console.print(f"\n[bold]Context Budget[/bold]")
    ctx_tokens = manifest.context_tokens_formatted if hasattr(manifest, 'context_tokens_formatted') else "minimal"
    console.print(f"  Context Tokens: {ctx_tokens}")
    if hasattr(manifest, 'min_model_tier'):
        console.print(f"  Min Model Tier: {manifest.min_model_tier.value}")
        # Show compatibility
        compatible = manifest.get_compatible_tiers() if hasattr(manifest, 'get_compatible_tiers') else []
        if compatible:
            tier_icons = {
                "small": "7B",
                "medium": "14B",
                "large": "32B+",
            }
            compat_str = " | ".join(
                f"[green]{tier_icons.get(t.value, t.value)}[/green]" if t in compatible
                else f"[dim]{tier_icons.get(t.value, t.value)}[/dim]"
                for t in [manifest.min_model_tier.__class__.SMALL,
                          manifest.min_model_tier.__class__.MEDIUM,
                          manifest.min_model_tier.__class__.LARGE]
            )
            console.print(f"  Compatible: {compat_str}")

    if manifest.agent_class:
        console.print(f"  Agent Class: {manifest.agent_class}")

    if manifest.profile_class:
        console.print(f"  Profile Class: {manifest.profile_class}")

    if manifest.retriever_class:
        console.print(f"  Retriever Class: {manifest.retriever_class}")

    if manifest.requires:
        console.print(f"\n[bold]Dependencies[/bold]")
        for dep in manifest.requires:
            console.print(f"  - {dep}")

    if manifest.conflicts_with:
        console.print(f"\n[bold]Conflicts With[/bold]")
        for conflict in manifest.conflicts_with:
            console.print(f"  - {conflict}")

    if manifest.keywords:
        console.print(f"\n[bold]Keywords[/bold]: {', '.join(manifest.keywords)}")


@extensions_app.command("enable")
def enable(
    name: str = typer.Argument(..., help="Extension name to enable"),
) -> None:
    """Enable a disabled extension.

    Example:
        acf extensions enable secrets-scan
    """
    loader = get_loader()
    loader.discover()

    if loader.enable_extension(name):
        console.print(f"[green]✓ Enabled extension: {name}[/green]")
    else:
        console.print(f"[red]Extension '{name}' not found[/red]")
        raise typer.Exit(1)


@extensions_app.command("disable")
def disable(
    name: str = typer.Argument(..., help="Extension name to disable"),
) -> None:
    """Disable an extension without uninstalling.

    Example:
        acf extensions disable secrets-scan
    """
    loader = get_loader()
    loader.discover()

    if loader.disable_extension(name):
        console.print(f"[green]✓ Disabled extension: {name}[/green]")
    else:
        console.print(f"[red]Extension '{name}' not found[/red]")
        raise typer.Exit(1)


@extensions_app.command("uninstall")
def uninstall(
    name: str = typer.Argument(..., help="Extension name to uninstall"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation",
    ),
) -> None:
    """Uninstall an extension.

    Example:
        acf extensions uninstall secrets-scan
    """
    installer = get_installer()

    # Check if installed
    manifest = installer.get_installed(name)
    if not manifest:
        console.print(f"[red]Extension '{name}' is not installed[/red]")
        raise typer.Exit(1)

    if not yes:
        if not typer.confirm(f"Uninstall {name} v{manifest.version}?"):
            raise typer.Exit(0)

    if installer.uninstall(name):
        console.print(f"[green]✓ Uninstalled: {name}[/green]")
    else:
        console.print(f"[red]Failed to uninstall: {name}[/red]")
        raise typer.Exit(1)


@extensions_app.command("update")
def update(
    name: Optional[str] = typer.Argument(
        None,
        help="Extension name (omit for all)",
    ),
    all_exts: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Update all extensions",
    ),
) -> None:
    """Update extension(s) to latest version.

    Examples:
        acf extensions update secrets-scan
        acf extensions update --all
    """
    from extensions import ExtensionInstaller, MarketplaceClient
    from extensions.installer import MarketplaceError

    try:
        client = MarketplaceClient()
    except Exception as e:
        console.print(f"[yellow]Marketplace unavailable: {e}[/yellow]")
        raise typer.Exit(1)

    installer = ExtensionInstaller(marketplace_client=client)

    if not name and not all_exts:
        console.print("[red]Specify extension name or use --all[/red]")
        raise typer.Exit(1)

    if all_exts:
        updates = installer.check_updates()

        if not updates:
            console.print("[green]All extensions are up to date[/green]")
            return

        console.print(f"[bold]Found {len(updates)} updates available:[/bold]")
        for installed, latest in updates:
            console.print(f"  {installed.name}: {installed.version} → {latest.version}")

        if not typer.confirm("Update all?"):
            raise typer.Exit(0)

        for installed, latest in updates:
            try:
                console.print(f"Updating {installed.name}...")
                installer.install_from_marketplace(installed.name, force=True)
                console.print(f"[green]✓ Updated {installed.name}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to update {installed.name}: {e}[/red]")

    else:
        installed = installer.get_installed(name)
        if not installed:
            console.print(f"[red]Extension '{name}' is not installed[/red]")
            raise typer.Exit(1)

        try:
            latest = client.get_extension(name)
        except MarketplaceError:
            console.print(f"[yellow]Extension '{name}' not found in marketplace[/yellow]")
            raise typer.Exit(1)

        if installer._version_compare(latest.version, installed.version) <= 0:
            console.print(f"[green]{name} is already at latest version ({installed.version})[/green]")
            return

        console.print(f"Updating {name}: {installed.version} → {latest.version}")
        installer.install_from_marketplace(name, force=True)
        console.print(f"[green]✓ Updated {name}[/green]")


@extensions_app.command("create")
def create(
    name: str = typer.Argument(..., help="Extension name"),
    ext_type: str = typer.Option(
        "agent",
        "--type",
        "-t",
        help="Extension type (agent, profile, rag)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: current directory)",
    ),
) -> None:
    """Create a new extension scaffold.

    Examples:
        acf extensions create my-agent
        acf extensions create my-profile --type profile
        acf extensions create my-rag --type rag -o ./extensions
    """
    from extensions import ExtensionInstaller
    from extensions.manifest import ExtensionType

    try:
        type_enum = ExtensionType(ext_type)
    except ValueError:
        console.print(f"[red]Invalid type: {ext_type}. Use: agent, profile, or rag[/red]")
        raise typer.Exit(1)

    installer = ExtensionInstaller()
    ext_path = installer.create_extension_scaffold(name, type_enum, output)

    console.print(f"[green]✓ Created extension scaffold: {ext_path}[/green]")
    console.print()
    console.print("[bold]Files created:[/bold]")
    for file in ext_path.iterdir():
        console.print(f"  {file.name}")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  1. Edit {ext_path}/manifest.yaml")
    console.print(f"  2. Implement your {ext_type} logic")
    console.print(f"  3. Test locally with: acf extensions install-local {ext_path}")
    console.print(f"  4. Submit to marketplace: acf marketplace submit {ext_path}")


@extensions_app.command("install-local")
def install_local(
    path: Path = typer.Argument(..., help="Path to extension directory"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if already installed",
    ),
) -> None:
    """Install an extension from a local directory.

    Example:
        acf extensions install-local ./my-extension
    """
    installer = get_installer()

    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    try:
        manifest = installer.install_from_path(path, force=force)
        console.print(f"[green]✓ Installed {manifest.name} v{manifest.version}[/green]")
        console.print(f"[dim]Type: {manifest.type.value}[/dim]")
    except Exception as e:
        console.print(f"[red]Installation failed: {e}[/red]")
        raise typer.Exit(1)


@extensions_app.command("requirements")
def requirements(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: stdout)",
    ),
) -> None:
    """Show combined requirements for all installed extensions.

    Examples:
        acf extensions requirements
        acf extensions requirements -o ext-requirements.txt
    """
    loader = get_loader()
    loader.discover()

    reqs = loader.get_requirements()

    if not reqs:
        console.print("[yellow]No extension requirements[/yellow]")
        return

    if output:
        output.write_text("\n".join(reqs))
        console.print(f"[green]✓ Wrote {len(reqs)} requirements to {output}[/green]")
    else:
        console.print("[bold]Extension Requirements:[/bold]")
        for req in reqs:
            console.print(f"  {req}")


@extensions_app.command("check-conflicts")
def check_conflicts() -> None:
    """Check for conflicts between installed extensions.

    Example:
        acf extensions check-conflicts
    """
    loader = get_loader()
    loaded = loader.discover()

    if not loaded:
        console.print("[yellow]No extensions installed[/yellow]")
        return

    conflicts = loader.check_conflicts(loaded)

    if not conflicts:
        console.print("[green]✓ No conflicts detected[/green]")
    else:
        console.print("[bold red]Conflicts Detected:[/bold red]")
        for name1, name2 in conflicts:
            console.print(f"  • {name1} ↔ {name2}")
        console.print()
        console.print("[dim]Consider disabling one of the conflicting extensions[/dim]")


@extensions_app.command("init")
def init() -> None:
    """Initialize the extensions directory structure.

    Creates ~/.coding-factory/extensions/ with subdirectories.

    Example:
        acf extensions init
    """
    loader = get_loader()
    loader.ensure_extensions_dir()

    console.print("[green]✓ Extensions directory initialized[/green]")
    console.print(f"[dim]Location: {loader.extensions_dir}[/dim]")
    console.print()
    console.print("[bold]Directory structure:[/bold]")
    console.print("  ~/.coding-factory/extensions/")
    console.print("  ├── agents/      # Agent extensions")
    console.print("  ├── profiles/    # Profile extensions")
    console.print("  └── rag/         # RAG retriever extensions")
