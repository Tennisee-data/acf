"""Marketplace CLI commands for ACF Local Edition.

Browse, search, and install extensions from the ACF marketplace.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

marketplace_app = typer.Typer(
    name="marketplace",
    help="Browse and install extensions from the ACF marketplace.",
)


def get_marketplace_client():
    """Get marketplace client with optional authentication."""
    try:
        from extensions import MarketplaceClient

        return MarketplaceClient()
    except Exception as e:
        console.print(f"[yellow]Marketplace unavailable: {e}[/yellow]")
        raise typer.Exit(1)


def get_installer():
    """Get extension installer."""
    from extensions import ExtensionInstaller

    return ExtensionInstaller()


@marketplace_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    ext_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by type (agent, profile, rag)",
    ),
    free_only: bool = typer.Option(
        False,
        "--free",
        "-f",
        help="Only show free extensions",
    ),
) -> None:
    """Search for extensions in the marketplace.

    Examples:
        acf marketplace search security
        acf marketplace search vue --type profile
        acf marketplace search testing --free
    """
    from extensions.manifest import ExtensionType

    client = get_marketplace_client()

    type_filter = None
    if ext_type:
        try:
            type_filter = ExtensionType(ext_type)
        except ValueError:
            console.print(f"[red]Invalid type: {ext_type}. Use: agent, profile, or rag[/red]")
            raise typer.Exit(1)

    try:
        results = client.search(query, ext_type=type_filter, free_only=free_only)

        if not results:
            console.print(f"[yellow]No extensions found for: {query}[/yellow]")
            return

        table = Table(title=f"Search Results: {query}")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Version")
        table.add_column("Context", justify="right")
        table.add_column("Min", style="yellow")
        table.add_column("Price", justify="right")
        table.add_column("Description")

        for ext in results:
            price = "Free" if ext.is_free else f"${ext.price_usd:.2f}"
            desc = ext.description[:35] + "..." if len(ext.description) > 35 else ext.description
            # Context tokens info
            ctx = "-"
            if hasattr(ext, 'context_tokens') and ext.context_tokens:
                ctx = f"{ext.context_tokens / 1000:.1f}K" if ext.context_tokens >= 1000 else str(ext.context_tokens)
            tier = getattr(ext, 'min_model_tier', 'any') or "any"

            table.add_row(
                ext.name,
                ext.type,
                ext.version,
                ctx,
                tier,
                price,
                desc,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("featured")
def featured() -> None:
    """Show featured/curated extensions.

    Example:
        acf marketplace featured
    """
    client = get_marketplace_client()

    try:
        results = client.get_featured()

        if not results:
            console.print("[yellow]No featured extensions available[/yellow]")
            return

        console.print("[bold]Featured Extensions[/bold]\n")

        for ext in results:
            badge = "⭐ Official" if ext.is_official else ""
            price = "Free" if ext.is_free else f"${ext.price_usd:.2f}"

            panel = Panel(
                f"{ext.description}\n\n"
                f"[dim]Type:[/dim] {ext.type}  "
                f"[dim]Version:[/dim] {ext.version}  "
                f"[dim]Price:[/dim] {price}  "
                f"[dim]Downloads:[/dim] {ext.downloads}",
                title=f"[cyan]{ext.name}[/cyan] {badge}",
                border_style="blue",
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[red]Failed to fetch featured extensions: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("info")
def info(
    name: str = typer.Argument(..., help="Extension name"),
) -> None:
    """Show detailed information about an extension.

    Example:
        acf marketplace info secrets-scan
    """
    client = get_marketplace_client()

    try:
        ext = client.get_extension(name)

        price = "Free" if ext.is_free else f"${ext.price_usd:.2f}"
        rating = f"★ {ext.rating:.1f}" if ext.rating else "No ratings yet"

        console.print(f"\n[bold cyan]{ext.name}[/bold cyan] v{ext.version}")
        console.print(f"[dim]by {ext.author}[/dim]\n")

        console.print(f"{ext.description}\n")

        console.print(f"[bold]Details[/bold]")
        console.print(f"  Type: {ext.type}")
        console.print(f"  Price: {price}")
        console.print(f"  Downloads: {ext.downloads:,}")
        console.print(f"  Rating: {rating}")
        console.print(f"  Official: {'Yes' if ext.is_official else 'No'}")
        console.print(f"  Updated: {ext.updated_at.strftime('%Y-%m-%d')}")

        # Show context budget if available
        if hasattr(ext, 'context_tokens') and ext.context_tokens:
            console.print(f"\n[bold]Context Budget[/bold]")
            ctx = f"{ext.context_tokens / 1000:.1f}K" if ext.context_tokens >= 1000 else str(ext.context_tokens)
            console.print(f"  Context Tokens: {ctx}")
            if hasattr(ext, 'min_model_tier') and ext.min_model_tier:
                console.print(f"  Min Model Tier: {ext.min_model_tier}")
                # Show compatibility icons
                tier_compat = {
                    "any": "[green]7B[/green] | [green]14B[/green] | [green]32B+[/green]",
                    "small": "[green]7B[/green] | [green]14B[/green] | [green]32B+[/green]",
                    "medium": "[dim]7B[/dim] | [green]14B[/green] | [green]32B+[/green]",
                    "large": "[dim]7B[/dim] | [dim]14B[/dim] | [green]32B+[/green]",
                }
                compat = tier_compat.get(ext.min_model_tier, tier_compat["any"])
                console.print(f"  Compatible: {compat}")

        if ext.keywords:
            console.print(f"\n[bold]Keywords[/bold]: {', '.join(ext.keywords)}")

        console.print(f"\n[dim]Install with: acf marketplace install {name}[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to fetch extension info: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("install")
def install(
    name: str = typer.Argument(..., help="Extension name to install"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if already installed",
    ),
) -> None:
    """Install an extension from the marketplace.

    For paid extensions, you'll be prompted to purchase first.

    Examples:
        acf marketplace install secrets-scan
        acf marketplace install vue-profile --force
    """
    from extensions import ExtensionInstaller, MarketplaceClient
    from extensions.installer import MarketplaceError, InstallError

    client = get_marketplace_client()
    installer = ExtensionInstaller(marketplace_client=client)

    try:
        # Get extension info first
        ext = client.get_extension(name)

        if not ext.is_free:
            console.print(f"[yellow]Extension '{name}' costs ${ext.price_usd:.2f}[/yellow]")
            if not typer.confirm("Would you like to purchase it?"):
                raise typer.Exit(0)

            # Initiate purchase
            try:
                result = client.purchase(name)
                if "checkout_url" in result:
                    console.print(f"Complete purchase at: {result['checkout_url']}")
                    console.print("Run this command again after purchasing.")
                    raise typer.Exit(0)
            except MarketplaceError as e:
                console.print(f"[red]Purchase failed: {e}[/red]")
                raise typer.Exit(1)

        # Install
        console.print(f"Installing {name}...")
        manifest = installer.install_from_marketplace(name, force=force)

        console.print(f"[green]✓ Installed {manifest.name} v{manifest.version}[/green]")
        console.print(f"[dim]Type: {manifest.type.value}[/dim]")

        if manifest.type.value == "agent" and manifest.hook_point:
            console.print(f"[dim]Hook: {manifest.hook_point.value}[/dim]")

    except InstallError as e:
        console.print(f"[red]Installation failed: {e}[/red]")
        raise typer.Exit(1)
    except MarketplaceError as e:
        console.print(f"[red]Marketplace error: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("purchase")
def purchase(
    name: str = typer.Argument(..., help="Extension name to purchase"),
) -> None:
    """Purchase a paid extension.

    Example:
        acf marketplace purchase semantic-rag
    """
    client = get_marketplace_client()

    try:
        ext = client.get_extension(name)

        if ext.is_free:
            console.print(f"[green]'{name}' is free! Install with: acf marketplace install {name}[/green]")
            return

        console.print(f"[bold]Purchasing {name}[/bold]")
        console.print(f"Price: ${ext.price_usd:.2f}")

        if not typer.confirm("Proceed with purchase?"):
            raise typer.Exit(0)

        result = client.purchase(name)

        if "checkout_url" in result:
            console.print(f"\n[bold]Complete your purchase:[/bold]")
            console.print(result["checkout_url"])
        else:
            console.print(f"[green]✓ Purchase successful![/green]")
            console.print(f"Install with: acf marketplace install {name}")

    except Exception as e:
        console.print(f"[red]Purchase failed: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("purchases")
def purchases() -> None:
    """List your purchased extensions.

    Example:
        acf marketplace purchases
    """
    client = get_marketplace_client()

    try:
        results = client.get_purchases()

        if not results:
            console.print("[yellow]No purchased extensions yet[/yellow]")
            return

        table = Table(title="Your Purchased Extensions")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Version")
        table.add_column("Price", justify="right")
        table.add_column("Purchased")

        for ext in results:
            table.add_row(
                ext.name,
                ext.type,
                ext.version,
                f"${ext.price_usd:.2f}",
                ext.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to fetch purchases: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("submit")
def submit(
    path: Path = typer.Argument(..., help="Path to extension directory"),
    price: float = typer.Option(
        0.0,
        "--price",
        "-p",
        help="Price in USD (0 for free)",
    ),
) -> None:
    """Submit an extension for marketplace review.

    Example:
        acf marketplace submit ./my-extension --price 15.00
    """
    installer = get_installer()
    client = get_marketplace_client()

    try:
        # Validate extension
        manifest_path = path / "manifest.yaml"
        if not manifest_path.exists():
            console.print("[red]No manifest.yaml found in extension directory[/red]")
            raise typer.Exit(1)

        from extensions import ExtensionManifest

        manifest = ExtensionManifest.from_yaml(manifest_path)
        console.print(f"[bold]Submitting: {manifest.name} v{manifest.version}[/bold]")
        console.print(f"Type: {manifest.type.value}")
        console.print(f"Price: ${price:.2f}" if price > 0 else "Price: Free")

        if not typer.confirm("Submit for review?"):
            raise typer.Exit(0)

        # Package extension
        console.print("Packaging extension...")
        tarball_path = installer.package_extension(path)

        # Submit
        console.print("Uploading to marketplace...")
        result = client.submit_extension(tarball_path, price)

        # Clean up tarball
        tarball_path.unlink()

        console.print(f"\n[green]✓ Extension submitted successfully![/green]")
        console.print(f"Submission ID: {result.get('submission_id', 'N/A')}")
        console.print(f"Status: {result.get('status', 'pending_review')}")
        console.print("\nYou'll be notified when the review is complete.")

    except Exception as e:
        console.print(f"[red]Submission failed: {e}[/red]")
        raise typer.Exit(1)


@marketplace_app.command("status")
def status(
    submission_id: str = typer.Argument(..., help="Submission ID"),
) -> None:
    """Check status of a submitted extension.

    Example:
        acf marketplace status abc123
    """
    client = get_marketplace_client()

    try:
        result = client.get_submission_status(submission_id)

        console.print(f"\n[bold]Submission Status[/bold]")
        console.print(f"ID: {submission_id}")
        console.print(f"Status: {result.get('status', 'unknown')}")

        if result.get("feedback"):
            console.print(f"\n[bold]Feedback:[/bold]")
            console.print(result["feedback"])

        if result.get("approved_at"):
            console.print(f"\n[green]✓ Approved on {result['approved_at']}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to fetch status: {e}[/red]")
        raise typer.Exit(1)
