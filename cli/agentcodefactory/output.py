"""Rich console output utilities for AgentCodeFactory CLI."""

import time
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


console = Console()


# Pipeline stages in order
PIPELINE_STAGES = [
    ("SPEC", "Spec"),
    ("CONTEXT", "Context"),
    ("DESIGN", "Design"),
    ("IMPLEMENTATION", "Implement"),
    ("CODE_REVIEW", "Review"),
    ("DONE", "Done"),
]


class PipelineDisplay:
    """Live pipeline progress display with stage visualization."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.current_stage: Optional[str] = None
        self.completed_stages: set[str] = set()
        self.failed_stage: Optional[str] = None
        self.current_message: str = "Initializing..."
        self.start_time = time.time()
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        self._live: Optional[Live] = None

    def _format_elapsed(self) -> str:
        """Format elapsed time as MM:SS."""
        elapsed = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _get_stage_icon(self, stage_key: str) -> str:
        """Get icon for stage based on status."""
        if stage_key in self.completed_stages:
            return "[green]✓[/green]"
        elif self.failed_stage == stage_key:
            return "[red]✗[/red]"
        elif self.current_stage == stage_key:
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
            return f"[cyan]{self.spinner_frames[self.spinner_idx]}[/cyan]"
        else:
            return "[dim]○[/dim]"

    def _render_pipeline(self) -> Text:
        """Render the pipeline stages as a single line."""
        parts = []
        for i, (stage_key, stage_name) in enumerate(PIPELINE_STAGES):
            icon = self._get_stage_icon(stage_key)

            # Style the stage name based on status
            if stage_key in self.completed_stages:
                name_style = "green"
            elif self.failed_stage == stage_key:
                name_style = "red"
            elif self.current_stage == stage_key:
                name_style = "cyan bold"
            else:
                name_style = "dim"

            parts.append(f"{icon} [{name_style}]{stage_name}[/{name_style}]")

            # Add arrow between stages (except last)
            if i < len(PIPELINE_STAGES) - 1:
                if stage_key in self.completed_stages:
                    parts.append(" [green]→[/green] ")
                else:
                    parts.append(" [dim]→[/dim] ")

        text = Text.from_markup("".join(parts))
        return text

    def _render(self) -> Panel:
        """Render the full pipeline display."""
        # Pipeline visualization
        pipeline_line = self._render_pipeline()

        # Current status message
        if self.failed_stage:
            status_line = Text.from_markup(f"\n[red]✗ {self.current_message}[/red]")
        else:
            status_line = Text.from_markup(f"\n[dim]{self.current_message}[/dim]")

        # Timer
        elapsed = self._format_elapsed()
        timer_line = Text.from_markup(f"\n[dim]⏱  {elapsed}[/dim]")

        # Combine all elements
        content = Group(pipeline_line, status_line, timer_line)

        # Create panel
        return Panel(
            content,
            title=f"[bold cyan]ACF Pipeline[/bold cyan]",
            subtitle=f"[dim]Job: {self.job_id[:12]}...[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )

    def __enter__(self) -> "PipelineDisplay":
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        """Stop the live display."""
        if self._live:
            # Final render before exit
            self._live.update(self._render())
            self._live.__exit__(*args)

    def update(self, progress_message: str) -> None:
        """Update progress from API response."""
        self.current_message = progress_message

        # Parse stage from progress message
        # Messages look like: "Parsing requirements...", "Creating design ✓", etc.
        stage_map = {
            "Queued": None,
            "Initializing": None,
            "Parsing requirements": "SPEC",
            "Analyzing codebase": "CONTEXT",
            "Processing RAG": "CONTEXT",
            "Creating design": "DESIGN",
            "Approving design": "DESIGN",
            "Generating code": "IMPLEMENTATION",
            "Implementing": "IMPLEMENTATION",
            "Reviewing code": "CODE_REVIEW",
            "Enforcing policies": "CODE_REVIEW",
            "Running tests": "CODE_REVIEW",
            "Verifying": "CODE_REVIEW",
            "Storing": "DONE",
            "Complete": "DONE",
            "Generated": "DONE",
        }

        # Find matching stage
        for key, stage in stage_map.items():
            if key.lower() in progress_message.lower():
                if stage:
                    # Mark previous stage as completed
                    if self.current_stage and self.current_stage != stage:
                        self.completed_stages.add(self.current_stage)
                    self.current_stage = stage
                break

        # Check for completion markers
        if "✓" in progress_message or "complete" in progress_message.lower():
            if self.current_stage:
                self.completed_stages.add(self.current_stage)

        # Update display
        if self._live:
            self._live.update(self._render())

    def complete(self, file_count: int = 0) -> None:
        """Mark pipeline as complete."""
        # Mark all stages as complete
        for stage_key, _ in PIPELINE_STAGES:
            self.completed_stages.add(stage_key)
        self.current_stage = None
        if file_count > 0:
            self.current_message = f"✨ Generated {file_count} files"
        else:
            self.current_message = "✨ Complete"
        if self._live:
            self._live.update(self._render())

    def fail(self, error: str) -> None:
        """Mark pipeline as failed."""
        self.failed_stage = self.current_stage or "SPEC"
        self.current_message = error[:80] + "..." if len(error) > 80 else error
        if self._live:
            self._live.update(self._render())


error_console = Console(stderr=True)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    error_console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]→[/blue] {message}")


def print_json(data: Any) -> None:
    """Print formatted JSON."""
    import json

    console.print_json(json.dumps(data, indent=2, default=str))


def print_key_value(key: str, value: Any, key_style: str = "bold") -> None:
    """Print a key-value pair."""
    console.print(f"[{key_style}]{key}:[/{key_style}] {value}")


def print_config(config: dict[str, Any]) -> None:
    """Print configuration as a table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")

    for key, value in sorted(config.items()):
        # Mask API key
        if key == "api_key" and value:
            value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
        table.add_row(key, str(value))

    console.print(table)


def print_projects(projects: list[Any]) -> None:
    """Print projects as a table."""
    if not projects:
        print_info("No projects found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Iterations", justify="right")
    table.add_column("Storage", justify="right")
    table.add_column("Created")

    for p in projects:
        storage = format_bytes(p.storage_bytes) if hasattr(p, "storage_bytes") else "-"
        created = p.created_at[:10] if p.created_at else "-"
        table.add_row(
            p.id[:12] + "..." if len(p.id) > 15 else p.id,
            p.name,
            str(p.iteration_count),
            storage,
            created,
        )

    console.print(table)


def print_project_detail(project: Any, iterations: list[Any] = None) -> None:
    """Print detailed project info."""
    console.print(Panel(f"[bold]{project.name}[/bold]", subtitle=f"ID: {project.id}"))

    print_key_value("Description", project.description or "(none)")
    print_key_value("Iterations", project.iteration_count)
    print_key_value("Storage", format_bytes(project.storage_bytes))
    print_key_value("Created", project.created_at or "-")
    print_key_value("Updated", project.updated_at or "-")

    if iterations:
        console.print("\n[bold]Iterations:[/bold]")
        print_iterations(iterations)


def print_iterations(iterations: list[Any]) -> None:
    """Print iterations as a table."""
    if not iterations:
        print_info("No iterations found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Status")
    table.add_column("Prompt")
    table.add_column("Files")
    table.add_column("Created")

    for i in iterations:
        status_color = {
            "completed": "green",
            "running": "yellow",
            "pending": "blue",
            "failed": "red",
        }.get(i.status, "white")

        prompt_preview = i.prompt[:40] + "..." if len(i.prompt) > 43 else i.prompt
        created = i.created_at[:10] if i.created_at else "-"

        table.add_row(
            i.id[:12] + "...",
            f"v{i.version}",
            f"[{status_color}]{i.status}[/{status_color}]",
            prompt_preview,
            "Yes" if i.files_available else "No",
            created,
        )

    console.print(table)


def print_job_result(result: Any, verbose: bool = False) -> None:
    """Print job result summary."""
    status_color = "green" if result.status == "completed" else "red"
    console.print(
        Panel(
            f"[{status_color}]{result.status.upper()}[/{status_color}]",
            title="Generation Complete",
        )
    )

    if result.error:
        print_error(result.error)
        return

    print_key_value("Job ID", result.job_id)
    if result.project_id:
        print_key_value("Project ID", result.project_id)
    if result.iteration_id:
        print_key_value("Iteration ID", result.iteration_id)

    if result.usage:
        tokens = result.usage.get("input_tokens", 0) + result.usage.get("output_tokens", 0)
        print_key_value("Tokens Used", f"{tokens:,}")
    if result.cost_usd:
        print_key_value("Cost", f"${result.cost_usd:.4f}")

    console.print(f"\n[bold]Generated {len(result.files)} files:[/bold]")
    print_file_tree(result.files)

    if verbose and result.security_warnings:
        console.print("\n[bold yellow]Security Warnings:[/bold yellow]")
        print_json(result.security_warnings)


def print_file_tree(files: list[Any]) -> None:
    """Print files as a tree structure."""
    if not files:
        print_info("No files generated.")
        return

    # Build tree structure
    tree = Tree("[bold].[/bold]")
    paths: dict[str, Any] = {}

    for f in sorted(files, key=lambda x: x.path):
        parts = f.path.split("/")
        current = tree

        # Navigate/create path
        for i, part in enumerate(parts[:-1]):
            path_so_far = "/".join(parts[: i + 1])
            if path_so_far not in paths:
                paths[path_so_far] = current.add(f"[blue]{part}/[/blue]")
            current = paths[path_so_far]

        # Add file
        filename = parts[-1]
        size = len(f.content) if hasattr(f, "content") else 0
        current.add(f"[green]{filename}[/green] [dim]({format_bytes(size)})[/dim]")

    console.print(tree)


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def create_progress() -> Progress:
    """Create a progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def create_spinner(description: str = "Working...") -> Progress:
    """Create a simple spinner for indeterminate progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
