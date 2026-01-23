"""Checkpoint and approval management."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from schemas.pipeline_state import Checkpoint, PipelineState, Stage


class ApprovalResult(Enum):
    """Result of an approval checkpoint."""

    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"  # User wants to review later


@dataclass
class ApprovalResponse:
    """Response from approval prompt."""

    result: ApprovalResult
    notes: str | None = None
    approved_by: str = "user"


class CheckpointManager:
    """Manages human approval checkpoints in the pipeline.

    Supports:
    - CLI prompts (blocking)
    - File-based approvals (for async workflows)
    - Auto-approval based on criteria
    """

    def __init__(
        self,
        console: Console | None = None,
        auto_approve: bool = False,
        approval_callback: Callable[[Checkpoint, PipelineState], ApprovalResponse] | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            console: Rich console for output
            auto_approve: If True, auto-approve all checkpoints
            approval_callback: Custom approval handler
        """
        self.console = console or Console()
        self.auto_approve = auto_approve
        self.approval_callback = approval_callback

    def create_checkpoint(
        self,
        stage: Stage,
        state: PipelineState,
        title: str,
        description: str,
        artifact_paths: list[str] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            stage: Stage requiring approval
            state: Current pipeline state
            title: Checkpoint title
            description: What needs review
            artifact_paths: Paths to artifacts for review

        Returns:
            Checkpoint object
        """
        checkpoint = Checkpoint(
            stage=stage,
            title=title,
            description=description,
            artifact_paths=artifact_paths or [],
        )
        state.checkpoints.append(checkpoint)
        return checkpoint

    def request_approval(
        self,
        checkpoint: Checkpoint,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> ApprovalResponse:
        """Request approval for a checkpoint.

        Args:
            checkpoint: The checkpoint requiring approval
            state: Current pipeline state
            artifacts_dir: Directory containing artifacts

        Returns:
            ApprovalResponse with decision
        """
        # Auto-approve if configured
        if self.auto_approve:
            return ApprovalResponse(
                result=ApprovalResult.APPROVED,
                notes="Auto-approved",
                approved_by="auto",
            )

        # Use custom callback if provided
        if self.approval_callback:
            return self.approval_callback(checkpoint, state)

        # Default: CLI prompt
        return self._cli_approval(checkpoint, state, artifacts_dir)

    def _cli_approval(
        self,
        checkpoint: Checkpoint,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> ApprovalResponse:
        """Interactive CLI approval prompt.

        Args:
            checkpoint: The checkpoint
            state: Pipeline state
            artifacts_dir: Artifacts directory

        Returns:
            ApprovalResponse
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold yellow]Approval Required: {checkpoint.title}[/bold yellow]",
                title="Checkpoint",
                border_style="yellow",
            )
        )

        # Show description
        self.console.print()
        self.console.print(Markdown(checkpoint.description))
        self.console.print()

        # Show artifacts to review
        if checkpoint.artifact_paths:
            self.console.print("[bold]Artifacts to review:[/bold]")
            for path in checkpoint.artifact_paths:
                full_path = artifacts_dir / path
                self.console.print(f"  - {full_path}")

                # Show content preview for markdown files
                if full_path.suffix == ".md" and full_path.exists():
                    content = full_path.read_text()
                    if len(content) > 2000:
                        content = content[:2000] + "\n\n... (truncated)"
                    self.console.print()
                    self.console.print(Panel(Markdown(content), title=path))

            self.console.print()

        # Prompt for decision
        self.console.print("[bold]Options:[/bold]")
        self.console.print("  [green]y/yes[/green] - Approve and continue")
        self.console.print("  [red]n/no[/red] - Reject and stop")
        self.console.print("  [yellow]d/defer[/yellow] - Save state and exit (resume later)")
        self.console.print("  [blue]v/view[/blue] - View artifact contents")
        self.console.print()

        while True:
            choice = Prompt.ask(
                "Your decision",
                choices=["y", "yes", "n", "no", "d", "defer", "v", "view"],
                default="y",
            )

            if choice in ("y", "yes"):
                notes = Prompt.ask("Any notes? (optional)", default="")
                return ApprovalResponse(
                    result=ApprovalResult.APPROVED,
                    notes=notes if notes else None,
                )

            elif choice in ("n", "no"):
                reason = Prompt.ask("Reason for rejection")
                return ApprovalResponse(
                    result=ApprovalResult.REJECTED,
                    notes=reason,
                )

            elif choice in ("d", "defer"):
                self.console.print(
                    "[yellow]State saved. Run the same command to resume.[/yellow]"
                )
                return ApprovalResponse(result=ApprovalResult.DEFERRED)

            elif choice in ("v", "view"):
                self._view_artifacts(checkpoint, artifacts_dir)

    def _view_artifacts(self, checkpoint: Checkpoint, artifacts_dir: Path) -> None:
        """Display artifact contents."""
        if not checkpoint.artifact_paths:
            self.console.print("[dim]No artifacts to view[/dim]")
            return

        for path in checkpoint.artifact_paths:
            full_path = artifacts_dir / path
            if full_path.exists():
                content = full_path.read_text()
                self.console.print()
                self.console.print(
                    Panel(
                        Markdown(content) if full_path.suffix == ".md" else content,
                        title=str(path),
                        border_style="blue",
                    )
                )
            else:
                self.console.print(f"[red]File not found: {path}[/red]")

    def get_design_checkpoint(self, state: PipelineState) -> Checkpoint:
        """Create the design approval checkpoint.

        Args:
            state: Pipeline state

        Returns:
            Checkpoint for design review
        """
        return self.create_checkpoint(
            stage=Stage.DESIGN_APPROVAL,
            state=state,
            title="Design Review",
            description="""
## Design Proposal Review

Please review the proposed design for this feature:

1. **Architecture**: Does the proposed architecture align with the codebase?
2. **Patterns**: Are the patterns appropriate?
3. **Dependencies**: Are new dependencies acceptable?
4. **Risks**: Are the identified risks manageable?

If you approve, the implementation phase will begin.
""",
            artifact_paths=["design_proposal.md", "context_report.md"],
        )

    def get_final_checkpoint(self, state: PipelineState) -> Checkpoint:
        """Create the final approval checkpoint.

        Args:
            state: Pipeline state

        Returns:
            Checkpoint for final review
        """
        return self.create_checkpoint(
            stage=Stage.FINAL_APPROVAL,
            state=state,
            title="Final Review & Deploy",
            description="""
## Final Review

The implementation is complete. Please review:

1. **Changes**: Review the diff and implementation notes
2. **Tests**: All tests passing?
3. **Verification**: Black-box checks passed?
4. **Release notes**: Ready for release?

If you approve, changes will be deployed.
""",
            artifact_paths=[
                "diff.patch",
                "implementation_notes.md",
                "test_report.md",
                "verification_report.md",
            ],
        )
