"""Git operations tool using GitPython."""

import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult, ToolStatus


class GitTool(BaseTool):
    """Tool for Git repository operations.

    Provides:
    - Repository status and info
    - Branch management
    - Diff generation and application
    - Commit operations
    """

    name = "git"
    description = "Git repository operations"

    def __init__(self, repo_path: Path | str | None = None) -> None:
        """Initialize Git tool.

        Args:
            repo_path: Path to repository (default: current directory)
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def execute(self, operation: str, **kwargs: Any) -> ToolResult:
        """Execute a Git operation.

        Args:
            operation: Operation name (status, branch, diff, commit, apply_patch, etc.)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with operation output
        """
        operations = {
            "status": self._status,
            "branch": self._branch,
            "create_branch": self._create_branch,
            "checkout": self._checkout,
            "diff": self._diff,
            "diff_staged": self._diff_staged,
            "add": self._add,
            "commit": self._commit,
            "apply_patch": self._apply_patch,
            "log": self._log,
            "list_files": self._list_files,
            "show_file": self._show_file,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            return ToolResult(status=ToolStatus.FAILURE, error=str(e))

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )

    def _status(self) -> ToolResult:
        """Get repository status."""
        result = self._run_git("status", "--porcelain", check=False)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output={
                "clean": len(result.stdout.strip()) == 0,
                "files": result.stdout.strip().split("\n") if result.stdout.strip() else [],
            },
        )

    def _branch(self) -> ToolResult:
        """Get current branch name."""
        result = self._run_git("branch", "--show-current", check=False)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=result.stdout.strip(),
        )

    def _create_branch(self, name: str, checkout: bool = True) -> ToolResult:
        """Create a new branch."""
        if checkout:
            result = self._run_git("checkout", "-b", name, check=False)
        else:
            result = self._run_git("branch", name, check=False)

        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)

        return ToolResult(status=ToolStatus.SUCCESS, output=name)

    def _checkout(self, ref: str) -> ToolResult:
        """Checkout a branch or commit."""
        result = self._run_git("checkout", ref, check=False)
        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)
        return ToolResult(status=ToolStatus.SUCCESS, output=ref)

    def _diff(self, path: str | None = None) -> ToolResult:
        """Get unstaged diff."""
        args = ["diff"]
        if path:
            args.append(path)
        result = self._run_git(*args, check=False)
        return ToolResult(status=ToolStatus.SUCCESS, output=result.stdout)

    def _diff_staged(self) -> ToolResult:
        """Get staged diff."""
        result = self._run_git("diff", "--staged", check=False)
        return ToolResult(status=ToolStatus.SUCCESS, output=result.stdout)

    def _add(self, paths: list[str] | str = ".") -> ToolResult:
        """Stage files for commit."""
        if isinstance(paths, str):
            paths = [paths]
        result = self._run_git("add", *paths, check=False)
        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)
        return ToolResult(status=ToolStatus.SUCCESS, output=paths)

    def _commit(self, message: str) -> ToolResult:
        """Create a commit."""
        result = self._run_git("commit", "-m", message, check=False)
        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)

        # Get commit hash
        hash_result = self._run_git("rev-parse", "HEAD", check=False)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output={"message": message, "hash": hash_result.stdout.strip()},
        )

    def _apply_patch(self, patch_content: str, check_only: bool = False) -> ToolResult:
        """Apply a patch to the repository."""
        args = ["apply"]
        if check_only:
            args.append("--check")
        args.append("-")

        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            input=patch_content,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="Patch applied" if not check_only else "Patch can be applied",
        )

    def _log(self, count: int = 10, oneline: bool = True) -> ToolResult:
        """Get commit log."""
        args = ["log", f"-{count}"]
        if oneline:
            args.append("--oneline")
        result = self._run_git(*args, check=False)
        return ToolResult(status=ToolStatus.SUCCESS, output=result.stdout.strip())

    def _list_files(self, pattern: str | None = None) -> ToolResult:
        """List tracked files."""
        args = ["ls-files"]
        if pattern:
            args.append(pattern)
        result = self._run_git(*args, check=False)
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return ToolResult(status=ToolStatus.SUCCESS, output=files)

    def _show_file(self, path: str, ref: str = "HEAD") -> ToolResult:
        """Show file content at a specific ref."""
        result = self._run_git("show", f"{ref}:{path}", check=False)
        if result.returncode != 0:
            return ToolResult(status=ToolStatus.FAILURE, error=result.stderr)
        return ToolResult(status=ToolStatus.SUCCESS, output=result.stdout)
