"""Local Git versioner for ACF Local Edition.

Provides automatic git-based versioning for pipeline iterations,
allowing users to track and restore generated code without external dependencies.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ACF_COMMIT_PREFIX = "[ACF]"


@dataclass
class Iteration:
    """Represents a single pipeline iteration."""

    sha: str
    message: str
    date: datetime
    stage: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None


class GitError(Exception):
    """Raised when a git operation fails."""

    pass


class LocalGitVersioner:
    """Auto-commit iterations to local git repository.

    This class manages versioning of pipeline-generated code using the local
    git repository. Each iteration is committed with a standardized prefix
    and metadata for easy tracking and restoration.

    Example:
        >>> versioner = LocalGitVersioner(Path("/my/project"))
        >>> sha = versioner.save_iteration(
        ...     files={"src/main.py": "print('hello')"},
        ...     message="Initial implementation",
        ...     stage="implementation",
        ...     run_id="abc123"
        ... )
        >>> iterations = versioner.list_iterations()
        >>> versioner.restore_iteration(sha)
    """

    def __init__(self, project_dir: Path):
        """Initialize the versioner.

        Args:
            project_dir: Path to the git repository root.

        Raises:
            GitError: If the directory is not a git repository.
        """
        self.project_dir = Path(project_dir).resolve()
        self._validate_repo()

    def _validate_repo(self) -> None:
        """Validate that the project directory is a git repository."""
        git_dir = self.project_dir / ".git"
        if not git_dir.exists():
            raise GitError(
                f"Not a git repository: {self.project_dir}. "
                "Initialize with 'git init' first."
            )

    def _run_git(
        self, *args: str, check: bool = True, capture: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the project directory.

        Args:
            *args: Git command arguments (without 'git' prefix).
            check: Whether to raise on non-zero exit code.
            capture: Whether to capture stdout/stderr.

        Returns:
            CompletedProcess with stdout and stderr.

        Raises:
            GitError: If the command fails and check=True.
        """
        cmd = ["git", *args]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=capture,
                text=True,
                check=False,
            )
            if check and result.returncode != 0:
                raise GitError(
                    f"Git command failed: {' '.join(cmd)}\n"
                    f"stderr: {result.stderr}"
                )
            return result
        except FileNotFoundError:
            raise GitError("Git is not installed or not in PATH")

    def _encode_metadata(self, metadata: dict[str, Any]) -> str:
        """Encode metadata as a commit message trailer."""
        return f"\n\nACF-Metadata: {json.dumps(metadata, separators=(',', ':'))}"

    def _decode_metadata(self, message: str) -> dict[str, Any] | None:
        """Decode metadata from a commit message."""
        marker = "ACF-Metadata: "
        for line in message.split("\n"):
            if line.startswith(marker):
                try:
                    return json.loads(line[len(marker) :])
                except json.JSONDecodeError:
                    return None
        return None

    def save_iteration(
        self,
        files: dict[str, str],
        message: str,
        stage: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write files and commit as a pipeline iteration.

        Args:
            files: Dictionary mapping relative file paths to content.
            message: Commit message describing the changes.
            stage: Pipeline stage name (e.g., "implementation", "testing").
            run_id: Pipeline run identifier.
            metadata: Additional metadata to store in commit.

        Returns:
            The commit SHA.

        Raises:
            GitError: If the commit operation fails.
        """
        if not files:
            raise GitError("No files to commit")

        # Write all files
        for rel_path, content in files.items():
            file_path = self.project_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Stage all files
        self._run_git("add", *list(files.keys()))

        # Check if there are changes to commit
        status = self._run_git("status", "--porcelain")
        if not status.stdout.strip():
            # No changes - return current HEAD
            result = self._run_git("rev-parse", "HEAD")
            return result.stdout.strip()

        # Build commit message with metadata
        full_metadata = {
            "stage": stage,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        commit_message = f"{ACF_COMMIT_PREFIX} {message}"
        commit_message += self._encode_metadata(full_metadata)

        # Commit
        self._run_git("commit", "-m", commit_message)

        # Get commit SHA
        result = self._run_git("rev-parse", "HEAD")
        return result.stdout.strip()

    def list_iterations(
        self,
        run_id: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[Iteration]:
        """List all ACF commits in the repository.

        Args:
            run_id: Filter by pipeline run ID.
            stage: Filter by pipeline stage.
            limit: Maximum number of iterations to return.

        Returns:
            List of Iteration objects, newest first.
        """
        # Get commits with ACF prefix
        result = self._run_git(
            "log",
            f"--max-count={limit * 2}",  # Fetch extra in case of filtering
            "--pretty=format:%H|%s|%aI|%B<<<END>>>",
            f"--grep={ACF_COMMIT_PREFIX}",
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return []

        iterations: list[Iteration] = []
        commits = result.stdout.split("<<<END>>>")

        for commit_data in commits:
            commit_data = commit_data.strip()
            if not commit_data:
                continue

            lines = commit_data.split("|", 3)
            if len(lines) < 4:
                continue

            sha, subject, date_str, body = lines

            # Parse metadata from body
            metadata = self._decode_metadata(body)

            # Remove prefix from subject
            message = subject
            if message.startswith(ACF_COMMIT_PREFIX):
                message = message[len(ACF_COMMIT_PREFIX) :].strip()

            # Parse date
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                date = datetime.utcnow()

            iteration = Iteration(
                sha=sha,
                message=message,
                date=date,
                stage=metadata.get("stage") if metadata else None,
                run_id=metadata.get("run_id") if metadata else None,
                metadata=metadata,
            )

            # Apply filters
            if run_id and iteration.run_id != run_id:
                continue
            if stage and iteration.stage != stage:
                continue

            iterations.append(iteration)

            if len(iterations) >= limit:
                break

        return iterations

    def restore_iteration(self, sha: str, create_branch: bool = False) -> None:
        """Restore the working directory to a specific iteration.

        Args:
            sha: The commit SHA to restore.
            create_branch: If True, create a new branch instead of detaching HEAD.

        Raises:
            GitError: If the restore operation fails.
        """
        # Validate SHA exists
        result = self._run_git("cat-file", "-t", sha, check=False)
        if result.returncode != 0:
            raise GitError(f"Invalid commit SHA: {sha}")

        # Check for uncommitted changes
        status = self._run_git("status", "--porcelain")
        if status.stdout.strip():
            raise GitError(
                "Working directory has uncommitted changes. "
                "Please commit or stash them before restoring."
            )

        if create_branch:
            # Create a new branch at the iteration
            branch_name = f"acf-restore-{sha[:8]}"
            self._run_git("checkout", "-b", branch_name, sha)
        else:
            # Checkout the commit (detached HEAD)
            self._run_git("checkout", sha)

    def get_iteration(self, sha: str) -> Iteration | None:
        """Get details of a specific iteration.

        Args:
            sha: The commit SHA.

        Returns:
            Iteration object or None if not found/not an ACF commit.
        """
        result = self._run_git(
            "log",
            "-1",
            "--pretty=format:%H|%s|%aI|%B",
            sha,
            check=False,
        )

        if result.returncode != 0:
            return None

        lines = result.stdout.split("|", 3)
        if len(lines) < 4:
            return None

        sha_full, subject, date_str, body = lines

        if not subject.startswith(ACF_COMMIT_PREFIX):
            return None

        metadata = self._decode_metadata(body)
        message = subject[len(ACF_COMMIT_PREFIX) :].strip()

        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            date = datetime.utcnow()

        return Iteration(
            sha=sha_full,
            message=message,
            date=date,
            stage=metadata.get("stage") if metadata else None,
            run_id=metadata.get("run_id") if metadata else None,
            metadata=metadata,
        )

    def get_diff(self, sha1: str, sha2: str | None = None) -> str:
        """Get the diff between two iterations.

        Args:
            sha1: First commit SHA.
            sha2: Second commit SHA. If None, diff against working directory.

        Returns:
            Unified diff string.
        """
        if sha2:
            result = self._run_git("diff", sha1, sha2)
        else:
            result = self._run_git("diff", sha1)
        return result.stdout

    def get_changed_files(self, sha: str) -> list[str]:
        """Get list of files changed in a commit.

        Args:
            sha: The commit SHA.

        Returns:
            List of file paths that were changed.
        """
        result = self._run_git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", sha
        )
        return [f for f in result.stdout.strip().split("\n") if f]

    def tag_iteration(self, sha: str, tag_name: str, message: str = "") -> None:
        """Create a git tag for an iteration.

        Args:
            sha: The commit SHA to tag.
            tag_name: Name for the tag.
            message: Optional tag message.

        Raises:
            GitError: If tagging fails.
        """
        if message:
            self._run_git("tag", "-a", tag_name, sha, "-m", message)
        else:
            self._run_git("tag", tag_name, sha)

    def get_current_branch(self) -> str | None:
        """Get the current branch name.

        Returns:
            Branch name or None if in detached HEAD state.
        """
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD", check=False)
        if result.returncode != 0:
            return None
        branch = result.stdout.strip()
        return None if branch == "HEAD" else branch

    def stash_changes(self, message: str = "ACF auto-stash") -> bool:
        """Stash uncommitted changes.

        Args:
            message: Stash message.

        Returns:
            True if changes were stashed, False if working directory was clean.
        """
        status = self._run_git("status", "--porcelain")
        if not status.stdout.strip():
            return False

        self._run_git("stash", "push", "-m", message)
        return True

    def pop_stash(self) -> bool:
        """Pop the most recent stash.

        Returns:
            True if stash was popped, False if no stashes exist.
        """
        result = self._run_git("stash", "list", check=False)
        if not result.stdout.strip():
            return False

        self._run_git("stash", "pop")
        return True
