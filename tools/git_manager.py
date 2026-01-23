"""Git manager for version control of generated projects."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    message: str
    output: str = ""


class GitManager:
    """Manages git operations for generated projects.

    Handles:
    - Repository initialization
    - Commits after each pipeline stage
    - Feature branches per run
    - Iteration tracking on same branch

    SSH Key Support:
        Set GIT_SSH_KEY environment variable with the private key content
        for headless/CI environments. The key will be used automatically
        for all remote operations (push, pull, fetch).

        Example:
            export GIT_SSH_KEY="-----BEGIN OPENSSH PRIVATE KEY-----
            ...key content...
            -----END OPENSSH PRIVATE KEY-----"
    """

    def __init__(self, project_dir: Path) -> None:
        """Initialize git manager.

        Args:
            project_dir: Path to the generated project directory
        """
        self.project_dir = project_dir
        self._ssh_key_file: Path | None = None

    def _get_git_env(self) -> dict[str, str] | None:
        """Get environment variables for git commands.

        If GIT_SSH_KEY is set, creates a temporary key file and
        configures GIT_SSH_COMMAND to use it.

        Returns:
            Environment dict or None to use default
        """
        ssh_key = os.environ.get("GIT_SSH_KEY")
        if not ssh_key:
            return None

        # Create temporary key file if not already created
        if self._ssh_key_file is None or not self._ssh_key_file.exists():
            # Create temp file with restricted permissions
            fd, key_path = tempfile.mkstemp(prefix="git_ssh_", suffix=".key")
            self._ssh_key_file = Path(key_path)

            # Write key content
            with os.fdopen(fd, "w") as f:
                f.write(ssh_key)
                if not ssh_key.endswith("\n"):
                    f.write("\n")

            # Set permissions to 600 (required by SSH)
            os.chmod(self._ssh_key_file, 0o600)

        # Build environment with custom SSH command
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = (
            f"ssh -i {self._ssh_key_file} "
            "-o StrictHostKeyChecking=accept-new "
            "-o UserKnownHostsFile=/dev/null"
        )
        return env

    def _cleanup_ssh_key(self) -> None:
        """Clean up temporary SSH key file."""
        if self._ssh_key_file and self._ssh_key_file.exists():
            try:
                self._ssh_key_file.unlink()
            except OSError:
                pass
            self._ssh_key_file = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._cleanup_ssh_key()

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the project directory.

        Automatically uses GIT_SSH_KEY if set in environment.

        Args:
            *args: Git command arguments
            check: Whether to raise on non-zero exit

        Returns:
            CompletedProcess result
        """
        return subprocess.run(
            ["git", *args],
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            check=check,
            env=self._get_git_env(),
        )

    def is_repo(self) -> bool:
        """Check if project directory is a git repository."""
        git_dir = self.project_dir / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def init_repo(self, branch_name: str = "main") -> GitResult:
        """Initialize a new git repository.

        Args:
            branch_name: Name for the initial branch

        Returns:
            GitResult with success status
        """
        if self.is_repo():
            return GitResult(
                success=True,
                message="Repository already initialized",
            )

        try:
            # Initialize repo
            self._run_git("init", "-b", branch_name)

            # Configure local user (for commits without global config)
            self._run_git("config", "user.email", "coding-factory@local")
            self._run_git("config", "user.name", "Coding Factory")

            return GitResult(
                success=True,
                message=f"Initialized git repository on branch '{branch_name}'",
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to initialize repository: {e.stderr}",
                output=e.stderr,
            )

    def create_branch(self, branch_name: str, checkout: bool = True) -> GitResult:
        """Create a new branch.

        Args:
            branch_name: Name of the branch to create
            checkout: Whether to checkout the new branch

        Returns:
            GitResult with success status
        """
        try:
            if checkout:
                # Create and checkout
                result = self._run_git("checkout", "-b", branch_name, check=False)
                if result.returncode != 0:
                    # Branch might exist, try just checking out
                    result = self._run_git("checkout", branch_name, check=False)
                    if result.returncode != 0:
                        return GitResult(
                            success=False,
                            message=f"Failed to create/checkout branch: {result.stderr}",
                            output=result.stderr,
                        )
                    return GitResult(
                        success=True,
                        message=f"Checked out existing branch '{branch_name}'",
                    )
            else:
                self._run_git("branch", branch_name)

            return GitResult(
                success=True,
                message=f"Created branch '{branch_name}'",
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to create branch: {e.stderr}",
                output=e.stderr,
            )

    def get_current_branch(self) -> str | None:
        """Get the current branch name."""
        try:
            result = self._run_git("branch", "--show-current")
            return result.stdout.strip() or None
        except subprocess.CalledProcessError:
            return None

    def stage_all(self) -> GitResult:
        """Stage all changes for commit."""
        try:
            self._run_git("add", "-A")
            return GitResult(success=True, message="Staged all changes")
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to stage changes: {e.stderr}",
                output=e.stderr,
            )

    def commit(self, message: str, allow_empty: bool = False) -> GitResult:
        """Create a commit with the given message.

        Args:
            message: Commit message
            allow_empty: Whether to allow empty commits

        Returns:
            GitResult with success status
        """
        try:
            # Stage all changes first
            self.stage_all()

            # Check if there are changes to commit
            status = self._run_git("status", "--porcelain")
            if not status.stdout.strip() and not allow_empty:
                return GitResult(
                    success=True,
                    message="No changes to commit",
                )

            # Commit
            args = ["commit", "-m", message]
            if allow_empty:
                args.append("--allow-empty")

            result = self._run_git(*args)

            # Get the commit hash
            hash_result = self._run_git("rev-parse", "--short", "HEAD")
            commit_hash = hash_result.stdout.strip()

            return GitResult(
                success=True,
                message=f"Committed: {commit_hash}",
                output=result.stdout,
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to commit: {e.stderr}",
                output=e.stderr,
            )

    def commit_stage(
        self,
        stage: str,
        run_id: str,
        feature_summary: str = "",
    ) -> GitResult:
        """Commit changes for a specific pipeline stage.

        Args:
            stage: Pipeline stage name (e.g., 'implementation', 'scaffold')
            run_id: Pipeline run ID
            feature_summary: Short description of the feature

        Returns:
            GitResult with success status
        """
        # Build commit message
        stage_emoji = {
            "init": "🎬",
            "implementation": "💻",
            "scaffold": "📦",
            "testing": "🧪",
            "verification": "✅",
            "iteration": "🔄",
        }.get(stage, "📝")

        summary = feature_summary[:50] if feature_summary else "Pipeline update"
        message = f"{stage_emoji} [{stage}] {summary}\n\nRun ID: {run_id}"

        return self.commit(message)

    def get_log(self, max_count: int = 10) -> list[dict]:
        """Get commit history.

        Args:
            max_count: Maximum number of commits to return

        Returns:
            List of commit dictionaries with hash, message, date
        """
        try:
            result = self._run_git(
                "log",
                f"--max-count={max_count}",
                "--format=%H|%s|%ai",
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        commits.append({
                            "hash": parts[0][:8],
                            "message": parts[1],
                            "date": parts[2],
                        })
            return commits
        except subprocess.CalledProcessError:
            return []

    def get_diff_stat(self, from_ref: str = "HEAD~1", to_ref: str = "HEAD") -> str:
        """Get diff statistics between two refs.

        Args:
            from_ref: Starting reference
            to_ref: Ending reference

        Returns:
            Diff stat output
        """
        try:
            result = self._run_git("diff", "--stat", from_ref, to_ref)
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def tag(self, tag_name: str, message: str = "") -> GitResult:
        """Create a git tag.

        Args:
            tag_name: Name of the tag
            message: Optional tag message (creates annotated tag)

        Returns:
            GitResult with success status
        """
        try:
            if message:
                self._run_git("tag", "-a", tag_name, "-m", message)
            else:
                self._run_git("tag", tag_name)

            return GitResult(
                success=True,
                message=f"Created tag '{tag_name}'",
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to create tag: {e.stderr}",
                output=e.stderr,
            )

    def get_tags(self) -> list[str]:
        """Get all tags in the repository.

        Returns:
            List of tag names, sorted by version
        """
        try:
            result = self._run_git("tag", "--sort=-v:refname")
            tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
            return tags
        except subprocess.CalledProcessError:
            return []

    def get_latest_tag(self) -> str | None:
        """Get the most recent tag.

        Returns:
            Latest tag name or None
        """
        tags = self.get_tags()
        return tags[0] if tags else None

    def get_next_version(self, bump: str = "patch") -> str:
        """Calculate next semantic version based on existing tags.

        Args:
            bump: Type of version bump - 'major', 'minor', or 'patch'

        Returns:
            Next version string (e.g., 'v1.0.1')
        """
        import re

        latest = self.get_latest_tag()

        if not latest:
            return "v0.1.0"

        # Parse version (v1.2.3 or 1.2.3)
        match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", latest)
        if not match:
            return "v0.1.0"

        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

        if bump == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1

        return f"v{major}.{minor}.{patch}"

    def push_tag(self, tag_name: str, remote: str = "origin") -> GitResult:
        """Push a tag to remote.

        Args:
            tag_name: Tag to push
            remote: Remote name

        Returns:
            GitResult with success status
        """
        try:
            self._run_git("push", remote, tag_name)
            return GitResult(
                success=True,
                message=f"Pushed tag '{tag_name}' to {remote}",
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to push tag: {e.stderr}",
                output=e.stderr,
            )

    def create_release_tag(
        self,
        message: str,
        bump: str = "patch",
        push: bool = False,
        remote: str = "origin",
    ) -> GitResult:
        """Create a release tag with auto-incremented version.

        Args:
            message: Tag/release message
            bump: Version bump type ('major', 'minor', 'patch')
            push: Whether to push the tag to remote
            remote: Remote to push to

        Returns:
            GitResult with the created tag name
        """
        version = self.get_next_version(bump)

        result = self.tag(version, message)
        if not result.success:
            return result

        if push:
            push_result = self.push_tag(version, remote)
            if not push_result.success:
                return GitResult(
                    success=True,
                    message=f"Created tag '{version}' (push failed: {push_result.message})",
                    output=version,
                )
            return GitResult(
                success=True,
                message=f"Created and pushed tag '{version}'",
                output=version,
            )

        return GitResult(
            success=True,
            message=f"Created tag '{version}'",
            output=version,
        )

    def create_pull_request(
        self,
        title: str,
        body: str = "",
        base: str = "main",
        head: str | None = None,
        draft: bool = False,
    ) -> GitResult:
        """Create a GitHub Pull Request using gh CLI.

        Requires: gh CLI installed and authenticated (gh auth login)

        Args:
            title: PR title
            body: PR description
            base: Target branch (default: main)
            head: Source branch (default: current branch)
            draft: Create as draft PR

        Returns:
            GitResult with PR URL
        """
        head = head or self.get_current_branch()
        if not head:
            return GitResult(
                success=False,
                message="No branch to create PR from",
            )

        try:
            # Check if gh is available
            check = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if check.returncode != 0:
                return GitResult(
                    success=False,
                    message="gh CLI not installed. Install from https://cli.github.com",
                )

            # Build gh pr create command
            args = [
                "gh", "pr", "create",
                "--title", title,
                "--base", base,
                "--head", head,
            ]

            if body:
                args.extend(["--body", body])

            if draft:
                args.append("--draft")

            result = subprocess.run(
                args,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                # Check for common errors
                if "gh auth login" in result.stderr:
                    return GitResult(
                        success=False,
                        message="Not authenticated. Run 'gh auth login' first.",
                        output=result.stderr,
                    )
                if "already exists" in result.stderr.lower():
                    return GitResult(
                        success=False,
                        message=f"PR already exists for branch '{head}'",
                        output=result.stderr,
                    )
                return GitResult(
                    success=False,
                    message=f"Failed to create PR: {result.stderr}",
                    output=result.stderr,
                )

            # Extract PR URL from output
            pr_url = result.stdout.strip()

            return GitResult(
                success=True,
                message=f"Created PR: {pr_url}",
                output=pr_url,
            )

        except FileNotFoundError:
            return GitResult(
                success=False,
                message="gh CLI not found. Install from https://cli.github.com",
            )
        except Exception as e:
            return GitResult(
                success=False,
                message=f"Error creating PR: {e}",
            )

    def has_remote(self, name: str = "origin") -> bool:
        """Check if a remote exists."""
        try:
            result = self._run_git("remote", "get-url", name, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def add_remote(self, url: str, name: str = "origin") -> GitResult:
        """Add a remote repository.

        Args:
            url: Remote repository URL
            name: Remote name

        Returns:
            GitResult with success status
        """
        try:
            if self.has_remote(name):
                # Update existing remote
                self._run_git("remote", "set-url", name, url)
                return GitResult(
                    success=True,
                    message=f"Updated remote '{name}' to {url}",
                )
            else:
                self._run_git("remote", "add", name, url)
                return GitResult(
                    success=True,
                    message=f"Added remote '{name}': {url}",
                )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to add remote: {e.stderr}",
                output=e.stderr,
            )

    def fetch(self, remote: str = "origin") -> GitResult:
        """Fetch from remote repository.

        Args:
            remote: Remote name

        Returns:
            GitResult with success status
        """
        try:
            self._run_git("fetch", remote)
            return GitResult(success=True, message=f"Fetched from {remote}")
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to fetch: {e.stderr}",
                output=e.stderr,
            )

    def is_behind_remote(self, remote: str = "origin", branch: str | None = None) -> tuple[bool, int]:
        """Check if local branch is behind remote.

        Args:
            remote: Remote name
            branch: Branch to check (defaults to current)

        Returns:
            Tuple of (is_behind, commits_behind)
        """
        try:
            branch = branch or self.get_current_branch()
            if not branch:
                return False, 0

            # Fetch first to get latest remote state
            self.fetch(remote)

            # Count commits we're behind
            result = self._run_git(
                "rev-list",
                "--count",
                f"{branch}..{remote}/{branch}",
                check=False,
            )

            if result.returncode != 0:
                # Remote branch doesn't exist yet
                return False, 0

            behind = int(result.stdout.strip())
            return behind > 0, behind

        except Exception:
            return False, 0

    def pull_rebase(self, remote: str = "origin", branch: str | None = None) -> GitResult:
        """Pull from remote with rebase.

        Args:
            remote: Remote name
            branch: Branch to pull (defaults to current)

        Returns:
            GitResult with success status
        """
        try:
            branch = branch or self.get_current_branch()
            result = self._run_git("pull", "--rebase", remote, branch, check=False)

            if result.returncode != 0:
                # Check if it's a conflict
                if "CONFLICT" in result.stdout or "CONFLICT" in result.stderr:
                    # Abort the rebase
                    self._run_git("rebase", "--abort", check=False)
                    return GitResult(
                        success=False,
                        message="Rebase conflict - manual resolution required",
                        output=result.stderr,
                    )
                return GitResult(
                    success=False,
                    message=f"Pull failed: {result.stderr}",
                    output=result.stderr,
                )

            return GitResult(
                success=True,
                message=f"Pulled and rebased from {remote}/{branch}",
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to pull: {e.stderr}",
                output=e.stderr,
            )

    def push(self, remote: str = "origin", branch: str | None = None, set_upstream: bool = True) -> GitResult:
        """Push to remote repository.

        Args:
            remote: Remote name
            branch: Branch to push (defaults to current)
            set_upstream: Whether to set upstream tracking

        Returns:
            GitResult with success status
        """
        try:
            branch = branch or self.get_current_branch()
            if not branch:
                return GitResult(
                    success=False,
                    message="No branch to push",
                )

            args = ["push"]
            if set_upstream:
                args.extend(["-u", remote, branch])
            else:
                args.extend([remote, branch])

            result = self._run_git(*args)

            return GitResult(
                success=True,
                message=f"Pushed to {remote}/{branch}",
                output=result.stdout,
            )
        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                message=f"Failed to push: {e.stderr}",
                output=e.stderr,
            )

    def sync_and_push(self, remote: str = "origin", branch: str | None = None) -> GitResult:
        """Sync with remote (fetch, rebase if needed) then push.

        Handles the case where remote has new commits.

        Args:
            remote: Remote name
            branch: Branch to sync (defaults to current)

        Returns:
            GitResult with success status
        """
        branch = branch or self.get_current_branch()
        if not branch:
            return GitResult(success=False, message="No branch to push")

        # Check if remote exists
        if not self.has_remote(remote):
            return GitResult(
                success=False,
                message=f"Remote '{remote}' not configured",
            )

        # Fetch latest from remote
        fetch_result = self.fetch(remote)
        if not fetch_result.success:
            # Remote might not have this branch yet, try pushing directly
            return self.push(remote, branch)

        # Check if we're behind
        is_behind, commits_behind = self.is_behind_remote(remote, branch)

        if is_behind:
            # Pull with rebase first
            pull_result = self.pull_rebase(remote, branch)
            if not pull_result.success:
                return pull_result

        # Now push
        return self.push(remote, branch)


def init_project_repo(
    project_dir: Path,
    run_id: str,
    feature_description: str,
) -> GitResult:
    """Initialize git repo for a new generated project.

    Creates repo with initial commit on main branch.

    Args:
        project_dir: Path to the generated project
        run_id: Pipeline run ID
        feature_description: Short description of the feature

    Returns:
        GitResult with success status
    """
    git = GitManager(project_dir)

    if git.is_repo():
        return GitResult(
            success=True,
            message="Repository already initialized",
        )

    # Initialize repo on main branch
    result = git.init_repo(branch_name="main")
    if not result.success:
        return result

    # Initial commit on main
    summary = feature_description[:50] if feature_description else "Initial project"
    return git.commit_stage("init", run_id, summary)


def commit_iteration(
    project_dir: Path,
    run_id: str,
    improvement_description: str,
    base_run_id: str | None = None,
) -> GitResult:
    """Commit an iteration on a new branch.

    Creates a new branch for the iteration and commits changes.

    Args:
        project_dir: Path to the generated project
        run_id: Pipeline run ID for this iteration
        improvement_description: Description of improvements made
        base_run_id: Run ID of the base project (for branch naming)

    Returns:
        GitResult with success status
    """
    git = GitManager(project_dir)

    if not git.is_repo():
        return GitResult(
            success=False,
            message="Not a git repository - cannot commit iteration",
        )

    # Create iteration branch from current HEAD
    branch_name = f"iteration/{run_id}"
    result = git.create_branch(branch_name, checkout=True)
    if not result.success:
        return result

    # Commit the iteration changes
    summary = improvement_description[:50] if improvement_description else "Project iteration"
    return git.commit_stage("iteration", run_id, summary)
