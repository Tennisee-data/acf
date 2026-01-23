"""Rollback Tools for Production Deployments.

Provides rollback capabilities for various deployment strategies:
- Docker image rollback (retag previous version)
- Kubernetes rollback (kubectl rollout undo)
- Git-based rollback (checkout previous tag)
- Custom rollback scripts
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    success: bool
    target: str
    from_version: str
    to_version: str
    message: str
    duration_seconds: float = 0.0
    logs: str = ""


@dataclass
class DeploymentRecord:
    """Record of a deployment for rollback tracking."""

    version: str
    timestamp: str
    target: str
    success: bool
    image: str = ""
    commit_sha: str = ""
    metadata: dict = field(default_factory=dict)


class RollbackManager:
    """Manages rollbacks for production deployments.

    Tracks deployment history and provides rollback capabilities
    for various deployment strategies.
    """

    def __init__(self, history_file: Path | None = None):
        """Initialize rollback manager.

        Args:
            history_file: Path to deployment history JSON file
        """
        self.history_file = history_file
        self._history: list[DeploymentRecord] = []

        if history_file and history_file.exists():
            self._load_history()

    def _load_history(self) -> None:
        """Load deployment history from file."""
        if self.history_file and self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self._history = [
                    DeploymentRecord(**record) for record in data
                ]
            except (json.JSONDecodeError, TypeError):
                self._history = []

    def _save_history(self) -> None:
        """Save deployment history to file."""
        if self.history_file:
            data = [
                {
                    "version": r.version,
                    "timestamp": r.timestamp,
                    "target": r.target,
                    "success": r.success,
                    "image": r.image,
                    "commit_sha": r.commit_sha,
                    "metadata": r.metadata,
                }
                for r in self._history
            ]
            self.history_file.write_text(json.dumps(data, indent=2))

    def record_deployment(
        self,
        version: str,
        target: str,
        success: bool,
        image: str = "",
        commit_sha: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Record a new deployment.

        Args:
            version: Version that was deployed
            target: Deployment target (docker, k8s, etc.)
            success: Whether deployment succeeded
            image: Docker image (if applicable)
            commit_sha: Git commit SHA
            metadata: Additional metadata
        """
        record = DeploymentRecord(
            version=version,
            timestamp=datetime.now().isoformat(),
            target=target,
            success=success,
            image=image,
            commit_sha=commit_sha,
            metadata=metadata or {},
        )
        self._history.append(record)
        self._save_history()

    def get_previous_version(self, current_version: str | None = None) -> DeploymentRecord | None:
        """Get the previous successful deployment.

        Args:
            current_version: Current version (to find what came before)

        Returns:
            Previous deployment record or None
        """
        successful = [r for r in self._history if r.success]

        if not successful:
            return None

        if current_version:
            # Find the version before the current one
            for i, record in enumerate(successful):
                if record.version == current_version and i > 0:
                    return successful[i - 1]

        # Return second-to-last successful deployment
        if len(successful) >= 2:
            return successful[-2]

        return None

    def get_version_by_tag(self, version: str) -> DeploymentRecord | None:
        """Get deployment record by version tag.

        Args:
            version: Version to look up

        Returns:
            Deployment record or None
        """
        for record in reversed(self._history):
            if record.version == version:
                return record
        return None

    def rollback(
        self,
        project_dir: Path,
        target_version: str | None = None,
        strategy: str = "docker",
        config: dict | None = None,
    ) -> RollbackResult:
        """Perform rollback to previous or specified version.

        Args:
            project_dir: Project directory
            target_version: Version to roll back to (None = previous)
            strategy: Rollback strategy
            config: Strategy-specific configuration

        Returns:
            RollbackResult with outcome
        """
        config = config or {}
        start_time = datetime.now()

        # Determine target version
        if target_version:
            target_record = self.get_version_by_tag(target_version)
        else:
            target_record = self.get_previous_version()

        if not target_record:
            return RollbackResult(
                success=False,
                target=strategy,
                from_version="current",
                to_version=target_version or "unknown",
                message="No previous version found for rollback",
            )

        # Get current version
        current = self._history[-1] if self._history else None
        current_version = current.version if current else "unknown"

        try:
            if strategy == "docker":
                result = self._rollback_docker(target_record, config)
            elif strategy == "k8s":
                result = self._rollback_kubernetes(target_record, config)
            elif strategy == "git":
                result = self._rollback_git(project_dir, target_record)
            elif strategy == "fly":
                result = self._rollback_fly(target_record, config)
            elif strategy == "custom":
                result = self._rollback_custom(project_dir, target_record, config)
            else:
                result = RollbackResult(
                    success=False,
                    target=strategy,
                    from_version=current_version,
                    to_version=target_record.version,
                    message=f"Unknown rollback strategy: {strategy}",
                )

            result.from_version = current_version
            result.to_version = target_record.version
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Record rollback as a deployment
            if result.success:
                self.record_deployment(
                    version=f"{target_record.version}-rollback",
                    target=strategy,
                    success=True,
                    metadata={"rollback_from": current_version},
                )

            return result

        except Exception as e:
            return RollbackResult(
                success=False,
                target=strategy,
                from_version=current_version,
                to_version=target_record.version,
                message=f"Rollback error: {str(e)}",
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _rollback_docker(
        self,
        target: DeploymentRecord,
        config: dict,
    ) -> RollbackResult:
        """Rollback Docker deployment by retagging image.

        Args:
            target: Target deployment record
            config: Configuration with registry info

        Returns:
            RollbackResult
        """
        registry = config.get("registry", "")
        image_name = config.get("image_name", "")

        if not target.image:
            # Reconstruct image name
            if registry and image_name:
                target_image = f"{registry}/{image_name}:{target.version}"
            else:
                return RollbackResult(
                    success=False,
                    target="docker",
                    from_version="",
                    to_version=target.version,
                    message="No image information available for rollback",
                )
        else:
            target_image = target.image

        logs = []

        # Pull the previous version
        pull_cmd = ["docker", "pull", target_image]
        result = subprocess.run(pull_cmd, capture_output=True, text=True)
        logs.append(f"Pull: {result.stdout}\n{result.stderr}")

        if result.returncode != 0:
            return RollbackResult(
                success=False,
                target="docker",
                from_version="",
                to_version=target.version,
                message=f"Failed to pull {target_image}",
                logs="\n".join(logs),
            )

        # Retag as latest
        if registry and image_name:
            latest_image = f"{registry}/{image_name}:latest"
            tag_cmd = ["docker", "tag", target_image, latest_image]
            result = subprocess.run(tag_cmd, capture_output=True, text=True)
            logs.append(f"Tag: {result.stdout}\n{result.stderr}")

            # Push latest tag
            push_cmd = ["docker", "push", latest_image]
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            logs.append(f"Push: {result.stdout}\n{result.stderr}")

            if result.returncode != 0:
                return RollbackResult(
                    success=False,
                    target="docker",
                    from_version="",
                    to_version=target.version,
                    message=f"Failed to push {latest_image}",
                    logs="\n".join(logs),
                )

        return RollbackResult(
            success=True,
            target="docker",
            from_version="",
            to_version=target.version,
            message=f"Rolled back to {target_image}",
            logs="\n".join(logs),
        )

    def _rollback_kubernetes(
        self,
        target: DeploymentRecord,
        config: dict,
    ) -> RollbackResult:
        """Rollback Kubernetes deployment.

        Args:
            target: Target deployment record
            config: Configuration with k8s settings

        Returns:
            RollbackResult
        """
        namespace = config.get("namespace", "default")
        deployment = config.get("deployment", "")
        context = config.get("context", "")

        if not deployment:
            return RollbackResult(
                success=False,
                target="k8s",
                from_version="",
                to_version=target.version,
                message="No deployment name configured",
            )

        # Use kubectl rollout undo
        cmd = ["kubectl", "rollout", "undo", f"deployment/{deployment}"]
        cmd.extend(["-n", namespace])

        if context:
            cmd.extend(["--context", context])

        # If we have a specific revision, use it
        if target.metadata.get("k8s_revision"):
            cmd.extend(["--to-revision", str(target.metadata["k8s_revision"])])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Wait for rollout
            wait_cmd = ["kubectl", "rollout", "status", f"deployment/{deployment}"]
            wait_cmd.extend(["-n", namespace])
            if context:
                wait_cmd.extend(["--context", context])

            wait_result = subprocess.run(
                wait_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            return RollbackResult(
                success=wait_result.returncode == 0,
                target="k8s",
                from_version="",
                to_version=target.version,
                message=f"Rolled back {deployment} in {namespace}",
                logs=f"{result.stdout}\n{wait_result.stdout}",
            )
        else:
            return RollbackResult(
                success=False,
                target="k8s",
                from_version="",
                to_version=target.version,
                message=f"Kubectl rollback failed: {result.stderr[:200]}",
                logs=result.stderr,
            )

    def _rollback_git(
        self,
        project_dir: Path,
        target: DeploymentRecord,
    ) -> RollbackResult:
        """Rollback by checking out previous git tag/commit.

        Args:
            project_dir: Project directory
            target: Target deployment record

        Returns:
            RollbackResult
        """
        # Determine what to checkout
        checkout_ref = target.commit_sha or f"v{target.version}" or target.version

        # Stash any changes
        stash_cmd = ["git", "stash"]
        subprocess.run(stash_cmd, cwd=project_dir, capture_output=True)

        # Checkout the target
        checkout_cmd = ["git", "checkout", checkout_ref]
        result = subprocess.run(
            checkout_cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return RollbackResult(
                success=True,
                target="git",
                from_version="",
                to_version=target.version,
                message=f"Checked out {checkout_ref}",
                logs=result.stdout,
            )
        else:
            return RollbackResult(
                success=False,
                target="git",
                from_version="",
                to_version=target.version,
                message=f"Git checkout failed: {result.stderr[:200]}",
                logs=result.stderr,
            )

    def _rollback_fly(
        self,
        target: DeploymentRecord,
        config: dict,
    ) -> RollbackResult:
        """Rollback Fly.io deployment.

        Args:
            target: Target deployment record
            config: Configuration with app name

        Returns:
            RollbackResult
        """
        app_name = config.get("app_name", "")

        if not app_name:
            return RollbackResult(
                success=False,
                target="fly",
                from_version="",
                to_version=target.version,
                message="No Fly app name configured",
            )

        # Use flyctl releases to rollback
        cmd = ["flyctl", "releases", "--app", app_name, "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return RollbackResult(
                success=False,
                target="fly",
                from_version="",
                to_version=target.version,
                message="Failed to get Fly releases",
                logs=result.stderr,
            )

        # Find the release to rollback to
        try:
            releases = json.loads(result.stdout)
            target_release = None

            for release in releases:
                if release.get("ImageRef", "").endswith(f":{target.version}"):
                    target_release = release.get("Version")
                    break

            if not target_release:
                # Just rollback to previous
                rollback_cmd = ["flyctl", "releases", "rollback", "--app", app_name]
            else:
                rollback_cmd = ["flyctl", "releases", "rollback", str(target_release), "--app", app_name]

            result = subprocess.run(rollback_cmd, capture_output=True, text=True)

            return RollbackResult(
                success=result.returncode == 0,
                target="fly",
                from_version="",
                to_version=target.version,
                message="Rolled back on Fly.io",
                logs=f"{result.stdout}\n{result.stderr}",
            )

        except json.JSONDecodeError:
            return RollbackResult(
                success=False,
                target="fly",
                from_version="",
                to_version=target.version,
                message="Failed to parse Fly releases",
            )

    def _rollback_custom(
        self,
        project_dir: Path,
        target: DeploymentRecord,
        config: dict,
    ) -> RollbackResult:
        """Run custom rollback script.

        Args:
            project_dir: Project directory
            target: Target deployment record
            config: Configuration with script path

        Returns:
            RollbackResult
        """
        script = config.get("rollback_script", "rollback.sh")
        script_path = project_dir / script

        if not script_path.exists():
            return RollbackResult(
                success=False,
                target="custom",
                from_version="",
                to_version=target.version,
                message=f"Rollback script not found: {script}",
            )

        import os
        env = os.environ.copy()
        env["ROLLBACK_VERSION"] = target.version
        env["ROLLBACK_IMAGE"] = target.image
        env["ROLLBACK_SHA"] = target.commit_sha

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return RollbackResult(
            success=result.returncode == 0,
            target="custom",
            from_version="",
            to_version=target.version,
            message="Custom rollback " + ("completed" if result.returncode == 0 else "failed"),
            logs=f"{result.stdout}\n{result.stderr}",
        )

    def get_history(self, limit: int = 10) -> list[DeploymentRecord]:
        """Get recent deployment history.

        Args:
            limit: Maximum records to return

        Returns:
            List of deployment records (most recent first)
        """
        return list(reversed(self._history[-limit:]))


def generate_rollback_script(strategy: str = "docker", config: dict | None = None) -> str:
    """Generate a rollback.sh script.

    Args:
        strategy: Rollback strategy
        config: Strategy-specific configuration

    Returns:
        Shell script content
    """
    config = config or {}

    if strategy == "docker":
        registry = config.get("registry", "your-registry.com")
        image = config.get("image", "your-app")
        return f"""#!/bin/bash
# Auto-generated rollback script
# Strategy: docker

set -e

VERSION="${{1:?Usage: $0 <version>}}"
REGISTRY="{registry}"
IMAGE="{image}"

echo "Rolling back to version: $VERSION"

# Pull the previous version
docker pull "$REGISTRY/$IMAGE:$VERSION"

# Retag as latest
docker tag "$REGISTRY/$IMAGE:$VERSION" "$REGISTRY/$IMAGE:latest"

# Push latest
docker push "$REGISTRY/$IMAGE:latest"

echo "Rollback complete: $REGISTRY/$IMAGE:$VERSION is now latest"
"""

    elif strategy == "k8s":
        namespace = config.get("namespace", "default")
        deployment = config.get("deployment", "your-deployment")
        return f"""#!/bin/bash
# Auto-generated rollback script
# Strategy: kubernetes

set -e

NAMESPACE="{namespace}"
DEPLOYMENT="{deployment}"

echo "Rolling back $DEPLOYMENT in $NAMESPACE..."

# Rollback to previous revision
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

echo "Rollback complete"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT
"""

    elif strategy == "fly":
        app_name = config.get("app_name", "your-app")
        return f"""#!/bin/bash
# Auto-generated rollback script
# Strategy: fly.io

set -e

APP="{app_name}"

echo "Rolling back $APP on Fly.io..."

# List recent releases
flyctl releases --app $APP

# Rollback to previous
flyctl releases rollback --app $APP

echo "Rollback complete"
flyctl status --app $APP
"""

    else:
        return """#!/bin/bash
# Auto-generated rollback script
# Strategy: custom

set -e

VERSION="${1:-previous}"

echo "Rolling back to: $VERSION"

# Add your rollback commands here
# Example:
# git checkout tags/v$VERSION
# ./deploy.sh

echo "Rollback complete"
"""
