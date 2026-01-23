"""Production Deployment Tools.

Configurable deployment scripts for various targets:
- Docker registry push
- Cloud platforms (Render, Fly.io, Railway)
- Kubernetes
- SSH/rsync to server
- Custom scripts
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DeployResult:
    """Result of a deployment operation."""

    success: bool
    target: str
    message: str
    url: str | None = None
    version: str | None = None
    duration_seconds: float = 0.0
    logs: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DeployConfig:
    """Deployment configuration."""

    strategy: str = "docker-push"  # docker-push, render, fly, k8s, ssh, custom
    registry: str = ""  # Docker registry URL
    image_name: str = ""  # Image name (without tag)

    # Cloud platform settings
    render_service_id: str = ""
    fly_app_name: str = ""
    railway_project_id: str = ""

    # Kubernetes settings
    k8s_namespace: str = "default"
    k8s_deployment: str = ""
    k8s_context: str = ""

    # SSH settings
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""
    ssh_deploy_path: str = "/app"

    # Custom script
    custom_script: str = ""

    # General settings
    health_check_url: str = ""
    health_check_timeout: int = 120
    rollback_on_failure: bool = True


class DeploymentManager:
    """Manages production deployments with multiple strategies."""

    def __init__(self, config: DeployConfig):
        """Initialize deployment manager.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self._deployment_history: list[dict] = []

    def deploy(
        self,
        project_dir: Path,
        version: str,
        env: dict[str, str] | None = None,
    ) -> DeployResult:
        """Deploy project to production.

        Args:
            project_dir: Path to project directory
            version: Version/tag to deploy
            env: Environment variables for deployment

        Returns:
            DeployResult with deployment outcome
        """
        strategy = self.config.strategy
        start_time = datetime.now()

        try:
            if strategy == "docker-push":
                result = self._deploy_docker_push(project_dir, version)
            elif strategy == "render":
                result = self._deploy_render(project_dir, version, env)
            elif strategy == "fly":
                result = self._deploy_fly(project_dir, version, env)
            elif strategy == "k8s":
                result = self._deploy_kubernetes(project_dir, version)
            elif strategy == "ssh":
                result = self._deploy_ssh(project_dir, version, env)
            elif strategy == "custom":
                result = self._deploy_custom(project_dir, version, env)
            else:
                result = DeployResult(
                    success=False,
                    target=strategy,
                    message=f"Unknown deployment strategy: {strategy}",
                )

            # Calculate duration
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Record deployment
            self._record_deployment(result, version)

            # Health check if URL configured
            if result.success and self.config.health_check_url:
                health_ok = self._health_check(self.config.health_check_url)
                if not health_ok:
                    result.success = False
                    result.message += " (Health check failed)"
                    if self.config.rollback_on_failure:
                        self._trigger_rollback(project_dir)

            return result

        except Exception as e:
            return DeployResult(
                success=False,
                target=strategy,
                message=f"Deployment error: {str(e)}",
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _deploy_docker_push(self, project_dir: Path, version: str) -> DeployResult:
        """Push Docker image to registry.

        Args:
            project_dir: Project directory with Dockerfile
            version: Version tag

        Returns:
            DeployResult
        """
        registry = self.config.registry
        image_name = self.config.image_name or project_dir.name

        if not registry:
            return DeployResult(
                success=False,
                target="docker-push",
                message="No registry configured. Set deploy.registry in config.",
            )

        full_image = f"{registry}/{image_name}:{version}"
        latest_image = f"{registry}/{image_name}:latest"

        logs = []

        # Build image
        build_cmd = ["docker", "build", "-t", full_image, "-t", latest_image, "."]
        result = subprocess.run(
            build_cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        logs.append(f"Build: {result.stdout}\n{result.stderr}")

        if result.returncode != 0:
            return DeployResult(
                success=False,
                target="docker-push",
                message=f"Build failed: {result.stderr[:200]}",
                logs="\n".join(logs),
            )

        # Push versioned tag
        push_cmd = ["docker", "push", full_image]
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        logs.append(f"Push {version}: {result.stdout}\n{result.stderr}")

        if result.returncode != 0:
            return DeployResult(
                success=False,
                target="docker-push",
                message=f"Push failed: {result.stderr[:200]}",
                logs="\n".join(logs),
            )

        # Push latest tag
        push_latest_cmd = ["docker", "push", latest_image]
        result = subprocess.run(push_latest_cmd, capture_output=True, text=True)
        logs.append(f"Push latest: {result.stdout}\n{result.stderr}")

        return DeployResult(
            success=True,
            target="docker-push",
            message=f"Pushed {full_image}",
            version=version,
            logs="\n".join(logs),
            metadata={"image": full_image, "latest": latest_image},
        )

    def _deploy_render(
        self,
        project_dir: Path,
        version: str,
        env: dict[str, str] | None,
    ) -> DeployResult:
        """Deploy to Render.com.

        Uses Render API or render.yaml blueprint.

        Args:
            project_dir: Project directory
            version: Version tag
            env: Environment variables

        Returns:
            DeployResult
        """
        service_id = self.config.render_service_id
        render_api_key = os.environ.get("RENDER_API_KEY")

        if not service_id:
            return DeployResult(
                success=False,
                target="render",
                message="No Render service ID configured. Set deploy.render_service_id",
            )

        if not render_api_key:
            return DeployResult(
                success=False,
                target="render",
                message="RENDER_API_KEY environment variable not set",
            )

        # Trigger deploy via Render API
        import requests

        try:
            response = requests.post(
                f"https://api.render.com/v1/services/{service_id}/deploys",
                headers={
                    "Authorization": f"Bearer {render_api_key}",
                    "Content-Type": "application/json",
                },
                json={"clearCache": False},
                timeout=30,
            )

            if response.status_code in (200, 201):
                deploy_data = response.json()
                deploy_id = deploy_data.get("id", "unknown")
                return DeployResult(
                    success=True,
                    target="render",
                    message=f"Deploy triggered: {deploy_id}",
                    version=version,
                    url=f"https://dashboard.render.com/web/{service_id}",
                    metadata={"deploy_id": deploy_id},
                )
            else:
                return DeployResult(
                    success=False,
                    target="render",
                    message=f"Render API error: {response.status_code} - {response.text[:200]}",
                )

        except requests.RequestException as e:
            return DeployResult(
                success=False,
                target="render",
                message=f"Render API request failed: {str(e)}",
            )

    def _deploy_fly(
        self,
        project_dir: Path,
        version: str,
        env: dict[str, str] | None,
    ) -> DeployResult:
        """Deploy to Fly.io.

        Uses flyctl CLI.

        Args:
            project_dir: Project directory with fly.toml
            version: Version tag
            env: Environment variables

        Returns:
            DeployResult
        """
        app_name = self.config.fly_app_name

        # Check for fly.toml
        fly_toml = project_dir / "fly.toml"
        if not fly_toml.exists() and not app_name:
            return DeployResult(
                success=False,
                target="fly",
                message="No fly.toml found and no app name configured",
            )

        # Build flyctl command
        cmd = ["flyctl", "deploy", "--remote-only"]
        if app_name:
            cmd.extend(["--app", app_name])

        # Add image tag
        cmd.extend(["--image-label", version])

        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )

        if result.returncode == 0:
            # Extract URL from output
            url = None
            for line in result.stdout.split("\n"):
                if "https://" in line and ".fly.dev" in line:
                    url = line.strip()
                    break

            return DeployResult(
                success=True,
                target="fly",
                message="Deployed to Fly.io",
                version=version,
                url=url,
                logs=result.stdout,
            )
        else:
            return DeployResult(
                success=False,
                target="fly",
                message=f"Fly deploy failed: {result.stderr[:200]}",
                logs=f"{result.stdout}\n{result.stderr}",
            )

    def _deploy_kubernetes(self, project_dir: Path, version: str) -> DeployResult:
        """Deploy to Kubernetes cluster.

        Updates deployment image tag.

        Args:
            project_dir: Project directory
            version: Version tag

        Returns:
            DeployResult
        """
        namespace = self.config.k8s_namespace
        deployment = self.config.k8s_deployment
        context = self.config.k8s_context
        registry = self.config.registry
        image_name = self.config.image_name or project_dir.name

        if not deployment:
            return DeployResult(
                success=False,
                target="k8s",
                message="No Kubernetes deployment configured. Set deploy.k8s_deployment",
            )

        full_image = f"{registry}/{image_name}:{version}" if registry else f"{image_name}:{version}"

        # Build kubectl command
        cmd = ["kubectl", "set", "image", f"deployment/{deployment}"]
        cmd.append(f"{deployment}={full_image}")
        cmd.extend(["-n", namespace])

        if context:
            cmd.extend(["--context", context])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Wait for rollout
            rollout_cmd = ["kubectl", "rollout", "status", f"deployment/{deployment}", "-n", namespace]
            if context:
                rollout_cmd.extend(["--context", context])

            rollout_result = subprocess.run(
                rollout_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            return DeployResult(
                success=rollout_result.returncode == 0,
                target="k8s",
                message=f"Deployed to {namespace}/{deployment}",
                version=version,
                logs=f"{result.stdout}\n{rollout_result.stdout}",
                metadata={"namespace": namespace, "deployment": deployment},
            )
        else:
            return DeployResult(
                success=False,
                target="k8s",
                message=f"Kubectl failed: {result.stderr[:200]}",
                logs=result.stderr,
            )

    def _deploy_ssh(
        self,
        project_dir: Path,
        version: str,
        env: dict[str, str] | None,
    ) -> DeployResult:
        """Deploy via SSH/rsync to server.

        Args:
            project_dir: Project directory
            version: Version tag
            env: Environment variables

        Returns:
            DeployResult
        """
        host = self.config.ssh_host
        user = self.config.ssh_user
        key_path = self.config.ssh_key_path
        deploy_path = self.config.ssh_deploy_path

        if not host or not user:
            return DeployResult(
                success=False,
                target="ssh",
                message="SSH host and user required. Set deploy.ssh_host and deploy.ssh_user",
            )

        logs = []

        # Build rsync command
        rsync_cmd = [
            "rsync", "-avz", "--delete",
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", ".env",
            "--exclude", "venv",
        ]

        if key_path:
            rsync_cmd.extend(["-e", f"ssh -i {key_path}"])

        rsync_cmd.extend([
            f"{project_dir}/",
            f"{user}@{host}:{deploy_path}",
        ])

        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        logs.append(f"Rsync: {result.stdout}\n{result.stderr}")

        if result.returncode != 0:
            return DeployResult(
                success=False,
                target="ssh",
                message=f"Rsync failed: {result.stderr[:200]}",
                logs="\n".join(logs),
            )

        # Run deploy script on remote
        ssh_cmd = ["ssh"]
        if key_path:
            ssh_cmd.extend(["-i", key_path])
        ssh_cmd.append(f"{user}@{host}")

        # Remote commands: install deps and restart
        remote_script = f"""
cd {deploy_path}
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
if [ -f run.sh ]; then
    chmod +x run.sh
fi
# Try to restart with systemd or docker
if systemctl is-active --quiet app 2>/dev/null; then
    sudo systemctl restart app
elif docker ps -q -f name=app 2>/dev/null | grep -q .; then
    docker-compose down && docker-compose up -d
fi
echo "Deploy complete: {version}"
"""
        ssh_cmd.append(remote_script)

        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
        logs.append(f"SSH: {result.stdout}\n{result.stderr}")

        return DeployResult(
            success=result.returncode == 0,
            target="ssh",
            message=f"Deployed to {host}:{deploy_path}",
            version=version,
            logs="\n".join(logs),
            metadata={"host": host, "path": deploy_path},
        )

    def _deploy_custom(
        self,
        project_dir: Path,
        version: str,
        env: dict[str, str] | None,
    ) -> DeployResult:
        """Run custom deployment script.

        Args:
            project_dir: Project directory
            version: Version tag
            env: Environment variables

        Returns:
            DeployResult
        """
        script = self.config.custom_script

        if not script:
            return DeployResult(
                success=False,
                target="custom",
                message="No custom script configured. Set deploy.custom_script",
            )

        # Check if script exists
        script_path = project_dir / script
        if not script_path.exists():
            # Try as absolute path
            script_path = Path(script)

        if not script_path.exists():
            return DeployResult(
                success=False,
                target="custom",
                message=f"Custom script not found: {script}",
            )

        # Prepare environment
        run_env = os.environ.copy()
        run_env["DEPLOY_VERSION"] = version
        run_env["PROJECT_DIR"] = str(project_dir)
        if env:
            run_env.update(env)

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=project_dir,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=600,
        )

        return DeployResult(
            success=result.returncode == 0,
            target="custom",
            message=f"Custom script {'completed' if result.returncode == 0 else 'failed'}",
            version=version,
            logs=f"{result.stdout}\n{result.stderr}",
        )

    def _health_check(self, url: str, timeout: int = 120) -> bool:
        """Check if deployed service is healthy.

        Args:
            url: Health check URL
            timeout: Max wait time in seconds

        Returns:
            True if healthy, False otherwise
        """
        import time
        import requests

        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)

        return False

    def _record_deployment(self, result: DeployResult, version: str) -> None:
        """Record deployment in history.

        Args:
            result: Deployment result
            version: Version deployed
        """
        self._deployment_history.append({
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "target": result.target,
            "success": result.success,
            "message": result.message,
            "duration_seconds": result.duration_seconds,
        })

    def _trigger_rollback(self, project_dir: Path) -> None:
        """Trigger rollback to previous version.

        Args:
            project_dir: Project directory
        """
        # Get previous successful deployment
        successful = [d for d in self._deployment_history if d["success"]]
        if len(successful) < 2:
            return  # No previous version to roll back to

        previous = successful[-2]
        # TODO: Implement actual rollback logic
        # This would depend on the deployment strategy

    def get_deployment_history(self) -> list[dict]:
        """Get deployment history.

        Returns:
            List of deployment records
        """
        return self._deployment_history.copy()


def generate_deploy_script(
    project_dir: Path,
    strategy: str = "docker-push",
    config: dict | None = None,
) -> str:
    """Generate a deploy.sh script for manual deployment.

    Args:
        project_dir: Project directory
        strategy: Deployment strategy
        config: Additional configuration

    Returns:
        Shell script content
    """
    config = config or {}
    project_name = project_dir.name

    if strategy == "docker-push":
        registry = config.get("registry", "your-registry.com")
        return f"""#!/bin/bash
# Auto-generated deployment script
# Strategy: docker-push

set -e

VERSION="${{1:-latest}}"
REGISTRY="{registry}"
IMAGE="{project_name}"

echo "Building image..."
docker build -t "$REGISTRY/$IMAGE:$VERSION" -t "$REGISTRY/$IMAGE:latest" .

echo "Pushing to registry..."
docker push "$REGISTRY/$IMAGE:$VERSION"
docker push "$REGISTRY/$IMAGE:latest"

echo "Deployed $REGISTRY/$IMAGE:$VERSION"
"""

    elif strategy == "fly":
        app_name = config.get("app_name", project_name)
        return f"""#!/bin/bash
# Auto-generated deployment script
# Strategy: fly.io

set -e

VERSION="${{1:-$(date +%Y%m%d%H%M%S)}}"

echo "Deploying to Fly.io..."
flyctl deploy --remote-only --app {app_name} --image-label "$VERSION"

echo "Deployed version $VERSION"
flyctl status --app {app_name}
"""

    elif strategy == "render":
        return f"""#!/bin/bash
# Auto-generated deployment script
# Strategy: render.com

set -e

SERVICE_ID="${{RENDER_SERVICE_ID:-your-service-id}}"
API_KEY="${{RENDER_API_KEY:?RENDER_API_KEY required}}"

echo "Triggering Render deploy..."
curl -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" \\
    -H "Authorization: Bearer $API_KEY" \\
    -H "Content-Type: application/json" \\
    -d '{{"clearCache": false}}'

echo "Deploy triggered. Check Render dashboard for status."
"""

    elif strategy == "ssh":
        host = config.get("host", "your-server.com")
        user = config.get("user", "deploy")
        path = config.get("path", "/app")
        return f"""#!/bin/bash
# Auto-generated deployment script
# Strategy: ssh/rsync

set -e

HOST="{host}"
USER="{user}"
DEPLOY_PATH="{path}"

echo "Syncing files to $HOST..."
rsync -avz --delete \\
    --exclude '.git' \\
    --exclude '__pycache__' \\
    --exclude '*.pyc' \\
    --exclude '.env' \\
    --exclude 'venv' \\
    ./ "$USER@$HOST:$DEPLOY_PATH"

echo "Running remote deploy script..."
ssh "$USER@$HOST" "cd $DEPLOY_PATH && ./run.sh"

echo "Deployed to $HOST:$DEPLOY_PATH"
"""

    else:
        return f"""#!/bin/bash
# Auto-generated deployment script
# Strategy: custom

set -e

VERSION="${{1:-latest}}"
PROJECT_DIR="$(pwd)"

echo "Custom deployment for {project_name}"
echo "Version: $VERSION"
echo "Project: $PROJECT_DIR"

# Add your deployment commands here
# Example:
# docker-compose down
# docker-compose pull
# docker-compose up -d

echo "Deploy complete"
"""
