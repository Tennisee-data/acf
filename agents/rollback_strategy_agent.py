"""Rollback Strategy Agent - Generates CI/CD rollback and canary deployment strategies.

This agent creates:
- Rollback jobs for GitHub Actions
- Canary deployment templates (blue-green, rolling, etc.)
- Rollback playbook documentation (ops.md)
"""

import logging
from pathlib import Path

from agents.base import BaseAgent
from llm_backend import LLMBackend
from schemas.rollback_strategy import (
    BlueGreenConfig,
    CanaryConfig,
    DeploymentPattern,
    GeneratedWorkflow,
    HealthCheck,
    PlaybookSection,
    RollbackAction,
    RollbackJob,
    RollbackPlaybook,
    RollbackStep,
    RollbackStrategyReport,
    RollbackTrigger,
    RollingConfig,
)

logger = logging.getLogger(__name__)


class RollbackStrategyAgent(BaseAgent):
    """Agent that generates rollback strategies and canary deployment configurations."""

    def __init__(self, llm: LLMBackend | None = None) -> None:
        """Initialize rollback strategy agent."""
        super().__init__(llm)
        self.name = "RollbackStrategyAgent"

    def default_system_prompt(self) -> str:
        """Return the default system prompt for rollback strategies."""
        return """You are a DevOps expert specializing in deployment strategies.
Generate rollback jobs, canary deployments, and operational playbooks for
safe production deployments."""

    def run(
        self,
        repo_path: Path,
        service_name: str = "app",
        deployment_pattern: DeploymentPattern = DeploymentPattern.ROLLING,
        has_database: bool = False,
        kubernetes: bool = False,
    ) -> RollbackStrategyReport:
        """Generate rollback strategy for a project.

        Args:
            repo_path: Path to the project
            service_name: Name of the service being deployed
            deployment_pattern: Preferred deployment pattern
            has_database: Whether the service has database migrations
            kubernetes: Whether deploying to Kubernetes

        Returns:
            Complete rollback strategy report
        """
        logger.info("Generating rollback strategy for %s", service_name)

        # Detect project characteristics
        project_info = self._analyze_project(repo_path)

        # Generate health checks
        health_checks = self._generate_health_checks(project_info)

        # Generate rollback jobs
        rollback_jobs = self._generate_rollback_jobs(
            service_name, has_database, project_info
        )

        # Generate pattern-specific config
        canary_config = None
        blue_green_config = None
        rolling_config = None

        if deployment_pattern == DeploymentPattern.CANARY:
            canary_config = self._generate_canary_config()
        elif deployment_pattern == DeploymentPattern.BLUE_GREEN:
            blue_green_config = self._generate_blue_green_config()
        elif deployment_pattern == DeploymentPattern.ROLLING:
            rolling_config = self._generate_rolling_config()

        # Generate workflows
        workflows = self._generate_workflows(
            service_name,
            deployment_pattern,
            rollback_jobs,
            health_checks,
            kubernetes,
            canary_config,
            blue_green_config,
        )

        # Generate playbook
        playbook = self._generate_playbook(
            service_name,
            deployment_pattern,
            rollback_jobs,
            has_database,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            deployment_pattern, has_database, kubernetes, project_info
        )

        report = RollbackStrategyReport(
            deployment_pattern=deployment_pattern,
            health_checks=health_checks,
            rollback_jobs=rollback_jobs,
            canary_config=canary_config,
            blue_green_config=blue_green_config,
            rolling_config=rolling_config,
            workflows=workflows,
            playbook=playbook,
            recommendations=recommendations,
        )

        logger.info(
            "Generated rollback strategy: %s pattern, %d workflows, %d rollback jobs",
            deployment_pattern.value,
            len(workflows),
            len(rollback_jobs),
        )

        return report

    def _analyze_project(self, repo_path: Path) -> dict:
        """Analyze project to detect deployment characteristics."""
        info = {
            "has_docker": False,
            "has_kubernetes": False,
            "has_helm": False,
            "has_terraform": False,
            "framework": None,
            "has_health_endpoint": False,
        }

        # Check for Docker
        if (repo_path / "Dockerfile").exists():
            info["has_docker"] = True

        # Check for Kubernetes
        k8s_patterns = ["k8s", "kubernetes", "manifests", "deploy"]
        for pattern in k8s_patterns:
            if (repo_path / pattern).exists():
                info["has_kubernetes"] = True
                break

        # Check for Helm
        if (repo_path / "charts").exists() or (repo_path / "helm").exists():
            info["has_helm"] = True

        # Check for Terraform
        tf_files = list(repo_path.glob("*.tf")) + list(repo_path.glob("**/*.tf"))
        if tf_files:
            info["has_terraform"] = True

        # Detect framework
        if (repo_path / "requirements.txt").exists():
            try:
                content = (repo_path / "requirements.txt").read_text()
                if "fastapi" in content.lower():
                    info["framework"] = "fastapi"
                elif "flask" in content.lower():
                    info["framework"] = "flask"
                elif "django" in content.lower():
                    info["framework"] = "django"
            except Exception:
                pass

        # Check for health endpoint in code
        for py_file in repo_path.glob("**/*.py"):
            try:
                content = py_file.read_text()
                if "/health" in content or "/ready" in content or "/live" in content:
                    info["has_health_endpoint"] = True
                    break
            except Exception:
                continue

        return info

    def _generate_health_checks(self, project_info: dict) -> list[HealthCheck]:
        """Generate health check configurations."""
        checks = []

        # Liveness check
        checks.append(
            HealthCheck(
                endpoint="/health" if project_info["has_health_endpoint"] else "/",
                expected_status=200,
                timeout_seconds=10,
                interval_seconds=10,
                failure_threshold=3,
            )
        )

        # Readiness check
        if project_info["has_health_endpoint"]:
            checks.append(
                HealthCheck(
                    endpoint="/ready",
                    expected_status=200,
                    timeout_seconds=30,
                    interval_seconds=5,
                    failure_threshold=3,
                )
            )

        return checks

    def _generate_rollback_jobs(
        self,
        service_name: str,
        has_database: bool,
        project_info: dict,
    ) -> list[RollbackJob]:
        """Generate rollback job configurations."""
        jobs = []

        # Manual rollback job
        manual_steps = [
            RollbackStep(
                order=1,
                action=RollbackAction.NOTIFY_TEAM,
                description="Notify team of rollback initiation",
                command="echo 'Rollback initiated for ${SERVICE_NAME}'",
            ),
            RollbackStep(
                order=2,
                action=RollbackAction.REVERT_DEPLOYMENT,
                description="Revert to previous deployment",
                command=self._get_revert_command(project_info),
                timeout_seconds=300,
            ),
        ]

        if has_database:
            manual_steps.append(
                RollbackStep(
                    order=3,
                    action=RollbackAction.RESTORE_DATABASE,
                    description="Restore database from backup (if needed)",
                    command="# Manual step: restore database if migration was applied",
                    on_failure="abort",
                )
            )

        manual_steps.extend([
            RollbackStep(
                order=len(manual_steps) + 1,
                action=RollbackAction.INVALIDATE_CACHE,
                description="Invalidate caches if needed",
                command="# Invalidate CDN/Redis cache if applicable",
            ),
            RollbackStep(
                order=len(manual_steps) + 2,
                action=RollbackAction.CREATE_INCIDENT,
                description="Create incident ticket",
                command="# Create incident in PagerDuty/Opsgenie",
            ),
        ])

        jobs.append(
            RollbackJob(
                name=f"rollback-{service_name}",
                trigger=RollbackTrigger.MANUAL,
                steps=manual_steps,
                environment="production",
                requires_approval=True,
                notification_channels=["#deployments", "#oncall"],
            )
        )

        # Automatic rollback on health check failure
        auto_steps = [
            RollbackStep(
                order=1,
                action=RollbackAction.NOTIFY_TEAM,
                description="Alert team of automatic rollback",
                command="echo '⚠️ Automatic rollback triggered due to health check failure'",
            ),
            RollbackStep(
                order=2,
                action=RollbackAction.REVERT_DEPLOYMENT,
                description="Revert to last known good version",
                command=self._get_revert_command(project_info),
                timeout_seconds=300,
            ),
            RollbackStep(
                order=3,
                action=RollbackAction.CREATE_INCIDENT,
                description="Create incident for investigation",
                command="# Auto-create P2 incident",
            ),
        ]

        jobs.append(
            RollbackJob(
                name=f"auto-rollback-{service_name}",
                trigger=RollbackTrigger.HEALTH_CHECK_FAILURE,
                steps=auto_steps,
                environment="production",
                requires_approval=False,
                notification_channels=["#alerts", "#oncall"],
            )
        )

        return jobs

    def _get_revert_command(self, project_info: dict) -> str:
        """Get appropriate revert command based on infrastructure."""
        if project_info.get("has_kubernetes") or project_info.get("has_helm"):
            return "kubectl rollout undo deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}"
        elif project_info.get("has_docker"):
            return "docker service update --rollback ${SERVICE_NAME}"
        else:
            return "git revert HEAD --no-edit && git push origin main"

    def _generate_canary_config(self) -> CanaryConfig:
        """Generate canary deployment configuration."""
        return CanaryConfig(
            initial_percentage=10,
            increment_percentage=20,
            increment_interval_minutes=15,
            success_threshold=0.99,
            error_rate_threshold=0.01,
            latency_p99_threshold_ms=500,
        )

    def _generate_blue_green_config(self) -> BlueGreenConfig:
        """Generate blue-green deployment configuration."""
        return BlueGreenConfig(
            blue_environment="blue",
            green_environment="green",
            switch_timeout_seconds=60,
            validation_period_seconds=300,
            keep_previous_version=True,
        )

    def _generate_rolling_config(self) -> RollingConfig:
        """Generate rolling deployment configuration."""
        return RollingConfig(
            max_surge="25%",
            max_unavailable="25%",
            min_ready_seconds=30,
        )

    def _generate_workflows(
        self,
        service_name: str,
        pattern: DeploymentPattern,
        rollback_jobs: list[RollbackJob],
        health_checks: list[HealthCheck],
        kubernetes: bool,
        canary_config: CanaryConfig | None,
        blue_green_config: BlueGreenConfig | None,
    ) -> list[GeneratedWorkflow]:
        """Generate GitHub Actions workflow files."""
        workflows = []

        # Rollback workflow
        rollback_workflow = self._generate_rollback_workflow(
            service_name, rollback_jobs
        )
        workflows.append(rollback_workflow)

        # Deployment workflow with rollback capability
        deploy_workflow = self._generate_deploy_workflow(
            service_name,
            pattern,
            health_checks,
            kubernetes,
            canary_config,
            blue_green_config,
        )
        workflows.append(deploy_workflow)

        return workflows

    def _generate_rollback_workflow(
        self,
        service_name: str,
        rollback_jobs: list[RollbackJob],
    ) -> GeneratedWorkflow:
        """Generate rollback workflow YAML."""
        workflow = f"""name: Rollback {service_name}

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to rollback'
        required: true
        default: 'production'
        type: choice
        options:
          - production
          - staging
      revision:
        description: 'Revision to rollback to (leave empty for previous)'
        required: false
        type: string
      reason:
        description: 'Reason for rollback'
        required: true
        type: string

env:
  SERVICE_NAME: {service_name}

jobs:
  rollback:
    name: Rollback Deployment
    runs-on: ubuntu-latest
    environment: ${{{{ github.event.inputs.environment }}}}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Notify Start
        run: |
          echo "🔄 Starting rollback of ${{{{ env.SERVICE_NAME }}}}"
          echo "Environment: ${{{{ github.event.inputs.environment }}}}"
          echo "Reason: ${{{{ github.event.inputs.reason }}}}"

      - name: Get Previous Revision
        id: revision
        run: |
          if [ -n "${{{{ github.event.inputs.revision }}}}" ]; then
            echo "revision=${{{{ github.event.inputs.revision }}}}" >> $GITHUB_OUTPUT
          else
            # Get previous successful deployment
            echo "revision=previous" >> $GITHUB_OUTPUT
          fi

      - name: Perform Rollback
        run: |
          echo "Rolling back to revision: ${{{{ steps.revision.outputs.revision }}}}"
          # Add your rollback command here
          # kubectl rollout undo deployment/${{{{ env.SERVICE_NAME }}}} \\
          #   -n ${{{{ github.event.inputs.environment }}}}

      - name: Verify Health
        run: |
          echo "Verifying service health..."
          # Add health check verification
          # curl -f http://localhost/health || exit 1

      - name: Create Incident
        if: always()
        run: |
          echo "📋 Creating incident record"
          echo "Rollback completed at $(date)"

      - name: Notify Complete
        if: success()
        run: |
          echo "✅ Rollback completed successfully"

      - name: Notify Failure
        if: failure()
        run: |
          echo "❌ Rollback failed - manual intervention required"
"""
        return GeneratedWorkflow(
            filename="rollback.yml",
            content=workflow,
            description="Manual rollback workflow with environment selection",
        )

    def _generate_deploy_workflow(
        self,
        service_name: str,
        pattern: DeploymentPattern,
        health_checks: list[HealthCheck],
        kubernetes: bool,
        canary_config: CanaryConfig | None,
        blue_green_config: BlueGreenConfig | None,
    ) -> GeneratedWorkflow:
        """Generate deployment workflow with rollback capability."""
        health_endpoint = health_checks[0].endpoint if health_checks else "/health"

        if pattern == DeploymentPattern.CANARY:
            deploy_steps = self._generate_canary_steps(canary_config)
        elif pattern == DeploymentPattern.BLUE_GREEN:
            deploy_steps = self._generate_blue_green_steps(blue_green_config)
        else:
            deploy_steps = self._generate_rolling_steps()

        workflow = f"""name: Deploy {service_name}

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  SERVICE_NAME: {service_name}
  HEALTH_ENDPOINT: {health_endpoint}

jobs:
  build:
    name: Build
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{{{ steps.meta.outputs.tags }}}}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{{{ env.SERVICE_NAME }}}}:${{{{ github.sha }}}}

      - name: Output image tag
        id: meta
        run: echo "tags=${{{{ env.SERVICE_NAME }}}}:${{{{ github.sha }}}}" >> $GITHUB_OUTPUT

  deploy:
    name: Deploy
    needs: build
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Checkout
        uses: actions/checkout@v4

{deploy_steps}

      - name: Verify Deployment
        id: verify
        run: |
          echo "Verifying deployment health..."
          for i in {{1..10}}; do
            if curl -sf "${{{{ env.HEALTH_ENDPOINT }}}}" > /dev/null; then
              echo "✅ Health check passed"
              exit 0
            fi
            echo "Waiting for service... (attempt $i/10)"
            sleep 10
          done
          echo "❌ Health check failed"
          exit 1

      - name: Rollback on Failure
        if: failure()
        run: |
          echo "🔄 Deployment failed, initiating rollback..."
          # kubectl rollout undo deployment/${{{{ env.SERVICE_NAME }}}}
          echo "Rollback completed"

  notify:
    name: Notify
    needs: [build, deploy]
    runs-on: ubuntu-latest
    if: always()

    steps:
      - name: Notify Success
        if: needs.deploy.result == 'success'
        run: echo "✅ Deployment successful"

      - name: Notify Failure
        if: needs.deploy.result == 'failure'
        run: echo "❌ Deployment failed and was rolled back"
"""
        return GeneratedWorkflow(
            filename="deploy.yml",
            content=workflow,
            description=f"Deployment workflow with {pattern.value} strategy and auto-rollback",
        )

    def _generate_canary_steps(self, config: CanaryConfig | None) -> str:
        """Generate canary deployment steps."""
        if not config:
            config = CanaryConfig()

        return f"""      - name: Deploy Canary ({config.initial_percentage}%)
        run: |
          echo "Deploying canary with {config.initial_percentage}% traffic..."
          # kubectl set image deployment/${{{{ env.SERVICE_NAME }}}}-canary ...

      - name: Monitor Canary
        run: |
          echo "Monitoring canary for {config.increment_interval_minutes} minutes..."
          sleep {config.increment_interval_minutes * 60}
          # Check error rate < {config.error_rate_threshold}
          # Check latency p99 < {config.latency_p99_threshold_ms}ms

      - name: Promote Canary
        run: |
          echo "Canary healthy, promoting to 100%..."
          # kubectl set image deployment/${{{{ env.SERVICE_NAME }}}} ..."""

    def _generate_blue_green_steps(self, config: BlueGreenConfig | None) -> str:
        """Generate blue-green deployment steps."""
        if not config:
            config = BlueGreenConfig()

        return f"""      - name: Deploy to {config.green_environment.title()} Environment
        run: |
          echo "Deploying to {config.green_environment} environment..."
          # kubectl apply -f k8s/{config.green_environment}/

      - name: Validate {config.green_environment.title()} Deployment
        run: |
          echo "Validating {config.green_environment} for {config.validation_period_seconds}s..."
          sleep {config.validation_period_seconds}

      - name: Switch Traffic
        run: |
          echo "Switching traffic to {config.green_environment}..."
          # kubectl patch service ${{{{ env.SERVICE_NAME }}}} \\
          #   -p '{{"spec":{{"selector":{{"version":"{config.green_environment}"}}}}}}'

      - name: Keep {config.blue_environment.title()} Running
        if: {str(config.keep_previous_version).lower()}
        run: |
          echo "Keeping {config.blue_environment} environment as fallback...\""""

    def _generate_rolling_steps(self) -> str:
        """Generate rolling deployment steps."""
        return """      - name: Rolling Update
        run: |
          echo "Starting rolling update..."
          # kubectl set image deployment/${SERVICE_NAME} app=${SERVICE_NAME}:${GITHUB_SHA}
          # kubectl rollout status deployment/${SERVICE_NAME} --timeout=300s"""

    def _generate_playbook(
        self,
        service_name: str,
        pattern: DeploymentPattern,
        rollback_jobs: list[RollbackJob],
        has_database: bool,
    ) -> RollbackPlaybook:
        """Generate rollback playbook documentation."""
        sections = []

        # When to rollback section
        sections.append(
            PlaybookSection(
                title="When to Rollback",
                content="""Initiate a rollback when:

- **Health checks fail** for more than 3 consecutive checks
- **Error rate exceeds** 1% of requests
- **P99 latency exceeds** 500ms threshold
- **Critical bug discovered** in production
- **Security vulnerability** identified in deployed code
- **Customer-impacting issue** reported by support

⚠️ When in doubt, rollback first, investigate later.
""",
            )
        )

        # Quick rollback section
        quick_commands = f"""### GitHub Actions (Recommended)

1. Go to Actions → "Rollback {service_name}"
2. Click "Run workflow"
3. Select environment and provide reason
4. Click "Run workflow"

### Manual CLI

```bash
# Kubernetes
kubectl rollout undo deployment/{service_name} -n production

# Docker Swarm
docker service update --rollback {service_name}

# Git-based
git revert HEAD --no-edit && git push origin main
```
"""
        sections.append(
            PlaybookSection(
                title="Quick Rollback Commands",
                content=quick_commands,
            )
        )

        # Step by step section
        step_by_step = """### Step-by-Step Rollback Procedure

1. **Assess the Situation** (1-2 minutes)
   - Check monitoring dashboards
   - Review error logs
   - Confirm rollback is necessary

2. **Notify the Team** (immediate)
   - Post in #incidents channel
   - Alert on-call if after hours

3. **Execute Rollback** (2-5 minutes)
   - Use GitHub Actions workflow OR manual commands
   - Monitor deployment status

4. **Verify Recovery** (5-10 minutes)
   - Check health endpoints
   - Verify error rates returning to normal
   - Confirm customer-facing functionality

5. **Document & Follow-up**
   - Create incident ticket
   - Schedule post-mortem
   - Update runbook if needed
"""
        sections.append(
            PlaybookSection(
                title="Step-by-Step Procedure",
                content=step_by_step,
            )
        )

        # Database rollback section (if applicable)
        if has_database:
            db_section = """### Database Rollback

⚠️ **Database rollbacks require extra caution**

If the deployment included database migrations:

1. **Check if migration is reversible**
   ```bash
   alembic history --verbose
   ```

2. **Rollback migration (if safe)**
   ```bash
   alembic downgrade -1
   ```

3. **Restore from backup (if needed)**
   ```bash
   # Restore from latest backup
   pg_restore -d $DATABASE_URL backup.sql
   ```

**Note:** Some migrations are not reversible. Consult with DBA before proceeding.
"""
            sections.append(
                PlaybookSection(
                    title="Database Rollback",
                    content=db_section,
                )
            )

        # Verification section
        verification = """### Post-Rollback Verification

Run these checks after rollback:

```bash
# Health check
curl -f https://api.example.com/health

# Smoke test
curl -f https://api.example.com/api/v1/status

# Check error rates (Datadog/Prometheus)
# Error rate should drop to < 0.1% within 5 minutes
```

### Success Criteria

- [ ] Health endpoint returns 200
- [ ] Error rate < 0.1%
- [ ] P99 latency < 200ms
- [ ] No new errors in logs
- [ ] Customer-facing features working
"""
        sections.append(
            PlaybookSection(
                title="Post-Rollback Verification",
                content=verification,
            )
        )

        # Escalation section
        escalation = """### Escalation Path

If rollback fails or issues persist:

1. **L1: On-call Engineer** - First 15 minutes
   - Attempt standard rollback procedures

2. **L2: Team Lead** - After 15 minutes
   - Escalate if rollback unsuccessful
   - Page: @team-lead

3. **L3: Platform Team** - After 30 minutes
   - Infrastructure-level issues
   - Page: @platform-oncall

4. **L4: Engineering Leadership** - Customer impact > 1 hour
   - Page: @eng-leadership
"""
        sections.append(
            PlaybookSection(
                title="Escalation Path",
                content=escalation,
            )
        )

        return RollbackPlaybook(
            title=f"Rollback Playbook: {service_name}",
            overview=f"""This playbook documents the rollback procedures for **{service_name}**.

**Deployment Pattern:** {pattern.value.replace('_', ' ').title()}

Use this document when you need to quickly revert a deployment due to issues in production.
""",
            prerequisites=[
                "Access to GitHub Actions",
                "kubectl configured for production cluster",
                "Access to monitoring dashboards",
                "Slack access for #incidents channel",
            ],
            sections=sections,
            contacts={
                "On-call": "@oncall in Slack",
                "Team Lead": "@team-lead",
                "Platform": "@platform-oncall",
            },
        )

    def _generate_recommendations(
        self,
        pattern: DeploymentPattern,
        has_database: bool,
        kubernetes: bool,
        project_info: dict,
    ) -> list[str]:
        """Generate strategy recommendations."""
        recommendations = []

        # Pattern-specific recommendations
        if pattern == DeploymentPattern.CANARY:
            recommendations.append(
                "Consider using service mesh (Istio/Linkerd) for fine-grained traffic control"
            )
            recommendations.append(
                "Set up automated canary analysis with Prometheus metrics"
            )
        elif pattern == DeploymentPattern.BLUE_GREEN:
            recommendations.append(
                "Ensure infrastructure supports running two full environments"
            )
            recommendations.append(
                "Configure DNS/load balancer for quick traffic switching"
            )

        # Database recommendations
        if has_database:
            recommendations.append(
                "Always backup database before deployments with migrations"
            )
            recommendations.append(
                "Use reversible migrations when possible (add columns, not drop)"
            )
            recommendations.append(
                "Consider blue-green for zero-downtime database migrations"
            )

        # Kubernetes recommendations
        if kubernetes or project_info.get("has_kubernetes"):
            recommendations.append(
                "Set PodDisruptionBudget to ensure availability during rollout"
            )
            recommendations.append(
                "Configure resource limits and requests for predictable scheduling"
            )

        # Health check recommendations
        if not project_info.get("has_health_endpoint"):
            recommendations.append(
                "Add dedicated /health and /ready endpoints for better monitoring"
            )

        # General recommendations
        recommendations.extend([
            "Run rollback drills quarterly to ensure procedures work",
            "Keep rollback documentation up-to-date with infrastructure changes",
            "Set up alerts for automatic rollback triggers",
        ])

        return recommendations

    def write_artifacts(
        self,
        report: RollbackStrategyReport,
        output_dir: Path,
    ) -> list[Path]:
        """Write generated artifacts to disk.

        Args:
            report: Rollback strategy report
            output_dir: Directory to write artifacts

        Returns:
            List of created file paths
        """
        created_files = []

        # Create .github/workflows directory
        workflows_dir = output_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # Write workflow files
        for workflow in report.workflows:
            workflow_path = workflows_dir / workflow.filename
            workflow_path.write_text(workflow.content)
            created_files.append(workflow_path)
            logger.info("Created workflow: %s", workflow_path)

        # Write playbook
        if report.playbook:
            docs_dir = output_dir / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)

            playbook_path = docs_dir / "ops.md"
            playbook_content = self._render_playbook_markdown(report.playbook)
            playbook_path.write_text(playbook_content)
            created_files.append(playbook_path)
            logger.info("Created playbook: %s", playbook_path)

        return created_files

    def _render_playbook_markdown(self, playbook: RollbackPlaybook) -> str:
        """Render playbook to markdown format."""
        lines = [
            f"# {playbook.title}",
            "",
            playbook.overview,
            "",
            "## Prerequisites",
            "",
        ]

        for prereq in playbook.prerequisites:
            lines.append(f"- {prereq}")

        lines.append("")

        for section in playbook.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        if playbook.contacts:
            lines.append("## Emergency Contacts")
            lines.append("")
            for role, contact in playbook.contacts.items():
                lines.append(f"- **{role}:** {contact}")
            lines.append("")

        if playbook.runbook_url:
            lines.append("## External Runbook")
            lines.append("")
            lines.append(f"See also: [{playbook.runbook_url}]({playbook.runbook_url})")
            lines.append("")

        lines.append("---")
        lines.append("*Generated by Coding Factory*")

        return "\n".join(lines)
