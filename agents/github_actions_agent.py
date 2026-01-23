"""GitHub Actions Agent - Reasons about CI/CD needs and generates workflows."""

import json
import re
from pathlib import Path
from dataclasses import dataclass

from agents.base import BaseAgent, AgentInput, AgentOutput
from llm_backend import get_backend
from pipeline.config import get_config


@dataclass
class WorkflowDecision:
    """Decision about GitHub Actions workflow."""

    needs_ci: bool = False
    needs_schedule: bool = False
    schedule_cron: str | None = None
    schedule_description: str | None = None
    reasoning: str = ""
    workflow_type: str = "none"  # none, ci, scheduled, ci_and_scheduled


# Common cron patterns
CRON_PATTERNS = {
    "hourly": "0 * * * *",
    "daily": "0 0 * * *",
    "weekly": "0 0 * * 0",  # Sunday
    "monthly": "0 0 1 * *",  # 1st of month
    "nightly": "0 2 * * *",  # 2 AM
    "every morning": "0 8 * * *",
    "every evening": "0 18 * * *",
    "twice daily": "0 8,18 * * *",
    "weekdays": "0 8 * * 1-5",
}


class GitHubActionsAgent(BaseAgent):
    """Agent that reasons about CI/CD needs and generates GitHub Actions.

    This agent:
    1. Analyzes the feature description and project context
    2. Reasons about whether GitHub Actions are appropriate
    3. Interprets scheduling requirements from natural language
    4. Generates workflow files based on the reasoning
    """

    def __init__(self) -> None:
        super().__init__("GitHubActionsAgent")
        self.config = get_config()
        self.backend = get_backend(
            self.config.llm.backend,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model_general,
            timeout=self.config.llm.timeout,
        )

    def default_system_prompt(self) -> str:
        return "You are an expert in CI/CD and GitHub Actions configuration."

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Analyze project and generate GitHub Actions if appropriate.

        Args:
            input_data: Contains:
                - feature_description: Original feature request
                - project_dir: Path to generated project
                - framework: Detected framework (flask, fastapi, etc.)

        Returns:
            AgentOutput with:
                - decision: WorkflowDecision reasoning
                - workflows_generated: List of created workflow files
        """
        feature = input_data.context.get("feature_description", "")
        project_dir = Path(input_data.context.get("project_dir", ""))
        framework = input_data.context.get("framework", "python")

        try:
            # Step 1: Reason about CI/CD needs
            decision = self._reason_about_actions(feature, framework, project_dir)

            # Step 2: Generate workflows if needed
            workflows = []
            if decision.needs_ci or decision.needs_schedule:
                workflows = self._generate_workflows(decision, project_dir, framework)

            return AgentOutput(
                success=True,
                data={
                    "decision": {
                        "needs_ci": decision.needs_ci,
                        "needs_schedule": decision.needs_schedule,
                        "schedule_cron": decision.schedule_cron,
                        "schedule_description": decision.schedule_description,
                        "reasoning": decision.reasoning,
                        "workflow_type": decision.workflow_type,
                    },
                    "workflows_generated": workflows,
                },
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[str(e)],
            )

    def _reason_about_actions(
        self,
        feature: str,
        framework: str,
        project_dir: Path,
    ) -> WorkflowDecision:
        """Use LLM to reason about whether GitHub Actions are needed.

        Args:
            feature: Feature description
            framework: Detected framework
            project_dir: Project directory

        Returns:
            WorkflowDecision with reasoning
        """
        # Check for existing test files
        has_tests = any(project_dir.rglob("test_*.py")) or any(project_dir.rglob("*_test.py"))

        prompt = f"""Analyze this project and decide if GitHub Actions CI/CD would be beneficial.

Feature Description: {feature}
Framework: {framework}
Has Tests: {has_tests}

Consider:
1. Does this project need continuous integration (run tests on push/PR)?
2. Does this project need scheduled tasks (daily data fetches, weekly reports, etc.)?
3. What scheduling frequency is implied by the feature description?

Respond in JSON format:
{{
    "needs_ci": true/false,
    "ci_reasoning": "Why CI is/isn't needed",
    "needs_schedule": true/false,
    "schedule_reasoning": "Why scheduling is/isn't needed",
    "schedule_frequency": "none/hourly/daily/weekly/monthly/custom",
    "schedule_description": "What the scheduled task does (if any)",
    "custom_cron": "cron expression if custom timing mentioned"
}}

Examples of schedule detection:
- "fetch news daily" → daily
- "generate weekly report" → weekly
- "sync data every hour" → hourly
- "monthly cleanup" → monthly
- "run every Monday at 9am" → custom: "0 9 * * 1"

Be conservative - only suggest CI if there are tests or it's a web app.
Only suggest scheduling if the feature explicitly mentions timing."""

        response = self.backend.chat([{"role": "user", "content": prompt}])

        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Default to basic CI for web frameworks
                data = {
                    "needs_ci": framework in ["flask", "fastapi", "django"],
                    "needs_schedule": False,
                }
        except json.JSONDecodeError:
            data = {
                "needs_ci": framework in ["flask", "fastapi", "django"] or has_tests,
                "needs_schedule": False,
            }

        # Build decision
        decision = WorkflowDecision()
        decision.needs_ci = data.get("needs_ci", False)
        decision.needs_schedule = data.get("needs_schedule", False)

        # Determine cron expression
        if decision.needs_schedule:
            freq = data.get("schedule_frequency", "daily")
            if freq == "custom" and data.get("custom_cron"):
                decision.schedule_cron = data["custom_cron"]
            else:
                decision.schedule_cron = CRON_PATTERNS.get(freq, CRON_PATTERNS["daily"])
            decision.schedule_description = data.get("schedule_description", "Scheduled task")

        # Build reasoning
        reasoning_parts = []
        if data.get("ci_reasoning"):
            reasoning_parts.append(f"CI: {data['ci_reasoning']}")
        if data.get("schedule_reasoning"):
            reasoning_parts.append(f"Schedule: {data['schedule_reasoning']}")
        decision.reasoning = " | ".join(reasoning_parts) or "Default analysis"

        # Determine workflow type
        if decision.needs_ci and decision.needs_schedule:
            decision.workflow_type = "ci_and_scheduled"
        elif decision.needs_ci:
            decision.workflow_type = "ci"
        elif decision.needs_schedule:
            decision.workflow_type = "scheduled"
        else:
            decision.workflow_type = "none"

        return decision

    def _generate_workflows(
        self,
        decision: WorkflowDecision,
        project_dir: Path,
        framework: str,
    ) -> list[str]:
        """Generate GitHub Actions workflow files.

        Args:
            decision: Workflow decision with reasoning
            project_dir: Project directory
            framework: Detected framework

        Returns:
            List of created workflow file paths
        """
        workflows_dir = project_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        created = []

        # Generate CI workflow
        if decision.needs_ci:
            ci_path = workflows_dir / "ci.yml"
            ci_content = self._generate_ci_workflow(framework)
            ci_path.write_text(ci_content)
            created.append(".github/workflows/ci.yml")

        # Generate scheduled workflow
        if decision.needs_schedule and decision.schedule_cron:
            schedule_path = workflows_dir / "scheduled.yml"
            schedule_content = self._generate_scheduled_workflow(
                decision.schedule_cron,
                decision.schedule_description or "Scheduled task",
                framework,
            )
            schedule_path.write_text(schedule_content)
            created.append(".github/workflows/scheduled.yml")

        return created

    def _generate_ci_workflow(self, framework: str) -> str:
        """Generate CI workflow based on framework."""

        # Base setup for Python
        python_setup = """      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest ruff"""

        # Test commands based on framework
        test_commands = {
            "flask": "pytest -v",
            "fastapi": "pytest -v",
            "django": "python manage.py test",
        }
        test_cmd = test_commands.get(framework, "pytest -v")

        return f"""name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

{python_setup}

      - name: Lint with ruff
        run: ruff check . --exit-zero

      - name: Run tests
        run: {test_cmd}
"""

    def _generate_scheduled_workflow(
        self,
        cron: str,
        description: str,
        framework: str,
    ) -> str:
        """Generate scheduled workflow."""

        return f"""name: Scheduled Task

on:
  schedule:
    - cron: '{cron}'
  workflow_dispatch:  # Allow manual trigger

jobs:
  scheduled-task:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run scheduled task
        run: |
          chmod +x run.sh
          ./run.sh
        env:
          # Add any required environment variables
          CI: true

# Schedule: {description}
# Cron: {cron}
#
# Cron format: minute hour day-of-month month day-of-week
# Examples:
#   '0 0 * * *'   - Daily at midnight
#   '0 0 * * 0'   - Weekly on Sunday
#   '0 0 1 * *'   - Monthly on the 1st
"""
