"""Runtime Decision Agent - determines optimal execution environment.

Implements Option C: Hybrid with Fallback
- Analyzes project characteristics to decide if Docker is warranted
- Considers framework, dependencies, complexity, and user intent
- Outputs: docker | local | venv with reasoning
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base import AgentInput, AgentOutput, BaseAgent


@dataclass
class RuntimeDecision:
    """Result of runtime environment analysis."""

    runtime: str  # "docker" | "local" | "venv"
    reason: str
    confidence: float  # 0.0 to 1.0
    signals: dict[str, Any]


# Thresholds for decision making
LINE_COUNT_DOCKER_THRESHOLD = 500  # Projects over this likely benefit from Docker
SERVICE_KEYWORDS = [
    "postgres", "postgresql", "mysql", "mariadb", "mongodb", "redis",
    "rabbitmq", "kafka", "elasticsearch", "celery", "memcached",
    "nginx", "gunicorn", "uvicorn", "sqlite",  # sqlite is local but often indicates DB usage
]

WEB_FRAMEWORKS = {
    "flask": {"docker_weight": 0.7, "type": "web"},
    "fastapi": {"docker_weight": 0.8, "type": "web"},
    "django": {"docker_weight": 0.9, "type": "web"},
    "starlette": {"docker_weight": 0.7, "type": "web"},
    "tornado": {"docker_weight": 0.6, "type": "web"},
    "bottle": {"docker_weight": 0.5, "type": "web"},
    "aiohttp": {"docker_weight": 0.7, "type": "web"},
}

CLI_FRAMEWORKS = {
    "typer": {"docker_weight": 0.2, "type": "cli"},
    "click": {"docker_weight": 0.2, "type": "cli"},
    "argparse": {"docker_weight": 0.1, "type": "cli"},
    "fire": {"docker_weight": 0.2, "type": "cli"},
}

# Keywords in feature description that suggest Docker
DOCKER_INTENT_KEYWORDS = [
    "deploy", "production", "container", "docker", "kubernetes", "k8s",
    "scale", "microservice", "api", "server", "service", "host",
    "cloud", "aws", "gcp", "azure", "heroku",
]

LOCAL_INTENT_KEYWORDS = [
    "script", "utility", "tool", "cli", "command-line", "local",
    "simple", "quick", "prototype", "library", "package",
]


class RuntimeDecisionAgent(BaseAgent):
    """Agent that decides the optimal runtime environment for a project.

    Analyzes:
    - Framework type (web vs CLI vs library)
    - External service dependencies
    - Code complexity (line count, file count)
    - User intent from feature description
    - Existing Docker configuration
    """

    def default_system_prompt(self) -> str:
        return """You are a DevOps expert that decides the optimal runtime environment for Python projects.

Given project characteristics, decide between:
- "docker": Containerized environment (best for web apps, services, complex dependencies)
- "venv": Virtual environment (good for CLI tools, moderate complexity)
- "local": Direct Python execution (simple scripts, libraries)

Consider:
1. Web frameworks almost always benefit from Docker
2. External services (databases, caches) require Docker for consistency
3. Simple CLI tools don't need Docker overhead
4. User intent matters - if they mention "deploy" or "production", prefer Docker

Respond with JSON:
{
    "runtime": "docker" | "venv" | "local",
    "reason": "Brief explanation",
    "confidence": 0.0-1.0
}"""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Analyze project and decide runtime environment.

        Args:
            input_data: Context with project_dir, feature_description, framework

        Returns:
            AgentOutput with RuntimeDecision data
        """
        context = input_data.context
        project_dir = Path(context.get("project_dir", "."))
        feature_description = context.get("feature_description", "")
        detected_framework = context.get("framework", "")

        # Gather signals
        signals = self._gather_signals(project_dir, feature_description, detected_framework)

        # Calculate Docker score (0.0 to 1.0)
        docker_score = self._calculate_docker_score(signals)

        # Make decision
        decision = self._make_decision(docker_score, signals)

        # Optionally use LLM for edge cases
        if 0.4 <= docker_score <= 0.6:
            decision = self._llm_tiebreaker(signals, feature_description, decision)

        return AgentOutput(
            success=True,
            data={
                "runtime": decision.runtime,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "signals": decision.signals,
                "docker_score": docker_score,
            },
        )

    def _gather_signals(
        self,
        project_dir: Path,
        feature_description: str,
        framework: str,
    ) -> dict[str, Any]:
        """Gather all signals for runtime decision."""
        signals = {
            "has_dockerfile": False,
            "has_docker_compose": False,
            "framework": framework.lower() if framework else None,
            "framework_type": None,
            "line_count": 0,
            "file_count": 0,
            "services_detected": [],
            "docker_intent_keywords": [],
            "local_intent_keywords": [],
            "imports": [],
        }

        # Check for existing Docker files
        if project_dir.exists():
            signals["has_dockerfile"] = (project_dir / "Dockerfile").exists()
            signals["has_docker_compose"] = (project_dir / "docker-compose.yml").exists()

            # Count lines and files
            signals["line_count"], signals["file_count"] = self._count_code(project_dir)

            # Analyze imports
            signals["imports"] = self._extract_imports(project_dir)

        # Detect framework type
        if signals["framework"]:
            if signals["framework"] in WEB_FRAMEWORKS:
                signals["framework_type"] = "web"
            elif signals["framework"] in CLI_FRAMEWORKS:
                signals["framework_type"] = "cli"

        # Detect service dependencies from imports and feature description
        all_text = " ".join(signals["imports"]) + " " + feature_description.lower()
        for service in SERVICE_KEYWORDS:
            if service in all_text:
                signals["services_detected"].append(service)

        # Analyze feature description intent
        desc_lower = feature_description.lower()
        for keyword in DOCKER_INTENT_KEYWORDS:
            if keyword in desc_lower:
                signals["docker_intent_keywords"].append(keyword)

        for keyword in LOCAL_INTENT_KEYWORDS:
            if keyword in desc_lower:
                signals["local_intent_keywords"].append(keyword)

        return signals

    def _count_code(self, project_dir: Path) -> tuple[int, int]:
        """Count total lines of Python code and number of files."""
        total_lines = 0
        file_count = 0

        for py_file in project_dir.rglob("*.py"):
            # Skip common non-source directories
            if any(part in py_file.parts for part in ["venv", ".venv", "__pycache__", ".git", "node_modules"]):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                # Count non-empty, non-comment lines
                lines = [
                    line for line in content.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
                total_lines += len(lines)
                file_count += 1
            except (OSError, UnicodeDecodeError):
                continue

        return total_lines, file_count

    def _extract_imports(self, project_dir: Path) -> list[str]:
        """Extract all import statements from Python files."""
        imports = set()
        import_pattern = re.compile(r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE)

        for py_file in project_dir.rglob("*.py"):
            if any(part in py_file.parts for part in ["venv", ".venv", "__pycache__", ".git"]):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for match in import_pattern.finditer(content):
                    imports.add(match.group(1).lower())
            except (OSError, UnicodeDecodeError):
                continue

        return list(imports)

    def _calculate_docker_score(self, signals: dict[str, Any]) -> float:
        """Calculate a 0.0-1.0 score for Docker preference."""
        score = 0.5  # Start neutral

        # Existing Docker files are strong signals
        if signals["has_dockerfile"]:
            score += 0.3
        if signals["has_docker_compose"]:
            score += 0.2

        # Framework influence
        framework = signals.get("framework", "")
        if framework in WEB_FRAMEWORKS:
            score += WEB_FRAMEWORKS[framework]["docker_weight"] * 0.3
        elif framework in CLI_FRAMEWORKS:
            score -= (1 - CLI_FRAMEWORKS[framework]["docker_weight"]) * 0.2

        # Services require Docker
        services = signals.get("services_detected", [])
        if services:
            # More services = stronger Docker signal
            score += min(len(services) * 0.15, 0.4)

        # Code complexity
        line_count = signals.get("line_count", 0)
        if line_count > LINE_COUNT_DOCKER_THRESHOLD:
            score += 0.15
        elif line_count < 100:
            score -= 0.15

        # User intent from feature description
        docker_keywords = len(signals.get("docker_intent_keywords", []))
        local_keywords = len(signals.get("local_intent_keywords", []))

        if docker_keywords > local_keywords:
            score += min(docker_keywords * 0.1, 0.25)
        elif local_keywords > docker_keywords:
            score -= min(local_keywords * 0.1, 0.25)

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _make_decision(self, docker_score: float, signals: dict[str, Any]) -> RuntimeDecision:
        """Convert docker score to a runtime decision."""
        if docker_score >= 0.6:
            runtime = "docker"
            reason = self._build_reason("docker", signals)
            confidence = min(docker_score, 0.95)
        elif docker_score <= 0.35:
            runtime = "local"
            reason = self._build_reason("local", signals)
            confidence = min(1.0 - docker_score, 0.95)
        else:
            runtime = "venv"
            reason = self._build_reason("venv", signals)
            confidence = 0.6  # Middle ground, less confident

        return RuntimeDecision(
            runtime=runtime,
            reason=reason,
            confidence=confidence,
            signals=signals,
        )

    def _build_reason(self, runtime: str, signals: dict[str, Any]) -> str:
        """Build a human-readable reason for the decision."""
        reasons = []

        if runtime == "docker":
            if signals.get("has_dockerfile"):
                reasons.append("existing Dockerfile")
            if signals.get("framework_type") == "web":
                reasons.append(f"{signals['framework']} web framework")
            if signals.get("services_detected"):
                reasons.append(f"services: {', '.join(signals['services_detected'][:3])}")
            if signals.get("docker_intent_keywords"):
                reasons.append("deployment intent detected")
            if signals.get("line_count", 0) > LINE_COUNT_DOCKER_THRESHOLD:
                reasons.append(f"complex codebase ({signals['line_count']} lines)")

            return f"Docker recommended: {'; '.join(reasons) or 'web application pattern'}"

        elif runtime == "local":
            if signals.get("framework_type") == "cli":
                reasons.append("CLI tool")
            if signals.get("line_count", 0) < 100:
                reasons.append("simple script")
            if signals.get("local_intent_keywords"):
                reasons.append("local/utility intent")
            if not signals.get("services_detected"):
                reasons.append("no external services")

            return f"Local execution: {'; '.join(reasons) or 'simple project'}"

        else:  # venv
            reasons.append("moderate complexity")
            if signals.get("framework_type") == "cli":
                reasons.append("CLI tool with dependencies")
            if not signals.get("services_detected"):
                reasons.append("no external services needed")

            return f"Virtual environment: {'; '.join(reasons)}"

    def _llm_tiebreaker(
        self,
        signals: dict[str, Any],
        feature_description: str,
        current_decision: RuntimeDecision,
    ) -> RuntimeDecision:
        """Use LLM to break ties in edge cases."""
        prompt = f"""Analyze this project and decide the best runtime environment.

Feature Description: {feature_description}

Signals detected:
- Framework: {signals.get('framework', 'none')} ({signals.get('framework_type', 'unknown')} type)
- Code size: {signals.get('line_count', 0)} lines across {signals.get('file_count', 0)} files
- Services detected: {signals.get('services_detected', [])}
- Has Dockerfile: {signals.get('has_dockerfile', False)}
- Docker intent keywords: {signals.get('docker_intent_keywords', [])}
- Local intent keywords: {signals.get('local_intent_keywords', [])}

Current leaning: {current_decision.runtime} (confidence: {current_decision.confidence:.2f})

Respond with JSON only:
{{"runtime": "docker" | "venv" | "local", "reason": "brief explanation", "confidence": 0.0-1.0}}"""

        try:
            response = self._chat(prompt)
            # Extract JSON from response
            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return RuntimeDecision(
                    runtime=data.get("runtime", current_decision.runtime),
                    reason=data.get("reason", current_decision.reason),
                    confidence=data.get("confidence", current_decision.confidence),
                    signals=signals,
                )
        except (json.JSONDecodeError, Exception):
            pass

        return current_decision


def decide_runtime(
    project_dir: Path | str,
    feature_description: str = "",
    framework: str = "",
    llm: Any = None,
) -> RuntimeDecision:
    """Convenience function to get runtime decision without full agent setup.

    Args:
        project_dir: Path to project directory
        feature_description: Feature being implemented
        framework: Detected framework (optional)
        llm: LLM backend (optional, for edge case tiebreaking)

    Returns:
        RuntimeDecision with runtime, reason, and confidence
    """
    # Create a minimal agent (LLM only used for tiebreaking)
    class MinimalLLM:
        def chat(self, messages, **kwargs):
            return "{}"

    agent = RuntimeDecisionAgent(llm or MinimalLLM())

    input_data = AgentInput(
        context={
            "project_dir": str(project_dir),
            "feature_description": feature_description,
            "framework": framework,
        }
    )

    result = agent.run(input_data)
    return RuntimeDecision(
        runtime=result.data["runtime"],
        reason=result.data["reason"],
        confidence=result.data["confidence"],
        signals=result.data["signals"],
    )
