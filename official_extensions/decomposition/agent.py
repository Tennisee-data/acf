"""Decomposition Agent - Break complex features into subtasks.

This official ACF extension analyzes feature complexity and breaks
large features into smaller, manageable implementation tasks.
"""

from dataclasses import dataclass, field
from typing import Any
import json
import re


@dataclass
class AgentOutput:
    """Output from the agent."""
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None


class DecompositionAgent:
    """Break complex features into smaller, manageable subtasks.

    Runs before the design stage to analyze feature complexity and
    create a work breakdown structure for large features.
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.llm = llm
        self.name = "decomposition"

    def default_system_prompt(self) -> str:
        return """You are a senior software architect specializing in breaking down complex features.

Your task is to analyze a feature request and determine if it should be decomposed into subtasks.

Guidelines:
- Features that touch 3+ files or have multiple distinct concerns should be decomposed
- Each subtask should be independently implementable
- Subtasks should have clear boundaries and acceptance criteria
- Order subtasks by dependency (foundational first)

Output JSON format:
{
    "should_decompose": true/false,
    "complexity_score": 1-10,
    "reasoning": "why decomposition is/isn't needed",
    "subtasks": [
        {
            "id": "task-1",
            "title": "Short title",
            "description": "What needs to be done",
            "acceptance_criteria": ["criterion 1", "criterion 2"],
            "depends_on": [],
            "estimated_files": ["file1.py", "file2.py"]
        }
    ]
}"""

    def run(self, input_data: Any) -> AgentOutput:
        """Analyze feature and decompose if complex.

        Args:
            input_data: Pipeline context with feature spec

        Returns:
            AgentOutput with decomposition results
        """
        context = input_data.context if hasattr(input_data, 'context') else input_data

        # Get feature spec
        spec = context.get("spec", {})
        feature_title = spec.get("title", "Unknown feature")
        feature_description = spec.get("description", "")
        requirements = spec.get("requirements", [])

        # Build prompt
        prompt = f"""Analyze this feature and determine if it should be decomposed into subtasks:

## Feature: {feature_title}

{feature_description}

## Requirements:
{json.dumps(requirements, indent=2) if requirements else "No specific requirements listed"}

Respond with JSON only."""

        try:
            # Call LLM
            messages = [
                {"role": "system", "content": self.default_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages)

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"should_decompose": False, "reasoning": "Could not parse response"}

            should_decompose = result.get("should_decompose", False)
            subtasks = result.get("subtasks", [])

            print(f"\n[Decomposition] Complexity: {result.get('complexity_score', '?')}/10")
            print(f"[Decomposition] Should decompose: {should_decompose}")
            if should_decompose and subtasks:
                print(f"[Decomposition] Subtasks: {len(subtasks)}")
                for task in subtasks:
                    print(f"  - {task.get('title', 'Untitled')}")

            return AgentOutput(
                success=True,
                data={
                    "decomposition": {
                        "should_decompose": should_decompose,
                        "complexity_score": result.get("complexity_score", 5),
                        "reasoning": result.get("reasoning", ""),
                        "subtasks": subtasks,
                    }
                },
                artifacts=["decomposition_report.json"] if should_decompose else None,
                agent_name=self.name,
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"Decomposition failed: {str(e)}"],
                agent_name=self.name,
            )
