"""Decomposition Agent for breaking features into sub-tasks.

Takes a FeatureSpec and decomposes it into a WorkPlan with
prioritized, dependent sub-tasks for incremental implementation.
"""

import json

from llm_backend import LLMBackend
from schemas.feature_spec import FeatureSpec
from schemas.workplan import (
    SubTask,
    TaskCategory,
    TaskPriority,
    TaskSize,
    WorkPlan,
)
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent

SYSTEM_PROMPT = """You are a software architect specializing in task decomposition. Your job is to break down feature specifications into small, traceable sub-tasks.

When given a feature spec, you must decompose it into sub-tasks that:

1. **Are Small & Focused**: Each task should be completable in a single PR (ideally < 200 lines changed)

2. **Have Clear Dependencies**: Identify what must be done first (e.g., config before implementation, backend before frontend)

3. **Are Categorized**: Tag each task by type:
   - frontend, backend, api, database, data_migration, infra, auth, testing, docs, config

4. **Are Prioritized**: Assign priority based on dependencies:
   - p0: Blocker - must be done first (e.g., dependencies, config)
   - p1: High - core functionality
   - p2: Normal - supporting features
   - p3: Nice to have - polish, optimization

5. **Have Size Estimates**:
   - xs: < 10 lines
   - s: 10-50 lines
   - m: 50-200 lines
   - l: 200-500 lines
   - xl: > 500 lines (should be split further!)

6. **Link to Acceptance Criteria**: Each task should map to one or more AC from the spec

DECOMPOSITION RULES:
- Start with foundational tasks (dependencies, config, schemas)
- Database changes before code that uses them
- Backend API before frontend that calls it
- Auth/security early if required
- Tests can be parallel with implementation
- Docs at the end

IMPORTANT: Respond ONLY with valid JSON matching this structure:
{
  "tasks": [
    {
      "id": "TASK-001",
      "title": "Short task title",
      "description": "What needs to be done",
      "category": "backend|frontend|api|database|data_migration|infra|auth|testing|docs|config",
      "priority": "p0|p1|p2|p3",
      "size": "xs|s|m|l|xl",
      "depends_on": ["TASK-IDs this depends on"],
      "blocks": ["TASK-IDs this blocks"],
      "acceptance_criteria": ["AC-IDs from spec"],
      "target_files": ["likely files to modify"],
      "implementation_notes": "Hints for implementation"
    }
  ],
  "execution_order": ["TASK-001", "TASK-002", "..."],
  "parallel_groups": [["TASK-003", "TASK-004"], ["TASK-005"]],
  "estimated_complexity": "trivial|simple|moderate|complex|very_complex",
  "decomposition_rationale": "Why split this way",
  "risks": ["Identified risks or blockers"]
}

Be thorough. A complex feature might have 5-15 tasks. Simple features might have 2-3."""


class DecompositionAgent(BaseAgent):
    """Agent for decomposing features into sub-tasks.

    Takes a FeatureSpec and produces a WorkPlan with:
    - Ordered sub-tasks with dependencies
    - Priority and size estimates
    - Execution order and parallel groups
    - Risk identification
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize DecompositionAgent.

        Args:
            llm: LLM backend for inference
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Decompose feature spec into workplan.

        Args:
            input_data: Must contain 'feature_spec' dict in context

        Returns:
            AgentOutput with WorkPlan data
        """
        spec_data = input_data.context.get("feature_spec", {})

        if not spec_data:
            return AgentOutput(
                success=False,
                data={},
                errors=["No feature spec provided"],
            )

        # Parse into FeatureSpec if dict
        if isinstance(spec_data, dict):
            try:
                feature_spec = FeatureSpec(**spec_data)
            except Exception as e:
                return AgentOutput(
                    success=False,
                    data={},
                    errors=[f"Invalid feature spec: {e}"],
                )
        else:
            feature_spec = spec_data

        # Build the user message with full spec context
        user_message = self._build_prompt(feature_spec)

        try:
            # Call LLM
            response = self._chat(user_message, temperature=0.3)

            # Parse the response
            plan_data = self._parse_response(response)

            # Create WorkPlan
            workplan = self._create_workplan(feature_spec, plan_data)

            return AgentOutput(
                success=True,
                data=workplan.model_dump(),
                artifacts=["workplan.json"],
            )

        except json.JSONDecodeError as e:
            return AgentOutput(
                success=False,
                data={"raw_response": response if "response" in dir() else ""},
                errors=[f"Failed to parse LLM response as JSON: {e}"],
            )
        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"DecompositionAgent error: {str(e)}"],
            )

    def _build_prompt(self, spec: FeatureSpec) -> str:
        """Build the decomposition prompt from feature spec.

        Args:
            spec: The feature specification

        Returns:
            Formatted prompt string
        """
        # Format acceptance criteria
        ac_list = "\n".join(
            f"  - {ac.id}: {ac.description}"
            for ac in spec.acceptance_criteria
        )

        # Format constraints
        constraints_list = "\n".join(
            f"  - [{c.type}] {c.description}"
            for c in spec.constraints
        ) or "  None specified"

        # Format domains
        domains = ", ".join(d.value for d in spec.domains) or "Not specified"

        return f"""Decompose this feature into sub-tasks:

FEATURE: {spec.title}
ID: {spec.id}

USER STORY:
{spec.user_story}

ACCEPTANCE CRITERIA:
{ac_list or "  None specified"}

DOMAINS AFFECTED: {domains}

CONSTRAINTS:
{constraints_list}

ESTIMATED COMPLEXITY: {spec.estimated_complexity or "Not specified"}

ASSUMPTIONS:
{chr(10).join(f"  - {a}" for a in spec.assumptions) or "  None"}

Break this into small, ordered sub-tasks. Consider:
1. What foundational work is needed first?
2. What can be done in parallel?
3. What depends on what?
4. Are there any XL tasks that should be split further?

Respond with ONLY valid JSON matching the required structure."""

    def _parse_response(self, response: str) -> dict:
        """Extract JSON from LLM response with repair.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed
        """
        result = parse_llm_json(response, default=None)

        if result is None:
            raise json.JSONDecodeError(
                "Could not parse JSON from response",
                response,
                0,
            )

        return result

    def _create_workplan(self, spec: FeatureSpec, data: dict) -> WorkPlan:
        """Create validated WorkPlan from parsed data.

        Args:
            spec: Source feature spec
            data: Parsed JSON data

        Returns:
            Validated WorkPlan
        """
        tasks = []
        for i, task_data in enumerate(data.get("tasks", [])):
            task = self._parse_task(task_data, i)
            tasks.append(task)

        # Calculate execution order if not provided
        execution_order = data.get("execution_order", [])
        if not execution_order:
            execution_order = self._calculate_execution_order(tasks)

        return WorkPlan(
            feature_id=spec.id,
            feature_title=spec.title,
            tasks=tasks,
            execution_order=execution_order,
            parallel_groups=data.get("parallel_groups", []),
            total_tasks=len(tasks),
            estimated_complexity=data.get("estimated_complexity", "moderate"),
            decomposition_rationale=data.get("decomposition_rationale", ""),
            risks=data.get("risks", []),
        )

    def _parse_task(self, data: dict, index: int) -> SubTask:
        """Parse a single sub-task from JSON data.

        Args:
            data: Task JSON data
            index: Task index for default ID

        Returns:
            Validated SubTask
        """
        # Parse category
        try:
            category = TaskCategory(data.get("category", "backend").lower())
        except ValueError:
            category = TaskCategory.BACKEND

        # Parse priority
        try:
            priority = TaskPriority(data.get("priority", "p1").lower())
        except ValueError:
            priority = TaskPriority.P1

        # Parse size
        try:
            size = TaskSize(data.get("size", "m").lower())
        except ValueError:
            size = TaskSize.M

        return SubTask(
            id=data.get("id", f"TASK-{index + 1:03d}"),
            title=data.get("title", "Untitled Task"),
            description=data.get("description", ""),
            category=category,
            priority=priority,
            size=size,
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            target_files=data.get("target_files", []),
            implementation_notes=data.get("implementation_notes"),
            status="pending",
        )

    def _calculate_execution_order(self, tasks: list[SubTask]) -> list[str]:
        """Calculate execution order based on dependencies.

        Uses topological sort to order tasks.

        Args:
            tasks: List of sub-tasks

        Returns:
            List of task IDs in execution order
        """
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        graph = {t.id: [] for t in tasks}

        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id in graph:
                    graph[dep_id].append(task.id)
                    in_degree[task.id] += 1

        # Topological sort with priority tie-breaker
        result = []
        available = [
            t.id for t in tasks
            if in_degree[t.id] == 0
        ]

        # Sort by priority (p0 first)
        available.sort(key=lambda x: task_map[x].priority.value)

        while available:
            # Take highest priority available task
            current = available.pop(0)
            result.append(current)

            # Update dependencies
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)
                    available.sort(key=lambda x: task_map[x].priority.value)

        # Add any remaining tasks (cycles or orphans)
        for task in tasks:
            if task.id not in result:
                result.append(task.id)

        return result

    def decompose(self, spec: FeatureSpec) -> WorkPlan | None:
        """Convenience method to decompose a feature spec.

        Args:
            spec: Feature specification to decompose

        Returns:
            WorkPlan if successful, None otherwise
        """
        input_data = AgentInput(
            context={"feature_spec": spec.model_dump()}
        )

        output = self.run(input_data)

        if output.success:
            return WorkPlan(**output.data)
        return None
