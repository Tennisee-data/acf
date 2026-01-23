"""Spec Agent for parsing feature descriptions into structured specifications."""

import json
import re
from datetime import datetime

from llm_backend import LLMBackend
from utils.json_repair import parse_llm_json
from schemas.feature_spec import (
    AcceptanceCriteria,
    Constraint,
    Domain,
    FeatureSpec,
    NonFunctionalRequirement,
    Priority,
)
from schemas.requirements import (
    Requirement,
    RequirementsTracker,
    RequirementType,
)

from .base import AgentInput, AgentOutput, BaseAgent


SYSTEM_PROMPT = """You are a software specification analyst. Your job is to take natural language feature descriptions and convert them into structured, actionable specifications.

When given a feature description, you must extract and generate:

1. **User Story**: Format as "As a [user type], I want [feature], so that [benefit]"

2. **Acceptance Criteria**: Specific, testable conditions that must be met. Each should:
   - Have a unique ID (AC-001, AC-002, etc.)
   - Be verifiable (can we write a test for it?)
   - Include a verification hint

3. **Domains**: Which parts of the system are affected:
   - frontend, backend, database, infra, api, auth, ui, cli, testing, docs

4. **Constraints**: Technical or business limitations to consider

5. **Non-Functional Requirements**: Performance, security, scalability needs

6. **Assumptions**: What you're assuming that wasn't explicitly stated

7. **Clarifications Needed**: Questions that would help refine the spec

8. **Priority**: critical, high, medium, or low

9. **Complexity**: trivial, simple, moderate, or complex

IMPORTANT: Respond ONLY with valid JSON matching this exact structure:
{
  "title": "Short feature title",
  "user_story": "As a..., I want..., so that...",
  "acceptance_criteria": [
    {
      "id": "AC-001",
      "description": "...",
      "testable": true,
      "verification_hint": "How to test this"
    }
  ],
  "constraints": [
    {
      "description": "...",
      "type": "technical|business|security",
      "impact": "What happens if violated"
    }
  ],
  "non_functional_requirements": [
    {
      "category": "performance|security|scalability|reliability|usability",
      "requirement": "...",
      "metric": "Measurable metric",
      "threshold": "Acceptable value"
    }
  ],
  "domains": ["backend", "api"],
  "priority": "medium",
  "estimated_complexity": "moderate",
  "assumptions": ["Assumption 1", "Assumption 2"],
  "clarifications_needed": ["Question 1?", "Question 2?"],
  "related_features": []
}

Be thorough but practical. If the description is vague, make reasonable assumptions and note them. Generate at least 3 acceptance criteria for any non-trivial feature."""


class SpecAgent(BaseAgent):
    """Agent for parsing feature descriptions into structured specifications.

    Takes natural language input and produces a FeatureSpec with:
    - User story format
    - Acceptance criteria
    - Domain classification
    - Constraints and NFRs
    - Assumptions and clarifications needed
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize SpecAgent.

        Args:
            llm: LLM backend for inference
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Parse feature description into structured spec.

        Args:
            input_data: Must contain 'feature_description' in context

        Returns:
            AgentOutput with FeatureSpec data
        """
        feature_description = input_data.context.get("feature_description", "")
        run_id = input_data.context.get("run_id", datetime.now().strftime("%Y%m%d%H%M%S"))

        if not feature_description:
            return AgentOutput(
                success=False,
                data={},
                errors=["No feature description provided"],
            )

        # Build the user message
        user_message = f"""Parse this feature description into a structured specification:

FEATURE DESCRIPTION:
{feature_description}

Remember to respond with ONLY valid JSON matching the required structure."""

        try:
            # Call LLM
            response = self._chat(user_message, temperature=0.3)

            # Parse the response
            spec_data = self._parse_response(response)

            # Add metadata
            spec_data["id"] = f"FEAT-{run_id}"
            spec_data["original_description"] = feature_description

            # Validate and create FeatureSpec
            feature_spec = self._create_feature_spec(spec_data)

            # Create RequirementsTracker from spec and original prompt
            tracker = self._create_requirements_tracker(
                feature_description,
                feature_spec
            )

            return AgentOutput(
                success=True,
                data={
                    **feature_spec.model_dump(),
                    "requirements_tracker": tracker.model_dump(),
                },
                artifacts=["feature_spec.json", "requirements_tracker.json"],
            )

        except json.JSONDecodeError as e:
            return AgentOutput(
                success=False,
                data={"raw_response": response if 'response' in dir() else ""},
                errors=[f"Failed to parse LLM response as JSON: {e}"],
            )
        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"SpecAgent error: {str(e)}"],
            )

    def _parse_response(self, response: str) -> dict:
        """Extract JSON from LLM response with repair.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed even after repair
        """
        result = parse_llm_json(response, default=None)

        if result is None:
            raise json.JSONDecodeError(
                "Could not parse JSON from response",
                response,
                0,
            )

        return result

    def _create_feature_spec(self, data: dict) -> FeatureSpec:
        """Create validated FeatureSpec from parsed data.

        Args:
            data: Parsed JSON data

        Returns:
            Validated FeatureSpec
        """
        # Parse acceptance criteria
        acceptance_criteria = []
        for ac in data.get("acceptance_criteria", []):
            acceptance_criteria.append(
                AcceptanceCriteria(
                    id=ac.get("id", f"AC-{len(acceptance_criteria)+1:03d}"),
                    description=ac.get("description", ""),
                    testable=ac.get("testable", True),
                    verification_hint=ac.get("verification_hint"),
                )
            )

        # Parse constraints
        constraints = []
        for c in data.get("constraints", []):
            constraints.append(
                Constraint(
                    description=c.get("description", ""),
                    type=c.get("type", "technical"),
                    impact=c.get("impact"),
                )
            )

        # Parse NFRs
        nfrs = []
        for nfr in data.get("non_functional_requirements", []):
            nfrs.append(
                NonFunctionalRequirement(
                    category=nfr.get("category", ""),
                    requirement=nfr.get("requirement", ""),
                    metric=nfr.get("metric"),
                    threshold=nfr.get("threshold"),
                )
            )

        # Parse domains
        domains = []
        for d in data.get("domains", []):
            try:
                domains.append(Domain(d.lower()))
            except ValueError:
                pass  # Skip invalid domains

        # Parse priority
        try:
            priority = Priority(data.get("priority", "medium").lower())
        except ValueError:
            priority = Priority.MEDIUM

        return FeatureSpec(
            id=data.get("id", "FEAT-000"),
            title=data.get("title", "Untitled Feature"),
            original_description=data.get("original_description", ""),
            user_story=data.get("user_story", ""),
            acceptance_criteria=acceptance_criteria,
            constraints=constraints,
            non_functional_requirements=nfrs,
            domains=domains,
            priority=priority,
            assumptions=data.get("assumptions", []),
            clarifications_needed=data.get("clarifications_needed", []),
            estimated_complexity=data.get("estimated_complexity"),
            related_features=data.get("related_features", []),
        )

    def _create_requirements_tracker(
        self,
        original_prompt: str,
        feature_spec: FeatureSpec
    ) -> RequirementsTracker:
        """Create RequirementsTracker from the spec and original prompt.

        Extracts trackable requirements from:
        1. Acceptance criteria in the spec
        2. Direct action items found in the original prompt
        3. Documentation requirements implied by the prompt

        Args:
            original_prompt: The original user prompt
            feature_spec: The parsed feature specification

        Returns:
            RequirementsTracker populated with requirements
        """
        tracker = RequirementsTracker(original_prompt=original_prompt)

        # 1. Convert acceptance criteria to requirements
        for ac in feature_spec.acceptance_criteria:
            req_type = self._infer_requirement_type(ac.description)
            tracker.add_requirement(
                description=ac.description,
                req_type=req_type,
                priority=1,
                source_text=ac.description,
                acceptance_criteria_id=ac.id,
            )

        # 2. Extract direct action items from the prompt
        action_items = self._extract_action_items(original_prompt)
        for item in action_items:
            # Don't add duplicates
            is_duplicate = any(
                item.lower() in r.description.lower() or
                r.description.lower() in item.lower()
                for r in tracker.requirements
            )
            if not is_duplicate:
                req_type = self._infer_requirement_type(item)
                tracker.add_requirement(
                    description=item,
                    req_type=req_type,
                    priority=1,
                    source_text=item,
                )

        # 3. Add documentation requirement if docs domain is affected
        if Domain.DOCS in feature_spec.domains:
            has_doc_req = any(
                r.type == RequirementType.DOCUMENTATION
                for r in tracker.requirements
            )
            if not has_doc_req:
                tracker.add_requirement(
                    description="Update documentation to reflect changes",
                    req_type=RequirementType.DOCUMENTATION,
                    priority=2,
                )

        return tracker

    def _extract_action_items(self, prompt: str) -> list[str]:
        """Extract explicit action items from the user prompt.

        Looks for patterns like:
        - "fix X", "correct X", "update X"
        - "add X", "create X", "implement X"
        - "remove X", "delete X"
        - Negative statements: "not X", "no X", "don't X"

        Args:
            prompt: The original user prompt

        Returns:
            List of extracted action items
        """
        action_items = []
        prompt_lower = prompt.lower()

        # Patterns for extracting actions
        action_patterns = [
            # Fix/correct patterns
            (r'(?:fix|correct|update|change)\s+(?:the\s+)?(.+?)(?:\.|,|$|\n)', 'fix'),
            # Add/create patterns
            (r'(?:add|create|implement|include)\s+(?:a\s+)?(.+?)(?:\.|,|$|\n)', 'add'),
            # Remove/delete patterns
            (r'(?:remove|delete|get rid of)\s+(?:the\s+)?(.+?)(?:\.|,|$|\n)', 'remove'),
            # Should/must patterns
            (r'(?:should|must|needs? to)\s+(?:be\s+)?(.+?)(?:\.|,|$|\n)', 'should'),
            # Negative patterns - things that should NOT happen
            (r"(?:don'?t|do not|should not|shouldn'?t|no)\s+(.+?)(?:\.|,|$|\n)", 'not'),
            # "refers to X that does not exist" pattern
            (r'refers?\s+to\s+(?:a\s+)?(.+?)\s+that\s+does\s+not\s+exist', 'missing'),
            # "there is no X" pattern
            (r'there\s+is\s+no\s+(.+?)(?:\.|,|$|\n)', 'missing'),
        ]

        for pattern, action_type in action_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                match = match.strip()
                if len(match) > 5 and len(match) < 200:  # Reasonable length
                    if action_type == 'fix':
                        action_items.append(f"Fix: {match}")
                    elif action_type == 'add':
                        action_items.append(f"Add: {match}")
                    elif action_type == 'remove':
                        action_items.append(f"Remove: {match}")
                    elif action_type == 'should':
                        action_items.append(f"Ensure: {match}")
                    elif action_type == 'not':
                        action_items.append(f"Ensure NOT: {match}")
                    elif action_type == 'missing':
                        action_items.append(f"Fix missing: {match}")

        return action_items

    def _infer_requirement_type(self, description: str) -> RequirementType:
        """Infer the requirement type from its description.

        Args:
            description: The requirement description

        Returns:
            Inferred RequirementType
        """
        desc_lower = description.lower()

        # Check for documentation keywords
        doc_keywords = ['readme', 'documentation', 'docs', 'comment', 'docstring']
        if any(kw in desc_lower for kw in doc_keywords):
            return RequirementType.DOCUMENTATION

        # Check for testing keywords
        test_keywords = ['test', 'coverage', 'spec', 'assertion', 'mock']
        if any(kw in desc_lower for kw in test_keywords):
            return RequirementType.TESTING

        # Check for security keywords
        security_keywords = ['security', 'auth', 'permission', 'encrypt', 'password', 'secret']
        if any(kw in desc_lower for kw in security_keywords):
            return RequirementType.SECURITY

        # Check for performance keywords
        perf_keywords = ['performance', 'speed', 'fast', 'slow', 'optimize', 'cache']
        if any(kw in desc_lower for kw in perf_keywords):
            return RequirementType.PERFORMANCE

        # Check for fix/correction keywords
        fix_keywords = ['fix', 'correct', 'wrong', 'error', 'bug', 'issue', 'missing']
        if any(kw in desc_lower for kw in fix_keywords):
            return RequirementType.FIX

        # Check for removal keywords
        remove_keywords = ['remove', 'delete', 'get rid', 'not']
        if any(kw in desc_lower for kw in remove_keywords):
            return RequirementType.REMOVAL

        # Check for style/UI keywords
        style_keywords = ['style', 'color', 'layout', 'design', 'ui', 'ux', 'display']
        if any(kw in desc_lower for kw in style_keywords):
            return RequirementType.STYLE

        # Default to functional
        return RequirementType.FUNCTIONAL

    def parse_feature(self, description: str, run_id: str | None = None) -> FeatureSpec | None:
        """Convenience method to parse a feature description.

        Args:
            description: Natural language feature description
            run_id: Optional run ID for the feature

        Returns:
            FeatureSpec if successful, None otherwise
        """
        input_data = AgentInput(
            context={
                "feature_description": description,
                "run_id": run_id or datetime.now().strftime("%Y%m%d%H%M%S"),
            }
        )

        output = self.run(input_data)

        if output.success:
            return FeatureSpec(**output.data)
        return None
