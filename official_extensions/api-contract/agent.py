"""API Contract Agent - Define API boundaries before implementation.

This official ACF extension generates API contracts (endpoints, request/response
schemas) before implementation to ensure consistent interfaces.
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


class ApiContractAgent:
    """Define API boundaries and contracts before implementation.

    Runs before implementation to generate clear API contracts
    that guide the implementation and ensure consistency.
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.llm = llm
        self.name = "api-contract"

    def default_system_prompt(self) -> str:
        return """You are an API architect. Define clear API contracts based on the feature design.

For each API endpoint, specify:
- HTTP method and path
- Request parameters/body schema
- Response schema
- Error responses
- Authentication requirements

Output JSON format:
{
    "endpoints": [
        {
            "method": "POST",
            "path": "/api/v1/resource",
            "summary": "Create a new resource",
            "auth_required": true,
            "request_body": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "required": true}
                }
            },
            "responses": {
                "201": {"description": "Created", "schema": {...}},
                "400": {"description": "Validation error"},
                "401": {"description": "Unauthorized"}
            }
        }
    ],
    "schemas": {
        "Resource": {
            "type": "object",
            "properties": {...}
        }
    }
}"""

    def run(self, input_data: Any) -> AgentOutput:
        """Generate API contracts from design.

        Args:
            input_data: Pipeline context with design proposal

        Returns:
            AgentOutput with API contracts
        """
        context = input_data.context if hasattr(input_data, 'context') else input_data

        # Get design proposal
        design = context.get("design", {})
        spec = context.get("spec", {})

        feature_title = spec.get("title", "Unknown feature")
        design_approach = design.get("approach", "")
        file_changes = design.get("file_changes", [])

        # Check if this feature involves API endpoints
        is_api_feature = any(
            keyword in feature_title.lower() or keyword in design_approach.lower()
            for keyword in ["api", "endpoint", "route", "rest", "http", "request"]
        )

        if not is_api_feature:
            print(f"\n[API Contract] Skipping: not an API feature")
            return AgentOutput(
                success=True,
                data={"api_contract": {"skipped": True, "reason": "Not an API feature"}},
                agent_name=self.name,
            )

        prompt = f"""Generate API contracts for this feature:

## Feature: {feature_title}

## Design Approach:
{design_approach}

## Files to be changed:
{json.dumps(file_changes, indent=2) if file_changes else "Not specified"}

Generate clear API contracts. Respond with JSON only."""

        try:
            messages = [
                {"role": "system", "content": self.default_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages)

            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"endpoints": [], "schemas": {}}

            endpoints = result.get("endpoints", [])
            schemas = result.get("schemas", {})

            print(f"\n[API Contract] Generated {len(endpoints)} endpoint(s)")
            for ep in endpoints:
                print(f"  {ep.get('method', '?')} {ep.get('path', '?')}")

            return AgentOutput(
                success=True,
                data={
                    "api_contract": {
                        "endpoints": endpoints,
                        "schemas": schemas,
                        "endpoint_count": len(endpoints),
                    }
                },
                artifacts=["api_contract.json"] if endpoints else None,
                agent_name=self.name,
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"API contract generation failed: {str(e)}"],
                agent_name=self.name,
            )
