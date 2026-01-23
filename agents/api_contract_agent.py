"""API Contract Agent for defining API boundaries before implementation.

Takes spec + design and generates:
- OpenAPI specification (openapi.yaml)
- Pydantic models (schemas.py)
- Validation of naming, versioning, documentation
"""

import json
import re

from llm_backend import LLMBackend
from schemas.api_contract import (
    APIContract,
    DataType,
    Endpoint,
    HTTPMethod,
    Parameter,
    ParameterLocation,
    PydanticModel,
    Response,
    SchemaField,
    ValidationIssue,
)
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent

SYSTEM_PROMPT = """You are an API architect specializing in REST API design. Your job is to define clear API contracts before implementation begins.

Given a feature spec and design proposal, you must generate:

1. **Pydantic Models**: Request/response schemas
   - Use PascalCase for model names
   - Include field descriptions
   - Mark required vs optional fields
   - Add validation constraints where appropriate

2. **API Endpoints**: RESTful endpoints
   - Use consistent naming (plural nouns for collections)
   - Include proper HTTP methods (GET, POST, PUT, PATCH, DELETE)
   - Version the API (e.g., /api/v1/...)
   - Generate unique operation_ids (snake_case)

3. **Validation**: Check for issues
   - Inconsistent naming
   - Missing documentation
   - Missing error responses
   - Security concerns

NAMING CONVENTIONS:
- Endpoints: /api/v1/{resource}s (plural, lowercase, kebab-case for multi-word)
- Models: {Resource}Create, {Resource}Update, {Resource}Response
- Operations: {verb}_{resource} (e.g., create_user, get_users, delete_user)

REQUIRED RESPONSES:
- 200/201 for success
- 400 for validation errors
- 401 for auth required
- 404 for not found
- 500 for server errors

IMPORTANT: Respond ONLY with valid JSON matching this structure:
{
  "title": "API title",
  "version": "1.0.0",
  "description": "API description",
  "base_path": "/api/v1",
  "models": [
    {
      "name": "ModelName",
      "description": "Model description",
      "fields": [
        {
          "name": "field_name",
          "type": "string|integer|number|boolean|array|object",
          "description": "Field description",
          "required": true,
          "example": "example value"
        }
      ]
    }
  ],
  "endpoints": [
    {
      "path": "/resources",
      "method": "get|post|put|patch|delete",
      "operation_id": "operation_name",
      "summary": "Short summary",
      "description": "Detailed description",
      "tags": ["tag1"],
      "parameters": [
        {
          "name": "param_name",
          "location": "path|query|header",
          "type": "string",
          "required": true
        }
      ],
      "request_body_ref": "ModelName",
      "responses": [
        {"status_code": 200, "description": "Success", "schema_ref": "ResponseModel"},
        {"status_code": 400, "description": "Validation error"}
      ],
      "requires_auth": false
    }
  ],
  "validation_issues": [
    {
      "severity": "warning|error|info",
      "location": "endpoint:/path or model:Name",
      "message": "Issue description",
      "suggestion": "How to fix"
    }
  ],
  "implementation_notes": ["Note for ImplementationAgent"]
}

Be thorough. Define all models needed for requests and responses. Include standard error responses."""


class APIContractAgent(BaseAgent):
    """Agent for generating API contracts.

    Takes feature spec and design proposal, outputs:
    - OpenAPI specification
    - Pydantic model definitions
    - Validation issues
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize APIContractAgent.

        Args:
            llm: LLM backend for inference
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate API contract from spec and design.

        Args:
            input_data: Must contain 'feature_spec' and 'design_proposal' in context

        Returns:
            AgentOutput with APIContract data
        """
        feature_spec = input_data.context.get("feature_spec", {})
        design_proposal = input_data.context.get("design_proposal", {})
        workplan = input_data.context.get("workplan", {})

        if not feature_spec:
            return AgentOutput(
                success=False,
                data={},
                errors=["No feature spec provided"],
            )

        # Build the prompt
        user_message = self._build_prompt(feature_spec, design_proposal, workplan)

        try:
            # Call LLM
            response = self._chat(user_message, temperature=0.2)

            # Parse the response
            contract_data = self._parse_response(response)

            # Create and validate contract
            contract = self._create_contract(contract_data)

            # Run additional validation
            self._validate_contract(contract)

            return AgentOutput(
                success=True,
                data=contract.model_dump(),
                artifacts=["api_contract.json", "openapi.yaml", "schemas.py"],
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
                errors=[f"APIContractAgent error: {str(e)}"],
            )

    def _build_prompt(
        self,
        spec: dict,
        design: dict,
        workplan: dict,
    ) -> str:
        """Build the prompt for contract generation."""
        # Extract key information
        title = spec.get("title", "API")
        description = spec.get("original_description", "")
        user_story = spec.get("user_story", "")

        # Extract acceptance criteria
        ac_list = ""
        for ac in spec.get("acceptance_criteria", []):
            ac_list += f"  - {ac.get('id', 'AC')}: {ac.get('description', '')}\n"

        # Extract design info
        design_summary = design.get("summary", "")
        file_changes = design.get("file_changes", [])
        api_files = [f for f in file_changes if "route" in str(f).lower() or "api" in str(f).lower()]

        # Extract workplan tasks related to API
        api_tasks = ""
        for task in workplan.get("tasks", []):
            if task.get("category") in ["api", "backend"]:
                api_tasks += f"  - {task.get('title', '')}\n"

        return f"""Design an API contract for this feature:

FEATURE: {title}

DESCRIPTION:
{description}

USER STORY:
{user_story}

ACCEPTANCE CRITERIA:
{ac_list or "  None specified"}

DESIGN SUMMARY:
{design_summary or "Not provided"}

API-RELATED FILES FROM DESIGN:
{chr(10).join(str(f) for f in api_files) or "  None identified"}

API-RELATED TASKS:
{api_tasks or "  None identified"}

Generate a complete API contract with:
1. All necessary Pydantic models (request/response)
2. RESTful endpoints with proper methods
3. Standard error responses
4. Validation issues if any

Respond with ONLY valid JSON matching the required structure."""

    def _parse_response(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        result = parse_llm_json(response, default=None)

        if result is None:
            raise json.JSONDecodeError(
                "Could not parse JSON from response",
                response,
                0,
            )

        return result

    def _create_contract(self, data: dict) -> APIContract:
        """Create validated APIContract from parsed data."""
        # Parse models
        models = []
        for model_data in data.get("models", []):
            fields = []
            for field_data in model_data.get("fields", []):
                try:
                    field_type = DataType(field_data.get("type", "string").lower())
                except ValueError:
                    field_type = DataType.STRING

                fields.append(
                    SchemaField(
                        name=field_data.get("name", ""),
                        type=field_type,
                        description=field_data.get("description", ""),
                        required=field_data.get("required", True),
                        default=field_data.get("default"),
                        example=field_data.get("example"),
                    )
                )

            models.append(
                PydanticModel(
                    name=model_data.get("name", "Model"),
                    description=model_data.get("description", ""),
                    fields=fields,
                )
            )

        # Parse endpoints
        endpoints = []
        for ep_data in data.get("endpoints", []):
            # Parse method
            try:
                method = HTTPMethod(ep_data.get("method", "get").lower())
            except ValueError:
                method = HTTPMethod.GET

            # Parse parameters
            parameters = []
            for param_data in ep_data.get("parameters", []):
                try:
                    location = ParameterLocation(
                        param_data.get("location", "query").lower()
                    )
                except ValueError:
                    location = ParameterLocation.QUERY

                try:
                    param_type = DataType(param_data.get("type", "string").lower())
                except ValueError:
                    param_type = DataType.STRING

                parameters.append(
                    Parameter(
                        name=param_data.get("name", ""),
                        location=location,
                        type=param_type,
                        description=param_data.get("description", ""),
                        required=param_data.get("required", True),
                        example=param_data.get("example"),
                    )
                )

            # Parse responses
            responses = []
            for resp_data in ep_data.get("responses", []):
                responses.append(
                    Response(
                        status_code=resp_data.get("status_code", 200),
                        description=resp_data.get("description", ""),
                        schema_ref=resp_data.get("schema_ref"),
                        example=resp_data.get("example"),
                    )
                )

            endpoints.append(
                Endpoint(
                    path=ep_data.get("path", "/"),
                    method=method,
                    operation_id=ep_data.get("operation_id", ""),
                    summary=ep_data.get("summary", ""),
                    description=ep_data.get("description", ""),
                    tags=ep_data.get("tags", []),
                    parameters=parameters,
                    request_body_ref=ep_data.get("request_body_ref"),
                    request_body_required=ep_data.get("request_body_required", True),
                    responses=responses,
                    requires_auth=ep_data.get("requires_auth", False),
                    required_scopes=ep_data.get("required_scopes", []),
                )
            )

        # Parse validation issues
        issues = []
        for issue_data in data.get("validation_issues", []):
            issues.append(
                ValidationIssue(
                    severity=issue_data.get("severity", "warning"),
                    location=issue_data.get("location", ""),
                    message=issue_data.get("message", ""),
                    suggestion=issue_data.get("suggestion", ""),
                )
            )

        return APIContract(
            title=data.get("title", "API"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            base_path=data.get("base_path", "/api/v1"),
            models=models,
            endpoints=endpoints,
            validation_issues=issues,
            implementation_notes=data.get("implementation_notes", []),
        )

    def _validate_contract(self, contract: APIContract) -> None:
        """Run additional validation on the contract."""
        issues = list(contract.validation_issues)

        # Check endpoint naming conventions
        for ep in contract.endpoints:
            # Check path format
            if not ep.path.startswith("/"):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        location=f"endpoint:{ep.path}",
                        message="Path must start with /",
                        suggestion=f"Change to /{ep.path}",
                    )
                )

            # Check operation_id format (should be snake_case)
            if ep.operation_id and not re.match(r"^[a-z][a-z0-9_]*$", ep.operation_id):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        location=f"endpoint:{ep.path}",
                        message=f"operation_id '{ep.operation_id}' should be snake_case",
                        suggestion="Use format: verb_resource (e.g., create_user)",
                    )
                )

            # Check for missing responses
            status_codes = {r.status_code for r in ep.responses}
            if ep.method == HTTPMethod.POST and 201 not in status_codes:
                if 200 not in status_codes:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            location=f"endpoint:{ep.path}",
                            message="POST endpoint missing success response",
                            suggestion="Add 201 Created response",
                        )
                    )

            if 400 not in status_codes:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        location=f"endpoint:{ep.path}",
                        message="Missing 400 Bad Request response",
                        suggestion="Add validation error response",
                    )
                )

        # Check model naming conventions
        for model in contract.models:
            if not re.match(r"^[A-Z][a-zA-Z0-9]*$", model.name):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        location=f"model:{model.name}",
                        message=f"Model name '{model.name}' should be PascalCase",
                        suggestion="Use format: UserCreate, UserResponse",
                    )
                )

            # Check for empty descriptions
            if not model.description:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        location=f"model:{model.name}",
                        message="Model missing description",
                        suggestion="Add a docstring describing the model",
                    )
                )

        contract.validation_issues = issues

    def generate_openapi_yaml(self, contract: APIContract) -> str:
        """Generate OpenAPI 3.0 YAML from contract."""
        import yaml

        openapi = {
            "openapi": "3.0.3",
            "info": {
                "title": contract.title,
                "version": contract.version,
                "description": contract.description,
            },
            "paths": {},
            "components": {
                "schemas": {},
            },
        }

        # Add schemas
        for model in contract.models:
            schema = {
                "type": "object",
                "description": model.description,
                "properties": {},
                "required": [],
            }

            for field in model.fields:
                prop = {"type": field.type.value}
                if field.description:
                    prop["description"] = field.description
                if field.example:
                    prop["example"] = field.example

                schema["properties"][field.name] = prop

                if field.required:
                    schema["required"].append(field.name)

            openapi["components"]["schemas"][model.name] = schema

        # Add paths
        for ep in contract.endpoints:
            path = contract.base_path + ep.path
            if path not in openapi["paths"]:
                openapi["paths"][path] = {}

            operation = {
                "operationId": ep.operation_id,
                "summary": ep.summary,
                "description": ep.description,
                "tags": ep.tags,
                "responses": {},
            }

            # Add parameters
            if ep.parameters:
                operation["parameters"] = []
                for param in ep.parameters:
                    operation["parameters"].append({
                        "name": param.name,
                        "in": param.location.value,
                        "required": param.required,
                        "schema": {"type": param.type.value},
                        "description": param.description,
                    })

            # Add request body
            if ep.request_body_ref:
                operation["requestBody"] = {
                    "required": ep.request_body_required,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{ep.request_body_ref}"
                            }
                        }
                    },
                }

            # Add responses
            for resp in ep.responses:
                resp_obj = {"description": resp.description}
                if resp.schema_ref:
                    resp_obj["content"] = {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{resp.schema_ref}"
                            }
                        }
                    }
                operation["responses"][str(resp.status_code)] = resp_obj

            openapi["paths"][path][ep.method.value] = operation

        return yaml.dump(openapi, default_flow_style=False, sort_keys=False)

    def generate_schemas_py(self, contract: APIContract) -> str:
        """Generate Pydantic models Python code from contract."""
        lines = [
            '"""Auto-generated Pydantic models from API contract.',
            "",
            "DO NOT EDIT DIRECTLY - regenerate from api_contract.json",
            '"""',
            "",
            "from pydantic import BaseModel, Field",
            "",
            "",
        ]

        for model in contract.models:
            # Class definition
            lines.append(f"class {model.name}(BaseModel):")

            # Docstring
            if model.description:
                lines.append(f'    """{model.description}"""')
                lines.append("")

            # Fields
            if not model.fields:
                lines.append("    pass")
            else:
                for field in model.fields:
                    field_type = self._python_type(field.type)

                    # Build Field() arguments
                    field_args = []
                    if field.description:
                        field_args.append(f'description="{field.description}"')
                    if field.example:
                        field_args.append(f'example="{field.example}"')
                    if field.default is not None:
                        field_args.append(f'default="{field.default}"')

                    if not field.required:
                        field_type = f"{field_type} | None"
                        if field.default is None:
                            field_args.insert(0, "None")

                    if field_args:
                        field_def = f"Field({', '.join(field_args)})"
                        lines.append(f"    {field.name}: {field_type} = {field_def}")
                    else:
                        if field.required:
                            lines.append(f"    {field.name}: {field_type}")
                        else:
                            lines.append(f"    {field.name}: {field_type} = None")

            lines.append("")
            lines.append("")

        return "\n".join(lines)

    def _python_type(self, data_type: DataType) -> str:
        """Convert DataType to Python type annotation."""
        type_map = {
            DataType.STRING: "str",
            DataType.INTEGER: "int",
            DataType.NUMBER: "float",
            DataType.BOOLEAN: "bool",
            DataType.ARRAY: "list",
            DataType.OBJECT: "dict",
        }
        return type_map.get(data_type, "str")
