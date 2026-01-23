"""API Contract schema for defining API boundaries.

Output of APIContractAgent: OpenAPI spec and Pydantic models
that ImplementationAgent must respect.
"""

from enum import Enum

from pydantic import BaseModel, Field


class HTTPMethod(str, Enum):
    """HTTP methods for API endpoints."""

    GET = "get"
    POST = "post"
    PUT = "put"
    PATCH = "patch"
    DELETE = "delete"


class ParameterLocation(str, Enum):
    """Where the parameter appears."""

    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    BODY = "body"


class DataType(str, Enum):
    """Common data types for schemas."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class SchemaField(BaseModel):
    """A single field in a schema/model."""

    name: str = Field(..., description="Field name")
    type: DataType = Field(..., description="Data type")
    description: str = Field("", description="Field description")
    required: bool = Field(True, description="Is field required")
    default: str | None = Field(None, description="Default value if any")
    example: str | None = Field(None, description="Example value")
    # For arrays
    items_type: DataType | None = Field(None, description="Type of array items")
    # For nested objects
    nested_fields: list["SchemaField"] = Field(
        default_factory=list,
        description="Nested fields for object types",
    )
    # Validation
    min_length: int | None = Field(None, description="Min length for strings")
    max_length: int | None = Field(None, description="Max length for strings")
    minimum: float | None = Field(None, description="Min value for numbers")
    maximum: float | None = Field(None, description="Max value for numbers")
    pattern: str | None = Field(None, description="Regex pattern for strings")


class PydanticModel(BaseModel):
    """A Pydantic model definition."""

    name: str = Field(..., description="Model class name (PascalCase)")
    description: str = Field("", description="Model docstring")
    fields: list[SchemaField] = Field(
        default_factory=list,
        description="Model fields",
    )
    base_class: str = Field("BaseModel", description="Base class to inherit from")


class Parameter(BaseModel):
    """API endpoint parameter."""

    name: str = Field(..., description="Parameter name")
    location: ParameterLocation = Field(..., description="Where parameter appears")
    type: DataType = Field(DataType.STRING, description="Parameter type")
    description: str = Field("", description="Parameter description")
    required: bool = Field(True, description="Is parameter required")
    example: str | None = Field(None, description="Example value")


class Response(BaseModel):
    """API response definition."""

    status_code: int = Field(..., description="HTTP status code")
    description: str = Field("", description="Response description")
    schema_ref: str | None = Field(None, description="Reference to response model")
    example: dict | None = Field(None, description="Example response body")


class Endpoint(BaseModel):
    """API endpoint definition."""

    path: str = Field(..., description="URL path (e.g., /api/v1/users/{id})")
    method: HTTPMethod = Field(..., description="HTTP method")
    operation_id: str = Field(..., description="Unique operation ID (snake_case)")
    summary: str = Field("", description="Short summary")
    description: str = Field("", description="Detailed description")
    tags: list[str] = Field(default_factory=list, description="API tags for grouping")

    # Parameters
    parameters: list[Parameter] = Field(
        default_factory=list,
        description="Path, query, header parameters",
    )

    # Request body
    request_body_ref: str | None = Field(
        None,
        description="Reference to request body model",
    )
    request_body_required: bool = Field(True, description="Is request body required")

    # Responses
    responses: list[Response] = Field(
        default_factory=list,
        description="Possible responses",
    )

    # Security
    requires_auth: bool = Field(False, description="Requires authentication")
    required_scopes: list[str] = Field(
        default_factory=list,
        description="Required OAuth scopes",
    )


class ValidationIssue(BaseModel):
    """An issue found during contract validation."""

    severity: str = Field("warning", description="error | warning | info")
    location: str = Field(..., description="Where the issue was found")
    message: str = Field(..., description="Description of the issue")
    suggestion: str = Field("", description="How to fix it")


class APIContract(BaseModel):
    """Complete API contract definition.

    Output of APIContractAgent - defines the API boundaries
    that ImplementationAgent must respect.
    """

    # Metadata
    title: str = Field(..., description="API title")
    version: str = Field("1.0.0", description="API version")
    description: str = Field("", description="API description")
    base_path: str = Field("/api/v1", description="Base path for all endpoints")

    # Schemas/Models
    models: list[PydanticModel] = Field(
        default_factory=list,
        description="Pydantic models for request/response bodies",
    )

    # Endpoints
    endpoints: list[Endpoint] = Field(
        default_factory=list,
        description="API endpoints",
    )

    # Tags for grouping
    tags: list[dict] = Field(
        default_factory=list,
        description="Tag definitions with descriptions",
    )

    # Security schemes
    security_schemes: dict = Field(
        default_factory=dict,
        description="Security scheme definitions (OAuth, API key, etc.)",
    )

    # Validation results
    validation_issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="Issues found during validation",
    )

    # Generation notes
    implementation_notes: list[str] = Field(
        default_factory=list,
        description="Notes for ImplementationAgent",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "User Management API",
                "version": "1.0.0",
                "base_path": "/api/v1",
                "models": [
                    {
                        "name": "UserCreate",
                        "description": "Request body for creating a user",
                        "fields": [
                            {"name": "email", "type": "string", "required": True},
                            {"name": "password", "type": "string", "required": True},
                        ],
                    },
                    {
                        "name": "UserResponse",
                        "description": "User response model",
                        "fields": [
                            {"name": "id", "type": "integer", "required": True},
                            {"name": "email", "type": "string", "required": True},
                        ],
                    },
                ],
                "endpoints": [
                    {
                        "path": "/users",
                        "method": "post",
                        "operation_id": "create_user",
                        "summary": "Create a new user",
                        "request_body_ref": "UserCreate",
                        "responses": [
                            {"status_code": 201, "schema_ref": "UserResponse"},
                            {"status_code": 400, "description": "Validation error"},
                        ],
                    }
                ],
            }
        }
