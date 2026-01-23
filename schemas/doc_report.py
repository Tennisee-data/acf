"""Documentation report schema.

Defines output structure for DocAgent including generated docs,
docstrings, ADRs, and sync status.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    """Type of documentation."""

    README = "readme"
    ARCHITECTURE = "architecture"
    API_REFERENCE = "api_reference"
    GETTING_STARTED = "getting_started"
    HOW_TO_RUN = "how_to_run"
    HOW_TO_TEST = "how_to_test"
    CONTRIBUTING = "contributing"
    CHANGELOG = "changelog"
    ADR = "adr"


class DocstringStyle(str, Enum):
    """Docstring format style."""

    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"


class SyncStatus(str, Enum):
    """Sync status between documentation sources."""

    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    MISSING = "missing"
    UPDATED = "updated"


class GeneratedDoc(BaseModel):
    """A generated documentation file."""

    path: str = Field(..., description="Relative file path")
    doc_type: DocType = Field(..., description="Type of documentation")
    content: str = Field(..., description="Generated content")
    description: str = Field(..., description="Brief description of the doc")
    is_new: bool = Field(True, description="Whether this is a new file")
    sections_updated: list[str] = Field(
        default_factory=list,
        description="Sections that were updated (if not new)",
    )


class DocstringUpdate(BaseModel):
    """A docstring addition or update."""

    file_path: str = Field(..., description="File containing the function/class")
    symbol_name: str = Field(..., description="Function or class name")
    symbol_type: str = Field(..., description="'function', 'class', or 'method'")
    line_number: int = Field(..., description="Line number of the symbol")
    old_docstring: str | None = Field(None, description="Previous docstring if any")
    new_docstring: str = Field(..., description="Generated docstring")
    style: DocstringStyle = Field(
        DocstringStyle.GOOGLE,
        description="Docstring style used",
    )


class ADRRecord(BaseModel):
    """An Architecture Decision Record."""

    number: int = Field(..., description="ADR number (e.g., 001)")
    title: str = Field(..., description="Decision title")
    status: str = Field("accepted", description="Status: proposed, accepted, deprecated")
    context: str = Field(..., description="Why this decision was needed")
    decision: str = Field(..., description="What was decided")
    consequences: str = Field(..., description="Positive and negative consequences")
    file_path: str = Field(..., description="Path to the ADR file")


class SyncCheck(BaseModel):
    """Check for documentation sync between sources."""

    source: str = Field(..., description="Source document (e.g., 'spec.json')")
    target: str = Field(..., description="Target document (e.g., 'README.md')")
    status: SyncStatus = Field(..., description="Sync status")
    discrepancies: list[str] = Field(
        default_factory=list,
        description="List of discrepancies found",
    )
    auto_fixed: bool = Field(False, description="Whether discrepancies were auto-fixed")


class MissingDocstring(BaseModel):
    """A public symbol missing documentation."""

    file_path: str = Field(..., description="File containing the symbol")
    symbol_name: str = Field(..., description="Function or class name")
    symbol_type: str = Field(..., description="'function', 'class', or 'method'")
    line_number: int = Field(..., description="Line number")
    signature: str = Field(..., description="Function/method signature")


class DocReport(BaseModel):
    """Report from the DocAgent."""

    # Generated documentation files
    docs_generated: list[GeneratedDoc] = Field(
        default_factory=list,
        description="Documentation files generated",
    )
    docs_updated: list[GeneratedDoc] = Field(
        default_factory=list,
        description="Documentation files updated",
    )

    # Docstrings
    docstrings_added: list[DocstringUpdate] = Field(
        default_factory=list,
        description="Docstrings added to functions/classes",
    )
    missing_docstrings: list[MissingDocstring] = Field(
        default_factory=list,
        description="Public symbols still missing docstrings",
    )
    docstring_style: DocstringStyle = Field(
        DocstringStyle.GOOGLE,
        description="Docstring style used",
    )

    # ADRs
    adrs_created: list[ADRRecord] = Field(
        default_factory=list,
        description="Architecture Decision Records created",
    )
    total_adrs: int = Field(0, description="Total ADRs in the project")

    # Sync status
    sync_checks: list[SyncCheck] = Field(
        default_factory=list,
        description="Documentation sync checks performed",
    )
    all_in_sync: bool = Field(True, description="Whether all docs are in sync")

    # Summary counts
    total_docs_generated: int = Field(0, description="Total new docs created")
    total_docs_updated: int = Field(0, description="Total docs updated")
    total_docstrings_added: int = Field(0, description="Total docstrings added")
    coverage_before: float = Field(0.0, description="Docstring coverage before")
    coverage_after: float = Field(0.0, description="Docstring coverage after")

    # Summary
    summary: str = Field("", description="Human-readable summary")

    def update_counts(self) -> None:
        """Update summary counts."""
        self.total_docs_generated = len(self.docs_generated)
        self.total_docs_updated = len(self.docs_updated)
        self.total_docstrings_added = len(self.docstrings_added)
        self.all_in_sync = all(
            check.status in (SyncStatus.IN_SYNC, SyncStatus.UPDATED)
            for check in self.sync_checks
        )

    class Config:
        json_schema_extra = {
            "example": {
                "total_docs_generated": 3,
                "total_docs_updated": 1,
                "total_docstrings_added": 15,
                "coverage_after": 85.0,
                "all_in_sync": True,
            }
        }
