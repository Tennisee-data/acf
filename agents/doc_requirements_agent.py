"""Documentation Requirements Agent - Detects when external docs are needed.

Uses inverse abstraction: maintains an allowlist of known-safe libraries.
Any library/API NOT in the allowlist requires external documentation.

This prevents hallucination when generating code for proprietary APIs
like Stripe, Twilio, Shopify, etc. that the LLM may not know correctly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "context_libraries.cfg"


@dataclass
class DocumentationRequirement:
    """A library/API that requires external documentation."""

    name: str
    confidence: float  # How confident we are this is a real library reference
    context: str  # Where it was detected (prompt, spec, etc.)
    suggestion: str = ""  # Suggested action


@dataclass
class DocumentationReport:
    """Report of documentation requirements for a feature."""

    feature_description: str
    detected_libraries: set[str] = field(default_factory=set)
    known_libraries: set[str] = field(default_factory=set)
    unknown_libraries: set[str] = field(default_factory=set)
    requirements: list[DocumentationRequirement] = field(default_factory=list)

    @property
    def needs_documentation(self) -> bool:
        """Return True if any unknown libraries were detected."""
        return len(self.unknown_libraries) > 0

    @property
    def risk_level(self) -> str:
        """Assess risk level based on unknown libraries."""
        if not self.unknown_libraries:
            return "low"
        # Payment/auth APIs are high risk
        high_risk_keywords = {"stripe", "paypal", "braintree", "adyen", "square",
                             "twilio", "sendgrid", "auth0", "okta", "firebase"}
        if self.unknown_libraries & high_risk_keywords:
            return "high"
        if len(self.unknown_libraries) > 2:
            return "medium"
        return "medium"

    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.needs_documentation:
            return "All detected libraries are known. No additional documentation required."

        lines = [
            f"Documentation required for {len(self.unknown_libraries)} unknown library(ies):",
            "",
        ]
        for lib in sorted(self.unknown_libraries):
            req = next((r for r in self.requirements if r.name == lib), None)
            if req and req.suggestion:
                lines.append(f"  - {lib}: {req.suggestion}")
            else:
                lines.append(f"  - {lib}")

        lines.append("")
        lines.append(f"Risk level: {self.risk_level}")

        if self.risk_level == "high":
            lines.append("")
            lines.append("WARNING: High-risk API detected. Generated code MUST be verified")
            lines.append("against official documentation before use.")

        return "\n".join(lines)


class DocumentationRequirementsAgent:
    """Agent that detects when external documentation is required.

    Uses an allowlist approach: any library not in the known-safe list
    is flagged as requiring documentation.
    """

    # Common API/library indicators in prompts
    LIBRARY_PATTERNS = [
        r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase (e.g., FastAPI, OpenAI)
        r'\b([a-z]+[-_][a-z]+)\b',  # snake_case or kebab-case packages
        r'(?:using|with|via|integrate)\s+(\w+)',  # "using Stripe", "with Twilio"
        r'(\w+)\s+(?:api|sdk|library|webhook|integration)',  # "Stripe API"
        r'(?:api|sdk|library|webhook)\s+(?:for|from)\s+(\w+)',  # "API for Stripe"
    ]

    # Well-known proprietary APIs with documentation URLs
    KNOWN_APIS_DOCS = {
        "stripe": "https://stripe.com/docs/api",
        "twilio": "https://www.twilio.com/docs",
        "sendgrid": "https://docs.sendgrid.com/",
        "paypal": "https://developer.paypal.com/docs/",
        "shopify": "https://shopify.dev/docs/api",
        "firebase": "https://firebase.google.com/docs",
        "aws": "https://docs.aws.amazon.com/",
        "gcp": "https://cloud.google.com/docs",
        "azure": "https://docs.microsoft.com/azure/",
        "openai": "https://platform.openai.com/docs",
        "anthropic": "https://docs.anthropic.com/",
        "slack": "https://api.slack.com/docs",
        "discord": "https://discord.com/developers/docs",
        "github": "https://docs.github.com/rest",
        "gitlab": "https://docs.gitlab.com/ee/api/",
        "jira": "https://developer.atlassian.com/cloud/jira/",
        "notion": "https://developers.notion.com/",
        "airtable": "https://airtable.com/developers/web/api",
        "supabase": "https://supabase.com/docs",
        "auth0": "https://auth0.com/docs/api",
        "okta": "https://developer.okta.com/docs/",
        "plaid": "https://plaid.com/docs/",
        "braintree": "https://developer.paypal.com/braintree/docs",
        "square": "https://developer.squareup.com/docs",
        "adyen": "https://docs.adyen.com/",
        "mailchimp": "https://mailchimp.com/developer/",
        "hubspot": "https://developers.hubspot.com/docs/api",
        "salesforce": "https://developer.salesforce.com/docs",
        "zendesk": "https://developer.zendesk.com/api-reference",
        "intercom": "https://developers.intercom.com/docs",
        "segment": "https://segment.com/docs/",
        "mixpanel": "https://developer.mixpanel.com/docs",
        "amplitude": "https://www.docs.developers.amplitude.com/",
        "datadog": "https://docs.datadoghq.com/api/",
        "sentry": "https://docs.sentry.io/api/",
        "cloudflare": "https://developers.cloudflare.com/api/",
        "vercel": "https://vercel.com/docs/rest-api",
        "netlify": "https://docs.netlify.com/api/",
        "heroku": "https://devcenter.heroku.com/articles/platform-api-reference",
        "digitalocean": "https://docs.digitalocean.com/reference/api/",
        "linode": "https://www.linode.com/docs/api/",
        "mongodb": "https://www.mongodb.com/docs/atlas/api/",
        "algolia": "https://www.algolia.com/doc/api-reference/",
        "elasticsearch": "https://www.elastic.co/guide/en/elasticsearch/reference/",
        "pinecone": "https://docs.pinecone.io/reference",
        "weaviate": "https://weaviate.io/developers/weaviate/api",
        "qdrant": "https://qdrant.tech/documentation/",
    }

    def __init__(self, config_path: Path | str | None = None):
        """Initialize with config file path.

        Args:
            config_path: Path to context_libraries.cfg. Uses default if None.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.known_libraries: set[str] = set()
        self._load_config()

    def _load_config(self) -> None:
        """Load known libraries from config file."""
        if not self.config_path.exists():
            logger.warning(
                "Config file not found: %s. Using empty allowlist.",
                self.config_path
            )
            return

        try:
            content = self.config_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and section headers
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                # Normalize to lowercase
                self.known_libraries.add(line.lower())

            logger.info(
                "Loaded %d known libraries from %s",
                len(self.known_libraries),
                self.config_path
            )
        except Exception as e:
            logger.error("Failed to load config: %s", e)

    def _extract_libraries(self, text: str) -> set[str]:
        """Extract potential library/API names from text.

        Only detects libraries/APIs that are in the KNOWN_APIS_DOCS list.
        This is intentionally conservative - we only flag APIs we know
        are proprietary and need documentation.

        Args:
            text: Text to analyze (prompt, spec, etc.)

        Returns:
            Set of potential library names (lowercase)
        """
        detected = set()

        # Lowercase for matching
        text_lower = text.lower()

        # Only check for known proprietary APIs by name
        # This avoids false positives from generic words
        for api_name in self.KNOWN_APIS_DOCS:
            # Use word boundary matching to avoid partial matches
            # e.g., "stripe" should match but not "stripes"
            if re.search(rf'\b{re.escape(api_name)}\b', text_lower):
                detected.add(api_name)

        return detected

    def analyze(
        self,
        feature_description: str,
        spec_content: str | None = None,
    ) -> DocumentationReport:
        """Analyze feature for documentation requirements.

        Args:
            feature_description: The feature prompt/description
            spec_content: Optional parsed spec content

        Returns:
            DocumentationReport with findings
        """
        # Combine all text for analysis
        full_text = feature_description
        if spec_content:
            full_text += "\n" + spec_content

        # Extract potential libraries
        detected = self._extract_libraries(full_text)

        # Separate known vs unknown
        known = detected & self.known_libraries
        unknown = detected - self.known_libraries

        # Build requirements for unknown libraries
        requirements = []
        for lib in unknown:
            suggestion = ""
            if lib in self.KNOWN_APIS_DOCS:
                suggestion = f"See: {self.KNOWN_APIS_DOCS[lib]}"

            requirements.append(DocumentationRequirement(
                name=lib,
                confidence=0.8 if lib in self.KNOWN_APIS_DOCS else 0.5,
                context="prompt",
                suggestion=suggestion,
            ))

        report = DocumentationReport(
            feature_description=feature_description[:200],
            detected_libraries=detected,
            known_libraries=known,
            unknown_libraries=unknown,
            requirements=requirements,
        )

        if report.needs_documentation:
            logger.warning(
                "Documentation required for: %s (risk: %s)",
                ", ".join(unknown),
                report.risk_level
            )

        return report

    def check_rag_coverage(
        self,
        unknown_libraries: set[str],
        rag_path: Path | None = None,
    ) -> dict[str, bool]:
        """Check if RAG folder has documentation for unknown libraries.

        Args:
            unknown_libraries: Libraries that need docs
            rag_path: Path to RAG folder

        Returns:
            Dict mapping library name to whether docs exist
        """
        if not rag_path or not rag_path.exists():
            return {lib: False for lib in unknown_libraries}

        coverage = {}
        rag_files = list(rag_path.rglob("*"))
        rag_content = " ".join(f.name.lower() for f in rag_files)

        for lib in unknown_libraries:
            # Check if library name appears in any RAG file names
            has_docs = lib in rag_content
            coverage[lib] = has_docs

        return coverage

    def get_missing_docs_prompt(
        self,
        report: DocumentationReport,
        rag_coverage: dict[str, bool] | None = None,
    ) -> str | None:
        """Generate prompt asking user for missing documentation.

        Args:
            report: Documentation report
            rag_coverage: Optional coverage check results

        Returns:
            Prompt string or None if no docs needed
        """
        if not report.needs_documentation:
            return None

        # Filter to libraries without RAG coverage
        missing = report.unknown_libraries
        if rag_coverage:
            missing = {lib for lib in missing if not rag_coverage.get(lib, False)}

        if not missing:
            return None

        lines = [
            "The following APIs/libraries were detected but are not in the known-safe list:",
            "",
        ]

        for lib in sorted(missing):
            req = next((r for r in report.requirements if r.name == lib), None)
            if req and req.suggestion:
                lines.append(f"  - {lib} ({req.suggestion})")
            else:
                lines.append(f"  - {lib}")

        lines.extend([
            "",
            "To ensure correct code generation, please either:",
            "1. Add documentation to your project's RAG folder",
            "2. Paste relevant API documentation or code examples",
            "3. Confirm you accept the risk of potential inaccuracies",
            "",
        ])

        if report.risk_level == "high":
            lines.extend([
                "WARNING: This includes payment/authentication APIs where",
                "incorrect implementation could cause security issues.",
                "",
            ])

        return "\n".join(lines)
