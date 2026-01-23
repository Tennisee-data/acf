# ACF Marketplace Developer Guide

Create and sell extensions for the ACF marketplace. This guide covers everything you need to know to build agents, profiles, and RAG kits.

## Table of Contents

1. [Extension Types](#extension-types)
2. [Project Structure](#project-structure)
3. [Creating an Agent](#creating-an-agent)
4. [Creating a Profile](#creating-a-profile)
5. [Creating a RAG Kit](#creating-a-rag-kit)
6. [Testing Your Extension](#testing-your-extension)
7. [Submitting to Marketplace](#submitting-to-marketplace)
8. [Security Requirements](#security-requirements)
9. [Pricing Guidelines](#pricing-guidelines)

---

## Extension Types

| Type | Purpose | Hook Points | Price Range |
|------|---------|-------------|-------------|
| **Agent** | Add pipeline stages | before/after any stage | Free - $49 |
| **Profile** | Tech stack templates | N/A (loaded at startup) | Free - $15 |
| **RAG Kit** | Custom code retrieval | N/A (replaces default RAG) | Free - $39 |

> **Note:** Many extensions are available for free. Contributors can choose to offer their extensions at no cost to build reputation or contribute to the community.

---

## Project Structure

All extensions follow this structure:

```
my-extension/
├── manifest.yaml      # Required: metadata & configuration
├── agent.py           # For agents
├── profile.py         # For profiles
├── retriever.py       # For RAG kits
├── requirements.txt   # Python dependencies (optional)
└── README.md          # Documentation (recommended)
```

---

## Creating an Agent

Agents hook into the pipeline at specific stages to add functionality.

### Step 1: Create manifest.yaml

```yaml
name: my-security-scanner
version: 1.0.0
type: agent
author: Your Name
description: Scan generated code for security vulnerabilities
license: free  # or "commercial" for paid

# When to run in pipeline
hook_point: "after:implementation"

# Class name in agent.py
agent_class: SecurityScannerAgent

# Python dependencies (optional)
requires:
  - bandit>=1.7.0

# Metadata
keywords:
  - security
  - scanner
  - vulnerabilities
priority: 50  # Lower = runs earlier (default: 50)
```

### Available Hook Points

```
before:spec          after:spec
before:context       after:context
before:design        after:design
before:implementation  after:implementation
before:testing       after:testing
before:verification  after:verification
before:code_review   after:code_review
before:deploy        after:deploy
```

### Step 2: Create agent.py

```python
"""Security Scanner Agent - Scan code for vulnerabilities."""

from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class AgentOutput:
    """Standard output format."""
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None


class SecurityScannerAgent:
    """Scan generated code for security vulnerabilities.

    This agent runs after implementation and checks for common
    security issues like SQL injection, XSS, and hardcoded secrets.
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        """Initialize agent.

        Args:
            llm: LLM backend (use for AI-powered analysis)
            **kwargs: Additional arguments from pipeline
        """
        self.llm = llm
        self.name = "security-scanner"

    def default_system_prompt(self) -> str:
        """System prompt for LLM calls (if needed)."""
        return "You are a security expert analyzing code for vulnerabilities."

    def run(self, input_data: Any) -> AgentOutput:
        """Execute the security scan.

        Args:
            input_data: Contains input_data.context with pipeline state

        Returns:
            AgentOutput with scan results
        """
        # Access pipeline context
        context = input_data.context if hasattr(input_data, 'context') else input_data

        # Get data from previous stages
        run_dir = context.get("run_dir", "")
        repo_path = context.get("repo_path", ".")
        spec = context.get("spec", {})
        design = context.get("design", {})

        # Your logic here
        vulnerabilities = self._scan_for_vulnerabilities(repo_path, run_dir)

        # Return results
        return AgentOutput(
            success=len(vulnerabilities) == 0,
            data={
                "security_scan": {
                    "vulnerabilities": vulnerabilities,
                    "passed": len(vulnerabilities) == 0,
                    "scanned_files": 10,
                }
            },
            artifacts=["security_report.json"],
            agent_name=self.name,
        )

    def _scan_for_vulnerabilities(self, repo_path: str, run_dir: str) -> list:
        """Scan code files for vulnerabilities."""
        # Implementation here
        return []
```

### Using the LLM

Your agent receives an LLM instance for AI-powered analysis:

```python
def run(self, input_data: Any) -> AgentOutput:
    # Build messages for LLM
    messages = [
        {"role": "system", "content": self.default_system_prompt()},
        {"role": "user", "content": f"Analyze this code:\n{code}"}
    ]

    # Call LLM
    response = self.llm.chat(messages)

    # Process response
    # ...
```

---

## Creating a Profile

Profiles define tech stack templates with conventions and dependencies.

### Step 1: Create manifest.yaml

```yaml
name: sveltekit
version: 1.0.0
type: profile
author: Your Name
description: SvelteKit + TypeScript + Tailwind project template
license: free

profile_class: SvelteKitProfile

keywords:
  - svelte
  - sveltekit
  - typescript
  - frontend
```

### Step 2: Create profile.py

```python
"""SvelteKit Profile - Modern frontend template."""

from typing import Any


class SvelteKitProfile:
    """SvelteKit + TypeScript + Tailwind project template.

    Provides conventions, file structure, and dependencies
    for SvelteKit projects.
    """

    name = "sveltekit"
    display_name = "SvelteKit"

    # Tech stack definition
    stack = {
        "framework": "SvelteKit",
        "language": "TypeScript",
        "styling": "Tailwind CSS",
        "testing": "Vitest + Playwright",
        "package_manager": "pnpm",
    }

    # Project structure
    file_structure = """
    src/
      routes/
        +page.svelte
        +layout.svelte
        +error.svelte
      lib/
        components/
        stores/
        utils/
    static/
    tests/
    svelte.config.js
    tailwind.config.js
    """

    # Coding conventions
    conventions = """
    - Use TypeScript strict mode
    - Components in src/lib/components/
    - Stores in src/lib/stores/ using Svelte stores
    - Server code in +page.server.ts / +server.ts
    - Use Tailwind for styling, no CSS files
    - Tests alongside components with .test.ts extension
    """

    # Dependencies
    dependencies = {
        "@sveltejs/kit": "^2.0.0",
        "svelte": "^4.0.0",
        "typescript": "^5.0.0",
        "tailwindcss": "^3.0.0",
        "vite": "^5.0.0",
    }

    dev_dependencies = {
        "vitest": "^1.0.0",
        "@playwright/test": "^1.40.0",
    }

    def get_guidance(self, feature_type: str = "") -> str:
        """Get profile-specific guidance for code generation.

        Args:
            feature_type: Type of feature being built

        Returns:
            Guidance string for the LLM
        """
        return f"""
## SvelteKit Conventions

{self.conventions}

## File Structure
{self.file_structure}

## Stack
Framework: {self.stack['framework']}
Language: {self.stack['language']}
Styling: {self.stack['styling']}
"""
```

---

## Creating a RAG Kit

RAG kits provide custom code retrieval systems.

### Step 1: Create manifest.yaml

```yaml
name: semantic-search
version: 1.0.0
type: rag
author: Your Name
description: Semantic code search using sentence-transformers
license: commercial

retriever_class: SemanticSearchRetriever

requires:
  - sentence-transformers>=2.2.0
  - faiss-cpu>=1.7.0

price_usd: 29.00

keywords:
  - semantic
  - embeddings
  - search
```

### Step 2: Create retriever.py

```python
"""Semantic Search RAG Kit - Vector-based code retrieval."""

from typing import Any, List, Tuple


class SemanticSearchRetriever:
    """Semantic code search using sentence-transformers.

    Provides better code retrieval than keyword matching by
    understanding semantic similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever.

        Args:
            model_name: Sentence-transformer model to use
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def index_codebase(self, files: List[dict]) -> int:
        """Index code files for search.

        Args:
            files: List of {"path": str, "content": str}

        Returns:
            Number of indexed documents
        """
        import faiss
        import numpy as np

        self.documents = files
        contents = [f["content"] for f in files]

        # Generate embeddings
        embeddings = self.model.encode(contents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Build FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        return len(files)

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        """Retrieve relevant code snippets.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of relevant documents
        """
        import numpy as np

        if self.index is None:
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(1 / (1 + distances[0][i]))
                results.append(doc)

        return results

    def retrieve_with_budget(
        self,
        query: str,
        token_budget: int,
        top_k: int = 50,
    ) -> Tuple[List[dict], str]:
        """Retrieve content within token budget.

        Args:
            query: Search query
            token_budget: Maximum tokens
            top_k: Maximum results to consider

        Returns:
            Tuple of (results, formatted_content)
        """
        results = self.retrieve(query, top_k)

        content_parts = []
        total_tokens = 0
        included = []

        for doc in results:
            # Estimate tokens (rough: 4 chars per token)
            doc_tokens = len(doc.get("content", "")) // 4

            if total_tokens + doc_tokens > token_budget:
                break

            content_parts.append(f"### {doc['path']}\n{doc['content']}")
            total_tokens += doc_tokens
            included.append(doc)

        return included, "\n\n".join(content_parts)
```

---

## Testing Your Extension

### 1. Install Locally

```bash
# Create extensions directory if needed
mkdir -p ~/.coding-factory/extensions/agents

# Copy your extension
cp -r my-extension ~/.coding-factory/extensions/agents/

# Verify it's discovered
acf extensions list
```

### 2. Run Pipeline

```bash
# Create a test project
mkdir test-project && cd test-project
acf init

# Run with your extension
acf run "Build a simple API" --auto-approve
```

### 3. Check Output

Your extension's output will appear in the pipeline logs and artifacts.

---

## Submitting to Marketplace

### 1. Package Your Extension

```bash
cd my-extension
tar -czvf my-extension-1.0.0.tar.gz .
```

### 2. Submit for Review

```bash
acf marketplace submit ./my-extension-1.0.0.tar.gz \
  --name "my-extension" \
  --version "1.0.0" \
  --type agent \
  --description "What it does" \
  --price 15.00  # Omit for free extensions
```

### 3. Review Process

1. **Automated security scan** - Checks for malicious patterns
2. **Manual review** - ACF team reviews quality and security
3. **Published** - Extension goes live on marketplace

---

## Security Requirements

All extensions undergo automated and manual security review before publication.

**General guidelines:**
- Follow secure coding practices
- Avoid dynamic code execution with untrusted input
- Use safe deserialization methods
- Keep file operations within the project directory
- Document any network calls your extension makes

Extensions that don't meet security standards will be rejected with feedback for improvement.

---

## Pricing Guidelines

| Complexity | Suggested Price | Examples |
|------------|-----------------|----------|
| Simple utility | Free - $10 | Formatters, linters |
| Standard feature | $10 - $25 | Security scanners, reviewers |
| Advanced feature | $25 - $49 | AI-powered analysis, complex workflows |
| Profile/Template | $5 - $15 | Framework templates |
| RAG Kit | $15 - $39 | Custom retrievers |

### Revenue Split

- **You keep 82.35%** of each sale
- 17.65% covers payment processing (Stripe/PayPal) and infrastructure

---

## Tips for Success

1. **Solve a real problem** - Focus on pain points developers face
2. **Good documentation** - Clear README with examples
3. **Start free** - Build reputation before paid extensions
4. **Respond to feedback** - Update based on user reviews
5. **Test thoroughly** - Ensure it works with different project types

---

## Examples

See the `official_extensions/` directory for complete examples:

- `decomposition/` - Breaks complex features into subtasks
- `api-contract/` - Defines API boundaries
- `code-review/` - Senior engineer review

These are fully functional extensions you can use as templates.
