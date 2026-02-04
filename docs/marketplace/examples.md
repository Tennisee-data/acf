# Example Extensions

Learn from working code. All official extensions are open source and can be used as templates.

## Official Extensions

### Skills

#### add-docstrings
Adds Google-style docstrings to Python functions and classes using AI.

- **Type:** Skill
- **Source:** [official_extensions/add-docstrings](https://github.com/Tennisee-data/acf/tree/main/official_extensions/add-docstrings)

```bash
# Usage
acf skill add-docstrings ./src
acf skill add-docstrings ./src --dry-run  # Preview only
```

Key features:
- AI-powered docstring generation
- Skips already-documented code
- Supports dry-run mode

---

### Agents

#### code-review
Senior engineer code review focusing on maintainability and best practices.

- **Type:** Agent
- **Hook:** `after:implementation`
- **Source:** [official_extensions/code-review](https://github.com/Tennisee-data/acf/tree/main/official_extensions/code-review)

Key features:
- Reviews generated code automatically
- Provides actionable feedback
- Checks for common issues

#### decomposition
Breaks complex features into manageable subtasks.

- **Type:** Agent
- **Hook:** `before:design`
- **Source:** [official_extensions/decomposition](https://github.com/Tennisee-data/acf/tree/main/official_extensions/decomposition)

Key features:
- Analyzes feature complexity
- Suggests task breakdown
- Skips simple features automatically

#### api-contract
Defines API boundaries and contracts before implementation.

- **Type:** Agent
- **Hook:** `before:implementation`
- **Source:** [official_extensions/api-contract](https://github.com/Tennisee-data/acf/tree/main/official_extensions/api-contract)

Key features:
- Generates OpenAPI-style contracts
- Validates endpoint consistency
- Documents request/response schemas

---

### RAG Kits

#### fastapi-patterns
Production-ready FastAPI patterns for common use cases.

- **Type:** RAG
- **Source:** [marketplace/official_extensions/rag/fastapi-patterns](https://github.com/Tennisee-data/acf/tree/main/marketplace/official_extensions/rag/fastapi-patterns)

Key features:
- Authentication patterns (JWT, OAuth)
- Database patterns (SQLAlchemy, async)
- Error handling patterns
- Testing patterns

---

## Extension Templates

Use these as starting points for your own extensions.

### Minimal Skill Template

```
minimal-skill/
├── manifest.yaml
└── skill.py
```

**manifest.yaml:**
```yaml
name: minimal-skill
version: 1.0.0
type: skill
author: Your Name
description: Minimal skill template
license: free
skill_class: MinimalSkill
input_type: files
output_type: modified_files
file_patterns: ["*.py"]
```

**skill.py:**
```python
from pathlib import Path

class MinimalSkill:
    name = "minimal-skill"

    def __init__(self, llm=None, **kwargs):
        self.llm = llm

    def run(self, files: list[Path], dry_run: bool = False, **kwargs) -> dict:
        results = {"modified": [], "skipped": []}
        for f in files:
            # Your logic here
            results["skipped"].append(str(f))
        return results
```

### Minimal Agent Template

```
minimal-agent/
├── manifest.yaml
└── agent.py
```

**manifest.yaml:**
```yaml
name: minimal-agent
version: 1.0.0
type: agent
author: Your Name
description: Minimal agent template
license: free
agent_class: MinimalAgent
hook_point: "after:implementation"
```

**agent.py:**
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentOutput:
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None

class MinimalAgent:
    def __init__(self, llm: Any, **kwargs):
        self.llm = llm
        self.name = "minimal-agent"

    def run(self, input_data: Any) -> AgentOutput:
        context = getattr(input_data, 'context', input_data)
        # Your logic here
        return AgentOutput(
            success=True,
            data={"message": "Agent completed"},
            agent_name=self.name,
        )
```

### Minimal Profile Template

```
minimal-profile/
├── manifest.yaml
└── profile.py
```

**manifest.yaml:**
```yaml
name: minimal-profile
version: 1.0.0
type: profile
author: Your Name
description: Minimal profile template
license: free
profile_class: MinimalProfile
```

**profile.py:**
```python
class MinimalProfile:
    name = "minimal-profile"
    display_name = "Minimal Profile"

    stack = {
        "framework": "Your Framework",
        "language": "Your Language",
    }

    conventions = """
    - Your conventions here
    """

    def get_guidance(self, feature_type: str = "") -> str:
        return f"## Conventions\n{self.conventions}"
```

### Minimal RAG Template

```
minimal-rag/
├── manifest.yaml
└── retriever.py
```

**manifest.yaml:**
```yaml
name: minimal-rag
version: 1.0.0
type: rag
author: Your Name
description: Minimal RAG template
license: free
retriever_class: MinimalRetriever
```

**retriever.py:**
```python
class MinimalRetriever:
    def __init__(self):
        self.documents = []

    def index_codebase(self, files: list[dict]) -> int:
        self.documents = files
        return len(files)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        # Simple keyword matching
        results = []
        for doc in self.documents:
            if query.lower() in doc.get("content", "").lower():
                results.append(doc)
        return results[:top_k]

    def retrieve_with_budget(
        self, query: str, token_budget: int, top_k: int = 50
    ) -> tuple[list[dict], str]:
        results = self.retrieve(query, top_k)
        content_parts = []
        total_tokens = 0
        selected = []

        for doc in results:
            tokens = len(doc.get("content", "")) // 4
            if total_tokens + tokens > token_budget:
                break
            selected.append(doc)
            total_tokens += tokens
            content_parts.append(f"### {doc['path']}\n{doc['content']}")

        return selected, "\n\n".join(content_parts)
```

## Community Examples

Have you built an extension? [Submit a PR](https://github.com/Tennisee-data/acf/pulls) to add it here!

## Next Steps

- [Getting Started](getting-started.md) - Build your first extension
- [Specification](specification.md) - Full manifest reference
- [Publishing](publishing.md) - Submit to marketplace
