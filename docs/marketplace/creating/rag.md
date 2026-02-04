# Creating RAG Kits

RAG (Retrieval-Augmented Generation) kits provide custom code retrieval systems that help the LLM understand your codebase.

## When to Use RAG Kits

- Add semantic code search (beyond keyword matching)
- Provide domain-specific patterns and examples
- Include documentation in context
- Custom relevance ranking

## Directory Structure

```
my-rag/
├── manifest.yaml      # Required: metadata
├── retriever.py       # Required: implementation
├── patterns/          # Optional: pattern files
├── requirements.txt   # Optional: dependencies
└── README.md          # Recommended: documentation
```

Place in: `~/.coding-factory/extensions/rag/my-rag/`

## Manifest Schema

```yaml
name: semantic-search
version: 1.0.0
type: rag
author: Your Name
description: Semantic code search using embeddings
license: commercial

# Required for RAG kits
retriever_class: SemanticRetriever    # Class name in retriever.py

# Dependencies
requires:
  - sentence-transformers>=2.2.0
  - faiss-cpu>=1.7.0

# Pricing (for commercial)
price_usd: 29.00

# Resource hints
context_tokens: 4000                  # Typical tokens returned
min_model_tier: medium                # any | small | medium | large

# Metadata
keywords:
  - semantic
  - embeddings
  - search
```

## Implementation

### Required Methods

Your retriever class must implement:

| Method | Description |
|--------|-------------|
| `index_codebase(files)` | Index code files for search |
| `retrieve(query, top_k)` | Retrieve relevant code |
| `retrieve_with_budget(query, token_budget, top_k)` | Retrieve within token limit |

### Basic Retriever

```python
"""Semantic Search RAG Kit - Vector-based code retrieval."""

from pathlib import Path
from typing import Any


class SemanticRetriever:
    """Semantic code search using sentence-transformers.

    Provides better code retrieval than keyword matching by
    understanding semantic similarity between queries and code.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever.

        Args:
            model_name: Sentence-transformer model to use
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: list[dict] = []

    def index_codebase(self, files: list[dict]) -> int:
        """Index code files for search.

        Args:
            files: List of {"path": str, "content": str}

        Returns:
            Number of indexed documents
        """
        import faiss
        import numpy as np

        self.documents = files

        # Extract content for embedding
        contents = [f["content"] for f in files]

        # Generate embeddings
        embeddings = self.model.encode(
            contents,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = np.array(embeddings).astype('float32')

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        return len(files)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve relevant code snippets.

        Args:
            query: Natural language search query
            top_k: Maximum number of results

        Returns:
            List of relevant documents with scores
        """
        import numpy as np

        if self.index is None or not self.documents:
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                # Convert distance to similarity score (0-1)
                doc["score"] = float(1 / (1 + distances[0][i]))
                results.append(doc)

        return results

    def retrieve_with_budget(
        self,
        query: str,
        token_budget: int,
        top_k: int = 50,
    ) -> tuple[list[dict], str]:
        """Retrieve content within token budget.

        Args:
            query: Search query
            token_budget: Maximum tokens to return
            top_k: Maximum candidates to consider

        Returns:
            Tuple of (selected documents, formatted content string)
        """
        # Get candidates
        candidates = self.retrieve(query, top_k)

        # Select within budget
        selected = []
        total_tokens = 0
        content_parts = []

        for doc in candidates:
            # Estimate tokens (rough: 4 chars per token)
            doc_tokens = len(doc.get("content", "")) // 4

            if total_tokens + doc_tokens > token_budget:
                break

            selected.append(doc)
            total_tokens += doc_tokens
            content_parts.append(
                f"### {doc['path']}\n```\n{doc['content']}\n```"
            )

        formatted_content = "\n\n".join(content_parts)
        return selected, formatted_content
```

### Pattern-Based Retriever

For domain-specific patterns:

```python
"""FastAPI Patterns RAG Kit - Production patterns for FastAPI."""

import json
from pathlib import Path
from typing import Any


class FastAPIPatternRetriever:
    """Retrieve FastAPI patterns based on query context.

    Provides production-ready patterns for authentication,
    database operations, error handling, and more.
    """

    def __init__(self):
        self.patterns: list[dict] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from bundled files."""
        # Patterns directory relative to this file
        patterns_dir = Path(__file__).parent / "patterns"

        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob("*.json"):
                with open(pattern_file) as f:
                    pattern = json.load(f)
                    self.patterns.append(pattern)

    def index_codebase(self, files: list[dict]) -> int:
        """Index is pre-built from patterns, but accept files for interface."""
        # Could optionally combine with project files
        return len(self.patterns)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve patterns matching the query.

        Uses keyword matching against pattern tags and descriptions.
        """
        query_lower = query.lower()
        scored_patterns = []

        for pattern in self.patterns:
            score = 0

            # Check tags
            for tag in pattern.get("tags", []):
                if tag.lower() in query_lower:
                    score += 2

            # Check keywords in description
            desc = pattern.get("description", "").lower()
            for word in query_lower.split():
                if word in desc:
                    score += 1

            if score > 0:
                scored_patterns.append({
                    "path": pattern.get("name", "pattern"),
                    "content": pattern.get("code", ""),
                    "description": pattern.get("description", ""),
                    "score": score,
                })

        # Sort by score and return top_k
        scored_patterns.sort(key=lambda x: x["score"], reverse=True)
        return scored_patterns[:top_k]

    def retrieve_with_budget(
        self,
        query: str,
        token_budget: int,
        top_k: int = 10,
    ) -> tuple[list[dict], str]:
        """Retrieve patterns within token budget."""
        candidates = self.retrieve(query, top_k)

        selected = []
        total_tokens = 0
        content_parts = []

        for pattern in candidates:
            tokens = len(pattern["content"]) // 4
            if total_tokens + tokens > token_budget:
                break

            selected.append(pattern)
            total_tokens += tokens

            content_parts.append(f"""
## {pattern['path']}
{pattern.get('description', '')}

```python
{pattern['content']}
```
""")

        return selected, "\n".join(content_parts)
```

### Pattern File Format

Store patterns in `patterns/*.json`:

```json
{
  "name": "jwt-authentication",
  "description": "JWT authentication with refresh tokens for FastAPI",
  "tags": ["auth", "jwt", "authentication", "security", "token"],
  "code": "from fastapi import Depends, HTTPException, status\nfrom fastapi.security import HTTPBearer\nfrom jose import JWTError, jwt\n\nsecurity = HTTPBearer()\n\nasync def get_current_user(\n    token: str = Depends(security)\n) -> User:\n    try:\n        payload = jwt.decode(\n            token.credentials,\n            settings.SECRET_KEY,\n            algorithms=[settings.ALGORITHM]\n        )\n        user_id = payload.get('sub')\n        if user_id is None:\n            raise HTTPException(\n                status_code=status.HTTP_401_UNAUTHORIZED,\n                detail='Invalid token'\n            )\n        return await User.get(user_id)\n    except JWTError:\n        raise HTTPException(\n            status_code=status.HTTP_401_UNAUTHORIZED,\n            detail='Invalid token'\n        )"
}
```

## Token Budget Management

RAG kits must respect token budgets to fit in LLM context:

```python
def retrieve_with_budget(
    self,
    query: str,
    token_budget: int,    # Respect this limit
    top_k: int = 50,
) -> tuple[list[dict], str]:
    """
    Args:
        token_budget: Maximum tokens to return (typically 2000-8000)

    Returns:
        - List of selected documents
        - Formatted string ready for LLM context
    """
    # Estimate tokens: ~4 characters per token
    # Include buffer for formatting
```

## Testing Your RAG Kit

```bash
# Install locally
cp -r my-rag ~/.coding-factory/extensions/rag/

# Verify
acf extensions list

# Test retrieval
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / '.coding-factory/extensions/rag/my-rag'))
from retriever import SemanticRetriever

r = SemanticRetriever()
r.index_codebase([
    {'path': 'auth.py', 'content': 'def login(user, password): ...'},
    {'path': 'api.py', 'content': 'def get_items(): return items'},
])
print(r.retrieve('authentication'))
"

# Test in pipeline
acf run "Add user login" --auto-approve
```

## Examples

See official RAG kits:
- [`fastapi-patterns`](https://github.com/Tennisee-data/acf/tree/main/marketplace/official_extensions/rag/fastapi-patterns) - FastAPI production patterns

## Next Steps

- [Specification](../specification.md) - Full manifest schema
- [Publishing](../publishing.md) - Submit to marketplace
