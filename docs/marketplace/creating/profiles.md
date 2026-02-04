# Creating Profiles

Profiles define tech stack templates with conventions, file structures, and dependencies.

## When to Use Profiles

- Define framework-specific conventions
- Provide project structure templates
- Specify default dependencies
- Give LLMs context about your stack

## Directory Structure

```
my-profile/
├── manifest.yaml      # Required: metadata
├── profile.py         # Required: implementation
└── README.md          # Recommended: documentation
```

Place in: `~/.coding-factory/extensions/profiles/my-profile/`

## Manifest Schema

```yaml
name: sveltekit
version: 1.0.0
type: profile
author: Your Name
description: SvelteKit + TypeScript + Tailwind project template
license: free

# Required for profiles
profile_class: SvelteKitProfile     # Class name in profile.py

# Metadata
keywords:
  - svelte
  - sveltekit
  - typescript
  - frontend
  - tailwind
```

## Implementation

### Basic Profile

```python
"""SvelteKit Profile - Modern frontend template."""

from typing import Any


class SvelteKitProfile:
    """SvelteKit + TypeScript + Tailwind project template.

    Provides conventions, file structure, and dependencies
    for SvelteKit projects.
    """

    # Identifiers
    name = "sveltekit"
    display_name = "SvelteKit"

    # Tech stack definition
    stack = {
        "framework": "SvelteKit",
        "language": "TypeScript",
        "styling": "Tailwind CSS",
        "testing": "Vitest + Playwright",
        "package_manager": "pnpm",
        "runtime": "Node.js",
    }

    # Project structure
    file_structure = """
    src/
      routes/
        +page.svelte          # Pages
        +layout.svelte        # Layouts
        +error.svelte         # Error pages
        +page.server.ts       # Server-side logic
      lib/
        components/           # Reusable components
        stores/               # Svelte stores
        utils/                # Utility functions
        server/               # Server-only code
    static/                   # Static assets
    tests/                    # E2E tests
    svelte.config.js          # Svelte config
    tailwind.config.js        # Tailwind config
    vite.config.ts            # Vite config
    """

    # Coding conventions
    conventions = """
    - Use TypeScript strict mode for all files
    - Components go in src/lib/components/ with PascalCase names
    - Stores go in src/lib/stores/ using Svelte's writable/readable stores
    - Server-only code in +page.server.ts or src/lib/server/
    - Use Tailwind for styling, avoid separate CSS files
    - Tests alongside components with .test.ts extension
    - Use SvelteKit's form actions for mutations
    - Prefer server-side rendering when possible
    """

    # Production dependencies
    dependencies = {
        "@sveltejs/kit": "^2.0.0",
        "svelte": "^4.0.0",
        "typescript": "^5.0.0",
        "tailwindcss": "^3.0.0",
        "vite": "^5.0.0",
    }

    # Development dependencies
    dev_dependencies = {
        "vitest": "^1.0.0",
        "@playwright/test": "^1.40.0",
        "@sveltejs/adapter-auto": "^3.0.0",
        "prettier": "^3.0.0",
        "prettier-plugin-svelte": "^3.0.0",
        "eslint": "^8.0.0",
    }

    def get_guidance(self, feature_type: str = "") -> str:
        """Get profile-specific guidance for code generation.

        This is injected into the LLM context to help it understand
        your project's conventions.

        Args:
            feature_type: Type of feature being built (optional)

        Returns:
            Guidance string for the LLM
        """
        base_guidance = f"""
## SvelteKit Project Conventions

{self.conventions}

## File Structure
{self.file_structure}

## Tech Stack
- Framework: {self.stack['framework']}
- Language: {self.stack['language']}
- Styling: {self.stack['styling']}
- Testing: {self.stack['testing']}
- Package Manager: {self.stack['package_manager']}
"""

        # Add feature-specific guidance
        if feature_type == "api":
            base_guidance += """
## API Conventions
- Use +server.ts for API routes
- Return JSON with proper status codes
- Handle errors with SvelteKit's error() helper
- Validate input with zod or similar
"""
        elif feature_type == "form":
            base_guidance += """
## Form Conventions
- Use SvelteKit form actions in +page.server.ts
- Implement progressive enhancement
- Handle validation server-side
- Use enhance for client-side form handling
"""

        return base_guidance

    def get_starter_files(self) -> dict[str, str]:
        """Get starter files for new projects.

        Returns:
            Dict mapping file paths to contents
        """
        return {
            "src/routes/+page.svelte": """<script lang="ts">
  // Page logic here
</script>

<h1>Welcome to SvelteKit</h1>

<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to learn more.</p>
""",
            "src/routes/+layout.svelte": """<script lang="ts">
  import '../app.css';
</script>

<slot />
""",
            "src/app.css": """@tailwind base;
@tailwind components;
@tailwind utilities;
""",
        }
```

### Profile with Framework Detection

```python
class FastAPIProfile:
    """FastAPI + SQLAlchemy + Pydantic profile."""

    name = "fastapi"
    display_name = "FastAPI"

    stack = {
        "framework": "FastAPI",
        "language": "Python",
        "orm": "SQLAlchemy",
        "validation": "Pydantic",
        "testing": "pytest",
    }

    file_structure = """
    app/
      __init__.py
      main.py               # FastAPI app entry
      models/               # SQLAlchemy models
      schemas/              # Pydantic schemas
      routers/              # API route modules
      services/             # Business logic
      core/
        config.py           # Settings
        database.py         # DB connection
    tests/
      conftest.py           # Fixtures
      test_*.py             # Test files
    alembic/                # Migrations
    """

    conventions = """
    - Use Pydantic models for all request/response schemas
    - SQLAlchemy models in app/models/ with Base class
    - One router per resource in app/routers/
    - Business logic in app/services/, not in routers
    - Dependency injection for database sessions
    - Use async/await for all database operations
    - pytest for testing with TestClient
    """

    def get_guidance(self, feature_type: str = "") -> str:
        guidance = f"""
## FastAPI Conventions

{self.conventions}

## File Structure
{self.file_structure}

## Patterns

### Router Pattern
```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")
async def list_items(db: Session = Depends(get_db)):
    return await ItemService.list(db)
```

### Schema Pattern
```python
from pydantic import BaseModel

class ItemCreate(BaseModel):
    name: str
    price: float

class ItemResponse(ItemCreate):
    id: int

    class Config:
        from_attributes = True
```
"""
        return guidance
```

## Profile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier |
| `display_name` | `str` | Human-readable name |
| `stack` | `dict` | Tech stack components |
| `file_structure` | `str` | Project structure diagram |
| `conventions` | `str` | Coding guidelines |
| `dependencies` | `dict` | Production packages |
| `dev_dependencies` | `dict` | Development packages |

## Required Methods

### get_guidance(feature_type: str) -> str

Returns context for the LLM about your project's conventions. Called during code generation.

```python
def get_guidance(self, feature_type: str = "") -> str:
    """Return conventions and patterns for LLM context."""
    return f"""
## My Framework Conventions
{self.conventions}
"""
```

## Optional Methods

### get_starter_files() -> dict[str, str]

Returns starter files for new projects:

```python
def get_starter_files(self) -> dict[str, str]:
    return {
        "src/main.py": "# Main entry point\n",
        "tests/__init__.py": "",
    }
```

### validate_structure(path: Path) -> list[str]

Validates a project matches your conventions:

```python
def validate_structure(self, path: Path) -> list[str]:
    """Return list of violations."""
    issues = []
    if not (path / "src").exists():
        issues.append("Missing src/ directory")
    return issues
```

## Using Profiles

```bash
# List available profiles
acf extensions list --type profile

# Use a specific profile
acf run "Build a dashboard" --profile sveltekit

# Set default profile in config
# ~/.coding-factory/config.yaml
pipeline:
  default_profile: fastapi
```

## Testing Your Profile

```bash
# Install locally
cp -r my-profile ~/.coding-factory/extensions/profiles/

# Verify
acf extensions list

# Test with a run
acf run "Build a simple page" --profile my-profile
```

## Examples

See official profiles in the repo:
- [`streamlit-profile`](https://github.com/Tennisee-data/acf/tree/main/official_extensions) - Streamlit data apps

## Next Steps

- [Specification](../specification.md) - Full manifest schema
- [Publishing](../publishing.md) - Submit to marketplace
