# Creating Extensions

Build extensions that add new capabilities to ACF.

## Extension Types

| Type | Purpose | When It Runs |
|------|---------|--------------|
| [Skills](skills.md) | Transform code files | On demand via `acf skill <name>` |
| [Agents](agents.md) | Add pipeline analysis | Automatically at hook points |
| [Profiles](profiles.md) | Define tech stack conventions | At pipeline initialization |
| [RAG Kits](rag.md) | Custom code retrieval | During context gathering |

## Choosing the Right Type

### Use a Skill when you want to:
- Transform or modify code files
- Run a one-off operation on demand
- Create a reusable code tool

**Examples:** Add docstrings, format imports, generate tests, refactor patterns

### Use an Agent when you want to:
- Hook into the pipeline at specific stages
- Add automated checks or analysis
- Modify pipeline behavior

**Examples:** Security scanning, code review, complexity analysis, API contract validation

### Use a Profile when you want to:
- Define project structure and conventions
- Specify tech stack defaults
- Provide framework-specific guidance

**Examples:** SvelteKit template, FastAPI conventions, React + TypeScript setup

### Use a RAG Kit when you want to:
- Customize how code is retrieved for context
- Add semantic search capabilities
- Provide domain-specific code patterns

**Examples:** Semantic code search, pattern libraries, documentation retrieval

## Common Structure

All extensions share this basic structure:

```
my-extension/
├── manifest.yaml      # Required: metadata and configuration
├── [main].py          # Required: implementation (agent.py, skill.py, etc.)
├── requirements.txt   # Optional: Python dependencies
└── README.md          # Recommended: documentation
```

## The Manifest File

Every extension needs a `manifest.yaml`:

```yaml
name: my-extension          # Unique identifier (lowercase, hyphens ok)
version: 1.0.0              # Semantic version
type: skill                 # skill | agent | profile | rag
author: Your Name
description: What it does
license: free               # free | commercial

# Type-specific fields...
```

See [Specification](../specification.md) for the complete schema.

## Quick Links

- [Skills Guide](skills.md) - Full skill creation walkthrough
- [Agents Guide](agents.md) - Pipeline hooks and analysis
- [Profiles Guide](profiles.md) - Tech stack templates
- [RAG Kits Guide](rag.md) - Custom code retrieval
- [Publishing](../publishing.md) - Submit to marketplace
