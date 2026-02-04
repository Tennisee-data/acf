# Browse Extensions

Discover and install extensions for your ACF pipeline.

## Command Line

### List Installed Extensions

```bash
acf extensions list
```

Output:
```
Agents:
  • code-review v1.0.0 - Senior engineer code review
  • decomposition v1.0.0 - Break complex features into subtasks
  • api-contract v1.0.0 - Define API boundaries

Skills:
  • add-docstrings v1.0.0 - Add Google-style docstrings to Python

RAG:
  • fastapi-patterns v1.0.0 - Production-ready FastAPI patterns

Profiles:
  • (none installed)
```

### Search Marketplace

```bash
acf marketplace search "security"
```

### View Extension Details

```bash
acf extensions show code-review
```

Output:
```
Name: code-review
Version: 1.0.0
Type: agent
Author: ACF Team
License: free

Description:
  Senior engineer code review with focus on maintainability,
  performance, and best practices.

Hook Point: after:implementation
Context Tokens: 1500
Min Model Tier: medium

Keywords: review, quality, best-practices
```

### Install from Marketplace

```bash
# Install free extension
acf extensions install semantic-search

# Install with dependencies
acf extensions install semantic-search --with-deps
```

## Official Extensions

These ship with ACF and are maintained by the core team:

### Agents

| Name | Hook Point | Description |
|------|------------|-------------|
| `code-review` | after:implementation | Senior engineer code review |
| `decomposition` | before:design | Breaks complex features into subtasks |
| `api-contract` | before:implementation | Defines API boundaries and contracts |

### Skills

| Name | Description |
|------|-------------|
| `add-docstrings` | Add Google-style docstrings to Python functions |

### RAG Kits

| Name | Description |
|------|-------------|
| `fastapi-patterns` | Production-ready FastAPI patterns and best practices |

## Categories

### By Use Case

**Code Quality**
- `code-review` - Comprehensive code review
- `add-docstrings` - Documentation generation

**Security**
- Coming soon: `security-scanner`, `secrets-detector`

**Architecture**
- `decomposition` - Task breakdown
- `api-contract` - API design

**Framework-Specific**
- `fastapi-patterns` - FastAPI best practices

### By Extension Type

- [Skills](creating/skills.md) - Code transformations
- [Agents](creating/agents.md) - Pipeline hooks
- [Profiles](creating/profiles.md) - Tech stack templates
- [RAG Kits](creating/rag.md) - Code retrieval

## Install Local Extensions

Extensions can also be installed from local directories:

```bash
# Install from local path
acf extensions install-local ./my-extension

# Or copy directly
cp -r ./my-extension ~/.coding-factory/extensions/skills/
```

## Uninstall

```bash
acf extensions uninstall security-scanner
```

## Check for Updates

```bash
acf extensions outdated
```
