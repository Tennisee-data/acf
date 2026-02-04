# ACF Marketplace

> Extend your AI coding pipeline with community-built agents, skills, profiles, and RAG kits.

## What is the Marketplace?

The ACF Marketplace is a collection of extensions that add new capabilities to your local AI coding pipeline. Extensions run entirely on your machine - your code never leaves your environment.

## Extension Types

| Type | What it does | Example |
|------|--------------|---------|
| **Skills** | Single-purpose tools that transform code | Add docstrings, format imports, generate tests |
| **Agents** | Pipeline stage hooks that add analysis/checks | Security scanner, code reviewer, decomposition |
| **Profiles** | Tech stack templates with conventions | SvelteKit, FastAPI, React + TypeScript |
| **RAG Kits** | Custom code retrieval systems | Semantic search, pattern matching |

## Quick Start

```bash
# List available extensions
acf extensions list

# Install from marketplace
acf extensions install security-scanner

# Run pipeline with your new extension
acf run "Add user authentication"
```

## Browse

- [Browse Extensions](browse.md) - Search and discover extensions
- [Official Extensions](https://github.com/Tennisee-data/acf/tree/main/official_extensions) - Free, maintained by ACF team

## Create Your Own

- [Getting Started](getting-started.md) - Your first extension in 10 minutes
- [Creating Skills](creating/skills.md) - Single-purpose code transformations
- [Creating Agents](creating/agents.md) - Pipeline hooks and analysis
- [Creating Profiles](creating/profiles.md) - Tech stack templates
- [Creating RAG Kits](creating/rag.md) - Custom code retrieval

## Reference

- [Specification](specification.md) - Complete manifest.yaml schema
- [Publishing Guide](publishing.md) - Submit to marketplace
- [Example Extensions](examples.md) - Learn from working code

## Why Local Extensions?

- **Privacy**: Extensions run on your machine, your code stays yours
- **Customizable**: Modify or fork any extension
- **Composable**: Mix and match extensions for your workflow
- **Shareable**: Publish once, others install and use
