# RAG Documentation Folder

This folder contains external API documentation used to provide context to the LLM during code generation.

## Purpose

When generating code for proprietary APIs (Stripe, Twilio, etc.), the LLM may hallucinate incorrect implementation details. This folder provides ground-truth documentation to prevent such errors.

## Structure

```
rag/
├── README.md           # This file
├── stripe/
│   ├── index.md        # Overview and safety notes
│   ├── llms.md         # LLM-optimized docs (from llms.txt)
│   ├── api_reference.md # Generated from OpenAPI
│   └── _meta.json      # Fetch metadata
├── twilio/
│   └── ...
└── ...
```

## Updating Documentation

### Manual Update

```bash
# Fetch all sources
python tools/rag_fetcher.py --all

# Fetch specific sources
python tools/rag_fetcher.py stripe twilio

# Force refresh (ignore cache)
python tools/rag_fetcher.py stripe --force

# Check status
python tools/rag_fetcher.py --status

# List available sources
python tools/rag_fetcher.py --list
```

### Automatic Updates

Documentation is automatically updated monthly via GitHub Actions:
- Schedule: 1st of each month at 03:00 UTC
- Creates a PR with updated documentation
- Can be manually triggered from the Actions tab

## Adding New Sources

Edit `rag_sources.toml` to add new API sources:

```toml
[new_api]
description = "Description of the API"
priority = "high"  # high, medium, low
llms_txt = "https://docs.example.com/llms.txt"  # Best source
openapi = "https://api.example.com/openapi.json"
urls = [
    "https://docs.example.com/webhooks",
]
topics = ["auth", "webhooks", "payments"]
safety_critical = ["webhook signature verification"]
```

## Source Priority

Sources are fetched in order of reliability:

1. **llms.txt** - LLM-optimized documentation (best)
2. **OpenAPI specs** - Structured API reference
3. **GitHub docs** - README and documentation files
4. **Direct URLs** - Documentation pages (fallback)

## Integration with Pipeline

The `DocumentationRequirementsAgent` checks this folder when external APIs are detected:

1. Detects API usage in feature request (e.g., "Add Stripe webhook")
2. Checks if documentation exists in `rag/stripe/`
3. If found, includes in context for code generation
4. If missing, warns user and suggests fetching docs

## Safety-Critical Patterns

Each source can define `safety_critical` patterns. These are highlighted in the documentation and trigger the `SafetyPatternsAgent` to inject implementation invariants.

Example for webhooks:
- Must use raw request body (bytes)
- Must verify signature before processing
- Must implement idempotency
