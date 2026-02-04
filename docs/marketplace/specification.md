# Specification

Complete reference for the `manifest.yaml` schema.

## Common Fields

All extension types share these fields:

```yaml
# Required
name: my-extension              # Unique identifier (lowercase, hyphens allowed)
version: 1.0.0                  # Semantic versioning (major.minor.patch)
type: skill                     # skill | agent | profile | rag
author: Your Name               # Author name or organization
description: What it does       # Short description (1-2 sentences)
license: free                   # free | commercial

# Optional
keywords:                       # For search/discovery
  - keyword1
  - keyword2
priority: 50                    # Execution order (lower = first, default: 50)
requires:                       # Python dependencies
  - package>=1.0.0
conflicts_with:                 # Incompatible extensions
  - other-extension
```

## Skill Fields

```yaml
type: skill

# Required
skill_class: MySkillClass       # Class name in skill.py
input_type: files               # files | text | code
output_type: modified_files     # modified_files | text | report

# Optional
file_patterns:                  # Glob patterns for target files
  - "*.py"
  - "src/**/*.ts"
  - "!**/test_*.py"             # Exclude with !
supports_dry_run: true          # Supports --dry-run flag
chain:                          # Chain multiple skills
  - skill1
  - skill2
```

### input_type Values

| Value | Description | Parameter Type |
|-------|-------------|----------------|
| `files` | Operate on files | `list[Path]` |
| `text` | Operate on text | `str` |
| `code` | Operate on parsed code | `dict` |

### output_type Values

| Value | Description | Return Format |
|-------|-------------|---------------|
| `modified_files` | Modify files in place | `{"modified": [], "skipped": []}` |
| `text` | Return text | `{"output": str}` |
| `report` | Return report | `{"report": dict}` |

## Agent Fields

```yaml
type: agent

# Required
agent_class: MyAgentClass       # Class name in agent.py
hook_point: "after:implementation"  # When to run

# Optional
context_tokens: 2000            # Tokens added to context
min_model_tier: medium          # any | small | medium | large
```

### Hook Points

```yaml
# Before stage
hook_point: "before:spec"
hook_point: "before:context"
hook_point: "before:design"
hook_point: "before:implementation"
hook_point: "before:testing"
hook_point: "before:verification"
hook_point: "before:code_review"
hook_point: "before:deploy"

# After stage
hook_point: "after:spec"
hook_point: "after:context"
hook_point: "after:design"
hook_point: "after:implementation"
hook_point: "after:testing"
hook_point: "after:verification"
hook_point: "after:code_review"
hook_point: "after:deploy"
```

## Profile Fields

```yaml
type: profile

# Required
profile_class: MyProfileClass   # Class name in profile.py
```

## RAG Fields

```yaml
type: rag

# Required
retriever_class: MyRetrieverClass  # Class name in retriever.py

# Optional
context_tokens: 4000            # Typical tokens returned
min_model_tier: medium          # any | small | medium | large
```

## Model Tier Reference

| Tier | Context Window | Example Models |
|------|----------------|----------------|
| `any` | 4K+ | Any model |
| `small` | 8K+ | qwen2.5:7b, llama3:8b, codellama:7b |
| `medium` | 32K+ | qwen3:14b, mistral:7b, mixtral:8x7b |
| `large` | 128K+ | qwen2.5:32b, gpt-4-turbo, claude-3 |

## Commercial Extensions

```yaml
license: commercial
price_usd: 29.00                # Price in USD
```

Revenue split: You keep 82.35%, 17.65% covers processing and infrastructure.

## Dependencies

```yaml
requires:
  - package>=1.0.0              # Minimum version
  - package>=1.0.0,<2.0.0       # Version range
  - package==1.2.3              # Exact version
```

Dependencies are installed when the extension loads (user prompted).

## Conflicts

```yaml
conflicts_with:
  - other-security-scanner      # Can't run with this extension
```

Pipeline will warn if conflicting extensions are enabled.

## Full Examples

### Skill Manifest

```yaml
name: add-docstrings
version: 1.0.0
type: skill
author: ACF Team
description: Add Google-style docstrings to Python functions and classes
license: free

skill_class: AddDocstringsSkill
input_type: files
output_type: modified_files
file_patterns:
  - "*.py"
  - "!**/__pycache__/**"
  - "!**/test_*.py"
supports_dry_run: true

keywords:
  - documentation
  - docstrings
  - python
  - google-style

priority: 50
context_tokens: 1000
min_model_tier: small
```

### Agent Manifest

```yaml
name: security-scanner
version: 1.0.0
type: agent
author: Security Team
description: Scan generated code for OWASP Top 10 vulnerabilities
license: commercial

agent_class: SecurityScannerAgent
hook_point: "after:implementation"

requires:
  - bandit>=1.7.0
  - safety>=2.0.0

price_usd: 19.00

keywords:
  - security
  - owasp
  - vulnerabilities
  - scanning

priority: 40
context_tokens: 1500
min_model_tier: medium
```

### Profile Manifest

```yaml
name: nextjs-enterprise
version: 1.0.0
type: profile
author: Frontend Team
description: Next.js 14 + TypeScript + Tailwind enterprise template
license: free

profile_class: NextJSEnterpriseProfile

keywords:
  - nextjs
  - react
  - typescript
  - tailwind
  - enterprise
```

### RAG Manifest

```yaml
name: semantic-codebase
version: 1.0.0
type: rag
author: ML Team
description: Semantic code search using sentence-transformers and FAISS
license: commercial

retriever_class: SemanticCodeRetriever

requires:
  - sentence-transformers>=2.2.0
  - faiss-cpu>=1.7.4

price_usd: 29.00

keywords:
  - semantic
  - embeddings
  - vector-search
  - faiss

context_tokens: 4000
min_model_tier: medium
```

## Validation

Validate your manifest:

```bash
acf extensions validate ./my-extension
```

Common errors:
- Missing required fields
- Invalid type value
- Invalid hook_point format
- Malformed version string
