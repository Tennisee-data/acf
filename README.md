# ACF - AgentCodeFactory Local Edition

AI-powered code generation pipeline. Run locally or use cloud APIs.

> **[Read the Manifesto](MANIFESTO.md)** — Why we believe local AI is the future of secure, private code generation.

## Features

- **Flexible Backends**: Ollama, LM Studio (local) or OpenAI, Anthropic (cloud)
- **7-Stage Pipeline**: SPEC → CONTEXT → DESIGN → IMPLEMENTATION → TESTING → REVIEW → DONE
- **Multi-Model Routing**: Automatically routes tasks to appropriate model sizes
- **Extension Marketplace**: Install premium agents, profiles, and RAG kits
- **Local Git Versioning**: Every iteration auto-committed for easy rollback

## Quick Start

```bash
# Install from source (if you cloned the repo)
pip install -e .

# Or with all optional features
pip install -e ".[full]"

# Run (requires a configured LLM backend - see below)
acf run "Add user authentication with JWT"
```

## Requirements

- Python 3.11+
- One of the following LLM backends:
  - [Ollama](https://ollama.ai) - Local, free, recommended
  - [LM Studio](https://lmstudio.ai) - Local, free, GUI
  - [OpenAI API](https://platform.openai.com) - Cloud, paid
  - [Anthropic API](https://console.anthropic.com) - Cloud, paid

### LLM Backend Setup

**Quick start with Ollama:**
```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull a model (in another terminal)
ollama pull qwen2.5-coder:7b
```

> **Important:** Update `config.toml` to match your installed models. See guides below.

**Detailed guides:**
- [Ollama Setup Guide](docs/OLLAMA_SETUP.md) - Recommended for CLI users
- [LM Studio Setup Guide](docs/LM_STUDIO_SETUP.md) - GUI alternative

## Installation

### From PyPI (when published)
```bash
pip install acf
```

### From Source (local development)
```bash
git clone https://github.com/Tennisee-data/acf.git
cd acf
pip install -e .
```

> **Note**: If you cloned the repo, you must use `pip install -e .` from the project directory. Running `pip install acf` will attempt to install from PyPI, not your local copy.

### Optional Dependencies

From source:
```bash
pip install -e ".[semantic]"   # Semantic RAG (better code retrieval)
pip install -e ".[openai]"     # OpenAI backend
pip install -e ".[anthropic]"  # Anthropic backend
pip install -e ".[full]"       # Everything
```

From PyPI (when published):
```bash
pip install "acf[semantic]"
pip install "acf[openai]"
pip install "acf[anthropic]"
pip install "acf[full]"
```

## Usage

### Basic Generation
```bash
acf run "Build a REST API with FastAPI"
acf run "Add user authentication" --repo ./my-project
acf run "Build an API" --output ./my-api    # Custom output directory
```

Each run creates a project directory with code at the root and history in `.acf/`:
```
2026-01-24-130821/                # Project directory (run ID by default)
├── app/                          # Your generated code at root level
│   └── main.py
├── requirements.txt
├── .gitignore                    # Includes .acf/
└── .acf/                         # Pipeline history (gitignored)
    └── runs/
        └── 2026-01-24-130821/    # Run artifacts
            ├── state.json        # Pipeline state
            ├── feature_spec.json # Parsed requirements
            ├── design_proposal.json
            ├── change_set.json
            ├── diff.patch
            └── *.md              # Reports
```

Use `--output` to specify a custom project directory:
```bash
acf run "Build REST API" --output ./my-project
```

### CLI Reference

```bash
acf run "feature" [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--repo PATH` | `-r` | Target repository (default: current directory) |
| `--output PATH` | `-o` | Output directory for generated project (default: ./{run_id}/) |
| `--profile NAME` | `-p` | Configuration profile (default: dev) |
| `--auto-approve` | `-y` | Skip all approval prompts |
| `--resume ID` | | Resume a paused run by ID |
| `--dry-run` | | Show what would be done without executing |

**Pipeline Stages (built-in):**

| Option | Description |
|--------|-------------|
| `--decompose` | Break feature into sub-tasks before design |
| `--api-contract` | Generate OpenAPI spec before implementation |
| `--coverage` | Enforce test coverage (default: 80%) |
| `--coverage-threshold N` | Set coverage threshold (e.g., 90) |
| `--secrets-scan` | Detect hardcoded secrets, auto-fix with env vars |
| `--dependency-audit` | Scan for CVEs and outdated packages |
| `--rollback-strategy` | Generate CI/CD rollback and canary templates |
| `--observability` | Inject logging, metrics, tracing scaffolding |
| `--docs` | Generate documentation and ADRs |
| `--code-review` | Senior engineer code review |
| `--policy` | Enforce policy rules before verification |
| `--pr-package` | Build PR with changelog and spec links |

**Marketplace Extensions (install separately):**

| Option | Description |
|--------|-------------|
| `--config` | Enforce 12-factor config layout |

**Issue Integration:**

| Option | Description |
|--------|-------------|
| `--jira PROJ-123` | Fetch requirements from JIRA issue |
| `--issue URL` | Fetch from GitHub/JIRA issue URL |

### Examples

```bash
# Basic run with auto-approval
acf run "Add login rate-limit" --auto-approve

# Full pipeline with all quality checks
acf run "Build payment API" --api-contract --coverage --secrets-scan --code-review

# Resume a paused run
acf run "Add feature" --resume 2026-01-24-130821

# Iterate on existing project
acf iterate 2026-01-24-130821 "Add error handling"
```

### Other Commands

```bash
# List all runs
acf list

# Show run details
acf show 2026-01-24-130821

# Extract generated code to a directory
acf extract 2026-01-24-130821 --output ./my-project

# Create git scaffold with proper structure
acf scaffold 2026-01-24-130821

# Generate tests for existing code
acf generate-tests 2026-01-24-130821

# Deploy generated project
acf deploy 2026-01-24-130821 --version v1.0.0
```

### Extension Marketplace
```bash
# Browse extensions
acf marketplace search "security"
acf marketplace featured

# Install extensions
acf marketplace install secrets-scan
acf marketplace install semantic-rag

# Manage installed extensions
acf extensions list
acf extensions enable secrets-scan
```

## Configuration

Create `config.toml` in your project. **Make sure models match what you have installed** (`ollama list`):

```toml
[llm]
backend = "auto"  # ollama, lmstudio, openai, anthropic
model_general = "qwen2.5-coder:7b"  # Must match an installed model
model_code = "qwen2.5-coder:7b"

[extensions]
extensions_dir = "~/.coding-factory/extensions"

[routing]
# Enable to use different models for different task complexities
# All models must be installed: ollama pull <model>
enabled = true
model_cheap = "qwen2.5-coder:7b"    # Simple tasks
model_medium = "yi-coder:9b"         # Medium complexity
model_premium = "qwen2.5-coder:32b"  # Complex tasks
```

> **Common error:** `404 Not Found` means the model in config.toml is not installed. Run `ollama list` to see available models.

---

## Marketplace: Build & Sell Extensions

The ACF Marketplace lets you create and sell extensions. Earn money by building agents, profiles, or RAG kits that solve real problems.

**[Browse the Marketplace](https://api.agentcodefactory.com/marketplace/)** | **[Full Developer Guide](docs/MARKETPLACE_GUIDE.md)**

### Extension Types

| Type | Purpose | Price Range |
|------|---------|-------------|
| **Agents** | Add pipeline stages (security scanning, code review, etc.) | Free - $49 |
| **Profiles** | Framework templates (Vue, Go, Flutter, etc.) | Free - $15 |
| **RAG Kits** | Custom code retrieval systems | Free - $39 |

### Official Free Extensions

Get started with our free official extensions in `official_extensions/`:

| Extension | Hook Point | Description |
|-----------|------------|-------------|
| **decomposition** | before:design | Break complex features into subtasks |
| **api-contract** | before:implementation | Define API boundaries and contracts |
| **code-review** | after:implementation | Senior engineer review with feedback |

Install them:
```bash
cp -r official_extensions/* ~/.coding-factory/extensions/agents/
acf extensions list
```

### Quick Start: Create Your Own

1. **Create extension directory**
```bash
mkdir my-agent && cd my-agent
```

2. **Create manifest.yaml**
```yaml
name: my-agent
version: 1.0.0
type: agent
author: Your Name
description: What it does
license: free
hook_point: "after:implementation"
agent_class: MyAgent
```

3. **Create agent.py**
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

class MyAgent:
    def __init__(self, llm: Any, **kwargs):
        self.llm = llm
        self.name = "my-agent"

    def run(self, input_data) -> AgentOutput:
        context = input_data.context
        # Your logic here
        return AgentOutput(
            success=True,
            data={"result": "done"},
            agent_name=self.name
        )
```

4. **Install and test**
```bash
cp -r my-agent ~/.coding-factory/extensions/agents/
acf run "Build an API" --auto-approve
```

**[See full guide with profile and RAG kit examples →](docs/MARKETPLACE_GUIDE.md)**

### Submit to the Marketplace

Share your extensions with the community and earn from your work.

**1. Create an account**

Sign up at https://agentcodefactory.com/signup.html

**2. Verify your email**

Check your inbox and click the verification link. This unlocks API access and marketplace submissions.

Didn't receive it? Log in and go to Settings to resend.

**3. Create an API key**

Go to Settings → Platform API Keys and click "Create Key". Copy it immediately—it's only shown once.

```bash
export ACF_MARKETPLACE_API_KEY=acf_sk_xxxx...
```

**4. Build your extension**

Create a directory with:
```
my-extension/
├── manifest.yaml    # Required: name, version, type, description
├── agent.py         # Your code
└── README.md        # Documentation
```

**5. Submit for review**

```bash
acf marketplace submit ./my-extension --price 15.00
# Use --price 0 for free extensions
```

### Trust Badges

<table>
<tr>
<td align="center" width="200">
<img src="https://raw.githubusercontent.com/Tennisee-data/acf/main/docs/badges/genuine_badge.png" width="80" alt="Genuine Badge"/>
<br/><strong>Genuine</strong>
<br/><em>Official ACF Extension</em>
</td>
<td align="center" width="200">
<img src="https://raw.githubusercontent.com/Tennisee-data/acf/main/docs/badges/checked_badge.png" width="80" alt="Checked Badge"/>
<br/><strong>Checked</strong>
<br/><em>Security Reviewed</em>
</td>
<td align="center" width="200">
<img src="https://raw.githubusercontent.com/Tennisee-data/acf/main/docs/badges/trusted_badge.png" width="80" alt="Trusted Badge"/>
<br/><strong>Trusted</strong>
<br/><em>Proven Contributor</em>
</td>
</tr>
</table>

### Revenue

- **You keep 82.35%** of each sale
- 17.65% covers payment processing (Stripe/PayPal) and infrastructure
- **Payments via PayPal or Stripe**

---

## Architecture

```
~/.coding-factory/
├── extensions/          # Installed extensions
│   ├── agents/
│   ├── profiles/
│   └── rag/
└── memory/              # Learning from past runs
```

## Links

- [Marketplace](https://marketplace.agentcodefactory.com)
- [Developer Guide](https://marketplace.agentcodefactory.com/docs/developers)
- [Documentation](https://agentcodefactory.com/docs)
- [Main Platform](https://agentcodefactory.com)

## License

MIT License - Free for personal and commercial use.
