# ACF - AgentCodeFactory Local Edition

100% local AI-powered code generation pipeline. No cloud required.

> **[Read the Manifesto](MANIFESTO.md)** — Why we believe local AI is the future of secure, private code generation.

## Features

- **Fully Offline**: Runs entirely on your machine with Ollama or LM Studio
- **7-Stage Pipeline**: SPEC → CONTEXT → DESIGN → IMPLEMENTATION → TESTING → REVIEW → DONE
- **Multi-Model Routing**: Automatically routes tasks to appropriate model sizes
- **Extension Marketplace**: Install premium agents, profiles, and RAG kits
- **Local Git Versioning**: Every iteration auto-committed for easy rollback

## Quick Start

```bash
# Install
pip install acf

# Or with all optional features
pip install acf[full]

# Run (requires Ollama running locally)
acf run "Add user authentication with JWT"
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) or [LM Studio](https://lmstudio.ai)
- Recommended model: `qwen2.5-coder:14b`

## Installation

### From PyPI
```bash
pip install acf
```

### From Source
```bash
git clone https://github.com/Tennisee-data/acf.git
cd acf
pip install -e .
```

### Optional Dependencies
```bash
# Semantic RAG (better code retrieval)
pip install acf[semantic]

# OpenAI backend
pip install acf[openai]

# Anthropic backend
pip install acf[anthropic]

# Everything
pip install acf[full]
```

## Usage

### Basic Generation
```bash
acf run "Build a REST API with FastAPI"
acf run "Add user authentication" --repo ./my-project
```

### With Pipeline Options
```bash
acf run "Add payments" --code-review --secrets-scan --coverage
acf run "Build feature" --decompose --api-contract
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

Create `config.toml` in your project:

```toml
[llm]
backend = "auto"  # ollama, lmstudio, openai, anthropic
model = "qwen2.5-coder:14b"

[extensions]
extensions_dir = "~/.coding-factory/extensions"

[routing]
enabled = true
model_cheap = "qwen2.5-coder:7b"
model_premium = "qwen2.5-coder:32b"
```

---

## Marketplace: Build & Sell Extensions

The ACF Marketplace lets you create and sell extensions. Earn money by building agents, profiles, or RAG kits that solve real problems.

**[Browse the Marketplace](https://api.agentcodefactory.com/marketplace/)** | **[Full Developer Guide](docs/MARKETPLACE_GUIDE.md)**

### Extension Types

| Type | Purpose | Price Range |
|------|---------|-------------|
| **Agents** | Add pipeline stages (security scanning, code review, etc.) | $5-$49 |
| **Profiles** | Framework templates (Vue, Go, Flutter, etc.) | $5-$15 |
| **RAG Kits** | Custom code retrieval systems | $10-$39 |

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

5. **Submit to marketplace**
```bash
tar -czvf my-agent-1.0.0.tar.gz .
acf marketplace submit ./my-agent-1.0.0.tar.gz --name my-agent --version 1.0.0 --type agent
```

**[See full guide with profile and RAG kit examples →](docs/MARKETPLACE_GUIDE.md)**

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
