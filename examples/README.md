# ACF Examples

This directory contains example extensions and configurations for ACF Local Edition.

## Extensions

### [hello-checker](extensions/hello-checker/)

A simple example extension agent that demonstrates:
- Extension manifest structure
- Agent implementation pattern
- Hook point registration
- Pipeline context access

**Install it:**
```bash
cp -r examples/extensions/hello-checker ~/.coding-factory/extensions/agents/
acf extensions list
```

## Creating Extensions

Extensions allow you to extend ACF's pipeline with custom functionality:

| Type | Purpose | Hook Points |
|------|---------|-------------|
| **Agents** | Add custom pipeline stages | before/after any stage |
| **Profiles** | Define tech stack templates | N/A |
| **RAG Kits** | Custom code retrieval | N/A |

### Quick Start

1. Create extension directory:
   ```bash
   mkdir -p ~/.coding-factory/extensions/agents/my-agent
   ```

2. Create `manifest.yaml`:
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

3. Create `agent.py`:
   ```python
   class MyAgent:
       def __init__(self, llm, **kwargs):
           self.llm = llm
           self.name = "my-agent"

       def run(self, input_data):
           # Your logic here
           return AgentOutput(
               success=True,
               data={"result": "done"},
               agent_name=self.name
           )
   ```

4. Verify:
   ```bash
   acf extensions list
   ```

## Marketplace

Once your extension works locally, you can submit it to the marketplace:

```bash
cd ~/.coding-factory/extensions/agents/my-agent
tar -czvf my-agent-1.0.0.tar.gz .
acf marketplace submit ./my-agent-1.0.0.tar.gz
```

See [README.md](../README.md) for full marketplace documentation.
