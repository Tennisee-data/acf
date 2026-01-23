# Hello Checker - Example Extension Agent

This is a simple example extension agent that demonstrates how to create marketplace extensions for ACF.

## What it Does

The Hello Checker agent runs **after the implementation stage** and checks the generated code for common patterns:

- Flask/FastAPI imports
- Route decorators
- JSON response handling
- `/hello` endpoint presence

## Installation

Copy this extension to your extensions directory:

```bash
mkdir -p ~/.coding-factory/extensions/agents/hello-checker
cp manifest.yaml agent.py ~/.coding-factory/extensions/agents/hello-checker/
```

## Verify Installation

```bash
acf extensions list
```

You should see:
```
┃ hello-checker │ agent │ 1.0.0   │ after:implementation │ free    │ ACF Team │
```

## How it Works

1. **Discovery**: ACF scans `~/.coding-factory/extensions/agents/` for directories with `manifest.yaml`
2. **Loading**: The manifest defines `agent_class: HelloCheckerAgent` which maps to the class in `agent.py`
3. **Execution**: The agent runs at `hook_point: after:implementation`, receiving the pipeline context
4. **Output**: Returns an `AgentOutput` with check results that get logged in the pipeline

## Extension Structure

```
hello-checker/
├── manifest.yaml    # Extension metadata and configuration
├── agent.py         # Agent implementation
└── README.md        # Documentation
```

## Key Files

### manifest.yaml

```yaml
name: hello-checker
version: 1.0.0
type: agent
hook_point: "after:implementation"
agent_class: HelloCheckerAgent
```

### agent.py

The agent class must:
- Accept `llm` in `__init__` (even if not used)
- Implement `run(input_data) -> AgentOutput`
- Return an `AgentOutput` with `success`, `data`, and optional `artifacts`

## Creating Your Own Extension

1. Copy this example as a starting point
2. Modify `manifest.yaml` with your extension name and hook point
3. Implement your logic in `agent.py`
4. Install to `~/.coding-factory/extensions/agents/<name>/`
5. Test with `acf run "your feature" --auto-approve`

## Available Hook Points

- `before:spec`, `after:spec`
- `before:context`, `after:context`
- `before:design`, `after:design`
- `before:implementation`, `after:implementation`
- `before:testing`, `after:testing`
- `before:verification`, `after:verification`
- And more... see `extensions/manifest.py` for full list
