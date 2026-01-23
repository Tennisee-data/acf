# Official Free Extensions

These are official ACF extensions available for free. Install them to extend your pipeline with additional capabilities.

## Available Extensions

| Extension | Hook Point | Description |
|-----------|------------|-------------|
| **decomposition** | before:design | Break complex features into smaller subtasks |
| **api-contract** | before:implementation | Define API boundaries and contracts |
| **code-review** | after:implementation | Senior engineer review with feedback |

## Installation

Copy the extension you want to your extensions directory:

```bash
# Install decomposition
cp -r official_extensions/decomposition ~/.coding-factory/extensions/agents/

# Install api-contract
cp -r official_extensions/api-contract ~/.coding-factory/extensions/agents/

# Install code-review
cp -r official_extensions/code-review ~/.coding-factory/extensions/agents/

# Or install all at once
cp -r official_extensions/* ~/.coding-factory/extensions/agents/
```

## Verify Installation

```bash
acf extensions list
```

You should see:
```
┃ decomposition │ agent │ 1.0.0   │ before:design         │ free │ ACF Team │
┃ api-contract  │ agent │ 1.0.0   │ before:implementation │ free │ ACF Team │
┃ code-review   │ agent │ 1.0.0   │ after:implementation  │ free │ ACF Team │
```

## Usage

Once installed, these extensions run automatically at their designated hook points:

1. **decomposition** - Runs before design to analyze complexity
2. **api-contract** - Runs before implementation to define API contracts
3. **code-review** - Runs after implementation to review generated code

## Customization

You can modify these extensions or use them as templates for your own:

1. Copy the extension to a new directory
2. Update `manifest.yaml` with your changes
3. Modify `agent.py` with your logic
4. Install to `~/.coding-factory/extensions/agents/`

## Creating Your Own Extensions

See the [main README](../README.md#creating-an-extension) for full documentation on creating marketplace extensions.
