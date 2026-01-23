# AgentCodeFactory CLI & SDK

A command-line interface and Python SDK for the AgentCodeFactory code generation platform.

## Installation

```bash
pip install agentcodefactory
```

Or install from source:

```bash
git clone https://github.com/agentcodefactory/cli
cd cli
pip install -e .
```

## Quick Start

### 1. Get Your API Key

Generate an API key at [agentcodefactory.com/dashboard/settings](https://agentcodefactory.com/dashboard/settings)

### 2. Configure the CLI

```bash
acf config set api_key acf_sk_your_key_here
```

### 3. Verify Authentication

```bash
acf whoami
```

### 4. Generate Code

```bash
acf generate "Build a REST API with user authentication using FastAPI"
```

## CLI Commands

### Configuration

```bash
# Show current configuration
acf config show

# Set configuration values
acf config set api_key <your-api-key>
acf config set default_model claude-sonnet-4-20250514
acf config set default_provider anthropic

# Get a specific value
acf config get api_key
```

Configuration is stored at `~/.agentcodefactory/config.toml`. You can also use environment variables:

- `ACF_API_KEY` - API key (overrides config file)
- `ACF_API_URL` - API URL (default: https://api.agentcodefactory.com)

### Authentication

```bash
# Verify API key and show account info
acf whoami
```

### Code Generation

```bash
# Basic generation
acf generate "Build a REST API with FastAPI"

# Read prompt from file
acf generate -f spec.md

# Specify tech stack
acf generate "Build an API" -t python -t fastapi -t postgresql

# Add to existing project
acf generate "Add user authentication" -p <project_id>

# Create a named project
acf generate "Build a TODO app" -n "my-todo-app"

# Output to specific directory
acf generate "Build an API" -o ./output

# Don't write files, just show result
acf generate "Build an API" --no-write

# Output raw JSON
acf generate "Build an API" --json
```

#### Pipeline Options

Enable additional pipeline stages for more thorough generation:

```bash
# Generate with documentation
acf generate "Build an API" --docs

# Enable code review
acf generate "Build an API" --code-review

# Generate OpenAPI contract
acf generate "Build an API" --api-contract

# Scan for hardcoded secrets
acf generate "Build an API" --secrets-scan

# Enforce test coverage
acf generate "Build an API" --coverage

# Full pipeline with all checks
acf generate "Build an API" --docs --code-review --secrets-scan --coverage

# Use premium models for all stages
acf generate "Build an API" --quality

# Custom policy rules
acf generate "Build an API" --policy --policy-file rules.yaml
```

#### Reviewing Existing Code

```bash
# Review code without generating new code
acf generate "Review this code for security issues" -c ./src --code-review --no-decompose
```

### Cost Estimation

```bash
# Estimate cost before generation
acf estimate "Build a REST API with authentication"

# Estimate with tech stack
acf estimate "Build an API" -t python -t fastapi
```

### Projects

```bash
# List all projects
acf projects list

# Show project details with iterations
acf projects show <project_id>

# Delete a project
acf projects delete <project_id>

# Delete without confirmation
acf projects delete <project_id> -y
```

### Iterations

```bash
# List iterations for a project
acf iterations list <project_id>

# Show iteration details
acf iterations show <iteration_id>

# Download and extract iteration files
acf iterations download <iteration_id>

# Download to specific directory
acf iterations download <iteration_id> -d ./output

# Download as ZIP file
acf iterations download <iteration_id> --format zip

# Download to specific file
acf iterations download <iteration_id> --format zip -o myproject.zip
```

### Jobs

```bash
# Check job status
acf jobs status <job_id>

# Get status as JSON
acf jobs status <job_id> --json
```

## Python SDK

The SDK provides programmatic access to all AgentCodeFactory features.

### Basic Usage

```python
from agentcodefactory import ACFClient

# Initialize client
client = ACFClient(api_key="acf_sk_...")

# Generate code
job = client.generate(
    prompt="Build a REST API with user authentication",
    tech_stack=["python", "fastapi"],
    model="claude-sonnet-4-20250514"
)

# Wait for completion
result = job.wait()

# Write files to current directory
result.write_files()

# Or write to specific directory
result.write_files("./output")
```

### Streaming Progress

```python
job = client.generate("Build an API")

# Stream progress updates
for update in job.stream():
    print(f"Status: {update.status}")
    print(f"Progress: {update.progress}")

# Access result after streaming
result = job.result
```

### Working with Projects

```python
# List projects
projects = client.list_projects()
for p in projects:
    print(f"{p.id}: {p.name} ({p.iteration_count} iterations)")

# Get project details
project = client.get_project("project_id")

# List iterations
iterations = client.list_iterations("project_id")

# Add iteration to existing project
job = client.generate(
    prompt="Add user authentication",
    project_id="existing_project_id"
)
```

### Downloading Files

```python
# Get iteration details
iteration = client.get_iteration("iteration_id")

# Download as ZIP bytes
zip_content = client.download_iteration("iteration_id")

# Download to file
client.download_iteration_to_file("iteration_id", "output.zip")

# Extract to directory
files = client.extract_iteration("iteration_id", "./output")
print(f"Extracted {len(files)} files")
```

### Pipeline Options

```python
job = client.generate(
    prompt="Build a REST API",
    pipeline_options={
        "decompose": True,      # Break into sub-tasks
        "docs": True,           # Generate documentation
        "code_review": True,    # Senior engineer review
        "api_contract": True,   # OpenAPI contract
        "secrets_scan": True,   # Scan for secrets
        "coverage": True,       # Test coverage
        "policy": True,         # Custom policies
    },
    policy_rules="your-yaml-rules-here"
)
```

### Cost Estimation

```python
estimate = client.estimate(
    prompt="Build a REST API",
    tech_stack=["python", "fastapi"]
)

print(f"Estimated tokens: {estimate['estimated_input_tokens']:,}")
print(f"Estimated cost: ${estimate['estimated_cost_usd']:.4f}")
```

### Error Handling

```python
from agentcodefactory.client import (
    ACFError,
    AuthenticationError,
    RateLimitError,
    JobError
)

try:
    job = client.generate("Build an API")
    result = job.wait()
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, please wait")
except JobError as e:
    print(f"Generation failed: {e}")
except ACFError as e:
    print(f"API error: {e}")
```

### Context Manager

```python
# Client cleans up automatically
with ACFClient(api_key="acf_sk_...") as client:
    result = client.generate("Build an API").wait()
    result.write_files()
```

## Configuration

The CLI stores configuration in `~/.agentcodefactory/config.toml`:

```toml
api_key = "acf_sk_..."
api_url = "https://api.agentcodefactory.com"
default_model = "claude-sonnet-4-20250514"
default_provider = "anthropic"
auto_approve = true
```

### Available Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `api_key` | Your platform API key | (required) |
| `api_url` | API base URL | https://api.agentcodefactory.com |
| `default_model` | Default model for generation | claude-sonnet-4-20250514 |
| `default_provider` | AI provider (anthropic/openai) | anthropic |
| `auto_approve` | Auto-approve pipeline stages | true |

## Models

Available models:

- `claude-sonnet-4-20250514` (default) - Fast and capable
- `claude-opus-4-20250514` - Most capable, higher cost

## API Reference

Full API documentation available at [api.agentcodefactory.com/docs](https://api.agentcodefactory.com/docs)

## Support

- Documentation: [agentcodefactory.com/docs](https://agentcodefactory.com/docs)
- Issues: [github.com/agentcodefactory/cli/issues](https://github.com/agentcodefactory/cli/issues)
- Email: support@agentcodefactory.com

## License

MIT
