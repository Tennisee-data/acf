# Creating Agents

Agents hook into the pipeline at specific stages to add analysis, checks, or modifications.

## When to Use Agents

- Add automated code review
- Run security scans
- Validate architecture decisions
- Inject additional context
- Transform pipeline artifacts

## Directory Structure

```
my-agent/
├── manifest.yaml      # Required: metadata
├── agent.py           # Required: implementation
├── requirements.txt   # Optional: dependencies
└── README.md          # Recommended: documentation
```

Place in: `~/.coding-factory/extensions/agents/my-agent/`

## Manifest Schema

```yaml
name: security-scanner
version: 1.0.0
type: agent
author: Your Name
description: Scan generated code for security vulnerabilities
license: free

# Required for agents
agent_class: SecurityScannerAgent   # Class name in agent.py
hook_point: "after:implementation"  # When to run

# Dependencies
requires:
  - bandit>=1.7.0

# Metadata
keywords:
  - security
  - scanner
  - vulnerabilities

priority: 50                        # Execution order (lower = first)

# Resource hints
context_tokens: 2000                # Tokens added to context
min_model_tier: medium              # any | small | medium | large
```

## Hook Points

Agents run at specific points in the pipeline:

```
before:spec          after:spec
before:context       after:context
before:design        after:design
before:implementation  after:implementation
before:testing       after:testing
before:verification  after:verification
before:code_review   after:code_review
before:deploy        after:deploy
```

### Choosing a Hook Point

| Hook Point | Use Case |
|------------|----------|
| `before:spec` | Modify or validate the feature request |
| `after:spec` | Add requirements, flag concerns |
| `before:design` | Inject architectural constraints |
| `after:design` | Review design, add patterns |
| `before:implementation` | Provide additional context |
| `after:implementation` | Code review, security scan |
| `after:testing` | Validate test coverage |
| `before:deploy` | Final checks before deployment |

## Implementation

### Agent Output Format

All agents must return an `AgentOutput`:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentOutput:
    """Standard output format for agents."""

    success: bool                           # Did the agent succeed?
    data: dict[str, Any]                    # Results data
    errors: list[str] | None = None         # Error messages
    artifacts: list[str] | None = None      # Created file names
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None           # Your agent's name
```

### Basic Agent

```python
"""Security Scanner Agent - Scan code for vulnerabilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentOutput:
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None


class SecurityScannerAgent:
    """Scan generated code for security vulnerabilities.

    Runs after implementation to check for common security
    issues like SQL injection, XSS, and hardcoded secrets.
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        """Initialize the agent.

        Args:
            llm: LLM backend for AI-powered analysis
            **kwargs: Additional config from pipeline
        """
        self.llm = llm
        self.name = "security-scanner"

    def default_system_prompt(self) -> str:
        """System prompt for LLM calls."""
        return """You are a security expert analyzing code for vulnerabilities.
Focus on:
- SQL injection
- XSS vulnerabilities
- Hardcoded secrets
- Insecure deserialization
- Path traversal"""

    def run(self, input_data: Any) -> AgentOutput:
        """Execute the security scan.

        Args:
            input_data: Pipeline context with artifacts

        Returns:
            AgentOutput with scan results
        """
        # Access pipeline context
        context = getattr(input_data, 'context', input_data)

        # Get paths from context
        run_dir = Path(context.get("run_dir", "."))
        repo_path = Path(context.get("repo_path", "."))

        # Get artifacts from previous stages
        spec = context.get("spec", {})
        design = context.get("design", {})
        implementation = context.get("implementation", {})

        # Run your analysis
        vulnerabilities = self._scan_code(run_dir, repo_path)

        # Save report
        report_path = run_dir / "security_report.json"
        self._save_report(report_path, vulnerabilities)

        return AgentOutput(
            success=len(vulnerabilities) == 0,
            data={
                "vulnerabilities": vulnerabilities,
                "scanned_files": self._count_files(repo_path),
                "passed": len(vulnerabilities) == 0,
            },
            artifacts=["security_report.json"],
            agent_name=self.name,
        )

    def _scan_code(self, run_dir: Path, repo_path: Path) -> list[dict]:
        """Scan code for vulnerabilities."""
        vulnerabilities = []

        # Example: Check for hardcoded secrets
        for py_file in repo_path.glob("**/*.py"):
            content = py_file.read_text()
            if "password=" in content.lower() or "secret=" in content.lower():
                vulnerabilities.append({
                    "file": str(py_file),
                    "type": "hardcoded_secret",
                    "severity": "high",
                    "message": "Possible hardcoded secret detected",
                })

        return vulnerabilities

    def _count_files(self, repo_path: Path) -> int:
        return len(list(repo_path.glob("**/*.py")))

    def _save_report(self, path: Path, vulnerabilities: list) -> None:
        import json
        path.write_text(json.dumps(vulnerabilities, indent=2))
```

### AI-Powered Agent

Use the LLM for intelligent analysis:

```python
class CodeReviewAgent:
    """Senior engineer code review using AI."""

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.llm = llm
        self.name = "code-review"

    def default_system_prompt(self) -> str:
        return """You are a senior software engineer conducting a code review.
Focus on:
- Code correctness and logic errors
- Performance issues
- Maintainability and readability
- Best practices for the language/framework
- Security concerns

Provide specific, actionable feedback."""

    def run(self, input_data: Any) -> AgentOutput:
        context = getattr(input_data, 'context', input_data)
        run_dir = Path(context.get("run_dir", "."))

        # Get the generated code
        code_files = self._get_code_files(run_dir)

        if not code_files:
            return AgentOutput(
                success=True,
                data={"message": "No code files to review"},
                agent_name=self.name,
            )

        # Review with LLM
        review = self._review_code(code_files)

        return AgentOutput(
            success=True,
            data={"review": review},
            artifacts=["code_review.md"],
            agent_name=self.name,
        )

    def _get_code_files(self, run_dir: Path) -> dict[str, str]:
        """Collect code files for review."""
        files = {}
        generated = run_dir / "generated_project"
        if generated.exists():
            for f in generated.glob("**/*.py"):
                files[str(f.relative_to(generated))] = f.read_text()
        return files

    def _review_code(self, code_files: dict[str, str]) -> str:
        """Review code using LLM."""
        code_summary = "\n\n".join(
            f"### {path}\n```python\n{content}\n```"
            for path, content in code_files.items()
        )

        messages = [
            {"role": "system", "content": self.default_system_prompt()},
            {"role": "user", "content": f"Review this code:\n\n{code_summary}"},
        ]

        return self.llm.chat(messages)
```

## Accessing Pipeline Context

The `input_data.context` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `run_dir` | `str` | Path to current run's artifact directory |
| `repo_path` | `str` | Path to the repository being worked on |
| `run_id` | `str` | Unique identifier for this run |
| `spec` | `dict` | Parsed feature specification |
| `design` | `dict` | Design proposal |
| `implementation` | `dict` | Implementation details |
| `artifacts_dir` | `str` | Path to artifacts directory |

## Priority and Ordering

Multiple agents at the same hook point run in priority order:

```yaml
priority: 30   # Runs before default (50)
priority: 50   # Default
priority: 70   # Runs after default
```

Lower numbers run first.

## Error Handling

Return `success: False` with errors to signal failure:

```python
def run(self, input_data: Any) -> AgentOutput:
    try:
        # Your logic
        result = self._analyze()
        return AgentOutput(success=True, data=result, agent_name=self.name)
    except Exception as e:
        return AgentOutput(
            success=False,
            data={},
            errors=[f"Analysis failed: {str(e)}"],
            agent_name=self.name,
        )
```

## Testing Your Agent

```bash
# Install locally
cp -r my-agent ~/.coding-factory/extensions/agents/

# Verify
acf extensions list

# Run pipeline to trigger agent
acf run "Build a simple API" --auto-approve
```

Watch for your agent in the output:
```
Extension: my-agent (after:implementation)
  ✓ my-agent completed
```

## Examples

See official agents:
- [`code-review`](https://github.com/Tennisee-data/acf/tree/main/official_extensions/code-review)
- [`decomposition`](https://github.com/Tennisee-data/acf/tree/main/official_extensions/decomposition)
- [`api-contract`](https://github.com/Tennisee-data/acf/tree/main/official_extensions/api-contract)

## Next Steps

- [Specification](../specification.md) - Full manifest schema
- [Publishing](../publishing.md) - Submit to marketplace
