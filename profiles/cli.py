"""CLI Application Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects CLI as their application type. It provides patterns for
Python CLIs using Click/Typer and Rust CLIs using Clap.
"""

PROFILE_NAME = "cli"
PROFILE_VERSION = "1.0"

# Technologies covered by this profile
TECHNOLOGIES = ["cli", "click", "typer", "argparse", "clap"]

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## CLI Application Expert Guidelines

You are generating a CLI application. Follow these patterns exactly:

### Python CLI with Typer (PREFERRED)

```python
# main.py
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(
    name="myapp",
    help="My awesome CLI application",
    add_completion=True,
)
console = Console()

@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file to process"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    format: str = typer.Option("json", "--format", "-f", help="Output format"),
):
    \"\"\"Process the input file and generate output.\"\"\"
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(code=1)

    if verbose:
        console.print(f"[blue]Processing:[/blue] {input_file}")

    with Progress() as progress:
        task = progress.add_task("Processing...", total=100)
        # Do work here
        progress.update(task, advance=100)

    console.print("[green]Done![/green]")

@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Directory to initialize"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
):
    \"\"\"Initialize a new project.\"\"\"
    config_file = directory / "config.toml"

    if config_file.exists() and not force:
        console.print("[yellow]Config already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(code=1)

    config_file.write_text('# Configuration\\n')
    console.print(f"[green]Initialized:[/green] {config_file}")

if __name__ == "__main__":
    app()
```

### Project Structure (Python CLI)

```
mycli/
├── src/
│   └── mycli/
│       ├── __init__.py
│       ├── __main__.py      # Entry point
│       ├── cli.py           # CLI commands
│       ├── commands/        # Subcommands
│       │   ├── __init__.py
│       │   ├── process.py
│       │   └── config.py
│       ├── utils/
│       └── config.py
├── tests/
├── pyproject.toml
└── README.md
```

### pyproject.toml (Python)

```toml
[project]
name = "mycli"
version = "0.1.0"
description = "My CLI application"
requires-python = ">=3.11"
dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.7.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

[project.scripts]
mycli = "mycli.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Configuration Handling

```python
# config.py
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    \"\"\"Application configuration from env/file.\"\"\"
    api_key: str = ""
    debug: bool = False
    output_dir: Path = Path("./output")

    class Config:
        env_file = ".env"
        env_prefix = "MYCLI_"

def load_config(config_path: Path | None = None) -> Config:
    \"\"\"Load config from file or environment.\"\"\"
    if config_path and config_path.exists():
        import tomllib
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
        return Config(**data)
    return Config()
```

### Error Handling

```python
# exceptions.py
class CLIError(Exception):
    \"\"\"Base CLI error with exit code.\"\"\"
    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)

class ConfigError(CLIError):
    \"\"\"Configuration error.\"\"\"
    pass

class InputError(CLIError):
    \"\"\"Invalid input error.\"\"\"
    pass

# In CLI
@app.command()
def run():
    try:
        do_work()
    except CLIError as e:
        console.print(f"[red]Error:[/red] {e.message}")
        raise typer.Exit(code=e.exit_code)
```

### Interactive Input

```python
from rich.prompt import Prompt, Confirm

@app.command()
def setup():
    \"\"\"Interactive setup wizard.\"\"\"
    name = Prompt.ask("Project name", default="myproject")
    include_tests = Confirm.ask("Include test suite?", default=True)

    console.print(f"Setting up {name}...")
```

### Progress and Output

```python
from rich.table import Table
from rich.progress import track

# Table output
def show_results(items: list[dict]):
    table = Table(title="Results")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")

    for item in items:
        table.add_row(item["name"], item["status"])

    console.print(table)

# Progress bar
for item in track(items, description="Processing..."):
    process(item)
```

### Testing CLI

```python
# tests/test_cli.py
from typer.testing import CliRunner
from mycli.cli import app

runner = CliRunner()

def test_process_success():
    result = runner.invoke(app, ["process", "input.txt"])
    assert result.exit_code == 0
    assert "Done" in result.stdout

def test_process_missing_file():
    result = runner.invoke(app, ["process", "nonexistent.txt"])
    assert result.exit_code == 1
    assert "not found" in result.stdout
```

### Common Mistakes to Avoid

1. **Not using type hints** - Typer relies on type hints for argument parsing
2. **Poor error messages** - Always explain what went wrong and how to fix
3. **No progress feedback** - Use progress bars for long operations
4. **Missing --help** - Typer adds this automatically, but write good docstrings
5. **Not handling Ctrl+C** - Use `try/except KeyboardInterrupt`
6. **Hardcoded paths** - Use Path objects and config files

### Shell Completion

```python
# Install completions
# mycli --install-completion

# Or generate completion script
# mycli --show-completion > ~/.zshrc
```

### Required Dependencies (Python)

```
typer[all]>=0.9.0
rich>=13.7.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
```
"""

DEPENDENCIES = [
    "typer[all]>=0.9.0",
    "rich>=13.7.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

OPTIONAL_DEPENDENCIES = {
    "http": ["httpx>=0.26.0"],
    "database": ["sqlalchemy>=2.0.0"],
    "testing": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
}

TRIGGER_KEYWORDS = [
    "cli",
    "command line",
    "terminal app",
    "typer",
    "click",
    "argparse",
    "shell tool",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied."""
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["cli", "typer", "click"]):
            return True

    # Check prompt keywords
    return any(kw in prompt_lower for kw in TRIGGER_KEYWORDS)


def get_guidance() -> str:
    """Get the full guidance text to inject into prompts."""
    return SYSTEM_GUIDANCE


def get_dependencies(features: list[str] | None = None) -> list[str]:
    """Get recommended dependencies."""
    deps = DEPENDENCIES.copy()

    if features:
        for feature in features:
            if feature in OPTIONAL_DEPENDENCIES:
                deps.extend(OPTIONAL_DEPENDENCIES[feature])

    return deps
