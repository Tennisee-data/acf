"""Project generator for scaffolding new projects."""

import os
import re
import subprocess
from pathlib import Path

from rich.console import Console

from .templates import ProjectTemplate, get_template, TEMPLATES


console = Console()


class ProjectGenerator:
    """Generates new projects from templates.

    Creates:
    - Directory structure
    - Configuration files
    - Source code
    - Tests
    - Documentation
    - Git repository
    - Docker configuration
    """

    def __init__(
        self,
        name: str,
        template: str = "fastapi",
        description: str | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize project generator.

        Args:
            name: Project name (e.g., "My API")
            template: Template name (fastapi, cli, minimal)
            description: Project description
            output_dir: Parent directory for project (default: current dir)
        """
        self.name = name
        self.slug = self._slugify(name)
        self.description = description or f"A {template} project"
        self.output_dir = output_dir or Path.cwd()
        self.project_dir = self.output_dir / self.slug

        self.template = get_template(template)
        if not self.template:
            available = ", ".join(TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template}. Available: {available}")

    def _slugify(self, name: str) -> str:
        """Convert name to slug (lowercase, hyphens)."""
        # Convert to lowercase
        slug = name.lower()
        # Replace spaces and underscores with hyphens
        slug = re.sub(r"[\s_]+", "-", slug)
        # Remove non-alphanumeric characters except hyphens
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        # Remove multiple consecutive hyphens
        slug = re.sub(r"-+", "-", slug)
        # Remove leading/trailing hyphens
        slug = slug.strip("-")
        return slug or "project"

    def generate(self, init_git: bool = True, install_deps: bool = False) -> Path:
        """Generate the project.

        Args:
            init_git: Initialize git repository
            install_deps: Install dependencies after creation

        Returns:
            Path to created project directory
        """
        console.print(f"\n[bold blue]Creating project:[/bold blue] {self.name}")
        console.print(f"[dim]Template: {self.template.name}[/dim]")
        console.print(f"[dim]Directory: {self.project_dir}[/dim]\n")

        # Check if directory exists
        if self.project_dir.exists():
            raise FileExistsError(f"Directory already exists: {self.project_dir}")

        # Create project directory
        self.project_dir.mkdir(parents=True)
        console.print(f"[green]Created:[/green] {self.project_dir}/")

        # Create subdirectories
        self._create_directories()

        # Create files from template
        self._create_files()

        # Initialize git
        if init_git:
            self._init_git()

        # Install dependencies
        if install_deps:
            self._install_deps()

        console.print(f"\n[bold green]Project created successfully![/bold green]")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  cd {self.slug}")
        console.print(f"  pip install -e \".[dev]\"")
        if self.template.name == "fastapi":
            console.print(f"  cp .env.example .env")
            console.print(f"  python -m src.main")
        elif self.template.name == "cli":
            console.print(f"  {self.slug} --help")
        else:
            console.print(f"  python -m src.main")

        return self.project_dir

    def _create_directories(self) -> None:
        """Create directory structure."""
        for directory in self.template.directories:
            dir_path = self.project_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created:[/green] {directory}/")

    def _create_files(self) -> None:
        """Create files from template."""
        # Prepare template variables
        variables = {
            "project_name": self.name,
            "project_slug": self.slug,
            "project_description": self.description,
            "dependencies": self._format_dependencies(self.template.dependencies),
            "dev_dependencies": self._format_dependencies(self.template.dev_dependencies),
        }

        for file_path, content in self.template.files.items():
            # Get content
            if callable(content):
                file_content = content(variables)
            else:
                file_content = content

            # Substitute variables
            file_content = self._substitute_variables(file_content, variables)

            # Write file
            full_path = self.project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_content)
            console.print(f"[green]Created:[/green] {file_path}")

    def _substitute_variables(self, content: str, variables: dict) -> str:
        """Substitute {variable} placeholders in content."""
        for key, value in variables.items():
            content = content.replace("{" + key + "}", str(value))
        return content

    def _format_dependencies(self, deps: list[str]) -> str:
        """Format dependencies for pyproject.toml."""
        if not deps:
            return ""
        return ",\n    ".join(f'"{dep}"' for dep in deps)

    def _init_git(self) -> None:
        """Initialize git repository."""
        try:
            # Initialize repo
            subprocess.run(
                ["git", "init"],
                cwd=self.project_dir,
                capture_output=True,
                check=True,
            )
            console.print(f"[green]Initialized:[/green] git repository")

            # Create initial commit
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_dir,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit from Coding Factory"],
                cwd=self.project_dir,
                capture_output=True,
                check=True,
            )
            console.print(f"[green]Created:[/green] initial commit")

        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Warning:[/yellow] Git initialization failed: {e}")
        except FileNotFoundError:
            console.print(f"[yellow]Warning:[/yellow] Git not found, skipping initialization")

    def _install_deps(self) -> None:
        """Install project dependencies."""
        console.print("\n[bold]Installing dependencies...[/bold]")
        try:
            subprocess.run(
                ["pip", "install", "-e", ".[dev]"],
                cwd=self.project_dir,
                check=True,
            )
            console.print(f"[green]Installed:[/green] dependencies")
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Warning:[/yellow] Dependency installation failed: {e}")


def list_templates() -> None:
    """List available project templates."""
    console.print("\n[bold]Available Templates:[/bold]\n")
    for name, template in TEMPLATES.items():
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    {template.description}")
        console.print(f"    Language: {template.language}, Framework: {template.framework}")
        console.print()
