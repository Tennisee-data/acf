"""Project templates for scaffolding.

Each template defines:
- File structure
- File contents (as templates with variables)
- Dependencies
- Configuration
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProjectTemplate:
    """Definition of a project template."""

    name: str
    description: str
    language: str
    framework: str
    files: dict[str, str | Callable]  # path -> content or generator function
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)


# =============================================================================
# FastAPI Template
# =============================================================================

FASTAPI_TEMPLATE = ProjectTemplate(
    name="fastapi",
    description="FastAPI REST API with Docker and tests",
    language="python",
    framework="fastapi",
    directories=[
        "src",
        "src/api",
        "src/api/routes",
        "src/core",
        "src/models",
        "src/services",
        "tests",
        "tests/api",
        "docs",
    ],
    dependencies=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
    ],
    dev_dependencies=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.23.0",
        "httpx>=0.26.0",
        "ruff>=0.1.0",
        "mypy>=1.8.0",
    ],
    files={
        # Main application
        "src/__init__.py": "",
        "src/main.py": '''"""Main application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, api
from src.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(api.router, prefix="/api/v1", tags=["api"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
''',
        # API routes
        "src/api/__init__.py": "",
        "src/api/routes/__init__.py": "",
        "src/api/routes/health.py": '''"""Health check endpoints."""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
def root():
    """Root endpoint."""
    return {"message": "Welcome to {project_name}"}
''',
        "src/api/routes/api.py": '''"""Main API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def api_root():
    """API root endpoint."""
    return {"message": "API v1"}
''',
        # Core configuration
        "src/core/__init__.py": "",
        "src/core/config.py": '''"""Application configuration."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Project
    PROJECT_NAME: str = "{project_name}"
    PROJECT_DESCRIPTION: str = "{project_description}"
    VERSION: str = "0.1.0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Database (optional)
    DATABASE_URL: str | None = None

    # Redis (optional)
    REDIS_URL: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
''',
        # Models
        "src/models/__init__.py": "",
        # Services
        "src/services/__init__.py": "",
        # Tests
        "tests/__init__.py": "",
        "tests/conftest.py": '''"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    with TestClient(app) as client:
        yield client
''',
        "tests/api/__init__.py": "",
        "tests/api/test_health.py": '''"""Tests for health endpoints."""


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
''',
        # Configuration files
        "pyproject.toml": '''[project]
name = "{project_slug}"
version = "0.1.0"
description = "{project_description}"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    {dependencies}
]

[project.optional-dependencies]
dev = [
    {dev_dependencies}
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
''',
        ".env.example": '''# {project_name} Environment Configuration
# Copy to .env and configure

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database (uncomment and configure as needed)
# DATABASE_URL=postgres://user:password@localhost:5432/dbname

# Redis (uncomment and configure as needed)
# REDIS_URL=redis://localhost:6379

# API Keys (add your secrets here)
# API_KEY=your-api-key
''',
        ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.eggs/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local
.env.*.local

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Build
dist/
build/

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
''',
        "Dockerfile": '''# {project_name} Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application
COPY src/ src/

# Runtime configuration (secrets passed at runtime, not here)
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',
        "docker-compose.yml": '''version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
    env_file:
      - .env
    volumes:
      - ./src:/app/src:ro
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: {project_slug}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
''',
        # Documentation
        "README.md": '''# {project_name}

{project_description}

## Setup

```bash
# 1. Unzip into your git repository (use -o to overwrite for iterations)
unzip -o {project_slug}.zip -d your-repo/

# 2. Install dependencies
cd your-repo
pip install -e ".[dev]"

# 3. Copy environment file and configure (first time only)
cp .env.example .env

# 4. Commit to git
git add .
git commit -m "Generation from AgentCodeFactory"
git push
```

## Quick Start

```bash
# Run development server
python -m src.main
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker

```bash
# Build and run
docker-compose up --build

# Or just the app
docker build -t {project_slug} .
docker run -p 8000:8000 --env-file .env {project_slug}
```

## Testing

```bash
pytest
pytest --cov=src
```

## Project Structure

```
{project_slug}/
├── src/
│   ├── api/
│   │   └── routes/      # API endpoints
│   ├── core/            # Configuration
│   ├── models/          # Data models
│   └── services/        # Business logic
├── tests/               # Test suite
├── docs/                # Documentation
├── Dockerfile
└── docker-compose.yml
```
''',
        "docs/ARCHITECTURE.md": '''# Architecture

## Overview

{project_name} is built with FastAPI following a modular architecture.

## Layers

```
┌─────────────────────────────────────┐
│           API Routes                │  ← HTTP endpoints
├─────────────────────────────────────┤
│           Services                  │  ← Business logic
├─────────────────────────────────────┤
│           Models                    │  ← Data structures
├─────────────────────────────────────┤
│           Core                      │  ← Configuration
└─────────────────────────────────────┘
```

## Directory Structure

- `src/api/routes/` - FastAPI route handlers
- `src/services/` - Business logic, external API calls
- `src/models/` - Pydantic models, database schemas
- `src/core/` - Settings, dependencies, utilities

## Configuration

All configuration via environment variables (12-factor app):
- Development: `.env` file
- Production: Platform environment (Render, Railway, etc.)

## Security

- Secrets NEVER in code or Docker images
- Environment variables injected at runtime
- CORS configured for allowed origins
''',
    },
)


# =============================================================================
# CLI Template
# =============================================================================

CLI_TEMPLATE = ProjectTemplate(
    name="cli",
    description="Python CLI tool with Typer",
    language="python",
    framework="typer",
    directories=[
        "src",
        "tests",
    ],
    dependencies=[
        "typer>=0.9.0",
        "rich>=13.0.0",
    ],
    dev_dependencies=[
        "pytest>=7.4.0",
        "ruff>=0.1.0",
        "mypy>=1.8.0",
    ],
    files={
        "src/__init__.py": "",
        "src/main.py": '''"""Main CLI entry point."""

import typer
from rich.console import Console

app = typer.Typer(help="{project_description}")
console = Console()


@app.command()
def hello(name: str = "World"):
    """Say hello."""
    console.print(f"[green]Hello, {name}![/green]")


@app.command()
def version():
    """Show version."""
    console.print("{project_name} v0.1.0")


if __name__ == "__main__":
    app()
''',
        "tests/__init__.py": "",
        "tests/test_main.py": '''"""Tests for CLI."""

from typer.testing import CliRunner
from src.main import app

runner = CliRunner()


def test_hello():
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
''',
        "pyproject.toml": '''[project]
name = "{project_slug}"
version = "0.1.0"
description = "{project_description}"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    {dependencies}
]

[project.optional-dependencies]
dev = [
    {dev_dependencies}
]

[project.scripts]
{project_slug} = "src.main:app"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
''',
        ".gitignore": '''__pycache__/
*.py[cod]
.venv/
venv/
.env
.mypy_cache/
.pytest_cache/
dist/
*.egg-info/
''',
        "README.md": '''# {project_name}

{project_description}

## Setup

```bash
# 1. Unzip into your git repository (use -o to overwrite for iterations)
unzip -o {project_slug}.zip -d your-repo/

# 2. Install dependencies
cd your-repo
pip install -e ".[dev]"

# 3. Commit to git
git add .
git commit -m "Generation from AgentCodeFactory"
git push
```

## Usage

```bash
{project_slug} hello
{project_slug} hello --name "Your Name"
{project_slug} version
```
''',
    },
)


# =============================================================================
# Minimal Template
# =============================================================================

MINIMAL_TEMPLATE = ProjectTemplate(
    name="minimal",
    description="Minimal Python project",
    language="python",
    framework="none",
    directories=["src", "tests"],
    dependencies=[],
    dev_dependencies=["pytest>=7.4.0", "ruff>=0.1.0"],
    files={
        "src/__init__.py": "",
        "src/main.py": '''"""{project_name} main module."""


def main():
    """Main entry point."""
    print("Hello from {project_name}!")


if __name__ == "__main__":
    main()
''',
        "tests/__init__.py": "",
        "tests/test_main.py": '''"""Tests for main module."""

from src.main import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello" in captured.out
''',
        "pyproject.toml": '''[project]
name = "{project_slug}"
version = "0.1.0"
description = "{project_description}"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = [
    {dev_dependencies}
]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
''',
        ".gitignore": '''__pycache__/
*.py[cod]
.venv/
.env
.pytest_cache/
''',
        "README.md": '''# {project_name}

{project_description}

## Setup

```bash
# 1. Unzip into your git repository (use -o to overwrite for iterations)
unzip -o {project_slug}.zip -d your-repo/

# 2. Install dependencies
cd your-repo
pip install -e ".[dev]"

# 3. Commit to git
git add .
git commit -m "Generation from AgentCodeFactory"
git push
```

## Run

```bash
python -m src.main
```

## Test

```bash
pytest
```
''',
    },
)


# =============================================================================
# Template Registry
# =============================================================================

BUILTIN_TEMPLATES: dict[str, ProjectTemplate] = {
    "fastapi": FASTAPI_TEMPLATE,
    "cli": CLI_TEMPLATE,
    "minimal": MINIMAL_TEMPLATE,
}

# Global registry including custom templates
TEMPLATES: dict[str, ProjectTemplate] = BUILTIN_TEMPLATES.copy()


def get_template(name: str) -> ProjectTemplate | None:
    """Get a template by name (built-in or custom)."""
    # Reload custom templates to pick up changes
    load_custom_templates()
    return TEMPLATES.get(name)


def list_templates() -> list[dict]:
    """List all available templates with metadata.

    Returns:
        List of template info dicts
    """
    load_custom_templates()
    result = []
    for name, template in TEMPLATES.items():
        result.append({
            "name": name,
            "description": template.description,
            "language": template.language,
            "framework": template.framework,
            "builtin": name in BUILTIN_TEMPLATES,
        })
    return sorted(result, key=lambda x: (not x["builtin"], x["name"]))


# =============================================================================
# Custom Template Support
# =============================================================================

import json
import shutil
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def get_templates_dir() -> Path:
    """Get the custom templates directory.

    Returns:
        Path to ~/.coding-factory/templates/
    """
    templates_dir = Path.home() / ".coding-factory" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def load_custom_templates() -> dict[str, ProjectTemplate]:
    """Load custom templates from the templates directory.

    Returns:
        Dict of template name to ProjectTemplate
    """
    templates_dir = get_templates_dir()
    custom_templates = {}

    for template_dir in templates_dir.iterdir():
        if not template_dir.is_dir():
            continue

        # Skip hidden directories
        if template_dir.name.startswith("."):
            continue

        try:
            template = load_template_from_dir(template_dir)
            if template:
                custom_templates[template.name] = template
        except Exception:
            # Skip invalid templates
            continue

    # Update global registry
    TEMPLATES.clear()
    TEMPLATES.update(BUILTIN_TEMPLATES)
    TEMPLATES.update(custom_templates)

    return custom_templates


def load_template_from_dir(template_dir: Path) -> ProjectTemplate | None:
    """Load a template from a directory.

    Template directory structure:
        my-template/
        ├── template.toml (or template.json)
        └── files/
            ├── src/
            │   └── main.py
            └── README.md

    Args:
        template_dir: Path to template directory

    Returns:
        ProjectTemplate or None if invalid
    """
    # Find manifest file
    manifest_path = None
    manifest_data = {}

    for manifest_name in ["template.toml", "template.json", "manifest.toml", "manifest.json"]:
        candidate = template_dir / manifest_name
        if candidate.exists():
            manifest_path = candidate
            break

    if manifest_path:
        if manifest_path.suffix == ".toml":
            with open(manifest_path, "rb") as f:
                manifest_data = tomllib.load(f)
        else:
            with open(manifest_path) as f:
                manifest_data = json.load(f)

    # Get template name (from manifest or directory name)
    name = manifest_data.get("name", template_dir.name)
    description = manifest_data.get("description", f"Custom template: {name}")
    language = manifest_data.get("language", "python")
    framework = manifest_data.get("framework", "custom")
    dependencies = manifest_data.get("dependencies", [])
    dev_dependencies = manifest_data.get("dev_dependencies", [])

    # Load files from 'files/' subdirectory or root
    files_dir = template_dir / "files"
    if not files_dir.exists():
        files_dir = template_dir

    files = {}
    directories = set()

    for file_path in files_dir.rglob("*"):
        if file_path.is_dir():
            continue

        # Skip manifest files
        if file_path.name in ["template.toml", "template.json", "manifest.toml", "manifest.json"]:
            continue

        # Get relative path
        try:
            rel_path = str(file_path.relative_to(files_dir))
        except ValueError:
            continue

        # Track directories
        parent = file_path.parent
        while parent != files_dir:
            try:
                directories.add(str(parent.relative_to(files_dir)))
            except ValueError:
                break
            parent = parent.parent

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
            files[rel_path] = content
        except Exception:
            # Skip binary or unreadable files
            continue

    if not files:
        return None

    return ProjectTemplate(
        name=name,
        description=description,
        language=language,
        framework=framework,
        files=files,
        dependencies=dependencies,
        dev_dependencies=dev_dependencies,
        directories=sorted(directories),
    )


def save_template(
    name: str,
    source_dir: Path,
    description: str = "",
    language: str = "python",
    framework: str = "custom",
    dependencies: list[str] | None = None,
    dev_dependencies: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> Path:
    """Save a project as a custom template.

    Args:
        name: Template name
        source_dir: Source project directory
        description: Template description
        language: Programming language
        framework: Framework used
        dependencies: List of dependencies
        dev_dependencies: List of dev dependencies
        exclude_patterns: Patterns to exclude from template

    Returns:
        Path to saved template
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".env", ".env.local", "*.pyc", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "dist", "build", "*.egg-info", ".coding-factory-index",
        ]

    templates_dir = get_templates_dir()
    template_dir = templates_dir / name

    # Clean existing template
    if template_dir.exists():
        shutil.rmtree(template_dir)

    template_dir.mkdir(parents=True)
    files_dir = template_dir / "files"
    files_dir.mkdir()

    # Copy files
    for src_path in source_dir.rglob("*"):
        if src_path.is_dir():
            continue

        # Check exclude patterns
        path_str = str(src_path)
        if any(pattern in path_str for pattern in exclude_patterns):
            continue

        # Get relative path
        try:
            rel_path = src_path.relative_to(source_dir)
        except ValueError:
            continue

        # Create target directory
        target_path = files_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        try:
            shutil.copy2(src_path, target_path)
        except Exception:
            continue

    # Create manifest
    manifest = {
        "name": name,
        "description": description or f"Custom template created from {source_dir.name}",
        "language": language,
        "framework": framework,
        "dependencies": dependencies or [],
        "dev_dependencies": dev_dependencies or [],
    }

    manifest_path = template_dir / "template.toml"
    with open(manifest_path, "w") as f:
        f.write(f'name = "{manifest["name"]}"\n')
        f.write(f'description = "{manifest["description"]}"\n')
        f.write(f'language = "{manifest["language"]}"\n')
        f.write(f'framework = "{manifest["framework"]}"\n')
        f.write(f'dependencies = {json.dumps(manifest["dependencies"])}\n')
        f.write(f'dev_dependencies = {json.dumps(manifest["dev_dependencies"])}\n')

    # Reload templates
    load_custom_templates()

    return template_dir


def delete_template(name: str) -> bool:
    """Delete a custom template.

    Args:
        name: Template name

    Returns:
        True if deleted, False if not found or built-in
    """
    if name in BUILTIN_TEMPLATES:
        return False

    templates_dir = get_templates_dir()
    template_dir = templates_dir / name

    if not template_dir.exists():
        return False

    shutil.rmtree(template_dir)
    load_custom_templates()
    return True
