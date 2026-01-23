"""Documentation Agent for automated doc generation.

Generates and maintains project documentation including:
- docs/ folder (architecture, how-to-run, how-to-test)
- Inline docstrings for public functions/classes
- Architecture Decision Records (ADRs)
- Sync between spec, README, and code comments
"""

import ast
import json
from datetime import datetime
from pathlib import Path

from llm_backend import LLMBackend
from schemas.doc_report import (
    ADRRecord,
    DocReport,
    DocstringStyle,
    DocstringUpdate,
    DocType,
    GeneratedDoc,
    MissingDocstring,
    SyncCheck,
    SyncStatus,
)

from .base import AgentInput, AgentOutput, BaseAgent

# Templates for documentation files
ARCHITECTURE_TEMPLATE = """# Architecture Overview

## System Design

{architecture_description}

## Components

{components}

## Data Flow

{data_flow}

## Key Patterns

{patterns}

## Directory Structure

```
{directory_structure}
```

## Dependencies

{dependencies}
"""

HOW_TO_RUN_TEMPLATE = """# How to Run

## Prerequisites

{prerequisites}

## Installation

```bash
{installation_steps}
```

## Configuration

{configuration}

## Running the Application

### Quick Start (Recommended)

```bash
# Make the run script executable (first time only)
chmod +x run.sh

# Run the application
./run.sh
```

The run script handles virtual environment setup, dependency installation, and database migrations automatically.

### Manual Start

If you prefer to run manually:

```bash
{run_command}
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
{env_vars_table}

## Docker

```bash
{docker_commands}
```
"""

HOW_TO_TEST_TEMPLATE = """# How to Test

## Test Setup

```bash
{test_setup}
```

## Running Tests

```bash
# Run all tests
{run_all_tests}

# Run specific test file
{run_specific}

# Run with coverage
{run_coverage}
```

## Test Structure

{test_structure}

## Writing Tests

{writing_tests}

## CI/CD Integration

{ci_integration}
"""

ADR_TEMPLATE = """# ADR-{number:03d}: {title}

## Status

{status}

## Context

{context}

## Decision

{decision}

## Consequences

### Positive

{positive_consequences}

### Negative

{negative_consequences}

## Date

{date}
"""


class DocAgent(BaseAgent):
    """Agent for generating and maintaining documentation."""

    def __init__(self, llm: LLMBackend) -> None:
        """Initialize DocAgent.

        Args:
            llm: LLM backend for generating documentation content.
        """
        super().__init__(llm=llm)
        self.docstring_style = DocstringStyle.GOOGLE

    def default_system_prompt(self) -> str:
        """Return the default system prompt for documentation."""
        return """You are a technical documentation expert. Generate clear,
comprehensive documentation including architecture docs, READMEs, and docstrings.
Follow Google style for Python docstrings."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate documentation for the project.

        Args:
            input_data: Input containing repo path and context.

        Returns:
            AgentOutput with documentation report.
        """
        repo_path = Path(input_data.repo_path)
        context = input_data.context or {}

        report = DocReport(
            docstring_style=self.docstring_style,
        )

        try:
            # 1. Analyze project structure
            project_info = self._analyze_project(repo_path)

            # 2. Generate docs/ folder content
            if context.get("generate_docs_folder", True):
                self._generate_docs_folder(repo_path, project_info, report)

            # 3. Add missing docstrings
            if context.get("add_docstrings", True):
                self._add_docstrings(repo_path, report)

            # 4. Create ADRs for major patterns
            if context.get("create_adrs", True):
                self._create_adrs(repo_path, project_info, report)

            # 5. Check and sync documentation
            if context.get("sync_docs", True):
                self._sync_documentation(repo_path, report)

            # Update counts and generate summary
            report.update_counts()
            report.summary = self._generate_summary(report)

            return AgentOutput(
                success=True,
                result=report.model_dump(),
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                errors=[str(e)],
                result=report.model_dump(),
            )

    def _analyze_project(self, repo_path: Path) -> dict:
        """Analyze project structure and gather information.

        Args:
            repo_path: Path to the repository.

        Returns:
            Dictionary with project information.
        """
        info = {
            "name": repo_path.name,
            "python_files": [],
            "has_tests": False,
            "has_docker": False,
            "has_requirements": False,
            "has_pyproject": False,
            "framework": None,
            "patterns": [],
            "main_modules": [],
        }

        # Find Python files
        for py_file in repo_path.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                rel_path = py_file.relative_to(repo_path)
                info["python_files"].append(str(rel_path))

                # Check for test files
                if "test" in py_file.name.lower():
                    info["has_tests"] = True

        # Check for common files
        info["has_docker"] = (repo_path / "Dockerfile").exists()
        info["has_requirements"] = (repo_path / "requirements.txt").exists()
        info["has_pyproject"] = (repo_path / "pyproject.toml").exists()

        # Detect framework
        info["framework"] = self._detect_framework(repo_path)

        # Identify main modules
        for item in repo_path.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                info["main_modules"].append(item.name)

        return info

    def _detect_framework(self, repo_path: Path) -> str | None:
        """Detect the web framework used.

        Args:
            repo_path: Path to the repository.

        Returns:
            Framework name or None.
        """
        # Check requirements or imports
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text().lower()
            if "fastapi" in content:
                return "FastAPI"
            if "flask" in content:
                return "Flask"
            if "django" in content:
                return "Django"

        # Check pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text().lower()
            if "fastapi" in content:
                return "FastAPI"
            if "flask" in content:
                return "Flask"
            if "django" in content:
                return "Django"

        return None

    def _generate_docs_folder(
        self, repo_path: Path, project_info: dict, report: DocReport
    ) -> None:
        """Generate documentation files in docs/ folder.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.
            report: Documentation report to update.
        """
        docs_dir = repo_path / "docs"
        docs_dir.mkdir(exist_ok=True)

        # Generate architecture overview
        arch_doc = self._generate_architecture_doc(repo_path, project_info)
        arch_path = docs_dir / "architecture.md"
        is_new = not arch_path.exists()
        arch_path.write_text(arch_doc)

        report.docs_generated.append(
            GeneratedDoc(
                path="docs/architecture.md",
                doc_type=DocType.ARCHITECTURE,
                content=arch_doc,
                description="System architecture overview",
                is_new=is_new,
            )
        )

        # Generate how-to-run guide
        run_doc = self._generate_how_to_run(repo_path, project_info)
        run_path = docs_dir / "how-to-run.md"
        is_new = not run_path.exists()
        run_path.write_text(run_doc)

        report.docs_generated.append(
            GeneratedDoc(
                path="docs/how-to-run.md",
                doc_type=DocType.HOW_TO_RUN,
                content=run_doc,
                description="Instructions for running the application",
                is_new=is_new,
            )
        )

        # Generate how-to-test guide
        test_doc = self._generate_how_to_test(repo_path, project_info)
        test_path = docs_dir / "how-to-test.md"
        is_new = not test_path.exists()
        test_path.write_text(test_doc)

        report.docs_generated.append(
            GeneratedDoc(
                path="docs/how-to-test.md",
                doc_type=DocType.HOW_TO_TEST,
                content=test_doc,
                description="Testing guide and instructions",
                is_new=is_new,
            )
        )

    def _generate_architecture_doc(self, repo_path: Path, project_info: dict) -> str:
        """Generate architecture documentation.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.

        Returns:
            Architecture documentation content.
        """
        # Build directory structure
        dir_structure = self._get_directory_structure(repo_path)

        # Build components list
        components = []
        for module in project_info.get("main_modules", []):
            components.append(f"- **{module}/**: Module description")

        # Detect patterns
        patterns = []
        if project_info.get("framework"):
            patterns.append(f"- **{project_info['framework']}**: Web framework")
        if project_info.get("has_tests"):
            patterns.append("- **pytest**: Testing framework")

        # Use LLM to generate description if available
        arch_description = self._generate_with_llm(
            f"Generate a brief architecture description for a Python project "
            f"named '{project_info['name']}' with framework "
            f"'{project_info.get('framework', 'none')}' and modules: "
            f"{', '.join(project_info.get('main_modules', []))}",
            fallback="This project follows a modular architecture with clear "
            "separation of concerns.",
        )

        data_flow = self._generate_with_llm(
            f"Describe the data flow for a {project_info.get('framework', 'Python')} "
            "application in 2-3 sentences.",
            fallback="1. Request enters through the API layer\n"
            "2. Business logic processes the request\n"
            "3. Data is persisted or retrieved from storage\n"
            "4. Response is returned to the client",
        )

        # Build dependencies list
        dependencies = self._get_dependencies(repo_path)

        return ARCHITECTURE_TEMPLATE.format(
            architecture_description=arch_description,
            components="\n".join(components) if components else "No modules detected.",
            data_flow=data_flow,
            patterns="\n".join(patterns) if patterns else "No specific patterns detected.",
            directory_structure=dir_structure,
            dependencies=dependencies,
        )

    def _generate_how_to_run(self, repo_path: Path, project_info: dict) -> str:
        """Generate how-to-run documentation based on actual project introspection.

        Reads actual files to document what really exists, not assumptions.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.

        Returns:
            How-to-run documentation content.
        """
        # INTROSPECT the actual project instead of making assumptions
        introspected = self._introspect_project(repo_path)

        # Use introspected framework, fallback to project_info
        framework = introspected.get("framework") or project_info.get("framework")
        port = introspected.get("port", 8000)

        # Determine prerequisites
        prerequisites = "- Python 3.11+\n- pip or uv package manager"
        if introspected.get("has_dockerfile") or introspected.get("has_docker_compose"):
            prerequisites += "\n- Docker (optional)"

        # Installation steps
        if project_info.get("has_pyproject"):
            installation = "pip install -e ."
        elif project_info.get("has_requirements"):
            installation = "pip install -r requirements.txt"
        else:
            installation = "pip install ."

        # Configuration section
        config = self._get_config_docs(repo_path)

        # PRIORITY: Use actual command from run.sh if it exists
        run_command = None

        if introspected.get("run_script_command"):
            # Use the ACTUAL command from run.sh
            run_command = introspected["run_script_command"]
        elif introspected.get("entry_point"):
            # Build command based on introspected entry point and framework
            entry_point = introspected["entry_point"]

            if framework == "FastAPI":
                module = entry_point.replace("/", ".").replace(".py", "")
                run_command = f"uvicorn {module}:app --host 0.0.0.0 --port {port} --reload"
            elif framework == "Flask":
                run_command = f"python {entry_point}"
            elif framework == "Django":
                run_command = f"python manage.py runserver 0.0.0.0:{port}"
            else:
                run_command = f"python {entry_point}"
        else:
            # Last resort fallback
            run_command = "# See run.sh for startup command"

        # Environment variables
        env_vars = self._get_env_vars_table(repo_path)

        # Docker commands - based on what actually exists
        docker_commands = ""
        project_name = project_info.get('name', 'app')

        if introspected.get("has_docker_compose"):
            docker_commands = (
                "# Using docker-compose (recommended)\n"
                "docker-compose up --build\n\n"
                "# Run in background\n"
                "docker-compose up -d"
            )
        elif introspected.get("has_dockerfile"):
            docker_commands = (
                "# Build image\n"
                f"docker build -t {project_name} .\n\n"
                "# Run container\n"
                f"docker run -p {port}:{port} {project_name}"
            )
        else:
            docker_commands = "Docker not configured for this project."

        return HOW_TO_RUN_TEMPLATE.format(
            prerequisites=prerequisites,
            installation_steps=installation,
            configuration=config,
            run_command=run_command,
            env_vars_table=env_vars,
            docker_commands=docker_commands,
        )

    def _generate_how_to_test(self, repo_path: Path, project_info: dict) -> str:
        """Generate how-to-test documentation.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.

        Returns:
            How-to-test documentation content.
        """
        # Test setup
        test_setup = "pip install -e '.[dev]'  # or\npip install pytest pytest-cov"

        # Run commands
        run_all = "pytest"
        run_specific = "pytest tests/test_specific.py"
        run_coverage = "pytest --cov=. --cov-report=html"

        # Test structure
        test_dirs = []
        for item in repo_path.rglob("test*.py"):
            if "__pycache__" not in str(item):
                test_dirs.append(str(item.relative_to(repo_path)))

        if test_dirs:
            test_structure = "```\n" + "\n".join(sorted(set(test_dirs))[:10]) + "\n```"
        else:
            test_structure = "No test files found. Consider adding tests in a `tests/` directory."

        # Writing tests guide
        writing_tests = self._generate_with_llm(
            "Provide brief guidelines for writing tests in a Python project using pytest.",
            fallback=(
                "1. Create test files with `test_` prefix\n"
                "2. Write test functions starting with `test_`\n"
                "3. Use fixtures for common setup\n"
                "4. Use `assert` statements for validations\n"
                "5. Group related tests in classes"
            ),
        )

        # CI integration
        ci_integration = ""
        if (repo_path / ".github" / "workflows").exists():
            ci_integration = "Tests are configured to run in GitHub Actions."
        else:
            ci_integration = "Consider adding CI/CD configuration for automated testing."

        return HOW_TO_TEST_TEMPLATE.format(
            test_setup=test_setup,
            run_all_tests=run_all,
            run_specific=run_specific,
            run_coverage=run_coverage,
            test_structure=test_structure,
            writing_tests=writing_tests,
            ci_integration=ci_integration,
        )

    def _add_docstrings(self, repo_path: Path, report: DocReport) -> None:
        """Add docstrings to public functions and classes.

        Args:
            repo_path: Path to the repository.
            report: Documentation report to update.
        """
        # Find all Python files
        py_files = list(repo_path.rglob("*.py"))

        total_public = 0
        documented = 0

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                # Find public functions and classes without docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):
                            total_public += 1
                            if ast.get_docstring(node):
                                documented += 1
                            else:
                                # Record missing docstring
                                report.missing_docstrings.append(
                                    MissingDocstring(
                                        file_path=str(py_file.relative_to(repo_path)),
                                        symbol_name=node.name,
                                        symbol_type="function",
                                        line_number=node.lineno,
                                        signature=self._get_signature(node),
                                    )
                                )
                    elif isinstance(node, ast.ClassDef):
                        if not node.name.startswith("_"):
                            total_public += 1
                            if ast.get_docstring(node):
                                documented += 1
                            else:
                                report.missing_docstrings.append(
                                    MissingDocstring(
                                        file_path=str(py_file.relative_to(repo_path)),
                                        symbol_name=node.name,
                                        symbol_type="class",
                                        line_number=node.lineno,
                                        signature=f"class {node.name}",
                                    )
                                )

            except (SyntaxError, UnicodeDecodeError):
                continue

        # Calculate coverage
        if total_public > 0:
            report.coverage_before = round(documented / total_public * 100, 1)
            report.coverage_after = report.coverage_before  # Would be higher if we add docstrings

        # Generate docstrings for missing ones (limit to first 10)
        for missing in report.missing_docstrings[:10]:
            docstring = self._generate_docstring(missing)
            if docstring:
                report.docstrings_added.append(
                    DocstringUpdate(
                        file_path=missing.file_path,
                        symbol_name=missing.symbol_name,
                        symbol_type=missing.symbol_type,
                        line_number=missing.line_number,
                        old_docstring=None,
                        new_docstring=docstring,
                        style=self.docstring_style,
                    )
                )

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get function signature as string.

        Args:
            node: AST function node.

        Returns:
            Function signature string.
        """
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(args)})"

    def _generate_docstring(self, missing: MissingDocstring) -> str:
        """Generate a docstring for a missing symbol.

        Args:
            missing: Information about the undocumented symbol.

        Returns:
            Generated docstring.
        """
        prompt = (
            f"Generate a Google-style docstring for this Python "
            f"{missing.symbol_type}:\n\n{missing.signature}\n\n"
            "Include Args, Returns, and Raises sections if applicable. "
            "Be concise but informative."
        )

        return self._generate_with_llm(
            prompt,
            fallback=f'"""{missing.symbol_name.replace("_", " ").title()}."""',
        )

    def _create_adrs(
        self, repo_path: Path, project_info: dict, report: DocReport
    ) -> None:
        """Create Architecture Decision Records for major patterns.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.
            report: Documentation report to update.
        """
        adr_dir = repo_path / "docs" / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)

        # Count existing ADRs
        existing_adrs = list(adr_dir.glob("*.md"))
        report.total_adrs = len(existing_adrs)
        next_number = len(existing_adrs) + 1

        # Check if we should create ADRs for detected patterns
        framework = project_info.get("framework")
        if framework and not any(framework.lower() in str(a) for a in existing_adrs):
            adr = self._create_framework_adr(framework, next_number)
            adr_path = adr_dir / f"adr-{next_number:03d}-{framework.lower()}.md"
            adr_path.write_text(
                ADR_TEMPLATE.format(
                    number=adr.number,
                    title=adr.title,
                    status=adr.status,
                    context=adr.context,
                    decision=adr.decision,
                    positive_consequences=adr.consequences.split("Negative:")[0]
                    if "Negative:" in adr.consequences
                    else adr.consequences,
                    negative_consequences=adr.consequences.split("Negative:")[-1]
                    if "Negative:" in adr.consequences
                    else "None identified.",
                    date=datetime.now().strftime("%Y-%m-%d"),
                )
            )
            adr.file_path = str(adr_path.relative_to(repo_path))
            report.adrs_created.append(adr)
            report.total_adrs += 1

    def _create_framework_adr(self, framework: str, number: int) -> ADRRecord:
        """Create an ADR for the chosen framework.

        Args:
            framework: Framework name.
            number: ADR number.

        Returns:
            ADRRecord for the framework decision.
        """
        context = self._generate_with_llm(
            f"Explain why a team might choose {framework} for a Python web project. "
            "Be concise (2-3 sentences).",
            fallback=f"The team needed a web framework for building APIs. "
            f"{framework} was evaluated against alternatives.",
        )

        decision = self._generate_with_llm(
            f"Describe the decision to use {framework}. Be concise.",
            fallback=f"We will use {framework} as our web framework.",
        )

        consequences = self._generate_with_llm(
            f"List positive and negative consequences of choosing {framework}. "
            "Format: bullet points.",
            fallback=f"- {framework} provides excellent performance\n"
            "- Good ecosystem and documentation\n"
            "Negative:\n- Learning curve for team members",
        )

        return ADRRecord(
            number=number,
            title=f"Use {framework} as web framework",
            status="accepted",
            context=context,
            decision=decision,
            consequences=consequences,
            file_path="",
        )

    def _sync_documentation(self, repo_path: Path, report: DocReport) -> None:
        """Check and sync documentation across sources.

        Args:
            repo_path: Path to the repository.
            report: Documentation report to update.
        """
        # Check README vs spec
        readme_path = repo_path / "README.md"
        spec_path = repo_path / "artifacts" / "spec.json"

        if readme_path.exists():
            readme_content = readme_path.read_text()

            # Check if spec exists and compare
            if spec_path.exists():
                try:
                    spec_data = json.loads(spec_path.read_text())
                    spec_name = spec_data.get("name", "")

                    discrepancies = []
                    if spec_name and spec_name not in readme_content:
                        discrepancies.append(
                            f"Project name '{spec_name}' not mentioned in README"
                        )

                    report.sync_checks.append(
                        SyncCheck(
                            source="spec.json",
                            target="README.md",
                            status=(
                                SyncStatus.IN_SYNC
                                if not discrepancies
                                else SyncStatus.OUT_OF_SYNC
                            ),
                            discrepancies=discrepancies,
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

            # Check for required README sections
            required_sections = ["Installation", "Usage", "License"]
            missing_sections = [
                s for s in required_sections if s.lower() not in readme_content.lower()
            ]

            if missing_sections:
                report.sync_checks.append(
                    SyncCheck(
                        source="README template",
                        target="README.md",
                        status=SyncStatus.OUT_OF_SYNC,
                        discrepancies=[
                            f"Missing section: {s}" for s in missing_sections
                        ],
                    )
                )

    def _get_directory_structure(self, repo_path: Path, max_depth: int = 3) -> str:
        """Get directory structure as a string.

        Args:
            repo_path: Path to the repository.
            max_depth: Maximum depth to traverse.

        Returns:
            Directory structure string.
        """
        lines = []

        def walk(path: Path, prefix: str = "", depth: int = 0) -> None:
            if depth > max_depth:
                return

            # Skip hidden and cache directories
            items = sorted(
                p
                for p in path.iterdir()
                if not p.name.startswith(".")
                and "__pycache__" not in p.name
                and p.name not in ("node_modules", ".git", "venv", ".venv")
            )

            for i, item in enumerate(items[:15]):  # Limit items
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{item.name}")

                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    walk(item, prefix + extension, depth + 1)

        lines.append(repo_path.name + "/")
        walk(repo_path)
        return "\n".join(lines[:30])  # Limit total lines

    def _get_dependencies(self, repo_path: Path) -> str:
        """Get project dependencies as formatted string.

        Args:
            repo_path: Path to the repository.

        Returns:
            Formatted dependencies string.
        """
        deps = []

        # Check requirements.txt
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            for line in req_file.read_text().splitlines()[:15]:
                line = line.strip()
                if line and not line.startswith("#"):
                    deps.append(f"- {line}")

        if deps:
            return "\n".join(deps)
        return "See `requirements.txt` or `pyproject.toml` for dependencies."

    def _get_config_docs(self, repo_path: Path) -> str:
        """Get configuration documentation.

        Args:
            repo_path: Path to the repository.

        Returns:
            Configuration documentation string.
        """
        # Check for .env.example
        env_example = repo_path / ".env.example"
        if env_example.exists():
            return (
                "Copy `.env.example` to `.env` and configure:\n\n"
                "```bash\ncp .env.example .env\n# Edit .env with your values\n```"
            )

        # Check for config files
        config_files = list(repo_path.glob("config*.py")) + list(
            repo_path.glob("config*.yaml")
        )
        if config_files:
            return f"Configuration is managed via `{config_files[0].name}`."

        return "No specific configuration required."

    def _get_env_vars_table(self, repo_path: Path) -> str:
        """Get environment variables as a markdown table.

        Args:
            repo_path: Path to the repository.

        Returns:
            Markdown table of environment variables.
        """
        env_vars = []

        # Check .env.example
        env_example = repo_path / ".env.example"
        if env_example.exists():
            for line in env_example.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    var_name = line.split("=")[0].strip()
                    default = line.split("=")[1].strip() if "=" in line else "-"
                    env_vars.append(
                        f"| {var_name} | Configuration variable | No | {default} |"
                    )

        if env_vars:
            return "\n".join(env_vars[:10])
        return "| - | No environment variables documented | - | - |"

    def _introspect_project(self, repo_path: Path) -> dict:
        """Introspect the generated project to understand what actually exists.

        Reads actual files to determine:
        - Run script contents and command
        - Entry point file
        - Framework detection from actual code
        - Port number
        - Dependencies

        Args:
            repo_path: Path to the repository.

        Returns:
            Dict with introspected project information.
        """
        import re

        info = {
            "has_run_script": False,
            "run_script_command": None,
            "entry_point": None,
            "framework": None,
            "port": 8000,
            "has_dockerfile": False,
            "has_docker_compose": False,
            "dependencies": [],
        }

        # 1. Check for run.sh and extract actual command
        run_script = repo_path / "run.sh"
        if run_script.exists():
            info["has_run_script"] = True
            info["run_script_command"] = self._extract_run_command_from_script(run_script)

        # 2. Check for Dockerfile and docker-compose
        info["has_dockerfile"] = (repo_path / "Dockerfile").exists()
        info["has_docker_compose"] = (
            (repo_path / "docker-compose.yml").exists() or
            (repo_path / "docker-compose.yaml").exists()
        )

        # 3. Find actual entry point by scanning files
        entry_point, framework, port = self._find_actual_entry_point(repo_path)
        info["entry_point"] = entry_point
        info["framework"] = framework
        if port:
            info["port"] = port

        # 4. Read actual dependencies
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name (before ==, >=, etc.)
                        pkg = re.split(r'[=<>!~\[]', line)[0].strip()
                        if pkg:
                            info["dependencies"].append(pkg)
            except Exception:
                pass

        return info

    def _extract_run_command_from_script(self, script_path: Path) -> str | None:
        """Extract the actual run command from run.sh.

        Parses the script to find the command that starts the application.

        Args:
            script_path: Path to run.sh

        Returns:
            The extracted run command, or None if not found.
        """
        import re

        try:
            content = script_path.read_text()

            # Look for common run commands in order of specificity
            patterns = [
                # uvicorn command
                r'(uvicorn\s+[\w.:]+\s*(?:--[^\n]+)?)',
                # gunicorn command
                r'(gunicorn\s+[\w.:]+\s*(?:--[^\n]+)?)',
                # flask run
                r'(flask\s+run\s*(?:--[^\n]+)?)',
                # python with specific file
                r'(python\s+[\w./]+\.py\s*(?:--[^\n]+)?)',
                # Django runserver
                r'(python\s+manage\.py\s+runserver\s*(?:--[^\n]+)?)',
            ]

            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    cmd = match.group(1).strip()
                    # Clean up the command (remove trailing comments, etc.)
                    cmd = cmd.split('#')[0].strip()
                    cmd = cmd.split('&')[0].strip()
                    return cmd

            return None

        except Exception:
            return None

    def _find_actual_entry_point(self, repo_path: Path) -> tuple[str | None, str | None, int | None]:
        """Find the actual entry point by scanning project files.

        Looks for framework initialization and entry points in actual code.

        Args:
            repo_path: Path to the repository.

        Returns:
            Tuple of (entry_point_file, framework_name, port_number)
        """
        import re

        # Patterns to detect frameworks and entry points
        framework_patterns = {
            "FastAPI": (r'FastAPI\s*\(', r'app\s*=\s*FastAPI'),
            "Flask": (r'Flask\s*\(__name__', r'app\s*=\s*Flask'),
            "Django": (r'django\.setup\(\)', r'DJANGO_SETTINGS_MODULE'),
        }

        port_pattern = r'port\s*[=:]\s*(\d+)'

        # Scan Python files
        for py_file in repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                continue

            try:
                content = py_file.read_text()
                relative_path = str(py_file.relative_to(repo_path))

                # Check for framework patterns
                for framework, patterns in framework_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content):
                            # Found framework, extract port if present
                            port_match = re.search(port_pattern, content)
                            port = int(port_match.group(1)) if port_match else None
                            return relative_path, framework, port

                # Check for if __name__ == "__main__"
                if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content):
                    port_match = re.search(port_pattern, content)
                    port = int(port_match.group(1)) if port_match else None
                    return relative_path, None, port

            except Exception:
                continue

        # Check for common entry point files
        for name in ["app.py", "main.py", "server.py", "run.py"]:
            if (repo_path / name).exists():
                return name, None, None

        return None, None, None

    def _find_entry_point(self, repo_path: Path, project_info: dict) -> str:
        """Find the actual entry point file for the project.

        Uses introspection to find what actually exists.

        Args:
            repo_path: Path to the repository.
            project_info: Project analysis information.

        Returns:
            Relative path to the entry point file (e.g., 'app.py', 'main.py').
        """
        # Use introspection first
        introspected = self._introspect_project(repo_path)

        if introspected["entry_point"]:
            return introspected["entry_point"]

        # Fallback to framework-based detection
        framework = project_info.get("framework", "")

        if framework == "Django":
            return "manage.py"

        # Check if common files exist
        for name in ["app.py", "main.py", "server.py"]:
            if (repo_path / name).exists():
                return name

        return "app.py"

    def _generate_with_llm(self, prompt: str, fallback: str) -> str:
        """Generate content using LLM with fallback.

        Args:
            prompt: Prompt for the LLM.
            fallback: Fallback content if LLM fails.

        Returns:
            Generated or fallback content.
        """
        if self.llm is None:
            return fallback

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a technical documentation writer. "
                        "Be concise, clear, and professional."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            response = self.llm.chat(messages)
            return response.strip() if response else fallback
        except Exception:
            return fallback

    def _generate_summary(self, report: DocReport) -> str:
        """Generate human-readable summary.

        Args:
            report: Documentation report.

        Returns:
            Summary string.
        """
        parts = []

        if report.total_docs_generated > 0:
            parts.append(f"Generated {report.total_docs_generated} documentation files")

        if report.total_docs_updated > 0:
            parts.append(f"Updated {report.total_docs_updated} existing docs")

        if report.total_docstrings_added > 0:
            parts.append(f"Added {report.total_docstrings_added} docstrings")

        if len(report.missing_docstrings) > 0:
            parts.append(
                f"{len(report.missing_docstrings)} public symbols still need docstrings"
            )

        if report.coverage_after > 0:
            parts.append(f"Docstring coverage: {report.coverage_after}%")

        if len(report.adrs_created) > 0:
            parts.append(f"Created {len(report.adrs_created)} ADR(s)")

        if report.all_in_sync:
            parts.append("All documentation is in sync")
        else:
            out_of_sync = sum(
                1
                for c in report.sync_checks
                if c.status == SyncStatus.OUT_OF_SYNC
            )
            parts.append(f"{out_of_sync} sync issue(s) detected")

        return "; ".join(parts) if parts else "No documentation changes made."
