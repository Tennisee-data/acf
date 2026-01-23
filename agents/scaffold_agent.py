"""Project Scaffold Agent - Generates deployment-ready files."""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_backend import LLMBackend
from tools.dependency_resolver import DependencyResolver
from tools.security import scan_generated_files

from .base import AgentInput, AgentOutput, BaseAgent


@dataclass
class ScaffoldResult:
    """Result of scaffold generation."""
    requirements: list[str] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    framework: str = "python"
    entry_point: str = "main.py"
    port: int = 5000
    has_database: bool = False
    database_type: str = ""
    has_frontend: bool = False
    dependency_report: Any = None  # DependencyReport from resolver
    # Phase 8 additions
    frontend_framework: str = ""  # react, vue, svelte, angular
    bundler: str = ""  # vite, webpack, esbuild
    node_version: str = "20"
    has_package_json: bool = False
    package_json_data: dict = field(default_factory=dict)
    has_migrations: bool = False
    migration_tool: str = ""  # alembic, django
    cors_origins: list[str] = field(default_factory=list)
    frontend_port: int = 3000


class ProjectScaffoldAgent(BaseAgent):
    """Agent for generating deployment-ready project files.

    Analyzes generated code and produces:
    - requirements.txt
    - .env.example
    - Dockerfile (multi-stage for frontend projects)
    - docker-compose.yml
    - README.md
    - run.sh
    - package.json (for JS/frontend projects)
    - vite.config.js / webpack.config.js (bundler config)
    - migrations/ setup (Alembic/Django)
    - health endpoint injection
    """

    def __init__(self, llm: LLMBackend | None = None) -> None:
        """Initialize scaffold agent. LLM is optional."""
        self.llm = llm  # type: ignore[assignment]

    def default_system_prompt(self) -> str:
        return ""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate scaffold files from project code.

        Args:
            input_data: Must contain 'project_dir' in context
                        Optional 'runtime' to control Docker file generation

        Returns:
            AgentOutput with generated files
        """
        context = input_data.context
        project_dir = Path(context.get("project_dir", "."))
        feature_name = context.get("feature_name", "Generated Project")
        runtime = context.get("runtime", "docker")  # Default to docker for backwards compat

        if not project_dir.exists():
            return AgentOutput(
                success=False,
                data={},
                errors=["Project directory not found"]
            )

        # Analyze the project
        result = self._analyze_project(project_dir)

        # Generate all scaffold files
        files_generated = {}

        # requirements.txt (with smart dependency resolution)
        req_content = self._generate_requirements(result)
        files_generated["requirements.txt"] = req_content

        # .gitignore (framework-aware)
        gitignore = self._generate_gitignore(result)
        files_generated[".gitignore"] = gitignore

        # .env.example
        env_content = self._generate_env_example(result)
        files_generated[".env.example"] = env_content

        # Docker files only if runtime requires it
        if runtime == "docker":
            # Dockerfile
            dockerfile = self._generate_dockerfile(result)
            files_generated["Dockerfile"] = dockerfile

            # docker-compose.yml
            compose = self._generate_docker_compose(result, feature_name)
            files_generated["docker-compose.yml"] = compose

        # README.md (pass runtime for conditional Docker docs)
        readme = self._generate_readme(result, feature_name, runtime)
        files_generated["README.md"] = readme

        # run.sh
        run_script = self._generate_run_script(result)
        files_generated["run.sh"] = run_script

        # Phase 8: Frontend bundling setup
        if result.has_frontend or result.has_package_json:
            # package.json
            if not result.has_package_json:
                pkg_json = self._generate_package_json(result, feature_name)
                files_generated["package.json"] = pkg_json

            # Bundler config
            if result.bundler == "vite" or (result.frontend_framework and not result.bundler):
                vite_config = self._generate_vite_config(result)
                files_generated["vite.config.js"] = vite_config
            elif result.bundler == "webpack":
                webpack_config = self._generate_webpack_config(result)
                files_generated["webpack.config.js"] = webpack_config

        # Phase 8: Database migration setup
        if result.has_database and not result.has_migrations:
            migration_files = self._generate_migration_setup(result, project_dir)
            files_generated.update(migration_files)

        # Phase 8: Health endpoint injection (if not present)
        health_file = self._generate_health_endpoint(result, project_dir)
        if health_file:
            files_generated.update(health_file)

        # Write files to project directory
        for filename, content in files_generated.items():
            file_path = project_dir / filename
            # Create parent directories if needed (for migrations/, etc.)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        # Security scan generated files
        security_warnings = scan_generated_files(project_dir)
        all_warnings = []
        for filename, warnings in security_warnings.items():
            for warning in warnings:
                all_warnings.append(f"{filename}: {warning}")

        # Generate GitHub Actions if appropriate
        actions_result = self._generate_github_actions(
            project_dir=project_dir,
            feature_name=feature_name,
            framework=result.framework,
        )
        if actions_result.get("workflows_generated"):
            files_generated.update({
                f: "generated" for f in actions_result["workflows_generated"]
            })

        return AgentOutput(
            success=True,
            data={
                "files_generated": list(files_generated.keys()),
                "framework": result.framework,
                "requirements_count": len(result.requirements),
                "env_vars_count": len(result.env_vars),
                "has_database": result.has_database,
                "security_warnings": all_warnings,
                "github_actions": actions_result,
                # Phase 8 additions
                "frontend_framework": result.frontend_framework,
                "bundler": result.bundler,
                "has_frontend": result.has_frontend,
                "has_migrations": result.has_migrations,
                "migration_tool": result.migration_tool,
                "cors_origins": result.cors_origins,
            },
            artifacts=list(files_generated.keys()),
            errors=all_warnings if any("CRITICAL" in w for w in all_warnings) else [],
        )

    def _analyze_project(self, project_dir: Path) -> ScaffoldResult:
        """Analyze project files to detect dependencies and config."""
        result = ScaffoldResult()

        # Use smart dependency resolver
        resolver = DependencyResolver()
        dep_report = resolver.resolve(project_dir)

        # Store resolved requirements (already deduplicated and versioned)
        result.requirements = [
            f"{dep.name}{dep.version_spec}" if dep.version_spec else dep.name
            for dep in dep_report.dependencies
        ]
        result.dependency_report = dep_report

        # Extract env vars from all Python files
        all_env_vars = set()
        all_imports = set()

        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                env_vars = self._extract_env_vars(content)
                all_env_vars.update(env_vars)

                # Also track raw imports for framework detection
                imports = self._extract_imports(content)
                all_imports.update(imports)
            except Exception:
                continue

        # Scan HTML/JS files for frontend detection
        for html_file in project_dir.rglob("*.html"):
            result.has_frontend = True
        for js_file in project_dir.rglob("*.js"):
            result.has_frontend = True
        for ts_file in project_dir.rglob("*.ts"):
            result.has_frontend = True
        for tsx_file in project_dir.rglob("*.tsx"):
            result.has_frontend = True
        for jsx_file in project_dir.rglob("*.jsx"):
            result.has_frontend = True

        # Phase 8: Detect package.json and frontend framework
        package_json_path = project_dir / "package.json"
        if package_json_path.exists():
            result.has_package_json = True
            try:
                pkg_data = json.loads(package_json_path.read_text())
                result.package_json_data = pkg_data
                deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}

                # Detect frontend framework
                if "react" in deps or "@react" in " ".join(deps.keys()):
                    result.frontend_framework = "react"
                    result.has_frontend = True
                elif "vue" in deps:
                    result.frontend_framework = "vue"
                    result.has_frontend = True
                elif "svelte" in deps:
                    result.frontend_framework = "svelte"
                    result.has_frontend = True
                elif "@angular/core" in deps:
                    result.frontend_framework = "angular"
                    result.has_frontend = True

                # Detect bundler
                if "vite" in deps:
                    result.bundler = "vite"
                elif "webpack" in deps:
                    result.bundler = "webpack"
                elif "esbuild" in deps:
                    result.bundler = "esbuild"

                # Detect node version from engines
                engines = pkg_data.get("engines", {})
                if "node" in engines:
                    # Extract version number (e.g., ">=18" -> "18", "20.x" -> "20")
                    node_ver = engines["node"]
                    ver_match = re.search(r'(\d+)', node_ver)
                    if ver_match:
                        result.node_version = ver_match.group(1)
            except (json.JSONDecodeError, Exception):
                pass

        # Phase 8: Detect existing migrations
        alembic_dir = project_dir / "migrations"
        alembic_ini = project_dir / "alembic.ini"
        django_migrations = list(project_dir.rglob("*/migrations/__init__.py"))

        if alembic_dir.exists() or alembic_ini.exists():
            result.has_migrations = True
            result.migration_tool = "alembic"
        elif django_migrations:
            result.has_migrations = True
            result.migration_tool = "django"

        # Phase 8: Detect CORS configuration
        cors_patterns = [
            r"CORS_ORIGINS?\s*=\s*\[([^\]]+)\]",
            r"allow_origins?\s*=\s*\[([^\]]+)\]",
            r"origins\s*=\s*\[([^\]]+)\]",
        ]
        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in cors_patterns:
                    match = re.search(pattern, content)
                    if match:
                        origins = re.findall(r'["\']([^"\']+)["\']', match.group(1))
                        result.cors_origins.extend(origins)
            except Exception:
                continue

        # Detect framework from imports
        if 'fastapi' in all_imports:
            result.framework = "fastapi"
            result.entry_point = self._find_entry_point(project_dir, "fastapi")
            result.port = 8000
        elif 'flask' in all_imports:
            result.framework = "flask"
            result.entry_point = self._find_entry_point(project_dir, "flask")
            result.port = 5000
        elif 'django' in all_imports:
            result.framework = "django"
            result.entry_point = self._find_entry_point(project_dir, "django")
            result.port = 8000

        # Detect database
        if 'sqlalchemy' in all_imports or 'flask_sqlalchemy' in all_imports:
            result.has_database = True
            result.database_type = "sqlite"  # Default
        if 'psycopg2' in all_imports:
            result.database_type = "postgresql"
        if 'pymongo' in all_imports:
            result.database_type = "mongodb"

        # Store env vars
        result.env_vars = sorted(list(all_env_vars))

        return result

    def _extract_imports(self, content: str) -> set[str]:
        """Extract import names from Python code."""
        imports = set()

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level module
                        module = alias.name.split('.')[0]
                        imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        imports.add(module)
        except SyntaxError:
            # Fallback to regex
            import_pattern = r'^(?:from|import)\s+(\w+)'
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                imports.add(match.group(1))

        return imports

    def _find_entry_point(self, project_dir: Path, framework: str) -> str:
        """Find the actual entry point file for the framework.

        Searches for common patterns and returns relative path.

        Args:
            project_dir: Project directory
            framework: Framework name (flask, fastapi, django)

        Returns:
            Relative path to entry point file
        """
        # Common entry point names to search for
        if framework == "flask":
            candidates = ["app.py", "main.py", "run.py", "wsgi.py", "api/app.py", "src/app.py"]
            app_pattern = r"Flask\s*\("
        elif framework == "fastapi":
            candidates = ["main.py", "app.py", "api/main.py", "src/main.py"]
            app_pattern = r"FastAPI\s*\("
        elif framework == "django":
            candidates = ["manage.py", "wsgi.py"]
            app_pattern = r"execute_from_command_line|get_wsgi_application"
        else:
            return "main.py"

        # Check candidates in order
        for candidate in candidates:
            file_path = project_dir / candidate
            if file_path.exists():
                # Verify it contains the framework app
                try:
                    content = file_path.read_text()
                    if re.search(app_pattern, content):
                        return candidate
                except Exception:
                    continue

        # Search all Python files for the app pattern
        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                if re.search(app_pattern, content):
                    # Also check for if __name__ == "__main__" (entry point indicator)
                    if '__name__' in content and '__main__' in content:
                        return str(py_file.relative_to(project_dir))
            except Exception:
                continue

        # Default fallback
        defaults = {"flask": "app.py", "fastapi": "main.py", "django": "manage.py"}
        return defaults.get(framework, "main.py")

    def _extract_env_vars(self, content: str) -> set[str]:
        """Extract environment variable names from code."""
        env_vars = set()

        # os.environ['VAR'] or os.environ.get('VAR')
        patterns = [
            r"os\.environ\[['\"](\w+)['\"]\]",
            r"os\.environ\.get\(['\"](\w+)['\"]",
            r"os\.getenv\(['\"](\w+)['\"]",
            r"environ\.get\(['\"](\w+)['\"]",
            r"config\[['\"](\w+)['\"]\]",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                env_vars.add(match.group(1))

        return env_vars

    def _generate_requirements(self, result: ScaffoldResult) -> str:
        """Generate requirements.txt content using smart resolver."""
        if result.dependency_report:
            # Use the smart resolver output
            resolver = DependencyResolver()
            return resolver.generate_requirements(
                result.dependency_report,
                include_comments=True
            )

        # Fallback to simple list
        lines = ["# Auto-generated requirements", ""]
        for req in result.requirements:
            lines.append(req)
        return "\n".join(lines) + "\n"

    def _generate_gitignore(self, result: ScaffoldResult) -> str:
        """Generate .gitignore tailored to the project."""
        sections = []

        # Python
        sections.append("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST""")

        # Virtual environments
        sections.append("""# Virtual environments
venv/
ENV/
env/
.venv/
.env.local
.env.*.local""")

        # Environment and secrets
        sections.append("""# Environment and secrets
.env
.env.local
.env.production
*.pem
*.key
secrets.json
credentials.json""")

        # IDE
        sections.append("""# IDE
.idea/
.vscode/
*.swp
*.swo
*~
.project
.pydevproject
.settings/""")

        # OS
        sections.append("""# OS generated
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini""")

        # Testing
        sections.append("""# Testing
.tox/
.nox/
.coverage
.coverage.*
htmlcov/
.pytest_cache/
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/""")

        # Database
        if result.has_database:
            sections.append("""# Database
*.db
*.sqlite
*.sqlite3
instance/
migrations/versions/*.pyc""")

        # Framework specific
        if result.framework == "django":
            sections.append("""# Django
*.log
local_settings.py
db.sqlite3
media/
staticfiles/""")

        if result.framework == "flask":
            sections.append("""# Flask
instance/
.webassets-cache""")

        # Frontend
        if result.has_frontend:
            sections.append("""# Frontend / Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*
package-lock.json
yarn.lock
.next/
out/
.nuxt/
dist/""")

        # Uploads and generated
        sections.append("""# Uploads and generated files
uploads/
static/uploads/
media/uploads/
*.log
logs/""")

        # Docker
        sections.append("""# Docker (optional - uncomment if needed)
# .docker/""")

        return "\n\n".join(sections) + "\n"

    def _generate_env_example(self, result: ScaffoldResult) -> str:
        """Generate .env.example content with security guidance."""
        lines = [
            "# Environment Configuration",
            "# Copy this file to .env and fill in the values",
            "# SECURITY: Never commit .env to version control!",
            "",
        ]

        # Add detected env vars with placeholder values
        for var in result.env_vars:
            if 'SECRET' in var or 'KEY' in var or 'PASSWORD' in var:
                lines.append(f"{var}=your-secret-here")
            elif 'URL' in var or 'URI' in var:
                lines.append(f"{var}=http://localhost:8000")
            elif 'HOST' in var:
                lines.append(f"{var}=localhost")
            elif 'PORT' in var:
                lines.append(f"{var}={result.port}")
            elif 'DEBUG' in var:
                lines.append(f"{var}=true")
            else:
                lines.append(f"{var}=")

        # Add common defaults if not detected
        common_vars = [
            ("DEBUG", "true"),
            ("SECRET_KEY", "change-me-in-production-use-secrets-generator"),
        ]

        if result.has_database:
            if result.database_type == "postgresql":
                lines.append("")
                lines.append("# PostgreSQL (required for docker-compose)")
                common_vars.extend([
                    ("POSTGRES_USER", "app"),
                    ("POSTGRES_PASSWORD", "change-me-use-strong-password"),
                    ("POSTGRES_DB", "app"),
                    ("DATABASE_URL", "postgresql://app:change-me@db:5432/app"),
                ])
            elif result.database_type == "mongodb":
                lines.append("")
                lines.append("# MongoDB (required for docker-compose)")
                common_vars.extend([
                    ("MONGO_USER", "app"),
                    ("MONGO_PASSWORD", "change-me-use-strong-password"),
                    ("MONGODB_URI", "mongodb://app:change-me@db:27017/app"),
                ])
            else:
                common_vars.append(("DATABASE_URL", "sqlite:///app.db"))

        for var, default in common_vars:
            if var not in result.env_vars:
                lines.append(f"{var}={default}")

        # Phase 8: CORS and frontend configuration
        if result.has_frontend or result.cors_origins:
            lines.append("")
            lines.append("# CORS Configuration")
            if result.cors_origins:
                origins_str = ",".join(result.cors_origins)
                lines.append(f"CORS_ORIGINS={origins_str}")
            else:
                lines.append(f"CORS_ORIGINS=http://localhost:{result.frontend_port}")
            lines.append(f"FRONTEND_URL=http://localhost:{result.frontend_port}")

        if result.has_frontend:
            lines.append("")
            lines.append("# Frontend Configuration")
            lines.append(f"FRONTEND_PORT={result.frontend_port}")
            lines.append(f"VITE_API_URL=http://localhost:{result.port}")

        return "\n".join(lines) + "\n"

    def _generate_dockerfile(self, result: ScaffoldResult) -> str:
        """Generate secure Dockerfile.

        Security measures included:
        - Pinned Python version (not :latest)
        - Non-root user for runtime
        - No secrets in build args
        - Python-based healthcheck (no curl pipe)
        - Multi-stage build for frontend projects
        """
        entry = result.entry_point

        # Phase 8: Multi-stage build for frontend projects
        if result.has_frontend and result.frontend_framework:
            return self._generate_multistage_dockerfile(result)

        # Standard Python-only Dockerfile
        preamble = """# Auto-generated Dockerfile
# Security: runs as non-root user, no secrets in build

FROM python:3.11.7-slim-bookworm

# Security: don't run as root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser
"""

        if result.framework == "fastapi":
            # Convert path like api/main.py to api.main
            module = entry.replace("/", ".").replace(".py", "")
            return preamble + f"""
# Expose port
EXPOSE {result.port}

# Health check using Python (no curl security concerns)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; \\
    urllib.request.urlopen('http://localhost:{result.port}/health')" || exit 1

# Run application
CMD ["uvicorn", "{module}:app", "--host", "0.0.0.0", "--port", "{result.port}"]
"""
        elif result.framework == "flask":
            return preamble + f"""
# Expose port
EXPOSE {result.port}

# Health check using Python (no curl security concerns)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; \\
    urllib.request.urlopen('http://localhost:{result.port}/')" || exit 1

# Run application
CMD ["python", "{entry}"]
"""
        else:
            return preamble + f"""
# Expose port
EXPOSE {result.port}

# Run application
CMD ["python", "{entry}"]
"""

    def _generate_multistage_dockerfile(self, result: ScaffoldResult) -> str:
        """Generate multi-stage Dockerfile for frontend + backend projects.

        Stage 1: Build frontend assets
        Stage 2: Python backend with built assets
        """
        entry = result.entry_point
        node_version = result.node_version

        # Determine build command based on bundler
        if result.bundler == "vite":
            build_cmd = "npm run build"
            dist_dir = "dist"
        elif result.bundler == "webpack":
            build_cmd = "npm run build"
            dist_dir = "dist"
        else:
            build_cmd = "npm run build"
            dist_dir = "dist"

        # Determine static files destination based on framework
        if result.framework == "fastapi":
            static_dest = "/app/static"
            module = entry.replace("/", ".").replace(".py", "")
            run_cmd = (
                f'CMD ["uvicorn", "{module}:app", '
                f'"--host", "0.0.0.0", "--port", "{result.port}"]'
            )
        elif result.framework == "flask":
            static_dest = "/app/static"
            run_cmd = f'CMD ["python", "{entry}"]'
        else:
            static_dest = "/app/static"
            run_cmd = f'CMD ["python", "{entry}"]'

        return f"""# Auto-generated Dockerfile (Multi-stage)
# Stage 1: Build frontend assets
# Stage 2: Python backend with built assets

# ============ Stage 1: Frontend Build ============
FROM node:{node_version}-alpine AS frontend-builder

WORKDIR /frontend

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci --only=production=false

# Copy frontend source and build
COPY . .
RUN {build_cmd}

# ============ Stage 2: Python Backend ============
FROM python:3.11.7-slim-bookworm

# Security: don't run as root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appgroup . .

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder --chown=appuser:appgroup /frontend/{dist_dir} {static_dest}

# Switch to non-root user
USER appuser

# Expose port
EXPOSE {result.port}

# Health check using Python (no curl security concerns)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; \\
    urllib.request.urlopen('http://localhost:{result.port}/health')" || exit 1

# Run application
{run_cmd}
"""

    def _generate_docker_compose(self, result: ScaffoldResult, name: str) -> str:
        """Generate secure docker-compose.yml.

        Security measures:
        - No hardcoded passwords (uses env vars)
        - Database not exposed to host by default
        - Read-only volumes where possible
        - No privileged mode
        """
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

        compose = f"""# Auto-generated docker-compose.yml
# Security: no hardcoded secrets, minimal port exposure

services:
  app:
    build: .
    container_name: {slug}
    ports:
      - "{result.port}:{result.port}"
    env_file:
      - .env
    # Security: mount code read-only in production
    # For development, change to: - .:/app
    volumes:
      - .:/app:ro
    restart: unless-stopped
    # Security: drop all capabilities, add only needed ones
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
"""

        if result.has_database:
            if result.database_type == "postgresql":
                compose += """
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    # Security: credentials from env, not hardcoded
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-app}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}
      POSTGRES_DB: ${POSTGRES_DB:-app}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    # Security: only expose to host in development
    # Uncomment for local access:
    # ports:
    #   - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
"""
            elif result.database_type == "mongodb":
                compose += """
    depends_on:
      - db

  db:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USER:-app}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD:?Set MONGO_PASSWORD in .env}
    volumes:
      - mongo_data:/data/db
    # Security: only expose to host in development
    # Uncomment for local access:
    # ports:
    #   - "27017:27017"
    restart: unless-stopped

volumes:
  mongo_data:
"""

        return compose

    def _generate_readme(self, result: ScaffoldResult, name: str, runtime: str = "docker") -> str:
        """Generate README.md.

        Args:
            result: Scaffold analysis result
            name: Project name
            runtime: Runtime mode (docker, venv, local)
        """
        entry = result.entry_point
        # Use single source of truth for run command
        framework_cmd = self._get_run_command(result)

        # Build Docker section only if runtime is docker
        docker_section = ""
        if runtime == "docker":
            docker_section = """
### Running with Docker

```bash
# Build and run
docker-compose up --build

# Or just run
docker-compose up -d
```
"""

        # Build project structure based on runtime
        if runtime == "docker":
            structure = f"""```
.
├── {entry}             # Main application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
├── docker-compose.yml  # Container orchestration
├── .env.example        # Environment template
└── README.md           # This file
```"""
        else:
            structure = f"""```
.
├── {entry}             # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── run.sh              # Run script
└── README.md           # This file
```"""

        return f"""# {name}

Auto-generated project.

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Copy environment config
cp .env.example .env
# Edit .env with your values
```

### Running Locally (Recommended)

The easiest way to run the application is using the provided run script:

```bash
# Make executable (first time only)
chmod +x run.sh

# Run the application
./run.sh
```

The run script automatically:
- Creates a virtual environment if needed
- Installs dependencies
- Loads environment variables
- Runs database migrations (if applicable)
- Starts the application

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
{framework_cmd}
```

The app will be available at http://localhost:{result.port}
{docker_section}
## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Home page |
| GET | /health | Health check |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
{self._env_var_table(result)}

## Project Structure

{structure}

## License

MIT
"""

    def _env_var_table(self, result: ScaffoldResult) -> str:
        """Generate markdown table rows for env vars."""
        rows = []
        for var in result.env_vars:
            rows.append(f"| {var} | Configuration | - |")
        if not rows:
            rows.append("| DEBUG | Enable debug mode | true |")
            rows.append("| SECRET_KEY | Application secret | - |")
        return "\n".join(rows)

    def _get_run_command(self, result: ScaffoldResult) -> str:
        """Get the actual run command for this project.

        Single source of truth for the run command used in both
        run.sh and README documentation.

        Args:
            result: Scaffold analysis result

        Returns:
            The shell command to run the application
        """
        entry = result.entry_point

        if result.framework == "fastapi":
            # Convert path like api/main.py to api.main:app
            module = entry.replace("/", ".").replace(".py", "")
            return f"uvicorn {module}:app --host 0.0.0.0 --port {result.port} --reload"
        elif result.framework == "flask":
            return f"python {entry}"
        elif result.framework == "django":
            return f"python manage.py runserver 0.0.0.0:{result.port}"
        else:
            return f"python {entry}"

    def _generate_run_script(self, result: ScaffoldResult) -> str:
        """Generate run.sh script."""
        cmd = self._get_run_command(result)

        return f"""#!/bin/bash
# Auto-generated run script

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt

# Run database migrations if needed
if [ -f "migrations/env.py" ]; then
    echo "Running migrations..."
    alembic upgrade head
fi

# Start the application
echo "Starting application on port {result.port}..."
{cmd}
"""

    def _generate_github_actions(
        self,
        project_dir: Path,
        feature_name: str,
        framework: str,
    ) -> dict:
        """Generate GitHub Actions workflows if appropriate.

        Uses GitHubActionsAgent to reason about CI/CD needs.

        Args:
            project_dir: Project directory
            feature_name: Feature description for context
            framework: Detected framework

        Returns:
            Dict with decision and generated workflows
        """
        try:
            from agents.base import AgentInput
            from agents.github_actions_agent import GitHubActionsAgent

            agent = GitHubActionsAgent()
            output = agent.run(AgentInput(
                context={
                    "feature_description": feature_name,
                    "project_dir": str(project_dir),
                    "framework": framework,
                }
            ))

            if output.success:
                return output.data
            else:
                return {"error": output.errors, "workflows_generated": []}

        except Exception as e:
            # GitHub Actions generation is optional, don't fail scaffold
            return {"error": str(e), "workflows_generated": []}

    # ============ Phase 8: New Generator Methods ============

    def _generate_package_json(self, result: ScaffoldResult, name: str) -> str:
        """Generate package.json for frontend projects.

        Args:
            result: Scaffold analysis result
            name: Project name

        Returns:
            package.json content as string
        """
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

        # Determine dependencies based on framework
        deps = {}
        dev_deps = {}

        if result.frontend_framework == "react":
            deps["react"] = "^18.2.0"
            deps["react-dom"] = "^18.2.0"
            dev_deps["@types/react"] = "^18.2.0"
            dev_deps["@types/react-dom"] = "^18.2.0"
            dev_deps["@vitejs/plugin-react"] = "^4.2.0"
        elif result.frontend_framework == "vue":
            deps["vue"] = "^3.4.0"
            dev_deps["@vitejs/plugin-vue"] = "^5.0.0"
        elif result.frontend_framework == "svelte":
            deps["svelte"] = "^4.2.0"
            dev_deps["@sveltejs/vite-plugin-svelte"] = "^3.0.0"

        # Add bundler
        if result.bundler == "vite" or not result.bundler:
            dev_deps["vite"] = "^5.0.0"
            dev_deps["typescript"] = "^5.3.0"
        elif result.bundler == "webpack":
            dev_deps["webpack"] = "^5.89.0"
            dev_deps["webpack-cli"] = "^5.1.0"
            dev_deps["webpack-dev-server"] = "^4.15.0"

        # Build scripts based on bundler
        if result.bundler == "webpack":
            scripts = {
                "dev": "webpack serve --mode development",
                "build": "webpack --mode production",
                "preview": "webpack serve --mode production",
            }
        else:
            scripts = {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
            }

        package = {
            "name": slug,
            "version": "1.0.0",
            "type": "module",
            "scripts": scripts,
            "dependencies": deps,
            "devDependencies": dev_deps,
            "engines": {
                "node": f">={result.node_version}"
            }
        }

        return json.dumps(package, indent=2) + "\n"

    def _generate_vite_config(self, result: ScaffoldResult) -> str:
        """Generate Vite configuration file.

        Args:
            result: Scaffold analysis result

        Returns:
            vite.config.js content
        """
        # Determine plugin imports and usage
        if result.frontend_framework == "react":
            plugin_import = "import react from '@vitejs/plugin-react'"
            plugin_usage = "react()"
        elif result.frontend_framework == "vue":
            plugin_import = "import vue from '@vitejs/plugin-vue'"
            plugin_usage = "vue()"
        elif result.frontend_framework == "svelte":
            plugin_import = "import { svelte } from '@sveltejs/vite-plugin-svelte'"
            plugin_usage = "svelte()"
        else:
            plugin_import = "// No framework plugin needed"
            plugin_usage = ""

        plugins_array = f"[{plugin_usage}]" if plugin_usage else "[]"

        return f"""import {{ defineConfig }} from 'vite'
{plugin_import}

// https://vitejs.dev/config/
export default defineConfig({{
  plugins: {plugins_array},
  server: {{
    port: {result.frontend_port},
    proxy: {{
      '/api': {{
        target: 'http://localhost:{result.port}',
        changeOrigin: true,
      }},
    }},
  }},
  build: {{
    outDir: 'dist',
    sourcemap: true,
  }},
}})
"""

    def _generate_webpack_config(self, result: ScaffoldResult) -> str:
        """Generate Webpack configuration file.

        Args:
            result: Scaffold analysis result

        Returns:
            webpack.config.js content
        """
        # Determine entry and loaders based on framework
        if result.frontend_framework == "react":
            entry = "./src/index.jsx"
            loader_rules = """
      {
        test: /\\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
          },
        },
      },"""
        elif result.frontend_framework == "vue":
            entry = "./src/main.js"
            loader_rules = """
      {
        test: /\\.vue$/,
        loader: 'vue-loader',
      },"""
        else:
            entry = "./src/index.js"
            loader_rules = """
      {
        test: /\\.js$/,
        exclude: /node_modules/,
        use: 'babel-loader',
      },"""

        return f"""const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {{
  entry: '{entry}',
  output: {{
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].[contenthash].js',
    clean: true,
  }},
  module: {{
    rules: [{loader_rules}
      {{
        test: /\\.css$/,
        use: ['style-loader', 'css-loader'],
      }},
    ],
  }},
  plugins: [
    new HtmlWebpackPlugin({{
      template: './public/index.html',
    }}),
  ],
  devServer: {{
    static: './dist',
    port: {result.frontend_port},
    hot: true,
    proxy: [
      {{
        context: ['/api'],
        target: 'http://localhost:{result.port}',
        changeOrigin: true,
      }},
    ],
  }},
  resolve: {{
    extensions: ['.js', '.jsx', '.vue', '.ts', '.tsx'],
  }},
}};
"""

    def _generate_migration_setup(
        self, result: ScaffoldResult, project_dir: Path
    ) -> dict[str, str]:
        """Generate database migration setup files.

        Args:
            result: Scaffold analysis result
            project_dir: Project directory

        Returns:
            Dict of filename -> content for migration files
        """
        files = {}

        if result.framework == "django":
            # Django uses built-in migrations, just need management command reminder
            return {}

        # Default to Alembic for Flask/FastAPI with SQLAlchemy
        if result.has_database and result.database_type:
            # alembic.ini
            files["alembic.ini"] = self._generate_alembic_ini(result)

            # migrations/env.py
            files["migrations/env.py"] = self._generate_alembic_env(result)

            # migrations/script.py.mako
            files["migrations/script.py.mako"] = self._generate_alembic_template()

            # migrations/versions/.gitkeep
            files["migrations/versions/.gitkeep"] = ""

        return files

    def _generate_alembic_ini(self, result: ScaffoldResult) -> str:
        """Generate alembic.ini configuration."""
        return """# Auto-generated Alembic configuration
# See https://alembic.sqlalchemy.org/en/latest/tutorial.html

[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os

# Database URL (use environment variable in production)
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
# Enable Black formatting for migration scripts
# hooks = black
# black.type = console_scripts
# black.entrypoint = black

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

    def _generate_alembic_env(self, result: ScaffoldResult) -> str:
        """Generate migrations/env.py for Alembic."""
        return '''"""Alembic migrations environment configuration."""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models here for autogenerate support
# from app.models import Base
# target_metadata = Base.metadata

config = context.config

# Override database URL from environment
database_url = os.getenv("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set to your model's metadata for autogenerate
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

    def _generate_alembic_template(self) -> str:
        """Generate migrations/script.py.mako template."""
        return '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
'''

    def _generate_health_endpoint(
        self, result: ScaffoldResult, project_dir: Path
    ) -> dict[str, str] | None:
        """Generate health check endpoint if not present.

        Checks if a /health endpoint exists, and if not, generates one.

        Args:
            result: Scaffold analysis result
            project_dir: Project directory

        Returns:
            Dict with filename -> content, or None if health endpoint exists
        """
        # Check if health endpoint already exists
        health_patterns = [
            r'@app\.(get|route)\s*\(\s*["\']/?health["\']',
            r'path\s*\(\s*["\']health["\']',
            r'["\']/?health["\'].*GET',
        ]

        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in health_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Health endpoint already exists
                        return None
            except Exception:
                continue

        # Generate health endpoint based on framework
        if result.framework == "fastapi":
            return {"health.py": self._generate_fastapi_health()}
        elif result.framework == "flask":
            return {"health.py": self._generate_flask_health()}
        elif result.framework == "django":
            return {"health_views.py": self._generate_django_health()}

        return None

    def _generate_fastapi_health(self) -> str:
        """Generate FastAPI health check endpoint."""
        return '''"""Health check endpoint for FastAPI.

Import and include this router in your main app:
    from health import router as health_router
    app.include_router(health_router)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check - returns 200 when app is ready to serve traffic."""
    # Add database/cache connectivity checks here
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """Liveness check - returns 200 if the process is alive."""
    return {"alive": True}
'''

    def _generate_flask_health(self) -> str:
        """Generate Flask health check endpoint."""
        return '''"""Health check endpoint for Flask.

Import and register this blueprint in your main app:
    from health import health_bp
    app.register_blueprint(health_bp)
"""

from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    })


@health_bp.route("/ready")
def readiness_check():
    """Readiness check - returns 200 when app is ready to serve traffic."""
    # Add database/cache connectivity checks here
    return jsonify({"ready": True})


@health_bp.route("/live")
def liveness_check():
    """Liveness check - returns 200 if the process is alive."""
    return jsonify({"alive": True})
'''

    def _generate_django_health(self) -> str:
        """Generate Django health check views."""
        return '''"""Health check views for Django.

Add to your urls.py:
    from health_views import health_check, readiness_check, liveness_check

    urlpatterns = [
        path("health/", health_check, name="health"),
        path("ready/", readiness_check, name="ready"),
        path("live/", liveness_check, name="live"),
    ]
"""

from django.http import JsonResponse
from datetime import datetime


def health_check(request):
    """Health check endpoint for load balancers and monitoring."""
    return JsonResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    })


def readiness_check(request):
    """Readiness check - returns 200 when app is ready to serve traffic."""
    # Add database/cache connectivity checks here
    from django.db import connection
    try:
        connection.ensure_connection()
        db_ready = True
    except Exception:
        db_ready = False

    return JsonResponse({"ready": db_ready})


def liveness_check(request):
    """Liveness check - returns 200 if the process is alive."""
    return JsonResponse({"alive": True})
'''
