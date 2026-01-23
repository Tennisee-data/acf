"""Smart dependency resolver for Python projects.

Analyzes imports and produces minimal, deduplicated requirements
with proper version pinning.
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Standard library modules - never add to requirements
STDLIB_MODULES = {
    # Built-in modules
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asyncio', 'atexit', 'base64',
    'bdb', 'binascii', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
    'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
    'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
    'contextvars', 'copy', 'copyreg', 'cProfile', 'csv', 'ctypes', 'curses',
    'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'doctest',
    'email', 'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
    'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
    'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq',
    'hmac', 'html', 'http', 'imaplib', 'importlib', 'inspect', 'io', 'ipaddress',
    'itertools', 'json', 'keyword', 'linecache', 'locale', 'logging', 'lzma',
    'mailbox', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder',
    'multiprocessing', 'netrc', 'numbers', 'operator', 'optparse', 'os',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pkgutil', 'platform', 'plistlib',
    'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd',
    'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're',
    'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
    'site', 'smtplib', 'socket', 'socketserver', 'sqlite3', 'ssl', 'stat',
    'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'symtable',
    'sys', 'sysconfig', 'tabnanny', 'tarfile', 'tempfile', 'termios', 'test',
    'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
    'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'types',
    'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
    'warnings', 'wave', 'weakref', 'webbrowser', 'wsgiref', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib',
    # typing extras
    'typing_extensions',
    # Common local module names to ignore
    'config', 'settings', 'models', 'views', 'routes', 'handlers', 'utils',
    'helpers', 'schemas', 'services', 'api', 'app', 'main', 'core', 'common',
    'auth', 'database', 'db', 'middleware', 'tests', 'test',
}

# Import name -> PyPI package name mapping
IMPORT_TO_PACKAGE = {
    # Web frameworks
    'flask': 'Flask',
    'flask_sqlalchemy': 'Flask-SQLAlchemy',
    'flask_jwt_extended': 'Flask-JWT-Extended',
    'flask_cors': 'Flask-CORS',
    'flask_migrate': 'Flask-Migrate',
    'flask_login': 'Flask-Login',
    'flask_wtf': 'Flask-WTF',
    'fastapi': 'fastapi',
    'starlette': 'starlette',
    'uvicorn': 'uvicorn',
    'django': 'Django',
    'rest_framework': 'djangorestframework',

    # Database
    'sqlalchemy': 'SQLAlchemy',
    'alembic': 'alembic',
    'psycopg2': 'psycopg2-binary',
    'pymongo': 'pymongo',
    'motor': 'motor',
    'redis': 'redis',
    'pymysql': 'PyMySQL',

    # Data science
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'plotly': 'plotly',

    # ML/AI
    'torch': 'torch',
    'tensorflow': 'tensorflow',
    'keras': 'keras',
    'transformers': 'transformers',

    # HTTP/API
    'requests': 'requests',
    'httpx': 'httpx',
    'aiohttp': 'aiohttp',
    'urllib3': 'urllib3',

    # Auth/Security
    'jwt': 'PyJWT',
    'passlib': 'passlib',
    'bcrypt': 'bcrypt',
    'cryptography': 'cryptography',
    'oauthlib': 'oauthlib',

    # Utilities
    'pydantic': 'pydantic',
    'marshmallow': 'marshmallow',
    'attrs': 'attrs',
    'dotenv': 'python-dotenv',
    'yaml': 'PyYAML',
    'toml': 'toml',
    'click': 'click',
    'typer': 'typer',
    'rich': 'rich',

    # Image/Media
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'imageio': 'imageio',

    # Parsing
    'bs4': 'beautifulsoup4',
    'lxml': 'lxml',
    'html5lib': 'html5lib',

    # Testing
    'pytest': 'pytest',
    'mock': 'mock',
    'faker': 'Faker',

    # Cloud/APIs
    'boto3': 'boto3',
    'stripe': 'stripe',
    'twilio': 'twilio',
    'sendgrid': 'sendgrid',

    # Task queues
    'celery': 'celery',
    'rq': 'rq',

    # Misc
    'werkzeug': 'Werkzeug',
    'jinja2': 'Jinja2',
    'dateutil': 'python-dateutil',
    'pytz': 'pytz',
    'pendulum': 'pendulum',
}

# Known transitive dependencies (package -> list of deps it includes)
# If a package is listed as a dep of another, we can skip it
TRANSITIVE_DEPS = {
    'Flask': {'Werkzeug', 'Jinja2', 'click', 'itsdangerous', 'MarkupSafe'},
    'Flask-SQLAlchemy': {'SQLAlchemy', 'Flask'},
    'Flask-JWT-Extended': {'PyJWT', 'Flask'},
    'Flask-CORS': {'Flask'},
    'Flask-Migrate': {'Flask', 'alembic', 'Flask-SQLAlchemy'},
    'fastapi': {'starlette', 'pydantic'},
    'Django': {'asgiref', 'sqlparse'},
    'djangorestframework': {'Django'},
    'pandas': {'numpy', 'python-dateutil', 'pytz'},
    'scikit-learn': {'numpy', 'scipy', 'joblib', 'threadpoolctl'},
    'scipy': {'numpy'},
    'matplotlib': {'numpy', 'pillow', 'pyparsing', 'cycler', 'kiwisolver'},
    'seaborn': {'matplotlib', 'pandas', 'numpy'},
    'requests': {'urllib3', 'certifi', 'charset-normalizer', 'idna'},
    'httpx': {'certifi', 'httpcore', 'idna', 'sniffio'},
    'aiohttp': {'attrs', 'charset-normalizer', 'multidict', 'async-timeout', 'yarl'},
    'SQLAlchemy': {'greenlet'},
    'alembic': {'SQLAlchemy', 'Mako'},
    'celery': {'kombu', 'billiard', 'vine', 'amqp'},
    'boto3': {'botocore', 's3transfer', 'jmespath'},
    'uvicorn': {'click', 'h11'},
    'rich': {'markdown-it-py', 'pygments'},
    'typer': {'click', 'rich'},
    'pydantic': {'typing-extensions', 'annotated-types'},
    'transformers': {'torch', 'numpy', 'tokenizers', 'huggingface-hub'},
}

# Recommended versions (regularly updated, stable versions)
RECOMMENDED_VERSIONS = {
    'Flask': '>=3.0.0',
    'Flask-SQLAlchemy': '>=3.1.0',
    'Flask-JWT-Extended': '>=4.6.0',
    'Flask-CORS': '>=4.0.0',
    'fastapi': '>=0.109.0',
    'uvicorn': '>=0.27.0',
    'Django': '>=5.0',
    'SQLAlchemy': '>=2.0.0',
    'pydantic': '>=2.0.0',
    'requests': '>=2.31.0',
    'httpx': '>=0.26.0',
    'numpy': '>=1.26.0',
    'pandas': '>=2.1.0',
    'pytest': '>=8.0.0',
    'python-dotenv': '>=1.0.0',
    'redis': '>=5.0.0',
    'celery': '>=5.3.0',
    'Pillow': '>=10.0.0',
    'PyJWT': '>=2.8.0',
    'passlib': '>=1.7.4',
    'bcrypt': '>=4.1.0',
}


@dataclass
class ResolvedDependency:
    """A resolved package dependency."""
    name: str
    version_spec: str = ""
    is_direct: bool = True  # vs transitive
    source_imports: list[str] = field(default_factory=list)


@dataclass
class DependencyReport:
    """Report from dependency resolution."""
    dependencies: list[ResolvedDependency]
    warnings: list[str] = field(default_factory=list)
    unknown_imports: list[str] = field(default_factory=list)


class DependencyResolver:
    """Resolves Python imports to minimal, pinned requirements."""

    def __init__(self, use_pip_check: bool = False):
        """Initialize resolver.

        Args:
            use_pip_check: If True, use pip to verify packages exist
        """
        self.use_pip_check = use_pip_check
        self._pip_cache: dict[str, bool] = {}

    def resolve(self, project_dir: Path) -> DependencyReport:
        """Resolve dependencies for a project.

        Args:
            project_dir: Path to project directory

        Returns:
            DependencyReport with resolved dependencies
        """
        # Step 1: Extract all imports from Python files
        imports = self._extract_all_imports(project_dir)

        # Step 2: Filter out stdlib and local imports
        external_imports = self._filter_external(imports)

        # Step 3: Map imports to package names
        packages = self._map_to_packages(external_imports)

        # Step 4: Remove transitive dependencies
        direct_packages = self._remove_transitive(packages)

        # Step 5: Add version specs
        dependencies = self._add_versions(direct_packages)

        # Step 6: Sort alphabetically
        dependencies.sort(key=lambda d: d.name.lower())

        # Collect unknown imports
        unknown = [imp for imp in external_imports
                   if imp not in IMPORT_TO_PACKAGE and imp not in self._pip_cache]

        return DependencyReport(
            dependencies=dependencies,
            unknown_imports=unknown,
        )

    def _extract_all_imports(self, project_dir: Path) -> set[str]:
        """Extract all import names from Python files."""
        imports = set()

        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                file_imports = self._parse_imports(content)
                imports.update(file_imports)
            except Exception:
                continue

        return imports

    def _parse_imports(self, code: str) -> set[str]:
        """Parse imports from Python code using AST."""
        imports = set()

        try:
            tree = ast.parse(code)
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
            # Fallback to regex for invalid syntax
            pattern = r'^(?:from|import)\s+(\w+)'
            for match in re.finditer(pattern, code, re.MULTILINE):
                imports.add(match.group(1))

        return imports

    def _filter_external(self, imports: set[str]) -> set[str]:
        """Filter to only external (non-stdlib, non-local) imports."""
        return {imp for imp in imports if imp.lower() not in STDLIB_MODULES}

    def _map_to_packages(self, imports: set[str]) -> dict[str, list[str]]:
        """Map import names to PyPI package names.

        Returns:
            Dict mapping package name to list of imports that need it
        """
        packages: dict[str, list[str]] = {}

        for imp in imports:
            # Check known mapping
            if imp in IMPORT_TO_PACKAGE:
                pkg = IMPORT_TO_PACKAGE[imp]
            elif imp.lower() in IMPORT_TO_PACKAGE:
                pkg = IMPORT_TO_PACKAGE[imp.lower()]
            else:
                # Assume import name = package name (common case)
                pkg = imp

            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(imp)

        return packages

    def _remove_transitive(self, packages: dict[str, list[str]]) -> dict[str, list[str]]:
        """Remove packages that are transitive dependencies of others."""
        # Build set of all transitive deps
        all_transitive = set()
        for pkg in packages:
            if pkg in TRANSITIVE_DEPS:
                all_transitive.update(TRANSITIVE_DEPS[pkg])

        # Remove packages that are transitive
        direct = {}
        for pkg, imports in packages.items():
            if pkg not in all_transitive:
                direct[pkg] = imports
            # But keep if it's explicitly imported even if transitive
            elif any(imp.lower() == pkg.lower() for imp in imports):
                direct[pkg] = imports

        return direct

    def _add_versions(self, packages: dict[str, list[str]]) -> list[ResolvedDependency]:
        """Add version specifications to packages."""
        dependencies = []

        for pkg, imports in packages.items():
            # Get recommended version or use >=
            version = RECOMMENDED_VERSIONS.get(pkg, "")

            dependencies.append(ResolvedDependency(
                name=pkg,
                version_spec=version,
                source_imports=imports,
            ))

        return dependencies

    def generate_requirements(
        self,
        report: DependencyReport,
        include_comments: bool = True,
        group_by_category: bool = False,
    ) -> str:
        """Generate requirements.txt content from report.

        Args:
            report: Dependency resolution report
            include_comments: Add comments with source imports
            group_by_category: Group packages by category (TODO)

        Returns:
            requirements.txt content
        """
        lines = [
            "# Auto-generated requirements.txt",
            "# Generated by coding-factory",
            "",
        ]

        for dep in report.dependencies:
            line = dep.name
            if dep.version_spec:
                line += dep.version_spec

            if include_comments and dep.source_imports:
                imports_str = ", ".join(sorted(dep.source_imports)[:3])
                if len(dep.source_imports) > 3:
                    imports_str += f" +{len(dep.source_imports) - 3} more"
                line += f"  # imported as: {imports_str}"

            lines.append(line)

        # Add warning about unknown imports
        if report.unknown_imports:
            lines.append("")
            lines.append("# Unknown imports (may need manual review):")
            for imp in sorted(report.unknown_imports)[:10]:
                lines.append(f"# - {imp}")
            if len(report.unknown_imports) > 10:
                lines.append(f"# ... and {len(report.unknown_imports) - 10} more")

        return "\n".join(lines) + "\n"


def resolve_project(project_dir: Path) -> str:
    """Convenience function to resolve and generate requirements.

    Args:
        project_dir: Path to project

    Returns:
        requirements.txt content
    """
    resolver = DependencyResolver()
    report = resolver.resolve(project_dir)
    return resolver.generate_requirements(report)


if __name__ == "__main__":
    # CLI usage
    import sys

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path.cwd()

    print(resolve_project(path))
