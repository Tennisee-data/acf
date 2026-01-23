"""Test Generator Agent - Auto-generates pytest test stubs for generated code."""

import ast
import re
from pathlib import Path
from dataclasses import dataclass, field

from agents.base import BaseAgent, AgentInput, AgentOutput
from llm_backend import get_backend
from pipeline.config import get_config


@dataclass
class FunctionInfo:
    """Information about a function to test."""
    name: str
    args: list[str]
    returns: str | None
    docstring: str | None
    is_async: bool
    decorators: list[str]
    file_path: str
    line_number: int


@dataclass
class ClassInfo:
    """Information about a class to test."""
    name: str
    methods: list[FunctionInfo]
    file_path: str
    line_number: int


@dataclass
class EndpointInfo:
    """Information about an API endpoint."""
    method: str  # GET, POST, etc.
    path: str
    function_name: str
    file_path: str


@dataclass
class CodeAnalysis:
    """Analysis of code for test generation."""
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    endpoints: list[EndpointInfo] = field(default_factory=list)
    framework: str = "python"


class TestGeneratorAgent(BaseAgent):
    """Agent that generates pytest test stubs for generated code.

    Analyzes Python files to identify:
    - Functions and their signatures
    - Classes and methods
    - API endpoints (Flask/FastAPI)

    Generates appropriate test cases for each.
    """

    def __init__(self) -> None:
        super().__init__("TestGeneratorAgent")
        self.config = get_config()
        self.backend = get_backend(
            self.config.llm.backend,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model_code,
            timeout=self.config.llm.timeout,
        )

    def default_system_prompt(self) -> str:
        return """You are an expert Python test engineer.
You write clean, comprehensive pytest tests that cover edge cases and error conditions.
You follow best practices: arrange-act-assert pattern, descriptive test names, proper fixtures."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate test stubs for a project.

        Args:
            input_data: Contains:
                - project_dir: Path to generated project
                - framework: Detected framework (optional)

        Returns:
            AgentOutput with generated test files
        """
        project_dir = Path(input_data.context.get("project_dir", "."))
        framework = input_data.context.get("framework", "python")

        if not project_dir.exists():
            return AgentOutput(
                success=False,
                data={},
                errors=["Project directory not found"],
            )

        try:
            # Step 1: Analyze the codebase
            analysis = self._analyze_codebase(project_dir, framework)

            # Step 2: Generate test files
            test_files = self._generate_tests(analysis, project_dir)

            # Step 3: Write test files
            tests_dir = project_dir / "tests"
            tests_dir.mkdir(exist_ok=True)

            # Write conftest.py for fixtures
            conftest = self._generate_conftest(analysis, framework)
            (tests_dir / "conftest.py").write_text(conftest)

            files_written = ["tests/conftest.py"]

            for filename, content in test_files.items():
                file_path = tests_dir / filename
                file_path.write_text(content)
                files_written.append(f"tests/{filename}")

            return AgentOutput(
                success=True,
                data={
                    "files_generated": files_written,
                    "functions_found": len(analysis.functions),
                    "classes_found": len(analysis.classes),
                    "endpoints_found": len(analysis.endpoints),
                    "framework": framework,
                },
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[str(e)],
            )

    def _analyze_codebase(self, project_dir: Path, framework: str) -> CodeAnalysis:
        """Analyze Python files to find testable code.

        Args:
            project_dir: Project directory
            framework: Detected framework

        Returns:
            CodeAnalysis with functions, classes, and endpoints
        """
        analysis = CodeAnalysis(framework=framework)

        for py_file in project_dir.rglob("*.py"):
            # Skip test files and __pycache__
            if "test" in py_file.name.lower() or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                rel_path = str(py_file.relative_to(project_dir))

                for node in ast.walk(tree):
                    # Find functions
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        # Skip private functions and dunder methods
                        if node.name.startswith("_") and not node.name.startswith("__"):
                            continue

                        func_info = FunctionInfo(
                            name=node.name,
                            args=[arg.arg for arg in node.args.args if arg.arg != "self"],
                            returns=ast.unparse(node.returns) if node.returns else None,
                            docstring=ast.get_docstring(node),
                            is_async=isinstance(node, ast.AsyncFunctionDef),
                            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                            file_path=rel_path,
                            line_number=node.lineno,
                        )
                        analysis.functions.append(func_info)

                        # Check for API endpoints
                        endpoint = self._extract_endpoint(func_info, content)
                        if endpoint:
                            analysis.endpoints.append(endpoint)

                    # Find classes
                    elif isinstance(node, ast.ClassDef):
                        if node.name.startswith("_"):
                            continue

                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                                if not item.name.startswith("_") or item.name in ["__init__", "__call__"]:
                                    methods.append(FunctionInfo(
                                        name=item.name,
                                        args=[arg.arg for arg in item.args.args if arg.arg != "self"],
                                        returns=ast.unparse(item.returns) if item.returns else None,
                                        docstring=ast.get_docstring(item),
                                        is_async=isinstance(item, ast.AsyncFunctionDef),
                                        decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                                        file_path=rel_path,
                                        line_number=item.lineno,
                                    ))

                        if methods:
                            analysis.classes.append(ClassInfo(
                                name=node.name,
                                methods=methods,
                                file_path=rel_path,
                                line_number=node.lineno,
                            ))

            except SyntaxError:
                continue

        return analysis

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
            elif isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return ""

    def _extract_endpoint(self, func: FunctionInfo, content: str) -> EndpointInfo | None:
        """Extract API endpoint info from function decorators."""
        for decorator in func.decorators:
            # Flask patterns
            if decorator in ["route", "get", "post", "put", "delete", "patch"]:
                # Try to find the route path
                pattern = rf'@\w+\.{decorator}\s*\(\s*["\']([^"\']+)["\']'
                match = re.search(pattern, content)
                if match:
                    return EndpointInfo(
                        method=decorator.upper() if decorator != "route" else "GET",
                        path=match.group(1),
                        function_name=func.name,
                        file_path=func.file_path,
                    )

            # FastAPI patterns
            elif decorator in ["get", "post", "put", "delete", "patch"]:
                pattern = rf'@\w+\.{decorator}\s*\(\s*["\']([^"\']+)["\']'
                match = re.search(pattern, content)
                if match:
                    return EndpointInfo(
                        method=decorator.upper(),
                        path=match.group(1),
                        function_name=func.name,
                        file_path=func.file_path,
                    )

        return None

    def _generate_tests(self, analysis: CodeAnalysis, project_dir: Path) -> dict[str, str]:
        """Generate test files based on analysis.

        Args:
            analysis: Code analysis results
            project_dir: Project directory

        Returns:
            Dict of filename -> test content
        """
        test_files = {}

        # Group functions by file
        files_to_test = {}
        for func in analysis.functions:
            if func.file_path not in files_to_test:
                files_to_test[func.file_path] = {"functions": [], "classes": []}
            files_to_test[func.file_path]["functions"].append(func)

        for cls in analysis.classes:
            if cls.file_path not in files_to_test:
                files_to_test[cls.file_path] = {"functions": [], "classes": []}
            files_to_test[cls.file_path]["classes"].append(cls)

        # Generate test file for each source file
        for file_path, items in files_to_test.items():
            test_filename = f"test_{Path(file_path).stem}.py"
            test_content = self._generate_test_file(
                file_path,
                items["functions"],
                items["classes"],
                analysis.framework,
            )
            test_files[test_filename] = test_content

        # Generate API tests if endpoints found
        if analysis.endpoints:
            test_files["test_api.py"] = self._generate_api_tests(
                analysis.endpoints,
                analysis.framework,
            )

        return test_files

    def _generate_test_file(
        self,
        source_file: str,
        functions: list[FunctionInfo],
        classes: list[ClassInfo],
        framework: str,
    ) -> str:
        """Generate a test file for a source file.

        Args:
            source_file: Path to source file
            functions: Functions to test
            classes: Classes to test
            framework: Detected framework

        Returns:
            Test file content
        """
        module_path = source_file.replace("/", ".").replace(".py", "")

        lines = [
            '"""Auto-generated tests for {source_file}."""',
            "",
            "import pytest",
            "",
        ]

        # Add imports for functions
        func_names = [f.name for f in functions if not any(d in ["route", "get", "post", "put", "delete"] for d in f.decorators)]
        class_names = [c.name for c in classes]

        if func_names or class_names:
            imports = func_names + class_names
            lines.append(f"from {module_path} import {', '.join(imports)}")
            lines.append("")

        # Generate function tests
        for func in functions:
            # Skip route handlers (tested separately in API tests)
            if any(d in ["route", "get", "post", "put", "delete", "patch"] for d in func.decorators):
                continue

            lines.extend(self._generate_function_test(func))

        # Generate class tests
        for cls in classes:
            lines.extend(self._generate_class_test(cls))

        return "\n".join(lines).format(source_file=source_file)

    def _generate_function_test(self, func: FunctionInfo) -> list[str]:
        """Generate test for a function."""
        lines = []

        # Async test
        if func.is_async:
            lines.append("@pytest.mark.asyncio")
            lines.append(f"async def test_{func.name}():")
        else:
            lines.append(f"def test_{func.name}():")

        # Add docstring with info
        if func.docstring:
            lines.append(f'    """Test {func.name}: {func.docstring[:50]}..."""')
        else:
            lines.append(f'    """Test {func.name} function."""')

        # Generate test body based on signature
        if func.args:
            lines.append("    # Arrange")
            for arg in func.args:
                lines.append(f"    {arg} = None  # TODO: Set test value")
            lines.append("")
            lines.append("    # Act")
            args_str = ", ".join(func.args)
            if func.is_async:
                lines.append(f"    result = await {func.name}({args_str})")
            else:
                lines.append(f"    result = {func.name}({args_str})")
        else:
            lines.append("    # Act")
            if func.is_async:
                lines.append(f"    result = await {func.name}()")
            else:
                lines.append(f"    result = {func.name}()")

        lines.append("")
        lines.append("    # Assert")
        if func.returns:
            lines.append(f"    assert result is not None  # Expected: {func.returns}")
        else:
            lines.append("    assert result is not None  # TODO: Add proper assertion")

        lines.append("")
        lines.append("")

        return lines

    def _generate_class_test(self, cls: ClassInfo) -> list[str]:
        """Generate tests for a class."""
        lines = [
            f"class Test{cls.name}:",
            f'    """Tests for {cls.name} class."""',
            "",
        ]

        # Test fixture for class instance
        lines.append("    @pytest.fixture")
        lines.append(f"    def instance(self):")
        lines.append(f'        """Create {cls.name} instance for testing."""')
        lines.append(f"        return {cls.name}()  # TODO: Add constructor args if needed")
        lines.append("")

        # Generate test for each method
        for method in cls.methods:
            if method.name == "__init__":
                lines.extend(self._generate_init_test(cls))
            else:
                lines.extend(self._generate_method_test(method))

        return lines

    def _generate_init_test(self, cls: ClassInfo) -> list[str]:
        """Generate test for class __init__."""
        return [
            f"    def test_init(self):",
            f'        """Test {cls.name} initialization."""',
            f"        instance = {cls.name}()  # TODO: Add args",
            f"        assert instance is not None",
            "",
            "",
        ]

    def _generate_method_test(self, method: FunctionInfo) -> list[str]:
        """Generate test for a class method."""
        lines = []

        if method.is_async:
            lines.append("    @pytest.mark.asyncio")
            lines.append(f"    async def test_{method.name}(self, instance):")
        else:
            lines.append(f"    def test_{method.name}(self, instance):")

        lines.append(f'        """Test {method.name} method."""')

        # Setup args
        if method.args:
            for arg in method.args:
                lines.append(f"        {arg} = None  # TODO: Set test value")

            args_str = ", ".join(method.args)
            if method.is_async:
                lines.append(f"        result = await instance.{method.name}({args_str})")
            else:
                lines.append(f"        result = instance.{method.name}({args_str})")
        else:
            if method.is_async:
                lines.append(f"        result = await instance.{method.name}()")
            else:
                lines.append(f"        result = instance.{method.name}()")

        lines.append("        assert result is not None  # TODO: Add proper assertion")
        lines.append("")
        lines.append("")

        return lines

    def _generate_api_tests(self, endpoints: list[EndpointInfo], framework: str) -> str:
        """Generate API endpoint tests."""
        lines = [
            '"""Auto-generated API tests."""',
            "",
            "import pytest",
            "",
        ]

        if framework == "fastapi":
            lines.extend([
                "from fastapi.testclient import TestClient",
                "",
                "# TODO: Import your FastAPI app",
                "# from main import app",
                "",
                "@pytest.fixture",
                "def client():",
                '    """Create test client."""',
                "    # return TestClient(app)",
                "    pass  # TODO: Uncomment above",
                "",
                "",
            ])
        elif framework == "flask":
            lines.extend([
                "",
                "# TODO: Import your Flask app",
                "# from app import app",
                "",
                "@pytest.fixture",
                "def client():",
                '    """Create test client."""',
                "    # app.config['TESTING'] = True",
                "    # return app.test_client()",
                "    pass  # TODO: Uncomment above",
                "",
                "",
            ])

        # Generate test for each endpoint
        for endpoint in endpoints:
            lines.extend(self._generate_endpoint_test(endpoint))

        return "\n".join(lines)

    def _generate_endpoint_test(self, endpoint: EndpointInfo) -> list[str]:
        """Generate test for an API endpoint."""
        method_lower = endpoint.method.lower()
        test_name = f"test_{method_lower}_{endpoint.function_name}"

        lines = [
            f"def {test_name}(client):",
            f'    """Test {endpoint.method} {endpoint.path}."""',
            f"    # Arrange",
        ]

        if endpoint.method in ["POST", "PUT", "PATCH"]:
            lines.append("    payload = {}  # TODO: Add request body")
            lines.append("")
            lines.append("    # Act")
            lines.append(f'    response = client.{method_lower}("{endpoint.path}", json=payload)')
        else:
            lines.append("    # Act")
            lines.append(f'    response = client.{method_lower}("{endpoint.path}")')

        lines.extend([
            "",
            "    # Assert",
            "    assert response.status_code == 200  # TODO: Adjust expected status",
            "",
            "",
        ])

        return lines

    def _generate_conftest(self, analysis: CodeAnalysis, framework: str) -> str:
        """Generate conftest.py with common fixtures."""
        lines = [
            '"""Pytest configuration and fixtures."""',
            "",
            "import pytest",
            "",
        ]

        if framework == "fastapi":
            lines.extend([
                "# Uncomment to enable async tests",
                "# import asyncio",
                "#",
                "# @pytest.fixture(scope='session')",
                "# def event_loop():",
                "#     loop = asyncio.get_event_loop_policy().new_event_loop()",
                "#     yield loop",
                "#     loop.close()",
                "",
            ])

        lines.extend([
            "# Add your fixtures here",
            "",
            "@pytest.fixture",
            "def sample_data():",
            '    """Sample test data."""',
            "    return {",
            '        "id": 1,',
            '        "name": "test",',
            "    }",
            "",
        ])

        return "\n".join(lines)
