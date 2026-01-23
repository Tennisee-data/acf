"""Code Review Agent for simulating senior engineer review.

Reviews code with focus on:
- Clarity and naming
- Modularity and organization
- YAGNI / overengineering detection
- Consistency with project style
- Best practices and common pitfalls
"""

import ast
import re
import time
from pathlib import Path

from llm_backend import LLMBackend
from schemas.code_review_report import (
    AutoFix,
    CodeReviewReport,
    FileReview,
    IssueCategory,
    IssueSeverity,
    ReviewIssue,
    ShipStatus,
    StyleGuide,
)

from .base import AgentInput, AgentOutput, BaseAgent
from .prompts import CODE_PRINCIPLES, REVIEW_PRINCIPLES

# Common naming anti-patterns
BAD_NAMES = {
    "data", "info", "temp", "tmp", "foo", "bar", "baz",
    "x", "y", "z", "a", "b", "c", "i", "j", "k",
    "result", "ret", "val", "value", "item", "thing",
    "obj", "object", "stuff", "misc", "util", "utils",
    "helper", "helpers", "manager", "handler",
}

# Allowed short names in specific contexts
ALLOWED_SHORT_NAMES = {
    "i", "j", "k",  # Loop indices
    "e",  # Exception
    "f",  # File handle
    "n",  # Count
    "x", "y", "z",  # Coordinates
    "df",  # DataFrame
    "db",  # Database
    "id",  # Identifier
    "pk",  # Primary key
    "ok",  # Boolean flag
}

# Patterns indicating overengineering
OVERENGINEERING_PATTERNS = [
    r"class.*Factory.*:",
    r"class.*Builder.*:",
    r"class.*Singleton.*:",
    r"class.*Abstract.*:",
    r"class.*Interface.*:",
    r"def.*_strategy\(",
    r"@abstractmethod",
]

# Maximum recommended values
MAX_FUNCTION_LINES = 50
MAX_CLASS_METHODS = 15
MAX_PARAMETERS = 5
MAX_NESTED_DEPTH = 4
MAX_CYCLOMATIC_COMPLEXITY = 10


class CodeReviewAgent(BaseAgent):
    """Agent for performing code reviews like a senior engineer."""

    def __init__(self, llm: LLMBackend) -> None:
        """Initialize CodeReviewAgent.

        Args:
            llm: LLM backend for intelligent review.
        """
        super().__init__(llm)

    def default_system_prompt(self) -> str:
        """Return the default system prompt for code review."""
        return f"""{CODE_PRINCIPLES}

{REVIEW_PRINCIPLES}

## Role

You are a senior software engineer performing code review.
Focus on clarity, modularity, YAGNI violations, and consistency. Be direct
and actionable in your feedback."""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Review code and produce review report.

        Args:
            input_data: Input containing repo path and context.

        Returns:
            AgentOutput with code review report.
        """
        start_time = time.time()
        repo_path = Path(input_data.repo_path)
        context = input_data.context or {}

        report = CodeReviewReport()

        try:
            # 1. Detect project style
            report.style_guide = self._detect_style(repo_path)

            # 2. Get files to review
            files_to_review = self._get_files_to_review(repo_path, context)

            # 3. Review each file
            for file_path in files_to_review:
                file_review = self._review_file(
                    file_path, repo_path, report.style_guide, report
                )
                report.file_reviews.append(file_review)

            # 4. Run LLM-based review for deeper analysis
            if context.get("llm_review", True):
                self._llm_review(repo_path, files_to_review, report)

            # 5. Identify auto-fixable issues
            if context.get("auto_fix", False):
                self._apply_auto_fixes(repo_path, report)

            # 6. Update counts and determine status
            report.update_counts()
            report.determine_ship_status()
            report.review_time_seconds = time.time() - start_time

            # 7. Generate summary and top concerns
            report.summary = self._generate_summary(report)
            report.top_concerns = self._get_top_concerns(report)

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

    def _detect_style(self, repo_path: Path) -> StyleGuide:
        """Detect project coding style from existing code.

        Args:
            repo_path: Path to the repository.

        Returns:
            Detected StyleGuide.
        """
        style = StyleGuide()

        # Check for configuration files
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            # Check for line length config
            if "line-length" in content:
                match = re.search(r"line-length\s*=\s*(\d+)", content)
                if match:
                    style.max_line_length = int(match.group(1))

        # Check ruff.toml or .ruff.toml
        for ruff_config in ["ruff.toml", ".ruff.toml"]:
            ruff_path = repo_path / ruff_config
            if ruff_path.exists():
                content = ruff_path.read_text()
                if "line-length" in content:
                    match = re.search(r"line-length\s*=\s*(\d+)", content)
                    if match:
                        style.max_line_length = int(match.group(1))

        # Analyze existing code for patterns
        py_files = list(repo_path.rglob("*.py"))[:10]  # Sample first 10 files
        has_type_hints = False

        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text()
                # Check for type hints
                if "->" in content or ": str" in content or ": int" in content:
                    has_type_hints = True
                    break
            except (UnicodeDecodeError, OSError):
                continue

        style.type_hints = has_type_hints
        return style

    def _get_files_to_review(
        self, repo_path: Path, context: dict
    ) -> list[Path]:
        """Get list of files to review.

        Args:
            repo_path: Path to the repository.
            context: Context with optional file filters.

        Returns:
            List of file paths to review.
        """
        # Check if specific files are provided
        if "files" in context:
            return [Path(f) for f in context["files"]]

        # Check if diff is provided
        if "changed_files" in context:
            return [repo_path / f for f in context["changed_files"]]

        # Default: review all Python files (limit to recent/important)
        py_files = []
        for py_file in repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if "test" in py_file.name.lower():
                continue  # Skip test files for now
            if ".venv" in str(py_file) or "venv" in str(py_file):
                continue
            py_files.append(py_file)

        # Sort by modification time, review most recent first
        py_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return py_files[:20]  # Limit to 20 files

    def _review_file(
        self,
        file_path: Path,
        repo_path: Path,
        style: StyleGuide,
        report: CodeReviewReport,
    ) -> FileReview:
        """Review a single file.

        Args:
            file_path: Path to the file.
            repo_path: Repository root path.
            style: Style guide to check against.
            report: Report to add issues to.

        Returns:
            FileReview summary.
        """
        rel_path = str(file_path.relative_to(repo_path))
        review = FileReview(file_path=rel_path)

        try:
            content = file_path.read_text()
            lines = content.splitlines()
            review.lines_reviewed = len(lines)

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                report.issues.append(
                    ReviewIssue(
                        file_path=rel_path,
                        line_start=1,
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.BEST_PRACTICE,
                        title="Syntax error",
                        description="File has syntax errors and cannot be parsed",
                    )
                )
                return review

            # Check naming conventions
            self._check_naming(tree, rel_path, report)

            # Check function complexity
            self._check_complexity(tree, rel_path, content, report)

            # Check for overengineering patterns
            self._check_overengineering(content, rel_path, report)

            # Check style violations
            self._check_style(lines, rel_path, style, report)

            # Check for common anti-patterns
            self._check_antipatterns(tree, content, rel_path, report)

            # Update review counts
            file_issues = [i for i in report.issues if i.file_path == rel_path]
            review.issues_count = len(file_issues)
            review.critical_count = sum(
                1 for i in file_issues if i.severity == IssueSeverity.CRITICAL
            )
            review.major_count = sum(
                1 for i in file_issues if i.severity == IssueSeverity.MAJOR
            )
            review.minor_count = sum(
                1 for i in file_issues if i.severity == IssueSeverity.MINOR
            )
            review.nit_count = sum(
                1 for i in file_issues if i.severity == IssueSeverity.NIT
            )

        except (UnicodeDecodeError, OSError) as e:
            review.summary = f"Could not read file: {e}"

        return review

    def _check_naming(
        self, tree: ast.AST, file_path: str, report: CodeReviewReport
    ) -> None:
        """Check naming conventions.

        Args:
            tree: AST of the file.
            file_path: File path for reporting.
            report: Report to add issues to.
        """
        for node in ast.walk(tree):
            # Check function names
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if not name.startswith("_"):
                    # Check for bad generic names
                    if name.lower() in BAD_NAMES:
                        report.issues.append(
                            ReviewIssue(
                                file_path=file_path,
                                line_start=node.lineno,
                                severity=IssueSeverity.MINOR,
                                category=IssueCategory.NAMING,
                                title=f"Generic function name: {name}",
                                description=(
                                    f"'{name}' is too generic. Use a more "
                                    "descriptive name that explains what the "
                                    "function does."
                                ),
                                suggestion="Consider renaming to describe the action",
                            )
                        )

                    # Check for camelCase (should be snake_case)
                    if re.match(r"^[a-z]+[A-Z]", name):
                        report.issues.append(
                            ReviewIssue(
                                file_path=file_path,
                                line_start=node.lineno,
                                severity=IssueSeverity.NIT,
                                category=IssueCategory.NAMING,
                                title=f"camelCase function name: {name}",
                                description=(
                                    "Python convention is snake_case for "
                                    "function names."
                                ),
                                suggestion=self._to_snake_case(name),
                                auto_fixable=True,
                            )
                        )

            # Check class names
            elif isinstance(node, ast.ClassDef):
                name = node.name
                if not name[0].isupper():
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=node.lineno,
                            severity=IssueSeverity.MINOR,
                            category=IssueCategory.NAMING,
                            title=f"Class name not PascalCase: {name}",
                            description=(
                                "Python convention is PascalCase for class names."
                            ),
                            suggestion=name[0].upper() + name[1:],
                        )
                    )

            # Check variable assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Single letter variables (except allowed)
                        if (
                            len(name) == 1
                            and name not in ALLOWED_SHORT_NAMES
                            and not name.startswith("_")
                        ):
                            report.issues.append(
                                ReviewIssue(
                                    file_path=file_path,
                                    line_start=node.lineno,
                                    severity=IssueSeverity.NIT,
                                    category=IssueCategory.NAMING,
                                    title=f"Single-letter variable: {name}",
                                    description=(
                                        "Use descriptive variable names except "
                                        "for common conventions (i, j, k for loops)."
                                    ),
                                )
                            )

    def _check_complexity(
        self,
        tree: ast.AST,
        file_path: str,
        content: str,
        report: CodeReviewReport,
    ) -> None:
        """Check code complexity.

        Args:
            tree: AST of the file.
            file_path: File path for reporting.
            content: File content.
            report: Report to add issues to.
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check function length
                if hasattr(node, "end_lineno") and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > MAX_FUNCTION_LINES:
                        report.issues.append(
                            ReviewIssue(
                                file_path=file_path,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                                severity=IssueSeverity.MAJOR,
                                category=IssueCategory.COMPLEXITY,
                                title=f"Function too long: {node.name}",
                                description=(
                                    f"Function has {func_lines} lines. "
                                    f"Consider breaking it into smaller functions "
                                    f"(recommended max: {MAX_FUNCTION_LINES})."
                                ),
                            )
                        )

                # Check parameter count
                param_count = len(node.args.args)
                if param_count > MAX_PARAMETERS:
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=node.lineno,
                            severity=IssueSeverity.MINOR,
                            category=IssueCategory.COMPLEXITY,
                            title=f"Too many parameters: {node.name}",
                            description=(
                                f"Function has {param_count} parameters. "
                                f"Consider using a config object or dataclass "
                                f"(recommended max: {MAX_PARAMETERS})."
                            ),
                        )
                    )

                # Check nesting depth
                max_depth = self._get_max_depth(node)
                if max_depth > MAX_NESTED_DEPTH:
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=node.lineno,
                            severity=IssueSeverity.MAJOR,
                            category=IssueCategory.COMPLEXITY,
                            title=f"Deep nesting: {node.name}",
                            description=(
                                f"Function has nesting depth of {max_depth}. "
                                f"Consider early returns or extracting logic "
                                f"(recommended max: {MAX_NESTED_DEPTH})."
                            ),
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                # Check number of methods
                methods = [
                    n for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if len(methods) > MAX_CLASS_METHODS:
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=node.lineno,
                            severity=IssueSeverity.MINOR,
                            category=IssueCategory.MODULARITY,
                            title=f"Class has too many methods: {node.name}",
                            description=(
                                f"Class has {len(methods)} methods. "
                                f"Consider splitting into smaller classes "
                                f"(recommended max: {MAX_CLASS_METHODS})."
                            ),
                        )
                    )

    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth.

        Args:
            node: AST node.
            current_depth: Current depth level.

        Returns:
            Maximum nesting depth.
        """
        max_depth = current_depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._get_max_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._get_max_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _check_overengineering(
        self, content: str, file_path: str, report: CodeReviewReport
    ) -> None:
        """Check for overengineering patterns.

        Args:
            content: File content.
            file_path: File path for reporting.
            report: Report to add issues to.
        """
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            for pattern in OVERENGINEERING_PATTERNS:
                if re.search(pattern, line):
                    # Check if it's a small file (might be unnecessary)
                    if len(lines) < 100:
                        report.issues.append(
                            ReviewIssue(
                                file_path=file_path,
                                line_start=i,
                                severity=IssueSeverity.MINOR,
                                category=IssueCategory.YAGNI,
                                title="Possible overengineering",
                                description=(
                                    f"Pattern '{pattern}' found in small file. "
                                    "Consider if this abstraction is needed now "
                                    "or if simpler code would suffice."
                                ),
                                code_snippet=line.strip(),
                            )
                        )

    def _check_style(
        self,
        lines: list[str],
        file_path: str,
        style: StyleGuide,
        report: CodeReviewReport,
    ) -> None:
        """Check style violations.

        Args:
            lines: File lines.
            file_path: File path for reporting.
            style: Style guide to check against.
            report: Report to add issues to.
        """
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > style.max_line_length:
                report.issues.append(
                    ReviewIssue(
                        file_path=file_path,
                        line_start=i,
                        severity=IssueSeverity.NIT,
                        category=IssueCategory.CONSISTENCY,
                        title=f"Line too long ({len(line)} > {style.max_line_length})",
                        description="Consider breaking this line.",
                        code_snippet=line[:80] + "..." if len(line) > 80 else line,
                    )
                )

            # Check trailing whitespace
            if line.endswith(" ") or line.endswith("\t"):
                report.issues.append(
                    ReviewIssue(
                        file_path=file_path,
                        line_start=i,
                        severity=IssueSeverity.NIT,
                        category=IssueCategory.CONSISTENCY,
                        title="Trailing whitespace",
                        description="Remove trailing whitespace.",
                        auto_fixable=True,
                    )
                )

    def _check_antipatterns(
        self,
        tree: ast.AST,
        content: str,
        file_path: str,
        report: CodeReviewReport,
    ) -> None:
        """Check for common anti-patterns.

        Args:
            tree: AST of the file.
            content: File content.
            file_path: File path for reporting.
            report: Report to add issues to.
        """
        # Check for bare except
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=node.lineno,
                            severity=IssueSeverity.MAJOR,
                            category=IssueCategory.ERROR_HANDLING,
                            title="Bare except clause",
                            description=(
                                "Catching all exceptions can hide bugs. "
                                "Catch specific exceptions instead."
                            ),
                            suggestion=(
                                "except Exception:"
                                if node.name
                                else "except Exception as e:"
                            ),
                        )
                    )

            # Check for mutable default arguments
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        report.issues.append(
                            ReviewIssue(
                                file_path=file_path,
                                line_start=node.lineno,
                                severity=IssueSeverity.MAJOR,
                                category=IssueCategory.BEST_PRACTICE,
                                title=f"Mutable default argument in {node.name}",
                                description=(
                                    "Mutable default arguments are shared between "
                                    "calls. Use None and create inside function."
                                ),
                                suggestion="Use `= None` and `if arg is None: arg = []`",
                            )
                        )

        # Check for print statements (should use logging)
        if "print(" in content and "import logging" not in content:
            for i, line in enumerate(content.splitlines(), 1):
                if "print(" in line and not line.strip().startswith("#"):
                    report.issues.append(
                        ReviewIssue(
                            file_path=file_path,
                            line_start=i,
                            severity=IssueSeverity.NIT,
                            category=IssueCategory.BEST_PRACTICE,
                            title="print() instead of logging",
                            description=(
                                "Consider using logging module for production code."
                            ),
                            code_snippet=line.strip(),
                        )
                    )
                    break  # Only report once per file

        # Check for TODO/FIXME comments
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(r"#\s*(TODO|FIXME|XXX|HACK)", line, re.IGNORECASE):
                report.issues.append(
                    ReviewIssue(
                        file_path=file_path,
                        line_start=i,
                        severity=IssueSeverity.NIT,
                        category=IssueCategory.DOCUMENTATION,
                        title="Unresolved TODO/FIXME",
                        description="Consider addressing or creating an issue.",
                        code_snippet=line.strip(),
                    )
                )

    def _llm_review(
        self,
        repo_path: Path,
        files: list[Path],
        report: CodeReviewReport,
    ) -> None:
        """Use LLM for deeper code review.

        Args:
            repo_path: Repository path.
            files: Files to review.
            report: Report to add issues to.
        """
        if self.llm is None:
            return

        # Sample a few files for LLM review
        sample_files = files[:3]

        for file_path in sample_files:
            try:
                content = file_path.read_text()
                if len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"

                rel_path = str(file_path.relative_to(repo_path))

                prompt = f"""Review this Python code like a senior engineer.
Focus on:
1. Code clarity and readability
2. Naming quality
3. Potential bugs or edge cases
4. Unnecessary complexity or overengineering
5. Missing error handling

File: {rel_path}

```python
{content}
```

Provide 1-3 specific, actionable review comments. Format each as:
- [SEVERITY] TITLE: Description

Where SEVERITY is one of: CRITICAL, MAJOR, MINOR, NIT
"""

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior software engineer doing a code review. "
                            "Be constructive but direct. Focus on significant issues."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                response = self.llm.chat(messages)
                if response:
                    self._parse_llm_review(response, rel_path, report)

            except Exception:
                continue

    def _parse_llm_review(
        self, response: str, file_path: str, report: CodeReviewReport
    ) -> None:
        """Parse LLM review response into issues.

        Args:
            response: LLM response text.
            file_path: File being reviewed.
            report: Report to add issues to.
        """
        severity_map = {
            "CRITICAL": IssueSeverity.CRITICAL,
            "MAJOR": IssueSeverity.MAJOR,
            "MINOR": IssueSeverity.MINOR,
            "NIT": IssueSeverity.NIT,
        }

        for line in response.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue

            # Parse format: - [SEVERITY] TITLE: Description
            match = re.match(
                r"-\s*\[(\w+)\]\s*([^:]+):\s*(.+)",
                line,
            )
            if match:
                severity_str = match.group(1).upper()
                title = match.group(2).strip()
                description = match.group(3).strip()

                severity = severity_map.get(severity_str, IssueSeverity.MINOR)

                report.issues.append(
                    ReviewIssue(
                        file_path=file_path,
                        line_start=1,  # LLM doesn't give line numbers
                        severity=severity,
                        category=IssueCategory.BEST_PRACTICE,
                        title=title,
                        description=description,
                    )
                )

    def _apply_auto_fixes(
        self, repo_path: Path, report: CodeReviewReport
    ) -> None:
        """Apply safe auto-fixes.

        Args:
            repo_path: Repository path.
            report: Report with auto-fixable issues.
        """
        for issue in report.issues:
            if not issue.auto_fixable:
                continue

            file_path = repo_path / issue.file_path

            # Currently only support trailing whitespace removal
            if issue.title == "Trailing whitespace":
                try:
                    content = file_path.read_text()
                    lines = content.splitlines()
                    if issue.line_start <= len(lines):
                        old_line = lines[issue.line_start - 1]
                        new_line = old_line.rstrip()

                        report.auto_fixes.append(
                            AutoFix(
                                file_path=issue.file_path,
                                line_start=issue.line_start,
                                line_end=issue.line_start,
                                original_code=old_line,
                                fixed_code=new_line,
                                description="Remove trailing whitespace",
                                fix_type="format",
                                applied=True,
                            )
                        )

                        lines[issue.line_start - 1] = new_line
                        file_path.write_text("\n".join(lines) + "\n")
                        report.auto_fixes_applied += 1

                except Exception:
                    continue

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case.

        Args:
            name: Name to convert.

        Returns:
            snake_case version.
        """
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _generate_summary(self, report: CodeReviewReport) -> str:
        """Generate human-readable summary.

        Args:
            report: Code review report.

        Returns:
            Summary string.
        """
        status_emoji = {
            ShipStatus.SHIP: "✅",
            ShipStatus.SHIP_WITH_NITS: "⚠️",
            ShipStatus.DONT_SHIP: "❌",
        }

        parts = [
            f"{status_emoji.get(report.ship_status, '')} {report.ship_status.value.upper()}: "
            f"{report.ship_status_reason}",
            f"Reviewed {report.files_reviewed} files, {report.lines_reviewed} lines",
            f"Found {report.total_issues} issues "
            f"({report.critical_count} critical, {report.major_count} major, "
            f"{report.minor_count} minor, {report.nit_count} nits)",
        ]

        if report.auto_fixes_applied > 0:
            parts.append(f"Applied {report.auto_fixes_applied} auto-fixes")

        return "; ".join(parts)

    def _get_top_concerns(self, report: CodeReviewReport) -> list[str]:
        """Get top 3 concerns from the review.

        Args:
            report: Code review report.

        Returns:
            List of top concerns.
        """
        # Sort issues by severity
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.MAJOR: 1,
            IssueSeverity.MINOR: 2,
            IssueSeverity.NIT: 3,
        }

        sorted_issues = sorted(
            report.issues,
            key=lambda i: severity_order.get(i.severity, 99),
        )

        concerns = []
        for issue in sorted_issues[:3]:
            concerns.append(f"[{issue.severity.value}] {issue.title}: {issue.description[:100]}")

        return concerns
