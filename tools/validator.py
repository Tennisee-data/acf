"""Code validation tools for syntax, linting, and execution checks."""

import ast
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_output: str = ""


class CodeValidator:
    """Validate code for syntax errors, linting issues, and basic execution.

    Supports multiple validation strategies:
    - Python: ast.parse, ruff, mypy
    - JavaScript/TypeScript: eslint (if available)
    - Generic: basic syntax checks
    """

    def __init__(self, repo_path: Path | None = None):
        """Initialize validator.

        Args:
            repo_path: Base path for relative file operations
        """
        self.repo_path = repo_path or Path.cwd()

    def validate_python(self, code: str, file_path: str = "<string>") -> ValidationResult:
        """Validate Python code.

        Args:
            code: Python source code
            file_path: File path for error messages

        Returns:
            ValidationResult with any errors found
        """
        errors = []
        warnings = []
        error_output_parts = []

        # 1. Syntax check with ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            error_msg = f"{file_path}:{e.lineno}:{e.offset}: SyntaxError: {e.msg}"
            errors.append(error_msg)
            error_output_parts.append(error_msg)

        # 2. Ruff linting (if available)
        ruff_result = self._run_ruff(code, file_path)
        if ruff_result:
            errors.extend(ruff_result.get("errors", []))
            warnings.extend(ruff_result.get("warnings", []))
            if ruff_result.get("output"):
                error_output_parts.append(ruff_result["output"])

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            error_output="\n".join(error_output_parts),
        )

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            ValidationResult
        """
        if not file_path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {file_path}"],
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Could not read file: {e}"],
            )

        suffix = file_path.suffix.lower()

        if suffix == ".py":
            return self.validate_python(content, str(file_path))
        elif suffix in (".js", ".ts", ".jsx", ".tsx"):
            return self.validate_javascript(content, str(file_path))
        else:
            # Generic validation - just check it's valid UTF-8
            return ValidationResult(valid=True)

    def validate_javascript(self, code: str, file_path: str = "<string>") -> ValidationResult:
        """Validate JavaScript/TypeScript code.

        Args:
            code: JS/TS source code
            file_path: File path for error messages

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Basic syntax checks
        # Check for common issues
        if code.count("{") != code.count("}"):
            errors.append(f"{file_path}: Mismatched braces")
        if code.count("(") != code.count(")"):
            errors.append(f"{file_path}: Mismatched parentheses")
        if code.count("[") != code.count("]"):
            errors.append(f"{file_path}: Mismatched brackets")

        # Try eslint if available
        eslint_result = self._run_eslint(code, file_path)
        if eslint_result:
            errors.extend(eslint_result.get("errors", []))
            warnings.extend(eslint_result.get("warnings", []))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_files(self, file_contents: dict[str, str]) -> ValidationResult:
        """Validate multiple files.

        Args:
            file_contents: Dict mapping file paths to content

        Returns:
            Combined ValidationResult
        """
        all_errors = []
        all_warnings = []
        all_output = []

        for path, content in file_contents.items():
            suffix = Path(path).suffix.lower()

            if suffix == ".py":
                result = self.validate_python(content, path)
            elif suffix in (".js", ".ts", ".jsx", ".tsx"):
                result = self.validate_javascript(content, path)
            else:
                continue

            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if result.error_output:
                all_output.append(result.error_output)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            error_output="\n".join(all_output),
        )

    def _run_ruff(self, code: str, file_path: str) -> dict | None:
        """Run ruff linter on code.

        Returns:
            Dict with errors, warnings, output or None if ruff not available
        """
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            # Run ruff check
            result = subprocess.run(
                ["ruff", "check", temp_path, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return None

            # Parse output
            errors = []
            warnings = []
            output = result.stdout + result.stderr

            for line in output.splitlines():
                if temp_path in line:
                    # Replace temp path with original
                    line = line.replace(temp_path, file_path)

                if ": E" in line or ": F" in line:
                    errors.append(line)
                elif ": W" in line:
                    warnings.append(line)

            return {
                "errors": errors,
                "warnings": warnings,
                "output": output.replace(temp_path, file_path),
            }

        except FileNotFoundError:
            # ruff not installed
            return None
        except subprocess.TimeoutExpired:
            return {"errors": ["Ruff timed out"], "warnings": [], "output": ""}
        except Exception:
            return None

    def _run_eslint(self, code: str, file_path: str) -> dict | None:
        """Run eslint on JavaScript code.

        Returns:
            Dict with errors, warnings or None if eslint not available
        """
        try:
            suffix = Path(file_path).suffix or ".js"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["eslint", temp_path, "--format", "compact"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            Path(temp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return None

            errors = []
            for line in result.stdout.splitlines():
                if "Error" in line:
                    errors.append(line.replace(temp_path, file_path))

            return {"errors": errors, "warnings": []}

        except FileNotFoundError:
            return None
        except Exception:
            return None

    def try_execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 10,
    ) -> ValidationResult:
        """Try to execute code and capture any runtime errors.

        Args:
            code: Source code to execute
            language: Programming language
            timeout: Execution timeout in seconds

        Returns:
            ValidationResult with execution errors if any
        """
        if language != "python":
            return ValidationResult(
                valid=True,
                warnings=["Execution check only supported for Python"],
            )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.repo_path),
            )

            Path(temp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return ValidationResult(valid=True)

            return ValidationResult(
                valid=False,
                errors=[f"Runtime error: {result.stderr[:500]}"],
                error_output=result.stderr,
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                valid=False,
                errors=["Execution timed out"],
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Execution failed: {str(e)}"],
            )
