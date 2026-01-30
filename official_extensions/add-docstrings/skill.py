"""Add Google-style docstrings to Python functions and classes using AI."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from skills.base import BaseSkill, FileChange, SkillInput, SkillOutput

MAX_FILE_SIZE = 50 * 1024  # 50KB


class AddDocstringsSkill(BaseSkill):
    """Add missing docstrings to Python functions and classes.

    Uses an LLM to generate Google-style docstrings for any
    function, method, or class that lacks one.

    Config options:
        style: Docstring style - "google" (default), "numpy", or "sphinx"
        overwrite: If True, replace existing docstrings too (default: False)
    """

    def run(self, input_data: SkillInput) -> SkillOutput:
        if self.llm is None:
            return SkillOutput(
                success=False,
                changes=[],
                summary="LLM backend required but not available",
                errors=["No LLM backend configured. Set an API key or start a local model."],
            )

        style = self.config.get("style", "google")
        overwrite = self.config.get("overwrite", False)

        changes: list[FileChange] = []
        errors: list[str] = []

        for path in input_data.target_paths:
            result = self._process_file(path, style, overwrite)
            if result is None:
                continue
            if isinstance(result, str):
                errors.append(result)
                continue
            changes.append(result)

        if not changes and not errors:
            return SkillOutput(
                success=True,
                changes=[],
                summary="All functions and classes already have docstrings",
            )

        return SkillOutput(
            success=True,
            changes=changes,
            summary=f"Added docstrings to {len(changes)} file(s)",
            errors=errors if errors else None,
        )

    def _process_file(
        self, path: Path, style: str, overwrite: bool
    ) -> FileChange | str | None:
        """Process a single file. Returns FileChange, error string, or None (skip)."""
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return f"{path}: could not read file ({e})"

        if len(content.encode("utf-8")) > MAX_FILE_SIZE:
            return f"{path}: skipped (file exceeds 50KB)"

        # Parse AST to find items missing docstrings
        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as e:
            return f"{path}: syntax error ({e})"

        missing = self._find_missing_docstrings(tree, overwrite)
        if not missing:
            return None

        # Ask LLM to add docstrings
        modified = self._generate_docstrings(content, missing, style)
        if modified is None:
            return f"{path}: LLM did not return a valid response"

        # Validate the result parses as valid Python
        try:
            ast.parse(modified, filename=str(path))
        except SyntaxError:
            return f"{path}: LLM response was not valid Python, skipping"

        if modified == content:
            return None

        return FileChange(
            path=path,
            original_content=content,
            modified_content=modified,
            change_type="modified",
        )

    def _find_missing_docstrings(
        self, tree: ast.Module, overwrite: bool
    ) -> list[dict[str, Any]]:
        """Walk AST and find functions/classes without docstrings."""
        missing: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            has_docstring = (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            )

            if has_docstring and not overwrite:
                continue

            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            missing.append({
                "name": node.name,
                "kind": kind,
                "line": node.lineno,
                "has_existing": has_docstring,
            })

        return missing

    def _generate_docstrings(
        self, source: str, missing: list[dict[str, Any]], style: str
    ) -> str | None:
        """Call LLM to add docstrings to the source code."""
        items_desc = "\n".join(
            f"- {m['kind']} `{m['name']}` at line {m['line']}"
            + (" (replace existing docstring)" if m["has_existing"] else "")
            for m in missing
        )

        style_instruction = {
            "google": "Google-style (Args/Returns/Raises sections)",
            "numpy": "NumPy-style (Parameters/Returns/Raises sections with dashed underlines)",
            "sphinx": "Sphinx-style (:param/:returns/:raises: directives)",
        }.get(style, "Google-style (Args/Returns/Raises sections)")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python documentation expert. "
                    "Your task is to add docstrings to Python code. "
                    "Return ONLY the complete modified Python source code. "
                    "Do NOT wrap the output in markdown code fences. "
                    "Do NOT add any explanation before or after the code. "
                    "Preserve all existing code exactly — only add or replace docstrings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Add {style_instruction} docstrings to the following items:\n"
                    f"{items_desc}\n\n"
                    f"Source code:\n{source}"
                ),
            },
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                temperature=0.3,
            )
        except Exception:
            return None

        return self._strip_markdown_fences(response.strip())

    def _strip_markdown_fences(self, text: str) -> str:
        """Remove markdown code fences if present."""
        # Match ```python ... ``` or ``` ... ```
        match = re.match(
            r"^```(?:python)?\s*\n(.*?)```\s*$",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).rstrip("\n") + "\n"
        return text
