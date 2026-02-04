# Creating Skills

Skills are single-purpose tools that transform code files. They run on demand via the CLI.

## When to Use Skills

- Add or modify documentation (docstrings, comments)
- Refactor code patterns
- Generate boilerplate (tests, types)
- Format or lint code
- Any file transformation task

## Directory Structure

```
my-skill/
├── manifest.yaml      # Required: metadata
├── skill.py           # Required: implementation
├── requirements.txt   # Optional: dependencies
└── README.md          # Recommended: documentation
```

Place in: `~/.coding-factory/extensions/skills/my-skill/`

## Manifest Schema

```yaml
name: add-docstrings
version: 1.0.0
type: skill
author: Your Name
description: Add Google-style docstrings to Python functions and classes
license: free

# Required for skills
skill_class: AddDocstringsSkill    # Class name in skill.py
input_type: files                   # files | text | code
output_type: modified_files         # modified_files | text | report

# File filtering
file_patterns:                      # Glob patterns for target files
  - "*.py"

# Features
supports_dry_run: true              # Can preview without changes

# Optional metadata
keywords:
  - documentation
  - docstrings
  - python

priority: 50                        # Execution order (lower = first)

# For AI-powered skills
context_tokens: 1000                # Tokens added to LLM context
min_model_tier: small               # any | small | medium | large
```

## Implementation

### Basic Skill

```python
"""My Skill - Description of what it does."""

from pathlib import Path
from typing import Any


class MySkill:
    """One-line description.

    Longer description of what this skill does,
    when to use it, and any important notes.
    """

    name = "my-skill"

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        """Initialize the skill.

        Args:
            llm: LLM backend for AI-powered transformations
            **kwargs: Additional configuration
        """
        self.llm = llm

    def run(
        self,
        files: list[Path],
        dry_run: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Execute the skill on files.

        Args:
            files: List of file paths to process
            dry_run: If True, preview changes without applying
            **kwargs: Additional options

        Returns:
            Results dictionary with modified/skipped files
        """
        results = {
            "modified": [],
            "skipped": [],
            "errors": [],
        }

        for file_path in files:
            try:
                content = file_path.read_text()
                new_content = self._transform(content)

                if new_content == content:
                    results["skipped"].append(str(file_path))
                    continue

                if not dry_run:
                    file_path.write_text(new_content)

                results["modified"].append(str(file_path))

            except Exception as e:
                results["errors"].append({
                    "file": str(file_path),
                    "error": str(e),
                })

        return results

    def _transform(self, content: str) -> str:
        """Transform file content.

        Args:
            content: Original file content

        Returns:
            Transformed content
        """
        # Your transformation logic here
        return content
```

### AI-Powered Skill

Use the LLM for intelligent transformations:

```python
class DocstringSkill:
    """Add docstrings using AI."""

    name = "add-docstrings"

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        self.llm = llm

    def run(self, files: list[Path], dry_run: bool = False, **kwargs) -> dict:
        results = {"modified": [], "skipped": [], "errors": []}

        for file_path in files:
            content = file_path.read_text()

            # Use LLM to generate docstrings
            new_content = self._add_docstrings_with_llm(content)

            if new_content != content and not dry_run:
                file_path.write_text(new_content)
                results["modified"].append(str(file_path))
            else:
                results["skipped"].append(str(file_path))

        return results

    def _add_docstrings_with_llm(self, content: str) -> str:
        """Use LLM to add docstrings."""
        if not self.llm:
            return content

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python documentation expert. "
                    "Add Google-style docstrings to functions and classes "
                    "that are missing them. Return only the modified code."
                ),
            },
            {
                "role": "user",
                "content": f"Add docstrings to this code:\n\n```python\n{content}\n```",
            },
        ]

        response = self.llm.chat(messages)

        # Extract code from response
        code = self._extract_code(response)
        return code if code else content

    def _extract_code(self, response: str) -> str | None:
        """Extract code block from LLM response."""
        import re
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return match.group(1) if match else None
```

## Input/Output Types

### input_type

| Value | Description | `files` Parameter |
|-------|-------------|-------------------|
| `files` | Operate on file paths | `list[Path]` |
| `text` | Operate on text content | `str` |
| `code` | Operate on parsed code | `dict` (AST-like) |

### output_type

| Value | Description | Return Format |
|-------|-------------|---------------|
| `modified_files` | Modified files in place | `{"modified": [...], "skipped": [...]}` |
| `text` | Return transformed text | `{"output": "..."}` |
| `report` | Return analysis report | `{"report": {...}}` |

## File Patterns

Filter which files the skill processes:

```yaml
file_patterns:
  - "*.py"                    # All Python files
  - "src/**/*.ts"             # TypeScript in src/
  - "!**/test_*.py"           # Exclude test files
  - "!**/__pycache__/**"      # Exclude cache
```

## Dry Run Support

Always support `dry_run` for safe previewing:

```python
def run(self, files: list[Path], dry_run: bool = False, **kwargs) -> dict:
    results = {"modified": [], "would_modify": []}

    for file_path in files:
        content = file_path.read_text()
        new_content = self._transform(content)

        if new_content != content:
            if dry_run:
                results["would_modify"].append({
                    "file": str(file_path),
                    "preview": new_content[:500],  # First 500 chars
                })
            else:
                file_path.write_text(new_content)
                results["modified"].append(str(file_path))

    return results
```

## Testing Your Skill

```bash
# Install locally
cp -r my-skill ~/.coding-factory/extensions/skills/

# Verify
acf extensions list

# Test with dry run
acf skill my-skill ./src --dry-run

# Run for real
acf skill my-skill ./src
```

## Examples

See official skills:
- [`add-docstrings`](https://github.com/Tennisee-data/acf/tree/main/official_extensions/add-docstrings) - AI-powered docstring generation

## Next Steps

- [Specification](../specification.md) - Full manifest schema
- [Publishing](../publishing.md) - Submit to marketplace
- [Examples](../examples.md) - More working examples
