"""Filesystem operations tool."""

import fnmatch
import os
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult, ToolStatus


class FilesystemTool(BaseTool):
    """Tool for filesystem operations.

    Provides:
    - File reading and writing
    - Directory listing and creation
    - File search (glob patterns)
    - Content search (grep-like)
    """

    name = "filesystem"
    description = "Filesystem read/write operations"

    def __init__(self, base_path: Path | str | None = None) -> None:
        """Initialize filesystem tool.

        Args:
            base_path: Base path for operations (default: current directory)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def execute(self, operation: str, **kwargs: Any) -> ToolResult:
        """Execute a filesystem operation.

        Args:
            operation: Operation name (read, write, list, search, etc.)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with operation output
        """
        operations = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "delete": self._delete,
            "exists": self._exists,
            "list": self._list,
            "mkdir": self._mkdir,
            "glob": self._glob,
            "search": self._search,
            "tree": self._tree,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            return ToolResult(status=ToolStatus.FAILURE, error=str(e))

    def _resolve(self, path: str) -> Path:
        """Resolve path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def _read(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file content."""
        file_path = self._resolve(path)
        if not file_path.exists():
            return ToolResult(status=ToolStatus.FAILURE, error=f"File not found: {path}")
        if not file_path.is_file():
            return ToolResult(status=ToolStatus.FAILURE, error=f"Not a file: {path}")

        content = file_path.read_text(encoding=encoding)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=content,
            metadata={"path": str(file_path), "size": len(content)},
        )

    def _write(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        """Write content to file."""
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=str(file_path),
            metadata={"size": len(content)},
        )

    def _append(self, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        """Append content to file."""
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding=encoding) as f:
            f.write(content)
        return ToolResult(status=ToolStatus.SUCCESS, output=str(file_path))

    def _delete(self, path: str) -> ToolResult:
        """Delete file or empty directory."""
        file_path = self._resolve(path)
        if not file_path.exists():
            return ToolResult(status=ToolStatus.FAILURE, error=f"Path not found: {path}")

        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            file_path.rmdir()
        else:
            return ToolResult(status=ToolStatus.FAILURE, error=f"Cannot delete: {path}")

        return ToolResult(status=ToolStatus.SUCCESS, output=str(file_path))

    def _exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        file_path = self._resolve(path)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=file_path.exists(),
            metadata={
                "is_file": file_path.is_file() if file_path.exists() else False,
                "is_dir": file_path.is_dir() if file_path.exists() else False,
            },
        )

    def _list(self, path: str = ".", include_hidden: bool = False) -> ToolResult:
        """List directory contents."""
        dir_path = self._resolve(path)
        if not dir_path.exists():
            return ToolResult(status=ToolStatus.FAILURE, error=f"Directory not found: {path}")
        if not dir_path.is_dir():
            return ToolResult(status=ToolStatus.FAILURE, error=f"Not a directory: {path}")

        entries = []
        for entry in dir_path.iterdir():
            if not include_hidden and entry.name.startswith("."):
                continue
            entries.append(
                {
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                }
            )

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=sorted(entries, key=lambda x: (not x["is_dir"], x["name"])),
        )

    def _mkdir(self, path: str, parents: bool = True) -> ToolResult:
        """Create directory."""
        dir_path = self._resolve(path)
        dir_path.mkdir(parents=parents, exist_ok=True)
        return ToolResult(status=ToolStatus.SUCCESS, output=str(dir_path))

    def _glob(self, pattern: str, recursive: bool = True) -> ToolResult:
        """Find files matching glob pattern."""
        if recursive and "**" not in pattern:
            pattern = f"**/{pattern}"

        matches = list(self.base_path.glob(pattern))
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=[str(m.relative_to(self.base_path)) for m in matches],
        )

    def _search(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        max_results: int = 100,
    ) -> ToolResult:
        """Search for pattern in files (grep-like)."""
        search_path = self._resolve(path)
        results = []

        for root, _, files in os.walk(search_path):
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue

                file_path = Path(root) / filename
                try:
                    content = file_path.read_text(errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.lower() in line.lower():
                            results.append(
                                {
                                    "file": str(file_path.relative_to(self.base_path)),
                                    "line": i,
                                    "content": line.strip()[:200],
                                }
                            )
                            if len(results) >= max_results:
                                return ToolResult(
                                    status=ToolStatus.SUCCESS,
                                    output=results,
                                    metadata={"truncated": True},
                                )
                except (OSError, UnicodeDecodeError):
                    continue

        return ToolResult(status=ToolStatus.SUCCESS, output=results)

    def _tree(self, path: str = ".", max_depth: int = 3, max_items: int = 100) -> ToolResult:
        """Generate directory tree structure."""
        root_path = self._resolve(path)
        if not root_path.exists():
            return ToolResult(status=ToolStatus.FAILURE, error=f"Path not found: {path}")

        lines = []
        count = 0

        def walk(current: Path, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > max_depth or count >= max_items:
                return

            entries = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            entries = [e for e in entries if not e.name.startswith(".")]

            for i, entry in enumerate(entries):
                if count >= max_items:
                    lines.append(f"{prefix}... (truncated)")
                    return

                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
                count += 1

                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    walk(entry, prefix + extension, depth + 1)

        lines.append(str(root_path.name) + "/")
        walk(root_path, "", 1)

        return ToolResult(status=ToolStatus.SUCCESS, output="\n".join(lines))
