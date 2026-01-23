"""Tools module for agent operations.

Provides deterministic tool abstractions for:
- Git operations (branch, commit, diff)
- Filesystem (read, write, search)
- Shell commands (test runners, linters, docker)
- HTTP requests (health checks, API calls)
"""

from .base import BaseTool, ToolResult
from .git_tool import GitTool
from .filesystem_tool import FilesystemTool
from .shell_tool import ShellTool
from .http_tool import HttpTool
from .validator import CodeValidator, ValidationResult

__all__ = [
    "BaseTool",
    "ToolResult",
    "GitTool",
    "FilesystemTool",
    "ShellTool",
    "HttpTool",
    "CodeValidator",
    "ValidationResult",
]
