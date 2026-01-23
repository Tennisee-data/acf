"""Local storage module for ACF Local Edition.

This module provides local git-based versioning for pipeline iterations,
replacing cloud-based GitHub storage for fully offline operation.
"""

from local_storage.git_versioner import LocalGitVersioner

__all__ = ["LocalGitVersioner"]
