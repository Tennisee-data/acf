"""Skills module for ACF.

Skills are standalone, composable code transformations that can be
run independently of the pipeline. Unlike agents (which hook into
pipeline stages), skills operate directly on files and directories.

Extension type: skill
"""

from skills.base import BaseSkill, FileChange, SkillInput, SkillOutput
from skills.runner import SkillRunner
from skills.chain_runner import ChainRunner

__all__ = [
    "BaseSkill",
    "FileChange",
    "SkillInput",
    "SkillOutput",
    "SkillRunner",
    "ChainRunner",
]
