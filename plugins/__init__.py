"""Plugin system for extending the pipeline with custom agents.

Allows users to create custom agents that hook into the pipeline
at specific points (before/after any stage).

Plugin Structure:
    ~/.coding-factory/plugins/
    └── my_plugin/
        ├── plugin.yaml        # Manifest with metadata and hook config
        ├── agent.py           # Agent class extending BaseAgent
        └── requirements.txt   # Optional dependencies

Example plugin.yaml:
    name: compliance-checker
    version: 1.0.0
    description: Check regulatory compliance

    agent:
      class: agent.ComplianceAgent

    hook:
      point: after:code_review

    inputs: [code_review_report]
    outputs: [compliance_report]
"""

from .loader import PluginLoader
from .manifest import PluginManifest
from .registry import PluginRegistry

__all__ = ["PluginLoader", "PluginManifest", "PluginRegistry"]
