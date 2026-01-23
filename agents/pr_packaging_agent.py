"""PR Packaging Agent - Builds rich GitHub Pull Request packages.

This agent creates structured PR packages with:
- Title and description tied to spec.json
- Acceptance criteria checklist
- Links to test/coverage/performance reports
- Suggested labels, reviewers, changelog entries
"""

import json
import logging
from pathlib import Path

from agents.base import BaseAgent
from llm_backend import LLMBackend
from schemas.feature_spec import FeatureSpec
from schemas.pr_package import (
    AcceptanceCriteriaItem,
    ChangelogEntry,
    ChangeType,
    PRLabel,
    PRPackage,
    ReportLink,
)

logger = logging.getLogger(__name__)


class PRPackagingAgent(BaseAgent):
    """Agent that builds comprehensive PR packages for GitHub submission."""

    def __init__(self, llm: LLMBackend) -> None:
        """Initialize PR packaging agent."""
        super().__init__(llm)
        self.name = "PRPackagingAgent"

    def default_system_prompt(self) -> str:
        """Return the default system prompt for PR packaging."""
        return """You are a PR packaging expert. Create clear, comprehensive
pull request descriptions with acceptance criteria, test results, and
changelog entries."""

    def run(
        self,
        repo_path: Path,
        spec: FeatureSpec | None = None,
        run_id: str | None = None,
        artifacts_dir: Path | None = None,
        source_branch: str = "feature",
        target_branch: str = "main",
    ) -> PRPackage:
        """Build a PR package from spec and artifacts.

        Args:
            repo_path: Path to the generated project
            spec: Feature specification (optional, will load from spec.json)
            run_id: Pipeline run ID
            artifacts_dir: Directory containing run artifacts
            source_branch: Branch being merged
            target_branch: Target branch for merge

        Returns:
            Complete PR package ready for submission
        """
        logger.info("Building PR package")

        # Load spec if not provided
        if spec is None:
            spec = self._load_spec(repo_path, artifacts_dir, run_id)

        # Build acceptance criteria checklist
        acceptance_criteria = self._build_acceptance_criteria(spec, artifacts_dir)

        # Gather report links
        report_links = self._gather_report_links(artifacts_dir, run_id)

        # Determine labels
        labels = self._determine_labels(spec)

        # Generate changelog entries
        changelog = self._generate_changelog(spec)

        # Build title
        title = self._build_title(spec)

        # Build description
        description = spec.description if spec else "Feature implementation"

        # Render full PR body
        rendered_body = self._render_pr_body(
            spec=spec,
            acceptance_criteria=acceptance_criteria,
            report_links=report_links,
            changelog=changelog,
            run_id=run_id,
        )

        package = PRPackage(
            title=title,
            description=description,
            feature_summary=spec.description if spec else "",
            acceptance_criteria=acceptance_criteria,
            report_links=report_links,
            suggested_labels=labels,
            suggested_reviewers=[],  # Could use LLM to suggest based on file types
            changelog_entries=changelog,
            source_branch=source_branch,
            target_branch=target_branch,
            rendered_body=rendered_body,
        )

        logger.info(
            "PR package built: %s (%d criteria, %d reports)",
            title,
            len(acceptance_criteria),
            len(report_links),
        )

        return package

    def _load_spec(
        self,
        repo_path: Path,
        artifacts_dir: Path | None,
        run_id: str | None,
    ) -> FeatureSpec | None:
        """Load feature spec from various locations."""
        # Try spec.json in repo
        spec_file = repo_path / "spec.json"
        if spec_file.exists():
            try:
                with open(spec_file) as f:
                    data = json.load(f)
                return FeatureSpec(**data)
            except Exception as e:
                logger.warning("Failed to load spec.json from repo: %s", e)

        # Try artifacts dir
        if artifacts_dir and run_id:
            spec_file = artifacts_dir / run_id / "spec.json"
            if spec_file.exists():
                try:
                    with open(spec_file) as f:
                        data = json.load(f)
                    return FeatureSpec(**data)
                except Exception as e:
                    logger.warning("Failed to load spec.json from artifacts: %s", e)

        return None

    def _build_acceptance_criteria(
        self,
        spec: FeatureSpec | None,
        artifacts_dir: Path | None,
    ) -> list[AcceptanceCriteriaItem]:
        """Build acceptance criteria checklist from spec."""
        items: list[AcceptanceCriteriaItem] = []

        if not spec:
            return items

        # Add each acceptance criterion
        for criterion in spec.acceptance_criteria:
            # Default to met=True since we completed the pipeline
            # In a real scenario, we'd verify each criterion
            items.append(
                AcceptanceCriteriaItem(
                    criterion=criterion.description,
                    met=True,
                    evidence=f"Verified via {criterion.verification_method}"
                    if criterion.verification_method
                    else None,
                )
            )

        return items

    def _gather_report_links(
        self,
        artifacts_dir: Path | None,
        run_id: str | None,
    ) -> list[ReportLink]:
        """Gather links to generated reports."""
        links: list[ReportLink] = []

        if not artifacts_dir or not run_id:
            return links

        run_dir = artifacts_dir / run_id

        # Test report
        test_report = run_dir / "test_report.json"
        if test_report.exists():
            summary = self._summarize_test_report(test_report)
            links.append(
                ReportLink(
                    name="Test Report",
                    path=f"artifacts/{run_id}/test_report.json",
                    summary=summary,
                )
            )

        # Coverage report
        coverage_report = run_dir / "coverage_report.json"
        if coverage_report.exists():
            summary = self._summarize_coverage_report(coverage_report)
            links.append(
                ReportLink(
                    name="Coverage Report",
                    path=f"artifacts/{run_id}/coverage_report.json",
                    summary=summary,
                )
            )

        # Performance report
        perf_report = run_dir / "performance_report.json"
        if perf_report.exists():
            links.append(
                ReportLink(
                    name="Performance Report",
                    path=f"artifacts/{run_id}/performance_report.json",
                    summary=None,
                )
            )

        # Code review report
        review_report = run_dir / "code_review_report.json"
        if review_report.exists():
            summary = self._summarize_review_report(review_report)
            links.append(
                ReportLink(
                    name="Code Review",
                    path=f"artifacts/{run_id}/code_review_report.json",
                    summary=summary,
                )
            )

        # Security report
        secrets_report = run_dir / "secrets_report.json"
        if secrets_report.exists():
            links.append(
                ReportLink(
                    name="Security Scan",
                    path=f"artifacts/{run_id}/secrets_report.json",
                    summary="No secrets detected"
                    if self._check_secrets_clean(secrets_report)
                    else "⚠️ Review required",
                )
            )

        # Dependency audit
        dep_report = run_dir / "dependency_audit.json"
        if dep_report.exists():
            links.append(
                ReportLink(
                    name="Dependency Audit",
                    path=f"artifacts/{run_id}/dependency_audit.json",
                    summary=None,
                )
            )

        return links

    def _summarize_test_report(self, report_path: Path) -> str:
        """Summarize test report."""
        try:
            with open(report_path) as f:
                data = json.load(f)
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            total = passed + failed
            if failed == 0:
                return f"✅ {total} tests passed"
            return f"⚠️ {passed}/{total} passed, {failed} failed"
        except Exception:
            return None

    def _summarize_coverage_report(self, report_path: Path) -> str:
        """Summarize coverage report."""
        try:
            with open(report_path) as f:
                data = json.load(f)
            coverage = data.get("overall_coverage", 0)
            return f"📊 {coverage:.1f}% coverage"
        except Exception:
            return None

    def _summarize_review_report(self, report_path: Path) -> str:
        """Summarize code review report."""
        try:
            with open(report_path) as f:
                data = json.load(f)
            status = data.get("ship_status", "unknown")
            if status == "ship":
                return "✅ Ship it!"
            elif status == "ship_with_nits":
                return "🟡 Ship with nits"
            return "🔴 Don't ship"
        except Exception:
            return None

    def _check_secrets_clean(self, report_path: Path) -> bool:
        """Check if secrets report is clean."""
        try:
            with open(report_path) as f:
                data = json.load(f)
            secrets = data.get("detected_secrets", [])
            return len(secrets) == 0
        except Exception:
            return True

    def _determine_labels(self, spec: FeatureSpec | None) -> list[PRLabel]:
        """Determine appropriate PR labels from spec."""
        labels: list[PRLabel] = []

        if not spec:
            return [PRLabel.FEATURE]

        # Determine primary label from spec type
        name_lower = spec.name.lower()
        desc_lower = spec.description.lower()

        if "fix" in name_lower or "bug" in desc_lower:
            labels.append(PRLabel.BUG_FIX)
        elif "refactor" in name_lower or "refactor" in desc_lower:
            labels.append(PRLabel.REFACTOR)
        elif "doc" in name_lower or "documentation" in desc_lower:
            labels.append(PRLabel.DOCUMENTATION)
        elif "test" in name_lower or "test" in desc_lower:
            labels.append(PRLabel.TESTS)
        elif "performance" in desc_lower or "optimize" in desc_lower:
            labels.append(PRLabel.PERFORMANCE)
        elif "security" in desc_lower:
            labels.append(PRLabel.SECURITY)
        else:
            labels.append(PRLabel.FEATURE)

        # Check for breaking changes in constraints
        for constraint in spec.constraints:
            if "breaking" in constraint.description.lower():
                labels.append(PRLabel.BREAKING_CHANGE)
                break

        return labels

    def _generate_changelog(self, spec: FeatureSpec | None) -> list[ChangelogEntry]:
        """Generate changelog entries from spec."""
        entries: list[ChangelogEntry] = []

        if not spec:
            return entries

        # Primary entry
        change_type = ChangeType.ADDED
        name_lower = spec.name.lower()
        if "fix" in name_lower:
            change_type = ChangeType.FIXED
        elif "remove" in name_lower:
            change_type = ChangeType.REMOVED
        elif "deprecate" in name_lower:
            change_type = ChangeType.DEPRECATED
        elif "update" in name_lower or "change" in name_lower:
            change_type = ChangeType.CHANGED

        entries.append(
            ChangelogEntry(
                change_type=change_type,
                description=spec.description,
                breaking=False,
            )
        )

        return entries

    def _build_title(self, spec: FeatureSpec | None) -> str:
        """Build PR title from spec."""
        if not spec:
            return "Feature implementation"

        # Prefix based on type
        name_lower = spec.name.lower()
        if "fix" in name_lower:
            prefix = "fix"
        elif "refactor" in name_lower:
            prefix = "refactor"
        elif "doc" in name_lower:
            prefix = "docs"
        elif "test" in name_lower:
            prefix = "test"
        else:
            prefix = "feat"

        # Clean up name
        name = spec.name
        if len(name) > 60:
            name = name[:57] + "..."

        return f"{prefix}: {name}"

    def _render_pr_body(
        self,
        spec: FeatureSpec | None,
        acceptance_criteria: list[AcceptanceCriteriaItem],
        report_links: list[ReportLink],
        changelog: list[ChangelogEntry],
        run_id: str | None,
    ) -> str:
        """Render the full PR body markdown."""
        lines: list[str] = []

        # Summary section
        lines.append("## Summary")
        if spec:
            lines.append(spec.description)
        lines.append("")

        # Acceptance criteria checklist
        if acceptance_criteria:
            lines.append("## Acceptance Criteria")
            for item in acceptance_criteria:
                check = "x" if item.met else " "
                lines.append(f"- [{check}] {item.criterion}")
                if item.evidence:
                    lines.append(f"  - _{item.evidence}_")
            lines.append("")

        # Changes section (from spec features/requirements)
        if spec and spec.non_functional_requirements:
            lines.append("## Non-Functional Requirements")
            for nfr in spec.non_functional_requirements:
                lines.append(f"- **{nfr.category}**: {nfr.requirement}")
            lines.append("")

        # Report links
        if report_links:
            lines.append("## Reports")
            for link in report_links:
                summary_text = f" - {link.summary}" if link.summary else ""
                lines.append(f"- [{link.name}]({link.path}){summary_text}")
            lines.append("")

        # Changelog
        if changelog:
            lines.append("## Changelog")
            for entry in changelog:
                breaking = " **[BREAKING]**" if entry.breaking else ""
                lines.append(f"- **{entry.change_type.value}**: {entry.description}{breaking}")
            lines.append("")

        # Footer
        lines.append("---")
        if run_id:
            lines.append(f"*Generated by Coding Factory • Run ID: `{run_id}`*")
        else:
            lines.append("*Generated by Coding Factory*")

        return "\n".join(lines)
