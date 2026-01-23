"""Verification Agent for black-box testing and final summary."""

import re
import time
from pathlib import Path

from llm_backend import LLMBackend
from schemas.feature_spec import AcceptanceCriteria, FeatureSpec
from schemas.implementation import ChangeSet, ImplementationNotes
from schemas.requirements import (
    Requirement,
    RequirementsTracker,
    RequirementStatus,
    RequirementType,
)
from schemas.test_report import TestReport
from schemas.verification import (
    APICheck,
    CheckStatus,
    CriterionCheck,
    HealthCheck,
    VerificationReport,
)
from tools import HttpTool
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent


SYSTEM_PROMPT = """You are a verification and summary expert. Analyze the implementation and test results to provide a final assessment.
Output ONLY a JSON object. No markdown, no explanation.

{"technical_summary":"brief summary of technical changes","behavioral_summary":"observed behavior summary","residual_risks":["risk 1"],"open_questions":["question 1"],"recommendation":"approve","recommendation_rationale":"why this recommendation","pr_description":"PR description text","release_notes":"release notes entry","criteria_assessments":[{"criterion_id":"AC-1","status":"pass","evidence":"what proves it works","notes":"additional notes"}]}

IMPORTANT: recommendation must be one of: approve, reject, needs_review
Output starts with { and ends with }. No other text allowed."""


class VerifyAgent(BaseAgent):
    """Agent for black-box verification and final summary.

    Performs:
    - Health checks on endpoints
    - Acceptance criteria verification
    - API endpoint testing
    - Final summary and recommendation

    Generates:
    - PR description
    - Release notes
    - Deployment recommendation
    """

    def __init__(
        self,
        llm: LLMBackend,
        base_url: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize VerifyAgent.

        Args:
            llm: LLM backend for analysis
            base_url: Base URL for HTTP checks (e.g., http://localhost:8080)
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)
        self.base_url = base_url
        self.http = HttpTool(base_url=base_url) if base_url else None

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run verification and generate summary.

        Args:
            input_data: Must contain 'feature_id' and optionally:
                       'feature_spec', 'change_set', 'implementation_notes',
                       'test_report', 'container_id', 'base_url'

        Returns:
            AgentOutput with VerificationReport data
        """
        context = input_data.context
        feature_id = context.get("feature_id", "FEAT-000")

        # Extract inputs
        feature_spec = self._extract_feature_spec(context)
        change_set = self._extract_change_set(context)
        impl_notes = self._extract_impl_notes(context)
        test_report = self._extract_test_report(context)
        requirements_tracker = self._extract_requirements_tracker(context)

        # Container info
        container_id = context.get("container_id")
        image_tag = context.get("image_tag")

        # Project directory for file verification
        project_dir = context.get("project_dir")

        # Update base URL if provided
        if context.get("base_url"):
            self.base_url = context["base_url"]
            self.http = HttpTool(base_url=self.base_url)

        # Run health checks if we have an endpoint
        health_checks = []
        if self.http and self.base_url:
            health_checks = self._run_health_checks()

        # Run API checks if we have endpoints defined
        api_checks = []
        if self.http and self.base_url:
            api_checks = self._run_api_checks(feature_spec)

        # Verify requirements from tracker
        requirements_verification = None
        if requirements_tracker:
            requirements_verification = self._verify_requirements(
                requirements_tracker,
                project_dir,
                feature_spec,
            )

        # Use LLM to analyze and generate summaries
        llm_analysis = self._analyze_with_llm(
            feature_spec=feature_spec,
            change_set=change_set,
            impl_notes=impl_notes,
            test_report=test_report,
            health_checks=health_checks,
            api_checks=api_checks,
        )

        # Build criteria checks from LLM analysis
        criteria_checks = self._build_criteria_checks(
            feature_spec=feature_spec,
            llm_assessments=llm_analysis.get("criteria_assessments", []),
        )

        # Calculate criteria summary
        criteria_summary = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for check in criteria_checks:
            criteria_summary[check.status.value] = criteria_summary.get(check.status.value, 0) + 1

        all_criteria_met = criteria_summary["fail"] == 0 and criteria_summary["pass"] > 0

        # Determine recommendation
        recommendation = llm_analysis.get("recommendation", "needs_review")
        if recommendation not in ("approve", "reject", "needs_review"):
            recommendation = "needs_review"

        # Override recommendation based on hard checks
        if test_report:
            test_results = test_report if isinstance(test_report, dict) else test_report.model_dump()
            if test_results.get("test_results", {}).get("failed", 0) > 0:
                recommendation = "reject"
                llm_analysis["recommendation_rationale"] = "Tests are failing. " + llm_analysis.get("recommendation_rationale", "")

        # Check requirements verification
        if requirements_verification:
            unmet = requirements_verification.get("unmet_requirements", [])
            if unmet:
                recommendation = "reject"
                unmet_list = ", ".join([r["id"] for r in unmet[:5]])
                llm_analysis["recommendation_rationale"] = (
                    f"Unmet requirements: {unmet_list}. " +
                    llm_analysis.get("recommendation_rationale", "")
                )

        # Build the report
        report = VerificationReport(
            feature_id=feature_id,
            container_id=container_id,
            image_tag=image_tag,
            health_checks=health_checks,
            criteria_checks=criteria_checks,
            api_checks=api_checks,
            ui_checks=[],
            all_criteria_met=all_criteria_met,
            criteria_summary=criteria_summary,
            technical_summary=llm_analysis.get("technical_summary", "No technical summary available."),
            behavioral_summary=llm_analysis.get("behavioral_summary", "No behavioral summary available."),
            residual_risks=llm_analysis.get("residual_risks", []),
            open_questions=llm_analysis.get("open_questions", []),
            recommendation=recommendation,
            recommendation_rationale=llm_analysis.get("recommendation_rationale", ""),
            pr_description=llm_analysis.get("pr_description"),
            release_notes=llm_analysis.get("release_notes"),
        )

        # Include requirements verification in output
        output_data = report.model_dump()
        if requirements_verification:
            output_data["requirements_verification"] = requirements_verification

        return AgentOutput(
            success=True,
            data=output_data,
            artifacts=["verification_report.json", "verification_report.md"],
        )

    def _extract_requirements_tracker(self, context: dict) -> RequirementsTracker | None:
        """Extract RequirementsTracker from context."""
        data = context.get("requirements_tracker", {})
        if isinstance(data, RequirementsTracker):
            return data
        if isinstance(data, dict) and data:
            try:
                return RequirementsTracker(**data)
            except Exception:
                pass
        return None

    def _verify_requirements(
        self,
        tracker: RequirementsTracker,
        project_dir: str | None,
        feature_spec: FeatureSpec | None,
    ) -> dict:
        """Verify that requirements have been addressed.

        Uses multiple verification methods:
        1. File content grep for specific patterns
        2. Checking if files exist
        3. LLM analysis for semantic verification

        Args:
            tracker: The requirements tracker
            project_dir: Path to the generated project
            feature_spec: Feature specification for context

        Returns:
            Dict with verification results
        """
        results = {
            "total": len(tracker.requirements),
            "verified": 0,
            "failed": 0,
            "skipped": 0,
            "unmet_requirements": [],
            "verification_details": [],
        }

        project_path = Path(project_dir) if project_dir else None

        for req in tracker.requirements:
            verification = self._verify_single_requirement(req, project_path)
            results["verification_details"].append(verification)

            if verification["verified"]:
                results["verified"] += 1
                req.mark_verified(
                    agent_name="VerifyAgent",
                    method=verification["method"],
                    result=verification["evidence"],
                    success=True,
                )
            elif verification["skipped"]:
                results["skipped"] += 1
            else:
                results["failed"] += 1
                req.mark_failed(verification["reason"])
                results["unmet_requirements"].append({
                    "id": req.id,
                    "description": req.description,
                    "type": req.type.value,
                    "reason": verification["reason"],
                })

        # Update tracker with final status
        results["tracker_summary"] = tracker.completion_summary()

        return results

    def _verify_single_requirement(
        self,
        req: Requirement,
        project_path: Path | None,
    ) -> dict:
        """Verify a single requirement.

        Args:
            req: The requirement to verify
            project_path: Path to the project

        Returns:
            Dict with verification status and evidence
        """
        result = {
            "id": req.id,
            "description": req.description,
            "verified": False,
            "skipped": False,
            "method": None,
            "evidence": None,
            "reason": None,
        }

        # If already verified, skip
        if req.status == RequirementStatus.VERIFIED:
            result["verified"] = True
            result["method"] = "previous_verification"
            result["evidence"] = req.verification_result
            return result

        # If no project path, can't verify file-based requirements
        if not project_path or not project_path.exists():
            result["skipped"] = True
            result["reason"] = "No project directory available for verification"
            return result

        # Different verification strategies based on requirement type
        if req.type == RequirementType.DOCUMENTATION:
            return self._verify_documentation_requirement(req, project_path, result)
        elif req.type == RequirementType.FIX:
            return self._verify_fix_requirement(req, project_path, result)
        elif req.type == RequirementType.REMOVAL:
            return self._verify_removal_requirement(req, project_path, result)
        else:
            # For other types, check if it was marked as addressed
            if req.status == RequirementStatus.ADDRESSED and req.evidence:
                result["verified"] = True
                result["method"] = "agent_attestation"
                result["evidence"] = req.evidence
            else:
                result["reason"] = f"Requirement not addressed by any agent"
            return result

    def _verify_documentation_requirement(
        self,
        req: Requirement,
        project_path: Path,
        result: dict,
    ) -> dict:
        """Verify a documentation requirement."""
        # Check if README exists
        readme_files = list(project_path.glob("**/README.md")) + list(project_path.glob("**/README.txt"))

        if readme_files:
            result["verified"] = True
            result["method"] = "file_exists"
            result["evidence"] = f"Found README: {readme_files[0].name}"
        else:
            result["reason"] = "No README file found in project"

        return result

    def _verify_fix_requirement(
        self,
        req: Requirement,
        project_path: Path,
        result: dict,
    ) -> dict:
        """Verify a fix requirement by checking file contents."""
        desc_lower = req.description.lower()

        # Extract what should NOT be present (negative patterns)
        negative_patterns = []
        if "not" in desc_lower or "remove" in desc_lower:
            # Extract patterns like "main.py", "python main.py"
            matches = re.findall(r'(?:not|remove|no)\s+(?:reference\s+to\s+)?["\']?([^\s"\']+)["\']?', desc_lower)
            negative_patterns.extend(matches)

        # Check for specific patterns in the description
        if "main.py" in desc_lower:
            negative_patterns.append("python main.py")
            negative_patterns.append("main.py")

        if negative_patterns:
            # Search project files for these patterns
            for pattern in negative_patterns:
                found = self._grep_project(project_path, pattern)
                if found:
                    result["reason"] = f"Pattern '{pattern}' still found in: {', '.join(found[:3])}"
                    return result

            result["verified"] = True
            result["method"] = "grep_negative"
            result["evidence"] = f"Patterns {negative_patterns} not found in project files"
            return result

        # If addressed by an agent, trust it
        if req.status == RequirementStatus.ADDRESSED and req.evidence:
            result["verified"] = True
            result["method"] = "agent_attestation"
            result["evidence"] = req.evidence
        else:
            result["reason"] = "Could not verify fix - no evidence provided"

        return result

    def _verify_removal_requirement(
        self,
        req: Requirement,
        project_path: Path,
        result: dict,
    ) -> dict:
        """Verify a removal requirement."""
        desc_lower = req.description.lower()

        # Extract what should be removed
        patterns_to_check = []
        matches = re.findall(r'remove\s+(?:the\s+)?["\']?([^\s"\',.]+)["\']?', desc_lower)
        patterns_to_check.extend(matches)

        if patterns_to_check:
            for pattern in patterns_to_check:
                found = self._grep_project(project_path, pattern)
                if found:
                    result["reason"] = f"'{pattern}' still present in: {', '.join(found[:3])}"
                    return result

            result["verified"] = True
            result["method"] = "grep_negative"
            result["evidence"] = f"Removed items {patterns_to_check} not found"
            return result

        # Fallback to agent attestation
        if req.status == RequirementStatus.ADDRESSED and req.evidence:
            result["verified"] = True
            result["method"] = "agent_attestation"
            result["evidence"] = req.evidence
        else:
            result["reason"] = "Could not verify removal"

        return result

    def _grep_project(self, project_path: Path, pattern: str) -> list[str]:
        """Search project files for a pattern.

        Args:
            project_path: Path to search
            pattern: Text pattern to find

        Returns:
            List of file paths containing the pattern
        """
        found_in = []
        search_extensions = [".py", ".md", ".txt", ".html", ".js", ".ts", ".json", ".yaml", ".yml", ".sh"]

        for ext in search_extensions:
            for file_path in project_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(errors="ignore")
                    if pattern.lower() in content.lower():
                        found_in.append(str(file_path.relative_to(project_path)))
                except Exception:
                    continue

        return found_in

    def _extract_feature_spec(self, context: dict) -> FeatureSpec | None:
        """Extract FeatureSpec from context."""
        data = context.get("feature_spec", {})
        if isinstance(data, FeatureSpec):
            return data
        if isinstance(data, dict) and data:
            try:
                return FeatureSpec(**data)
            except Exception:
                pass
        return None

    def _extract_change_set(self, context: dict) -> ChangeSet | None:
        """Extract ChangeSet from context."""
        data = context.get("change_set", {})
        if isinstance(data, ChangeSet):
            return data
        if isinstance(data, dict) and data:
            try:
                return ChangeSet(**data)
            except Exception:
                pass
        return None

    def _extract_impl_notes(self, context: dict) -> ImplementationNotes | None:
        """Extract ImplementationNotes from context."""
        data = context.get("implementation_notes", {})
        if isinstance(data, ImplementationNotes):
            return data
        if isinstance(data, dict) and data:
            try:
                return ImplementationNotes(**data)
            except Exception:
                pass
        return None

    def _extract_test_report(self, context: dict) -> TestReport | dict | None:
        """Extract TestReport from context."""
        data = context.get("test_report", {})
        if isinstance(data, TestReport):
            return data
        if isinstance(data, dict) and data:
            return data  # Return as dict, parsing is optional
        return None

    def _run_health_checks(self) -> list[HealthCheck]:
        """Run health checks on common endpoints."""
        checks = []
        endpoints = [
            ("/health", 200),
            ("/", 200),
            ("/api/health", 200),
            ("/healthz", 200),
        ]

        for endpoint, expected_status in endpoints:
            start_time = time.time()
            result = self.http.execute("get", path=endpoint)
            response_time = (time.time() - start_time) * 1000

            if result.success:
                actual_status = result.output.get("status_code", 0) if result.output else 0
                status = CheckStatus.PASS if actual_status == expected_status else CheckStatus.WARN
                checks.append(
                    HealthCheck(
                        endpoint=endpoint,
                        status=status,
                        response_time_ms=response_time,
                        expected_status=expected_status,
                        actual_status=actual_status,
                    )
                )
                # Found a working health endpoint, stop checking
                if status == CheckStatus.PASS:
                    break
            else:
                # Only add if it's a real failure (not just 404)
                if result.output and result.output.get("status_code") == 404:
                    continue
                checks.append(
                    HealthCheck(
                        endpoint=endpoint,
                        status=CheckStatus.FAIL,
                        response_time_ms=response_time,
                        expected_status=expected_status,
                        error=result.error,
                    )
                )

        return checks

    def _run_api_checks(self, feature_spec: FeatureSpec | None) -> list[APICheck]:
        """Run API checks based on feature specification."""
        checks = []

        # Basic API check - just verify the root endpoint responds
        result = self.http.execute("get", path="/")
        if result.success:
            checks.append(
                APICheck(
                    method="GET",
                    endpoint="/",
                    description="Root endpoint accessibility",
                    status=CheckStatus.PASS,
                    actual_response=result.output.get("body") if result.output else None,
                )
            )
        elif result.output and result.output.get("status_code"):
            # Got a response, just not success
            checks.append(
                APICheck(
                    method="GET",
                    endpoint="/",
                    description="Root endpoint accessibility",
                    status=CheckStatus.WARN,
                    actual_response=result.output,
                    error=result.error,
                )
            )

        return checks

    def _build_criteria_checks(
        self,
        feature_spec: FeatureSpec | None,
        llm_assessments: list[dict],
    ) -> list[CriterionCheck]:
        """Build criteria checks from feature spec and LLM assessments."""
        checks = []

        if not feature_spec or not feature_spec.acceptance_criteria:
            # No criteria defined - create a generic check
            if llm_assessments:
                for assessment in llm_assessments:
                    status_str = assessment.get("status", "skip")
                    status = self._parse_status(status_str)
                    checks.append(
                        CriterionCheck(
                            criterion_id=assessment.get("criterion_id", "AC-unknown"),
                            description=assessment.get("description", "Unknown criterion"),
                            status=status,
                            evidence=assessment.get("evidence"),
                            notes=assessment.get("notes"),
                        )
                    )
            return checks

        # Match LLM assessments to acceptance criteria
        assessment_map = {a.get("criterion_id"): a for a in llm_assessments}

        for criterion in feature_spec.acceptance_criteria:
            assessment = assessment_map.get(criterion.id, {})
            status_str = assessment.get("status", "skip")
            status = self._parse_status(status_str)

            checks.append(
                CriterionCheck(
                    criterion_id=criterion.id,
                    description=criterion.description,
                    status=status,
                    evidence=assessment.get("evidence"),
                    notes=assessment.get("notes"),
                )
            )

        return checks

    def _parse_status(self, status_str: str) -> CheckStatus:
        """Parse status string to CheckStatus enum."""
        status_map = {
            "pass": CheckStatus.PASS,
            "passed": CheckStatus.PASS,
            "fail": CheckStatus.FAIL,
            "failed": CheckStatus.FAIL,
            "skip": CheckStatus.SKIP,
            "skipped": CheckStatus.SKIP,
            "warn": CheckStatus.WARN,
            "warning": CheckStatus.WARN,
        }
        return status_map.get(status_str.lower(), CheckStatus.SKIP)

    def _analyze_with_llm(
        self,
        feature_spec: FeatureSpec | None,
        change_set: ChangeSet | None,
        impl_notes: ImplementationNotes | None,
        test_report: TestReport | dict | None,
        health_checks: list[HealthCheck],
        api_checks: list[APICheck],
    ) -> dict:
        """Use LLM to analyze implementation and generate summaries."""
        parts = ["## Implementation Analysis\n"]

        # Feature info
        if feature_spec:
            parts.append("### Feature")
            parts.append(f"**Title:** {feature_spec.title}")
            parts.append(f"**Description:** {feature_spec.original_description}")

            if feature_spec.acceptance_criteria:
                parts.append("\n### Acceptance Criteria")
                for ac in feature_spec.acceptance_criteria:
                    parts.append(f"- [{ac.id}] {ac.description}")

        # Changes made
        if change_set:
            parts.append("\n### Changes Made")
            parts.append(f"- Files changed: {change_set.files_changed}")
            parts.append(f"- Insertions: +{change_set.insertions}")
            parts.append(f"- Deletions: -{change_set.deletions}")

        if impl_notes:
            if isinstance(impl_notes, ImplementationNotes):
                parts.append(f"\n**Summary:** {impl_notes.summary}")
                if impl_notes.new_functions:
                    parts.append(f"- New functions: {len(impl_notes.new_functions)}")
                if impl_notes.new_classes:
                    parts.append(f"- New classes: {len(impl_notes.new_classes)}")
                if impl_notes.tech_debt_items:
                    parts.append(f"- Tech debt items: {len(impl_notes.tech_debt_items)}")

        # Test results
        if test_report:
            test_data = test_report if isinstance(test_report, dict) else test_report.model_dump()
            test_results = test_data.get("test_results", {})
            parts.append("\n### Test Results")
            parts.append(f"- Total: {test_results.get('total', 0)}")
            parts.append(f"- Passed: {test_results.get('passed', 0)}")
            parts.append(f"- Failed: {test_results.get('failed', 0)}")
            parts.append(f"- Quality Score: {test_data.get('quality_score', 'N/A')}")

            blocking = test_data.get("blocking_issues", [])
            if blocking:
                parts.append("\n**Blocking Issues:**")
                for issue in blocking:
                    parts.append(f"- {issue}")

        # Health check results
        if health_checks:
            parts.append("\n### Health Checks")
            for check in health_checks:
                parts.append(f"- {check.endpoint}: {check.status.value}")

        # API check results
        if api_checks:
            parts.append("\n### API Checks")
            for check in api_checks:
                parts.append(f"- {check.method} {check.endpoint}: {check.status.value}")

        parts.append("\n## Task")
        parts.append("Analyze this implementation and provide:")
        parts.append("1. technical_summary: Brief technical changes summary")
        parts.append("2. behavioral_summary: How the system now behaves")
        parts.append("3. residual_risks: Any remaining risks")
        parts.append("4. open_questions: Questions for human review")
        parts.append("5. recommendation: approve, reject, or needs_review")
        parts.append("6. recommendation_rationale: Why this recommendation")
        parts.append("7. pr_description: Full PR description (markdown)")
        parts.append("8. release_notes: Brief release notes entry")

        if feature_spec and feature_spec.acceptance_criteria:
            parts.append("9. criteria_assessments: Array of {criterion_id, status (pass/fail/skip), evidence, notes}")

        user_message = "\n".join(parts)

        try:
            response = self._chat(user_message, temperature=0.2)
            result = parse_llm_json(response, default={})
            return result
        except Exception:
            return {
                "technical_summary": "Analysis unavailable",
                "behavioral_summary": "Analysis unavailable",
                "recommendation": "needs_review",
                "recommendation_rationale": "LLM analysis failed",
            }

    def generate_markdown_report(self, report: VerificationReport) -> str:
        """Generate human-readable markdown from verification report."""
        lines = [
            f"# Verification Report: {report.feature_id}",
            "",
        ]

        # Recommendation badge
        rec_emoji = {
            "approve": ":white_check_mark:",
            "reject": ":x:",
            "needs_review": ":warning:",
        }
        emoji = rec_emoji.get(report.recommendation, ":question:")
        lines.append(f"**Recommendation:** {emoji} **{report.recommendation.upper()}**")
        lines.append("")
        lines.append(f"> {report.recommendation_rationale}")
        lines.append("")

        # Summary section
        lines.extend([
            "## Summary",
            "",
            "### Technical Summary",
            "",
            report.technical_summary,
            "",
            "### Behavioral Summary",
            "",
            report.behavioral_summary,
            "",
        ])

        # Criteria checks
        if report.criteria_checks:
            lines.extend([
                "## Acceptance Criteria",
                "",
                "| Criterion | Status | Evidence |",
                "|-----------|--------|----------|",
            ])
            for check in report.criteria_checks:
                status_emoji = {
                    CheckStatus.PASS: ":white_check_mark:",
                    CheckStatus.FAIL: ":x:",
                    CheckStatus.SKIP: ":fast_forward:",
                    CheckStatus.WARN: ":warning:",
                }
                emoji = status_emoji.get(check.status, "")
                evidence = (check.evidence or "")[:50]
                lines.append(f"| {check.criterion_id} | {emoji} {check.status.value} | {evidence} |")
            lines.append("")

            # Criteria summary
            lines.append(f"**Summary:** {report.criteria_summary}")
            lines.append(f"**All Criteria Met:** {'Yes' if report.all_criteria_met else 'No'}")
            lines.append("")

        # Health checks
        if report.health_checks:
            lines.extend([
                "## Health Checks",
                "",
            ])
            for check in report.health_checks:
                status_str = f"{check.status.value}"
                if check.response_time_ms:
                    status_str += f" ({check.response_time_ms:.0f}ms)"
                lines.append(f"- `{check.endpoint}`: {status_str}")
            lines.append("")

        # API checks
        if report.api_checks:
            lines.extend([
                "## API Checks",
                "",
            ])
            for check in report.api_checks:
                lines.append(f"- **{check.method} {check.endpoint}**: {check.status.value}")
                lines.append(f"  - {check.description}")
                if check.error:
                    lines.append(f"  - Error: {check.error}")
            lines.append("")

        # Risks and questions
        if report.residual_risks:
            lines.extend([
                "## Residual Risks",
                "",
            ])
            for risk in report.residual_risks:
                lines.append(f"- :warning: {risk}")
            lines.append("")

        if report.open_questions:
            lines.extend([
                "## Open Questions",
                "",
            ])
            for question in report.open_questions:
                lines.append(f"- :question: {question}")
            lines.append("")

        # PR Description
        if report.pr_description:
            lines.extend([
                "## Generated PR Description",
                "",
                "```markdown",
                report.pr_description,
                "```",
                "",
            ])

        # Release Notes
        if report.release_notes:
            lines.extend([
                "## Release Notes",
                "",
                report.release_notes,
                "",
            ])

        # Container info
        if report.container_id or report.image_tag:
            lines.extend([
                "## Container Info",
                "",
            ])
            if report.container_id:
                lines.append(f"- Container ID: `{report.container_id}`")
            if report.image_tag:
                lines.append(f"- Image Tag: `{report.image_tag}`")
            lines.append("")

        return "\n".join(lines)
