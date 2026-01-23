"""Coverage Agent for enforcing test coverage thresholds.

Runs pytest --cov, parses coverage reports, and can trigger
TestGeneratorAgent to create additional tests for uncovered code.
"""

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from llm_backend import LLMBackend
from schemas.coverage_report import (
    CoverageReport,
    CoverageStatus,
    CoverageThreshold,
    DiffCoverage,
    FileCoverage,
    TestGenerationRequest,
    UncoveredBlock,
)

from .base import AgentInput, AgentOutput, BaseAgent

ANALYSIS_PROMPT = """You are a test coverage analyst. Given uncovered code blocks, suggest what tests would cover them.

For each uncovered block:
1. Identify what the code does
2. Suggest specific test cases that would exercise this code
3. Note any edge cases or error conditions to test

UNCOVERED BLOCKS:
{blocks}

For each block, provide:
- function_name: The function/method containing the code
- reason: Why this code might be untested (edge case, error path, complex logic)
- test_suggestion: A specific test approach

Respond with JSON:
{{
  "analysis": [
    {{
      "file": "path/to/file.py",
      "start_line": 10,
      "end_line": 15,
      "function_name": "function_name",
      "reason": "Error handling path",
      "test_suggestion": "Test with invalid input to trigger exception"
    }}
  ],
  "recommendations": ["Overall recommendation 1", "Recommendation 2"]
}}
"""


class CoverageAgent(BaseAgent):
    """Agent for running and analyzing test coverage.

    Runs pytest with coverage, parses results, and optionally
    triggers test generation for uncovered code.
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        threshold: CoverageThreshold | None = None,
        max_iterations: int = 2,
    ) -> None:
        """Initialize CoverageAgent.

        Args:
            llm: LLM backend for analyzing uncovered code
            threshold: Coverage thresholds to enforce
            max_iterations: Max test generation iterations
        """
        super().__init__(llm, system_prompt=None)
        self.threshold = threshold or CoverageThreshold()
        self.max_iterations = max_iterations

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return "You are a test coverage analyst helping improve test coverage."

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run coverage analysis.

        Args:
            input_data: Must contain:
                - repo_path: Path to repository
                - Optional: changed_files (list of changed file paths)
                - Optional: iteration (current iteration number)
                - Optional: previous_report (previous coverage data)

        Returns:
            AgentOutput with CoverageReport data
        """
        repo_path = Path(input_data.context.get("repo_path", "."))
        changed_files = input_data.context.get("changed_files", [])
        iteration = input_data.context.get("iteration", 1)
        previous_report = input_data.context.get("previous_report")

        try:
            # Run pytest with coverage
            cov_data = self._run_coverage(repo_path)

            if cov_data is None:
                return AgentOutput(
                    success=False,
                    data={},
                    errors=["Failed to run coverage - pytest-cov may not be installed"],
                )

            # Build coverage report
            report = self._build_report(cov_data, changed_files, iteration)

            # Track improvement from previous iteration
            if previous_report:
                prev_coverage = previous_report.get("overall_coverage", 0)
                report.previous_coverage = prev_coverage
                report.improvement = report.overall_coverage - prev_coverage

            # Determine status
            report.status = self._evaluate_status(report)

            # If failing and we have LLM, analyze uncovered blocks
            if report.status == CoverageStatus.FAILING and self.llm:
                self._analyze_uncovered(report, repo_path)

            # Generate test requests if below threshold
            if report.status in (CoverageStatus.FAILING, CoverageStatus.WARNING):
                self._generate_test_requests(report)

            # Add summary and recommendations
            self._add_summary(report)

            return AgentOutput(
                success=True,
                data=report.model_dump(),
                artifacts=["coverage_report.json"],
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"CoverageAgent error: {str(e)}"],
            )

    def _run_coverage(self, repo_path: Path) -> dict | None:
        """Run pytest with coverage and return parsed data."""
        # Check for pytest-cov
        result = subprocess.run(
            ["python", "-c", "import pytest_cov"],
            cwd=repo_path,
            capture_output=True,
        )
        if result.returncode != 0:
            return None

        # Run pytest with coverage, output to JSON
        cov_json = repo_path / ".coverage.json"
        cov_xml = repo_path / "coverage.xml"

        # Try to run with JSON output first
        result = subprocess.run(
            [
                "python", "-m", "pytest",
                "--cov=.",
                "--cov-report=json",
                f"--cov-report=json:{cov_json}",
                "--cov-report=xml",
                "-q",
                "--tb=no",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse JSON coverage if available
        if cov_json.exists():
            with open(cov_json) as f:
                return json.load(f)

        # Fallback to XML parsing
        if cov_xml.exists():
            return self._parse_coverage_xml(cov_xml)

        # Try parsing from stdout as last resort
        return self._parse_coverage_stdout(result.stdout)

    def _parse_coverage_xml(self, xml_path: Path) -> dict:
        """Parse coverage.xml into our format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        files = {}
        totals = {"covered_lines": 0, "total_lines": 0}

        for package in root.findall(".//package"):
            for cls in package.findall("classes/class"):
                filename = cls.get("filename", "")
                lines = cls.findall("lines/line")

                covered = sum(1 for l in lines if l.get("hits", "0") != "0")
                total = len(lines)
                missing = [
                    int(l.get("number", 0))
                    for l in lines
                    if l.get("hits", "0") == "0"
                ]

                files[filename] = {
                    "covered_lines": covered,
                    "num_statements": total,
                    "missing_lines": missing,
                }

                totals["covered_lines"] += covered
                totals["total_lines"] += total

        return {"files": files, "totals": totals}

    def _parse_coverage_stdout(self, stdout: str) -> dict:
        """Parse coverage from pytest stdout as fallback."""
        files = {}
        totals = {"covered_lines": 0, "total_lines": 0}

        # Look for coverage table in output
        # Format: Name    Stmts   Miss  Cover   Missing
        pattern = r"(\S+\.py)\s+(\d+)\s+(\d+)\s+(\d+)%\s*([\d,\-\s]*)"

        for match in re.finditer(pattern, stdout):
            filename = match.group(1)
            stmts = int(match.group(2))
            miss = int(match.group(3))
            covered = stmts - miss

            # Parse missing lines
            missing_str = match.group(5).strip()
            missing = []
            if missing_str:
                for part in missing_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = part.split("-")
                        missing.extend(range(int(start), int(end) + 1))
                    elif part.isdigit():
                        missing.append(int(part))

            files[filename] = {
                "covered_lines": covered,
                "num_statements": stmts,
                "missing_lines": missing,
            }

            totals["covered_lines"] += covered
            totals["total_lines"] += stmts

        return {"files": files, "totals": totals}

    def _build_report(
        self,
        cov_data: dict,
        changed_files: list[str],
        iteration: int,
    ) -> CoverageReport:
        """Build CoverageReport from parsed coverage data."""
        files_data = cov_data.get("files", {})
        totals = cov_data.get("totals", {})

        # Build file coverage list
        file_coverages = []
        for filepath, data in files_data.items():
            total = data.get("num_statements", 0)
            covered = data.get("covered_lines", 0)
            missing = data.get("missing_lines", [])

            pct = (covered / total * 100) if total > 0 else 0.0

            file_coverages.append(
                FileCoverage(
                    file_path=filepath,
                    total_lines=total,
                    covered_lines=covered,
                    missing_lines=missing,
                    coverage_percent=round(pct, 2),
                )
            )

        # Calculate overall coverage
        total_lines = totals.get("total_lines", 0)
        covered_lines = totals.get("covered_lines", 0)

        if total_lines == 0:
            # Fallback: sum from files
            total_lines = sum(f.total_lines for f in file_coverages)
            covered_lines = sum(f.covered_lines for f in file_coverages)

        overall_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

        # Build diff coverage if changed files provided
        diff_coverage = None
        if changed_files:
            diff_coverage = self._calculate_diff_coverage(
                file_coverages, changed_files
            )

        return CoverageReport(
            total_lines=total_lines,
            covered_lines=covered_lines,
            overall_coverage=round(overall_pct, 2),
            files=file_coverages,
            diff_coverage=diff_coverage,
            iteration=iteration,
            threshold=self.threshold,
        )

    def _calculate_diff_coverage(
        self,
        file_coverages: list[FileCoverage],
        changed_files: list[str],
    ) -> DiffCoverage:
        """Calculate coverage for changed files only."""
        new_lines = 0
        covered_new = 0
        uncovered = []

        for fc in file_coverages:
            # Check if file is in changed list
            if any(fc.file_path.endswith(cf) for cf in changed_files):
                new_lines += fc.total_lines
                covered_new += fc.covered_lines

                # Track uncovered blocks
                if fc.missing_lines:
                    blocks = self._group_missing_lines(fc.file_path, fc.missing_lines)
                    uncovered.extend(blocks)

        pct = (covered_new / new_lines * 100) if new_lines > 0 else 0.0

        return DiffCoverage(
            changed_files=changed_files,
            new_lines=new_lines,
            covered_new_lines=covered_new,
            diff_coverage_percent=round(pct, 2),
            uncovered_changes=uncovered,
        )

    def _group_missing_lines(
        self,
        file_path: str,
        missing: list[int],
    ) -> list[UncoveredBlock]:
        """Group consecutive missing lines into blocks."""
        if not missing:
            return []

        blocks = []
        missing = sorted(missing)
        start = missing[0]
        end = missing[0]

        for line in missing[1:]:
            if line == end + 1:
                end = line
            else:
                # Save current block
                blocks.append(
                    UncoveredBlock(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                    )
                )
                start = line
                end = line

        # Don't forget last block
        blocks.append(
            UncoveredBlock(
                file_path=file_path,
                start_line=start,
                end_line=end,
            )
        )

        return blocks

    def _evaluate_status(self, report: CoverageReport) -> CoverageStatus:
        """Determine if coverage meets thresholds."""
        overall = report.overall_coverage
        threshold = report.threshold.overall_min

        # Check diff coverage if available
        if report.diff_coverage:
            diff_pct = report.diff_coverage.diff_coverage_percent
            if diff_pct < report.threshold.diff_min:
                return CoverageStatus.FAILING

        # Check overall coverage
        if overall < threshold:
            return CoverageStatus.FAILING
        elif overall < threshold + 5:  # Within 5% of threshold
            return CoverageStatus.WARNING
        else:
            return CoverageStatus.PASSING

    def _analyze_uncovered(self, report: CoverageReport, repo_path: Path) -> None:
        """Use LLM to analyze uncovered blocks and suggest tests."""
        # Find significant uncovered blocks (3+ lines)
        blocks_to_analyze = []

        for fc in report.files:
            if fc.missing_lines:
                blocks = self._group_missing_lines(fc.file_path, fc.missing_lines)
                for block in blocks:
                    if block.end_line - block.start_line >= 2:
                        # Try to read the actual code
                        try:
                            file_path = repo_path / block.file_path
                            if file_path.exists():
                                lines = file_path.read_text().splitlines()
                                start = max(0, block.start_line - 1)
                                end = min(len(lines), block.end_line)
                                block.code_snippet = "\n".join(lines[start:end])
                        except Exception:
                            pass
                        blocks_to_analyze.append(block)

        if not blocks_to_analyze:
            return

        # Limit to top 10 blocks
        blocks_to_analyze = blocks_to_analyze[:10]

        # Format for LLM
        blocks_str = ""
        for b in blocks_to_analyze:
            blocks_str += f"\nFile: {b.file_path} (lines {b.start_line}-{b.end_line})\n"
            if b.code_snippet:
                blocks_str += f"```python\n{b.code_snippet}\n```\n"

        prompt = ANALYSIS_PROMPT.format(blocks=blocks_str)

        try:
            response = self._chat(prompt, temperature=0.3)

            # Parse response
            from utils.json_repair import parse_llm_json
            analysis = parse_llm_json(response, default={})

            # Update blocks with analysis
            for item in analysis.get("analysis", []):
                for block in blocks_to_analyze:
                    if (
                        block.file_path.endswith(item.get("file", ""))
                        and block.start_line == item.get("start_line")
                    ):
                        block.function_name = item.get("function_name")
                        block.reason = item.get("reason", "")
                        block.test_suggestion = item.get("test_suggestion", "")

            # Add recommendations
            report.recommendations = analysis.get("recommendations", [])
            report.uncovered_blocks = blocks_to_analyze

        except Exception:
            # LLM analysis failed, continue without it
            report.uncovered_blocks = blocks_to_analyze

    def _generate_test_requests(self, report: CoverageReport) -> None:
        """Generate test generation requests for uncovered code."""
        # Group uncovered blocks by file
        by_file: dict[str, list[UncoveredBlock]] = {}

        for fc in report.files:
            if fc.coverage_percent < self.threshold.diff_min:
                blocks = self._group_missing_lines(fc.file_path, fc.missing_lines)
                if blocks:
                    by_file[fc.file_path] = blocks

        # Also include blocks from uncovered_blocks analysis
        for block in report.uncovered_blocks:
            if block.file_path not in by_file:
                by_file[block.file_path] = []
            by_file[block.file_path].append(block)

        # Create requests, prioritized by coverage gap
        requests = []
        for file_path, blocks in by_file.items():
            # Find file coverage
            fc = next((f for f in report.files if f.file_path == file_path), None)
            gap = self.threshold.diff_min - (fc.coverage_percent if fc else 0)

            priority = 1 if gap > 20 else (2 if gap > 10 else 3)

            requests.append(
                TestGenerationRequest(
                    target_file=file_path,
                    uncovered_blocks=blocks,
                    priority=priority,
                    estimated_tests=len(blocks),
                )
            )

        # Sort by priority
        requests.sort(key=lambda r: r.priority)
        report.test_requests = requests[:5]  # Limit to top 5

    def _add_summary(self, report: CoverageReport) -> None:
        """Add human-readable summary and notes."""
        status_emoji = {
            CoverageStatus.PASSING: "PASS",
            CoverageStatus.WARNING: "WARN",
            CoverageStatus.FAILING: "FAIL",
            CoverageStatus.UNKNOWN: "????",
        }

        report.summary = (
            f"Coverage: {report.overall_coverage:.1f}% "
            f"({report.covered_lines}/{report.total_lines} lines) "
            f"[{status_emoji[report.status]}]"
        )

        if report.diff_coverage:
            report.summary += (
                f" | Diff: {report.diff_coverage.diff_coverage_percent:.1f}%"
            )

        if report.improvement != 0:
            sign = "+" if report.improvement > 0 else ""
            report.summary += f" | Change: {sign}{report.improvement:.1f}%"

        # Notes for VerifyAgent
        report.verify_notes = []

        if report.status == CoverageStatus.FAILING:
            report.verify_notes.append(
                f"Coverage {report.overall_coverage:.1f}% is below "
                f"threshold {report.threshold.overall_min}%"
            )
            if report.test_requests:
                report.verify_notes.append(
                    f"{len(report.test_requests)} files need additional tests"
                )

        if report.uncovered_blocks:
            report.verify_notes.append(
                f"{len(report.uncovered_blocks)} significant uncovered code blocks"
            )

        if report.status == CoverageStatus.PASSING:
            report.verify_notes.append("Coverage requirements met")

    def should_iterate(self, report: CoverageReport) -> bool:
        """Check if another coverage iteration should run.

        Returns True if:
        - Status is failing
        - Haven't exceeded max iterations
        - There are test requests to process
        """
        return (
            report.status == CoverageStatus.FAILING
            and report.iteration < self.max_iterations
            and len(report.test_requests) > 0
        )
