"""Policy Agent - Enforces rules and decides what needs approval.

This agent is the gatekeeper before final verification:
- Reads diff, test results, security results, metadata
- Applies rules from YAML configuration
- Evaluates code against PDF policy documents (natural language)
- Produces decisions: allow / require_approval / block
"""

import logging
import warnings
from fnmatch import fnmatch
from pathlib import Path

import yaml

from agents.base import BaseAgent
from llm_backend import LLMBackend

# Optional PDF support
try:
    import fitz  # pymupdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
from schemas.policy import (
    MatchedRule,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    PolicyRuleSet,
    RuleCondition,
)

logger = logging.getLogger(__name__)

# Default rules if no rules file exists
DEFAULT_RULES = """
version: "1.0"
description: "Default policy rules for Coding Factory"
rules:
  - id: block_secrets
    description: "Block if potential secrets detected"
    when:
      has_secrets: true
    action: block
    severity: critical
    message: "Hardcoded secrets detected. Remove them before proceeding."

  - id: min_coverage
    description: "Require 80% coverage on code"
    when:
      coverage_lt: 0.80
    action: require_approval
    severity: warning
    required_role: tech_lead
    message: "Coverage below 80%. Requires tech lead approval."

  - id: block_failed_tests
    description: "Block if tests are failing"
    when:
      tests_failed: true
    action: block
    severity: critical
    message: "Tests are failing. Fix them before proceeding."

  - id: block_critical_vulnerabilities
    description: "Block if critical CVEs detected"
    when:
      has_vulnerabilities: true
    action: block
    severity: critical
    message: "Security vulnerabilities detected. Fix before proceeding."

  - id: block_bandit_high
    description: "Block if high severity security issues"
    when:
      bandit_high_gt: 0
    action: block
    severity: critical
    message: "High severity security issues found by bandit."

  - id: large_diff_approval
    description: "Large changes require approval"
    when:
      files_changed_gt: 20
    action: require_approval
    severity: warning
    required_role: senior_engineer
    message: "Large change (>20 files). Requires senior engineer review."

  - id: infra_approval
    description: "Infrastructure changes require approval"
    when:
      paths_match:
        - "infra/**"
        - "terraform/**"
        - "k8s/**"
        - "kubernetes/**"
        - ".github/workflows/**"
    action: require_approval
    severity: warning
    required_role: platform_engineer
    message: "Infrastructure changes require platform engineer approval."

  - id: payment_approval
    description: "Payment code changes require senior review"
    when:
      paths_match:
        - "**/payments/**"
        - "**/billing/**"
        - "**/stripe/**"
        - "**/paypal/**"
    action: require_approval
    severity: warning
    required_role: senior_engineer
    message: "Payment-related changes require senior engineer approval."

  - id: auth_approval
    description: "Auth code changes require security review"
    when:
      paths_match:
        - "**/auth/**"
        - "**/authentication/**"
        - "**/authorization/**"
        - "**/oauth/**"
        - "**/jwt/**"
    action: require_approval
    severity: warning
    required_role: security_engineer
    message: "Authentication changes require security review."

  - id: docs_only_auto
    description: "Only docs/tests changed - auto allow"
    when:
      paths_subset_of:
        - "docs/**"
        - "*.md"
        - "tests/**"
        - "test_*.py"
        - "*_test.py"
    action: allow
    severity: info
    message: "Only documentation/tests changed. Auto-approved."

  - id: dont_ship_review
    description: "Block if code review says don't ship"
    when:
      ship_status: "dont_ship"
    action: block
    severity: error
    message: "Code review status is 'don't ship'. Address issues first."
"""


class PolicyAgent(BaseAgent):
    """Agent that applies policy rules to decide allow/block/require_approval."""

    def __init__(
        self,
        llm: LLMBackend | None = None,
        rules_path: str | Path | None = None,
        pdf_policies_dir: str | Path | None = None,
    ) -> None:
        """Initialize policy agent.

        Args:
            llm: LLM backend (required for PDF policy evaluation)
            rules_path: Path to policy_rules.yaml file
            pdf_policies_dir: Directory containing PDF policy documents
        """
        # Store LLM for PDF policy evaluation
        if llm is not None:
            super().__init__(llm)
        else:
            self.llm = None
            self.system_prompt = self.default_system_prompt()
        self.name = "PolicyAgent"
        self.rules_path = Path(rules_path) if rules_path else None
        self.pdf_policies_dir = Path(pdf_policies_dir) if pdf_policies_dir else None
        self.rule_set: PolicyRuleSet | None = None
        self.pdf_policies: list[dict] = []  # [{name, content}]

    def default_system_prompt(self) -> str:
        """Return the default system prompt.

        Note: PolicyAgent uses rule-based evaluation, not LLM inference.
        This method exists to satisfy the BaseAgent interface.
        """
        return "PolicyAgent: Rule-based policy enforcement (no LLM required)."

    def load_rules(self, rules_path: Path | None = None) -> PolicyRuleSet:
        """Load policy rules from YAML file.

        Args:
            rules_path: Path to rules file (uses default if not found)

        Returns:
            Loaded rule set
        """
        path = rules_path or self.rules_path

        if path and path.exists():
            logger.info("Loading policy rules from %s", path)
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self.rule_set = PolicyRuleSet(**data)
                return self.rule_set
            except Exception as e:
                logger.warning("Failed to load rules from %s: %s", path, e)

        # Use default rules
        logger.info("Using default policy rules")
        data = yaml.safe_load(DEFAULT_RULES)
        self.rule_set = PolicyRuleSet(**data)
        return self.rule_set

    def load_pdf_policies(self, pdf_path: Path | None = None) -> int:
        """Load policy documents from PDF files.

        Extracts text from PDFs to use for natural language policy evaluation.
        Can load a single PDF or all PDFs from the configured directory.

        Args:
            pdf_path: Optional path to a specific PDF file

        Returns:
            Number of PDF policies loaded
        """
        if not PDF_SUPPORT:
            warnings.warn(
                "PDF support not available. Install with: pip install pymupdf"
            )
            return 0

        paths_to_load = []

        if pdf_path:
            paths_to_load.append(Path(pdf_path))
        elif self.pdf_policies_dir and self.pdf_policies_dir.exists():
            paths_to_load.extend(self.pdf_policies_dir.glob("*.pdf"))

        for path in paths_to_load:
            try:
                content = self._extract_pdf_text(path)
                if content.strip():
                    self.pdf_policies.append({
                        "name": path.stem,
                        "path": str(path),
                        "content": content,
                    })
                    logger.info("Loaded PDF policy: %s (%d chars)", path.name, len(content))
            except Exception as e:
                logger.warning("Failed to load PDF policy %s: %s", path, e)

        return len(self.pdf_policies)

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        if not PDF_SUPPORT:
            return ""

        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        return "\n\n".join(text_parts)

    def evaluate_pdf_policies(
        self,
        code_files: list[dict],
        max_policies: int = 5,
    ) -> list[dict]:
        """Evaluate code against PDF policy documents using LLM.

        Args:
            code_files: List of {path, content} dicts
            max_policies: Max number of policies to evaluate at once

        Returns:
            List of violations found [{policy, violation, severity, file}]
        """
        if not self.pdf_policies:
            return []

        if not self.llm:
            logger.warning("No LLM configured for PDF policy evaluation")
            return []

        violations = []

        # Prepare code summary (truncate if too large)
        code_summary = []
        for f in code_files[:20]:  # Limit files
            content = f.get("content", "")
            if len(content) > 5000:
                content = content[:5000] + "\n... [truncated]"
            code_summary.append(f"### {f['path']}\n```\n{content}\n```")

        code_text = "\n\n".join(code_summary)

        # Evaluate against each PDF policy
        for policy in self.pdf_policies[:max_policies]:
            try:
                result = self._evaluate_single_pdf_policy(policy, code_text)
                if result:
                    violations.extend(result)
            except Exception as e:
                logger.warning("Failed to evaluate policy %s: %s", policy["name"], e)

        return violations

    def _evaluate_single_pdf_policy(
        self,
        policy: dict,
        code_text: str,
    ) -> list[dict]:
        """Evaluate code against a single PDF policy.

        Args:
            policy: Policy dict with name and content
            code_text: Formatted code to evaluate

        Returns:
            List of violations found
        """
        prompt = f"""You are a code policy compliance checker.

POLICY DOCUMENT: {policy['name']}
---
{policy['content'][:15000]}
---

CODE TO REVIEW:
{code_text[:30000]}

Analyze the code against the policy document above. List any violations found.

For each violation, output a JSON object on its own line:
{{"violation": "description", "severity": "critical|high|medium|low", "file": "path/to/file.py", "recommendation": "how to fix"}}

If no violations are found, output:
{{"status": "compliant"}}

Only output JSON, no other text."""

        response = self.llm.generate(prompt, max_tokens=2000)

        violations = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                import json
                data = json.loads(line)
                if "violation" in data:
                    data["policy"] = policy["name"]
                    violations.append(data)
            except json.JSONDecodeError:
                continue

        return violations

    def _match_condition(
        self,
        condition: RuleCondition,
        context: PolicyContext,
    ) -> bool:
        """Check if a rule condition matches the context.

        Args:
            condition: Rule condition to evaluate
            context: Pipeline context

        Returns:
            True if condition matches
        """
        # Coverage conditions
        if condition.coverage_lt is not None:
            if context.test_results.coverage >= condition.coverage_lt:
                return False

        if condition.coverage_gt is not None:
            if context.test_results.coverage <= condition.coverage_gt:
                return False

        # Security conditions
        if condition.has_secrets is not None:
            if context.security_results.has_secrets != condition.has_secrets:
                return False

        if condition.bandit_high_gt is not None:
            if context.security_results.bandit_high <= condition.bandit_high_gt:
                return False

        if condition.bandit_medium_gt is not None:
            if context.security_results.bandit_medium <= condition.bandit_medium_gt:
                return False

        if condition.has_vulnerabilities is not None:
            if context.security_results.has_vulnerabilities != condition.has_vulnerabilities:
                return False

        # Path-based conditions
        if condition.paths_match is not None:
            # Trigger if ANY file matches ANY pattern
            patterns = condition.paths_match
            files = context.files_changed
            matched = any(
                any(fnmatch(f, p) for p in patterns)
                for f in files
            )
            if not matched:
                return False

        if condition.paths_subset_of is not None:
            # Trigger if ALL files match at least one pattern
            patterns = condition.paths_subset_of
            files = context.files_changed
            if not files:  # No files changed, can't be subset
                return False
            all_match = all(
                any(fnmatch(f, p) for p in patterns)
                for f in files
            )
            if not all_match:
                return False

        if condition.paths_exclude is not None:
            # Trigger if NO files match these patterns
            patterns = condition.paths_exclude
            files = context.files_changed
            any_match = any(
                any(fnmatch(f, p) for p in patterns)
                for f in files
            )
            if any_match:
                return False

        # Diff conditions
        if condition.files_changed_gt is not None:
            if context.diff_stats.files_changed <= condition.files_changed_gt:
                return False

        if condition.lines_added_gt is not None:
            if context.diff_stats.lines_added <= condition.lines_added_gt:
                return False

        if condition.lines_deleted_gt is not None:
            if context.diff_stats.lines_deleted <= condition.lines_deleted_gt:
                return False

        # Test conditions
        if condition.tests_failed is not None:
            tests_failed = not context.test_results.passed
            if tests_failed != condition.tests_failed:
                return False

        if condition.tests_skipped_gt is not None:
            if context.test_results.skipped <= condition.tests_skipped_gt:
                return False

        # Code review conditions
        if condition.ship_status is not None:
            if context.code_review.ship_status != condition.ship_status:
                return False

        if condition.review_issues_gt is not None:
            if context.code_review.total_issues <= condition.review_issues_gt:
                return False

        # Branch conditions
        if condition.branch_match is not None:
            if not fnmatch(context.branch, condition.branch_match):
                return False

        if condition.is_main_branch is not None:
            is_main = context.branch in ("main", "master")
            if is_main != condition.is_main_branch:
                return False

        return True

    def _match_rule(
        self,
        rule: PolicyRule,
        context: PolicyContext,
    ) -> bool:
        """Check if a rule matches the context.

        Args:
            rule: Policy rule to evaluate
            context: Pipeline context

        Returns:
            True if rule matches
        """
        if not rule.enabled:
            return False

        return self._match_condition(rule.when, context)

    def run(self, context: PolicyContext) -> PolicyDecision:
        """Evaluate policy rules against context.

        Args:
            context: Pipeline context with all relevant information

        Returns:
            Policy decision with status and reasons
        """
        logger.info("Evaluating policy rules for run %s", context.run_id)

        # Load rules if not already loaded
        if self.rule_set is None:
            self.load_rules()

        matched_rules: list[MatchedRule] = []
        blocking_rules: list[str] = []
        approval_rules: list[str] = []

        # Evaluate each rule
        for rule in self.rule_set.rules:
            if self._match_rule(rule, context):
                matched = MatchedRule(
                    rule_id=rule.id,
                    action=rule.action,
                    severity=rule.severity,
                    description=rule.description,
                    message=rule.message,
                    required_role=rule.required_role,
                )
                matched_rules.append(matched)

                if rule.action == PolicyAction.BLOCK:
                    blocking_rules.append(rule.id)
                elif rule.action == PolicyAction.REQUIRE_APPROVAL:
                    approval_rules.append(rule.id)

                logger.info(
                    "Rule matched: %s (%s) -> %s",
                    rule.id,
                    rule.description,
                    rule.action.value,
                )

        # Determine final status (most severe wins)
        if blocking_rules:
            status = PolicyAction.BLOCK
        elif approval_rules:
            status = PolicyAction.REQUIRE_APPROVAL
        else:
            status = PolicyAction.ALLOW

        # Build reasons list
        reasons = []
        for matched in matched_rules:
            if matched.message:
                reasons.append(matched.message)
            else:
                reasons.append(matched.description)

        if not reasons:
            reasons = ["No policy rules matched. Default: allow."]

        # Get required role (from highest severity approval rule)
        required_role = None
        if approval_rules:
            for matched in matched_rules:
                if matched.rule_id in approval_rules and matched.required_role:
                    required_role = matched.required_role
                    break

        # Build summary
        summary = self._build_summary(status, matched_rules, context)

        decision = PolicyDecision(
            status=status,
            reasons=reasons,
            matched_rules=matched_rules,
            required_role=required_role,
            blocking_rules=blocking_rules,
            approval_rules=approval_rules,
            summary=summary,
        )

        logger.info(
            "Policy decision: %s (%d rules matched, %d blocking, %d approval)",
            status.value,
            len(matched_rules),
            len(blocking_rules),
            len(approval_rules),
        )

        return decision

    def _build_summary(
        self,
        status: PolicyAction,
        matched_rules: list[MatchedRule],
        context: PolicyContext,
    ) -> str:
        """Build a human-readable summary of the decision.

        Args:
            status: Final policy status
            matched_rules: List of matched rules
            context: Pipeline context

        Returns:
            Summary string
        """
        if status == PolicyAction.ALLOW:
            if matched_rules:
                return f"Allowed: {len(matched_rules)} rules matched, all permissive."
            return "Allowed: No policy rules triggered."

        if status == PolicyAction.BLOCK:
            blockers = [r for r in matched_rules if r.action == PolicyAction.BLOCK]
            return (
                f"Blocked: {len(blockers)} rule(s) blocking. "
                f"Fix issues before proceeding."
            )

        if status == PolicyAction.REQUIRE_APPROVAL:
            approvers = [r for r in matched_rules if r.action == PolicyAction.REQUIRE_APPROVAL]
            roles = {r.required_role for r in approvers if r.required_role}
            role_str = ", ".join(roles) if roles else "reviewer"
            return (
                f"Requires approval from {role_str}: "
                f"{len(approvers)} rule(s) require human review."
            )

        return "Unknown policy status."

    def generate_rules_template(self, output_path: Path) -> Path:
        """Generate a template policy_rules.yaml file.

        Args:
            output_path: Where to write the template

        Returns:
            Path to created file
        """
        output_path.write_text(DEFAULT_RULES)
        logger.info("Generated policy rules template: %s", output_path)
        return output_path

    def validate_rules(self, rules_path: Path) -> list[str]:
        """Validate a rules file.

        Args:
            rules_path: Path to rules file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            with open(rules_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            return [f"Failed to parse YAML: {e}"]

        if "rules" not in data:
            errors.append("Missing 'rules' key in file")
            return errors

        seen_ids = set()
        for i, rule in enumerate(data.get("rules", [])):
            prefix = f"Rule {i + 1}"

            if "id" not in rule:
                errors.append(f"{prefix}: Missing 'id' field")
            elif rule["id"] in seen_ids:
                errors.append(f"{prefix}: Duplicate rule ID '{rule['id']}'")
            else:
                seen_ids.add(rule["id"])

            if "when" not in rule:
                errors.append(f"{prefix}: Missing 'when' conditions")

            if "action" not in rule:
                errors.append(f"{prefix}: Missing 'action' field")
            elif rule["action"] not in ("allow", "require_approval", "block"):
                errors.append(
                    f"{prefix}: Invalid action '{rule['action']}'. "
                    f"Must be: allow, require_approval, block"
                )

        return errors
