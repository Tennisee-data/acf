"""Secrets Scan Agent for detecting hardcoded credentials.

Scans generated code for hardcoded secrets and can auto-fix by:
- Moving secrets to .env.example
- Replacing with os.getenv() calls
- Updating README with configuration instructions
"""

import re
import time
from pathlib import Path

from llm_backend import LLMBackend
from schemas.secrets_report import (
    AutoFix,
    DetectedSecret,
    SecretSeverity,
    SecretsReport,
    SecretStatus,
    SecretType,
)

from .base import AgentInput, AgentOutput, BaseAgent

# Secret detection patterns with their types and severities
SECRET_PATTERNS: list[tuple[str, SecretType, SecretSeverity]] = [
    # AWS
    (r'AKIA[0-9A-Z]{16}', SecretType.AWS_ACCESS_KEY, SecretSeverity.CRITICAL),
    (r'(?i)aws[_-]?secret[_-]?(?:access[_-]?)?key\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?',
     SecretType.AWS_SECRET_KEY, SecretSeverity.CRITICAL),

    # GitHub
    (r'ghp_[A-Za-z0-9]{36}', SecretType.GITHUB_PAT, SecretSeverity.CRITICAL),
    (r'gho_[A-Za-z0-9]{36}', SecretType.GITHUB_TOKEN, SecretSeverity.CRITICAL),
    (r'ghu_[A-Za-z0-9]{36}', SecretType.GITHUB_TOKEN, SecretSeverity.CRITICAL),
    (r'ghs_[A-Za-z0-9]{36}', SecretType.GITHUB_TOKEN, SecretSeverity.CRITICAL),
    (r'github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}', SecretType.GITHUB_PAT, SecretSeverity.CRITICAL),

    # GitLab
    (r'glpat-[A-Za-z0-9\-]{20}', SecretType.GITLAB_TOKEN, SecretSeverity.CRITICAL),

    # Slack
    (r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*',
     SecretType.SLACK_TOKEN, SecretSeverity.CRITICAL),
    (r'https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+',
     SecretType.SLACK_WEBHOOK, SecretSeverity.HIGH),

    # Stripe
    (r'sk_live_[A-Za-z0-9]{24,}', SecretType.STRIPE_KEY, SecretSeverity.CRITICAL),
    (r'sk_test_[A-Za-z0-9]{24,}', SecretType.STRIPE_KEY, SecretSeverity.MEDIUM),
    (r'pk_live_[A-Za-z0-9]{24,}', SecretType.STRIPE_KEY, SecretSeverity.HIGH),
    (r'pk_test_[A-Za-z0-9]{24,}', SecretType.STRIPE_KEY, SecretSeverity.LOW),

    # Twilio
    (r'SK[a-f0-9]{32}', SecretType.TWILIO_KEY, SecretSeverity.HIGH),

    # SendGrid
    (r'SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}',
     SecretType.SENDGRID_KEY, SecretSeverity.HIGH),

    # OpenAI
    (r'sk-[A-Za-z0-9]{48}', SecretType.OPENAI_KEY, SecretSeverity.HIGH),
    (r'sk-proj-[A-Za-z0-9\-_]{48,}', SecretType.OPENAI_KEY, SecretSeverity.HIGH),

    # Anthropic
    (r'sk-ant-[A-Za-z0-9\-_]{32,}', SecretType.ANTHROPIC_KEY, SecretSeverity.HIGH),

    # Google
    (r'AIza[0-9A-Za-z\-_]{35}', SecretType.GOOGLE_API_KEY, SecretSeverity.HIGH),

    # Firebase
    (r'(?i)firebase[_-]?(?:api[_-]?)?key\s*[=:]\s*["\']?([A-Za-z0-9\-_]{20,})["\']?',
     SecretType.FIREBASE_KEY, SecretSeverity.HIGH),

    # JWT/Auth secrets
    (r'(?i)(?:jwt|auth)[_-]?secret\s*[=:]\s*["\']([^"\']{16,})["\']',
     SecretType.JWT_SECRET, SecretSeverity.HIGH),

    # Database URLs with credentials
    (r'(?:postgres|mysql|mongodb)(?:ql)?://[^:]+:[^@]+@[^\s"\']+',
     SecretType.DATABASE_URL, SecretSeverity.CRITICAL),

    # Private keys
    (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
     SecretType.PRIVATE_KEY, SecretSeverity.CRITICAL),
    (r'-----BEGIN PGP PRIVATE KEY BLOCK-----',
     SecretType.PRIVATE_KEY, SecretSeverity.CRITICAL),

    # Generic patterns (lower confidence)
    (r'(?i)(?:api[_-]?key|apikey)\s*[=:]\s*["\']([A-Za-z0-9\-_]{20,})["\']',
     SecretType.API_KEY, SecretSeverity.MEDIUM),
    (r'(?i)(?:auth[_-]?token|access[_-]?token)\s*[=:]\s*["\']([A-Za-z0-9\-_]{20,})["\']',
     SecretType.AUTH_TOKEN, SecretSeverity.MEDIUM),
    (r'(?i)bearer\s+[A-Za-z0-9\-_\.]{20,}',
     SecretType.BEARER_TOKEN, SecretSeverity.MEDIUM),
    (r'(?i)password\s*[=:]\s*["\']([^"\']{8,})["\']',
     SecretType.PASSWORD, SecretSeverity.HIGH),
    (r'(?i)secret\s*[=:]\s*["\']([^"\']{16,})["\']',
     SecretType.GENERIC_SECRET, SecretSeverity.MEDIUM),
]

# Files/patterns to skip
SKIP_PATTERNS = [
    r'\.git/',
    r'__pycache__/',
    r'\.pyc$',
    r'node_modules/',
    r'\.env\.example$',
    r'\.env\.sample$',
    r'test_.*\.py$',
    r'.*_test\.py$',
    r'tests?/',
    r'fixtures?/',
    r'mocks?/',
]

# File extensions to scan
SCAN_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml',
    '.env', '.ini', '.cfg', '.conf', '.toml', '.xml', '.sh', '.bash',
    '.go', '.rs', '.java', '.kt', '.swift', '.rb', '.php',
}


class SecretsScanAgent(BaseAgent):
    """Agent for detecting and fixing hardcoded secrets.

    Scans source code for credentials and can auto-fix by moving
    them to environment variables.
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        auto_fix: bool = True,
        block_on_secrets: bool = True,
    ) -> None:
        """Initialize SecretsScanAgent.

        Args:
            llm: LLM backend for context-aware fixes
            auto_fix: Whether to automatically fix detected secrets
            block_on_secrets: Whether to block pipeline on unresolved secrets
        """
        super().__init__(llm, system_prompt=None)
        self.auto_fix = auto_fix
        self.block_on_secrets = block_on_secrets

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return "You are a security expert helping to identify and fix hardcoded secrets."

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Run secrets scan.

        Args:
            input_data: Must contain:
                - repo_path: Path to scan
                - scan_phase: "pre_docker" or "pre_verify"
                - Optional: auto_fix (override instance setting)

        Returns:
            AgentOutput with SecretsReport data
        """
        repo_path = Path(input_data.context.get("repo_path", "."))
        scan_phase = input_data.context.get("scan_phase", "pre_docker")
        auto_fix = input_data.context.get("auto_fix", self.auto_fix)

        start_time = time.time()

        try:
            # Scan for secrets
            secrets, files_scanned = self._scan_directory(repo_path)

            # Build initial report
            report = SecretsReport(
                scan_phase=scan_phase,
                files_scanned=files_scanned,
                secrets=secrets,
                total_detected=len(secrets),
            )

            # Count by severity
            for secret in secrets:
                if secret.severity == SecretSeverity.CRITICAL:
                    report.critical_count += 1
                elif secret.severity == SecretSeverity.HIGH:
                    report.high_count += 1
                elif secret.severity == SecretSeverity.MEDIUM:
                    report.medium_count += 1
                else:
                    report.low_count += 1

            # Apply auto-fixes if enabled
            if auto_fix and secrets:
                self._apply_auto_fixes(repo_path, report)

            # Update .env.example if fixes were applied
            if report.auto_fixes:
                self._update_env_example(repo_path, report)
                self._update_readme(repo_path, report)

            # Determine if pipeline should be blocked
            if self.block_on_secrets and report.should_block():
                report.blocked = True
                unresolved = [
                    s for s in report.secrets
                    if s.status in (SecretStatus.DETECTED, SecretStatus.MANUAL_REQUIRED)
                    and s.severity in (SecretSeverity.CRITICAL, SecretSeverity.HIGH)
                ]
                report.block_reason = (
                    f"{len(unresolved)} high-severity secret(s) require manual review"
                )

            # Calculate duration
            report.scan_duration_ms = int((time.time() - start_time) * 1000)

            # Add summary
            self._add_summary(report)

            return AgentOutput(
                success=True,
                data=report.model_dump(),
                artifacts=["secrets_report.json"],
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"SecretsScanAgent error: {str(e)}"],
            )

    def _scan_directory(
        self,
        repo_path: Path,
    ) -> tuple[list[DetectedSecret], int]:
        """Scan directory for secrets."""
        secrets = []
        files_scanned = 0

        for file_path in repo_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip based on patterns
            rel_path = str(file_path.relative_to(repo_path))
            if self._should_skip(rel_path):
                continue

            # Check extension
            if file_path.suffix.lower() not in SCAN_EXTENSIONS:
                continue

            files_scanned += 1

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_secrets = self._scan_file(rel_path, content)
                secrets.extend(file_secrets)
            except Exception:
                # Skip files that can't be read
                pass

        return secrets, files_scanned

    def _should_skip(self, path: str) -> bool:
        """Check if path should be skipped."""
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, path):
                return True
        return False

    def _scan_file(self, file_path: str, content: str) -> list[DetectedSecret]:
        """Scan a single file for secrets."""
        secrets = []
        lines = content.split('\n')

        for pattern, secret_type, severity in SECRET_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                # Find line number
                start_pos = match.start()
                line_num = content[:start_pos].count('\n') + 1

                # Get the matched value
                matched_text = match.group(0)
                # If there's a capture group, use that
                if match.groups():
                    matched_text = match.group(1) or matched_text

                # Redact the value
                redacted = self._redact_secret(matched_text)

                # Get context snippet
                context = ""
                if 0 < line_num <= len(lines):
                    context = self._redact_line(lines[line_num - 1])

                # Try to identify variable name
                source_line = lines[line_num - 1] if line_num <= len(lines) else ""
                var_name = self._extract_variable_name(source_line)

                # Suggest env var name
                suggested_env = self._suggest_env_var(secret_type, var_name)

                secrets.append(
                    DetectedSecret(
                        file_path=file_path,
                        line_number=line_num,
                        column_start=match.start() - content.rfind('\n', 0, match.start()) - 1,
                        column_end=match.end() - content.rfind('\n', 0, match.start()) - 1,
                        secret_type=secret_type,
                        severity=severity,
                        status=SecretStatus.DETECTED,
                        redacted_value=redacted,
                        variable_name=var_name,
                        context_snippet=context,
                        suggested_env_var=suggested_env,
                    )
                )

        return secrets

    def _redact_secret(self, value: str) -> str:
        """Redact a secret value for safe display."""
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "..." + value[-4:]

    def _redact_line(self, line: str) -> str:
        """Redact secrets in a line of code."""
        result = line
        for pattern, _, _ in SECRET_PATTERNS:
            result = re.sub(pattern, "[REDACTED]", result)
        return result

    def _extract_variable_name(self, line: str) -> str:
        """Try to extract variable name from assignment."""
        # Python: var = "value"
        match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
        if match:
            return match.group(1)

        # JS: const/let/var name = "value"
        match = re.match(r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
        if match:
            return match.group(1)

        return ""

    def _suggest_env_var(self, secret_type: SecretType, var_name: str) -> str:
        """Suggest an environment variable name."""
        type_to_env = {
            SecretType.AWS_ACCESS_KEY: "AWS_ACCESS_KEY_ID",
            SecretType.AWS_SECRET_KEY: "AWS_SECRET_ACCESS_KEY",
            SecretType.GITHUB_TOKEN: "GITHUB_TOKEN",
            SecretType.GITHUB_PAT: "GITHUB_TOKEN",
            SecretType.GITLAB_TOKEN: "GITLAB_TOKEN",
            SecretType.SLACK_TOKEN: "SLACK_TOKEN",
            SecretType.SLACK_WEBHOOK: "SLACK_WEBHOOK_URL",
            SecretType.STRIPE_KEY: "STRIPE_SECRET_KEY",
            SecretType.OPENAI_KEY: "OPENAI_API_KEY",
            SecretType.ANTHROPIC_KEY: "ANTHROPIC_API_KEY",
            SecretType.GOOGLE_API_KEY: "GOOGLE_API_KEY",
            SecretType.DATABASE_URL: "DATABASE_URL",
            SecretType.JWT_SECRET: "JWT_SECRET",
        }

        if secret_type in type_to_env:
            return type_to_env[secret_type]

        # Use variable name if available
        if var_name:
            return var_name.upper()

        # Fallback based on type
        return secret_type.value.upper()

    def _apply_auto_fixes(self, repo_path: Path, report: SecretsReport) -> None:
        """Apply auto-fixes to detected secrets."""
        # Group secrets by file
        by_file: dict[str, list[DetectedSecret]] = {}
        for secret in report.secrets:
            if secret.file_path not in by_file:
                by_file[secret.file_path] = []
            by_file[secret.file_path].append(secret)

        for file_path, secrets in by_file.items():
            full_path = repo_path / file_path
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text()
                lines = content.split('\n')
                modified = False

                # Process secrets in reverse order to maintain line numbers
                for secret in sorted(secrets, key=lambda s: -s.line_number):
                    if secret.status != SecretStatus.DETECTED:
                        continue

                    line_idx = secret.line_number - 1
                    if line_idx >= len(lines):
                        continue

                    original_line = lines[line_idx]
                    fixed_line = self._fix_line(
                        original_line,
                        secret,
                        file_path,
                    )

                    if fixed_line and fixed_line != original_line:
                        lines[line_idx] = fixed_line
                        modified = True
                        secret.status = SecretStatus.AUTO_FIXED
                        secret.fix_applied = True
                        env_var = secret.suggested_env_var
                        secret.fix_description = f"Replaced with os.getenv('{env_var}')"
                        report.auto_fixed_count += 1

                        report.auto_fixes.append(
                            AutoFix(
                                file_path=file_path,
                                original_line=self._redact_line(original_line),
                                fixed_line=fixed_line,
                                env_var_name=secret.suggested_env_var,
                            )
                        )
                    else:
                        # Can't auto-fix, needs manual review
                        secret.status = SecretStatus.MANUAL_REQUIRED
                        report.manual_required_count += 1

                if modified:
                    full_path.write_text('\n'.join(lines))

            except Exception:
                # Mark all secrets in this file as manual
                for secret in secrets:
                    if secret.status == SecretStatus.DETECTED:
                        secret.status = SecretStatus.MANUAL_REQUIRED
                        report.manual_required_count += 1

    def _fix_line(
        self,
        line: str,
        secret: DetectedSecret,
        file_path: str,
    ) -> str | None:
        """Attempt to fix a line containing a secret."""
        env_var = secret.suggested_env_var

        # Determine file type for appropriate fix
        if file_path.endswith('.py'):
            return self._fix_python_line(line, secret, env_var)
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            return self._fix_js_line(line, secret, env_var)
        elif file_path.endswith(('.yaml', '.yml')):
            return self._fix_yaml_line(line, secret, env_var)

        return None

    def _fix_python_line(
        self,
        line: str,
        secret: DetectedSecret,
        env_var: str,
    ) -> str:
        """Fix a Python line with hardcoded secret."""
        # Pattern: VAR = "secret" -> VAR = os.getenv("ENV_VAR")
        pattern = r'(["\'])([^"\']+)\1'

        def replace_secret(match: re.Match) -> str:
            value = match.group(2)
            # Check if this is the secret we're looking for
            if self._looks_like_secret(value, secret):
                return f'os.getenv("{env_var}")'
            return match.group(0)

        fixed = re.sub(pattern, replace_secret, line)

        # If we made a change, we might need to add os import
        # (handled at file level in practice)
        return fixed

    def _fix_js_line(
        self,
        line: str,
        secret: DetectedSecret,
        env_var: str,
    ) -> str:
        """Fix a JavaScript/TypeScript line with hardcoded secret."""
        # Pattern: const VAR = "secret" -> const VAR = process.env.ENV_VAR
        pattern = r'(["\'])([^"\']+)\1'

        def replace_secret(match: re.Match) -> str:
            value = match.group(2)
            if self._looks_like_secret(value, secret):
                return f'process.env.{env_var}'
            return match.group(0)

        return re.sub(pattern, replace_secret, line)

    def _fix_yaml_line(
        self,
        line: str,
        secret: DetectedSecret,
        env_var: str,
    ) -> str:
        """Fix a YAML line with hardcoded secret."""
        # Replace value with environment variable reference
        # key: "secret" -> key: ${ENV_VAR}
        pattern = r':\s*(["\']?)([^"\'#\n]+)\1'

        def replace_secret(match: re.Match) -> str:
            value = match.group(2).strip()
            if self._looks_like_secret(value, secret):
                return f': ${{{env_var}}}'
            return match.group(0)

        return re.sub(pattern, replace_secret, line)

    def _looks_like_secret(self, value: str, secret: DetectedSecret) -> bool:
        """Check if a value matches the detected secret."""
        # Compare redacted patterns
        if len(value) < 8:
            return False

        redacted = self._redact_secret(value)
        return redacted == secret.redacted_value

    def _update_env_example(self, repo_path: Path, report: SecretsReport) -> None:
        """Update .env.example with new environment variables."""
        env_example = repo_path / ".env.example"

        existing_vars = set()
        existing_content = ""

        if env_example.exists():
            existing_content = env_example.read_text()
            # Extract existing variable names
            for line in existing_content.split('\n'):
                if '=' in line and not line.strip().startswith('#'):
                    var_name = line.split('=')[0].strip()
                    existing_vars.add(var_name)

        # Add new variables
        new_vars = []
        for fix in report.auto_fixes:
            if fix.env_var_name not in existing_vars:
                new_vars.append("# Added by SecretsScanAgent")
                placeholder = fix.env_var_name.lower().replace('_', '-')
                new_vars.append(f"{fix.env_var_name}=your-{placeholder}-here")
                existing_vars.add(fix.env_var_name)

        if new_vars:
            new_content = existing_content.rstrip() + "\n\n" + "\n".join(new_vars) + "\n"
            env_example.write_text(new_content)
            report.env_example_updated = True

    def _update_readme(self, repo_path: Path, report: SecretsReport) -> None:
        """Update README with configuration instructions."""
        readme = repo_path / "README.md"

        if not readme.exists():
            return

        content = readme.read_text()

        # Check if configuration section exists
        if "## Configuration" in content or "## Environment Variables" in content:
            return  # Already has config section

        # Add configuration section
        config_section = "\n\n## Configuration\n\n"
        config_section += "Copy `.env.example` to `.env` and configure the following:\n\n"

        for fix in report.auto_fixes:
            desc = fix.env_var_name.lower().replace('_', ' ')
            config_section += f"- `{fix.env_var_name}`: Your {desc}\n"

        # Insert after first heading or at end
        if "\n## " in content:
            # Insert before first ## section
            parts = content.split("\n## ", 1)
            content = parts[0] + config_section + "\n## " + parts[1]
        else:
            content = content.rstrip() + config_section

        readme.write_text(content)
        report.readme_updated = True

    def _add_summary(self, report: SecretsReport) -> None:
        """Add human-readable summary."""
        if report.total_detected == 0:
            report.summary = "No secrets detected"
            report.verify_notes = ["Secrets scan passed - no hardcoded credentials found"]
            return

        parts = [f"Detected {report.total_detected} potential secret(s)"]

        if report.auto_fixed_count:
            parts.append(f"{report.auto_fixed_count} auto-fixed")

        if report.manual_required_count:
            parts.append(f"{report.manual_required_count} need manual review")

        report.summary = " | ".join(parts)

        # Notes for VerifyAgent
        report.verify_notes = []

        if report.blocked:
            report.verify_notes.append(f"BLOCKED: {report.block_reason}")

        if report.critical_count:
            report.verify_notes.append(
                f"{report.critical_count} CRITICAL severity secrets detected"
            )

        if report.auto_fixed_count:
            report.verify_notes.append(
                f"{report.auto_fixed_count} secrets moved to environment variables"
            )

        if report.env_example_updated:
            report.verify_notes.append(".env.example updated with new variables")

        if report.readme_updated:
            report.verify_notes.append("README updated with configuration instructions")
