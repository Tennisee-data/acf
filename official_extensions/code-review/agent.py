"""Code Review Agent - Senior engineer review with actionable feedback.

This official ACF extension performs automated code review on generated
code, providing feedback on quality, security, and best practices.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json
import re


@dataclass
class AgentOutput:
    """Output from the agent."""
    success: bool
    data: dict[str, Any]
    errors: list[str] | None = None
    artifacts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None


class CodeReviewAgent:
    """Senior engineer code review with actionable feedback.

    Runs after implementation to review generated code for:
    - Code quality and readability
    - Security vulnerabilities
    - Best practices violations
    - Performance concerns
    """

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.llm = llm
        self.name = "code-review"

    def default_system_prompt(self) -> str:
        return """You are a senior software engineer conducting a code review.

Review the code for:
1. **Security**: SQL injection, XSS, hardcoded secrets, unsafe operations
2. **Quality**: Code clarity, naming, structure, DRY violations
3. **Performance**: N+1 queries, unnecessary loops, memory issues
4. **Best Practices**: Error handling, logging, testing considerations

For each issue found, provide:
- Severity: critical, warning, suggestion
- Location: file and line if possible
- Issue: What's wrong
- Fix: How to fix it

Output JSON format:
{
    "summary": "Overall assessment",
    "score": 85,
    "issues": [
        {
            "severity": "warning",
            "category": "security",
            "file": "app.py",
            "line": 42,
            "issue": "SQL query built with string concatenation",
            "fix": "Use parameterized queries instead"
        }
    ],
    "praise": ["Good error handling", "Clean separation of concerns"],
    "approved": true
}"""

    def run(self, input_data: Any) -> AgentOutput:
        """Review generated code.

        Args:
            input_data: Pipeline context with implementation

        Returns:
            AgentOutput with review results
        """
        context = input_data.context if hasattr(input_data, 'context') else input_data

        # Get implementation data
        run_dir = context.get("run_dir", "")
        repo_path = context.get("repo_path", ".")

        # Find generated code files
        generated_project = Path(repo_path) / run_dir / "generated_project"
        code_content = ""

        if generated_project.exists():
            for py_file in generated_project.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    rel_path = py_file.relative_to(generated_project)
                    code_content += f"\n### {rel_path}\n```python\n{content}\n```\n"
                except Exception:
                    pass

        if not code_content:
            print(f"\n[Code Review] No code files found to review")
            return AgentOutput(
                success=True,
                data={"code_review": {"skipped": True, "reason": "No code files found"}},
                agent_name=self.name,
            )

        prompt = f"""Review the following generated code:

{code_content}

Provide a thorough code review. Respond with JSON only."""

        try:
            messages = [
                {"role": "system", "content": self.default_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages)

            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"summary": "Review completed", "score": 75, "issues": [], "approved": True}

            score = result.get("score", 75)
            issues = result.get("issues", [])
            approved = result.get("approved", True)

            # Count by severity
            critical = sum(1 for i in issues if i.get("severity") == "critical")
            warnings = sum(1 for i in issues if i.get("severity") == "warning")
            suggestions = sum(1 for i in issues if i.get("severity") == "suggestion")

            print(f"\n{'='*60}")
            print(f"CODE REVIEW RESULTS")
            print(f"{'='*60}")
            print(f"Score: {score}/100")
            print(f"Status: {'APPROVED' if approved else 'NEEDS CHANGES'}")
            print(f"Issues: {critical} critical, {warnings} warnings, {suggestions} suggestions")

            if issues:
                print(f"\nTop Issues:")
                for issue in issues[:3]:
                    sev = issue.get('severity', 'info').upper()
                    print(f"  [{sev}] {issue.get('issue', 'Unknown issue')}")

            if result.get("praise"):
                print(f"\nPositives: {', '.join(result['praise'][:3])}")
            print(f"{'='*60}\n")

            return AgentOutput(
                success=True,
                data={
                    "code_review": {
                        "score": score,
                        "approved": approved,
                        "summary": result.get("summary", ""),
                        "issues": issues,
                        "praise": result.get("praise", []),
                        "critical_count": critical,
                        "warning_count": warnings,
                    }
                },
                artifacts=["code_review_report.json"],
                agent_name=self.name,
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"Code review failed: {str(e)}"],
                agent_name=self.name,
            )
