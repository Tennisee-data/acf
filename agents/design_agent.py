"""Design Agent for proposing architecture and implementation plans."""

from pathlib import Path

from llm_backend import LLMBackend
from schemas.context_report import ContextReport
from schemas.design_proposal import (
    ChangeType,
    DataFlowStep,
    Dependency,
    DesignOption,
    DesignProposal,
    FileChange,
)
from schemas.feature_spec import FeatureSpec
from utils.json_repair import parse_llm_json

from .base import AgentInput, AgentOutput, BaseAgent
from .prompts import CODE_PRINCIPLES, DESIGN_PRINCIPLES


SYSTEM_PROMPT = f"""{CODE_PRINCIPLES}

{DESIGN_PRINCIPLES}

## Role

You are a senior software architect creating implementation designs.

Given a feature specification and codebase context, create a detailed design proposal.

Output a JSON object with EXACTLY this structure:

{{"summary":"One paragraph describing the overall design approach","architecture_sketch":"ASCII diagram showing component relationships","options":[{{"name":"Option Name","description":"Detailed description","pros":["advantage 1"],"cons":["disadvantage 1"],"estimated_effort":"low|medium|high","recommended":true}}],"chosen_approach":"Name of recommended option","rationale":"Why this approach was chosen","data_flow":[{{"order":1,"component":"ComponentName","action":"What happens","data_in":"input","data_out":"output"}}],"file_changes":[{{"path":"path/to/file.py","change_type":"create|modify|delete","description":"What changes","estimated_lines":50}}],"new_dependencies":[{{"name":"package-name","version":">=1.0","reason":"why needed","alternatives":["alt1"]}}],"patterns_to_follow":["pattern 1"],"patterns_to_avoid":["anti-pattern 1"],"risks":["risk 1"],"mitigations":["mitigation 1"],"testing_strategy":"How to test this implementation","requires_approval":true,"approval_notes":"Key decisions requiring human review"}}

Fill in with actual design based on the feature and context. Output ONLY the JSON object."""


class DesignAgent(BaseAgent):
    """Agent for proposing architecture and implementation plans.

    Takes a feature specification and context report, then produces:
    - Design options analysis
    - Recommended approach with rationale
    - File changes plan
    - Data flow description
    - Testing strategy
    - Risk assessment
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize DesignAgent.

        Args:
            llm: LLM backend for inference
            system_prompt: Override default system prompt
        """
        super().__init__(llm, system_prompt)

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Create design proposal from feature spec and context.

        Args:
            input_data: Must contain 'feature_spec' and 'context_report' in context

        Returns:
            AgentOutput with DesignProposal data
        """
        context = input_data.context

        # Extract feature spec
        feature_spec_data = context.get("feature_spec", {})
        if isinstance(feature_spec_data, FeatureSpec):
            feature_spec = feature_spec_data
        elif isinstance(feature_spec_data, dict) and feature_spec_data:
            try:
                feature_spec = FeatureSpec(**feature_spec_data)
            except Exception:
                feature_spec = None
        else:
            feature_spec = None

        # Extract context report
        context_report_data = context.get("context_report", {})
        if isinstance(context_report_data, ContextReport):
            context_report = context_report_data
        elif isinstance(context_report_data, dict) and context_report_data:
            try:
                context_report = ContextReport(**context_report_data)
            except Exception:
                context_report = None
        else:
            context_report = None

        feature_description = context.get("feature_description", "")
        feature_id = feature_spec.id if feature_spec else context.get("feature_id", "FEAT-000")

        # Extract workplan (from DecompositionAgent)
        workplan = context.get("workplan", {})

        if not feature_spec and not feature_description:
            return AgentOutput(
                success=False,
                data={},
                errors=["No feature specification or description provided"],
            )

        # Build the prompt
        user_message = self._build_prompt(
            feature_spec=feature_spec,
            context_report=context_report,
            feature_description=feature_description,
            workplan=workplan,
        )

        try:
            # Call LLM with adequate max_tokens for detailed design output
            response = self._chat(user_message, temperature=0.4, max_tokens=8192)

            # Parse the response
            design_data = self._parse_response(response)

            # Add feature ID reference
            design_data["feature_id"] = feature_id

            # Validate and create DesignProposal
            design_proposal = self._create_design_proposal(design_data)

            return AgentOutput(
                success=True,
                data=design_proposal.model_dump(),
                artifacts=["design_proposal.json", "design_proposal.md"],
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={"raw_response": response if "response" in dir() else ""},
                errors=[f"DesignAgent error: {str(e)}"],
            )

    def _build_prompt(
        self,
        feature_spec: FeatureSpec | None,
        context_report: ContextReport | None,
        feature_description: str,
        workplan: dict | None = None,
    ) -> str:
        """Build the user prompt for design generation."""
        parts = []

        # Feature specification
        parts.append("## Feature Specification")
        if feature_spec:
            parts.append(f"**Title:** {feature_spec.title}")
            parts.append(f"**Description:** {feature_spec.original_description}")
            parts.append(f"**User Story:** {feature_spec.user_story}")

            if feature_spec.acceptance_criteria:
                parts.append("\n**Acceptance Criteria:**")
                for ac in feature_spec.acceptance_criteria:
                    parts.append(f"- [{ac.id}] {ac.description}")

            if feature_spec.constraints:
                parts.append("\n**Constraints:**")
                for c in feature_spec.constraints:
                    parts.append(f"- [{c.type}] {c.description}")

            if feature_spec.non_functional_requirements:
                parts.append("\n**Non-Functional Requirements:**")
                for nfr in feature_spec.non_functional_requirements:
                    parts.append(f"- [{nfr.category}] {nfr.requirement}")

            if feature_spec.domains:
                domains = ", ".join([d.value for d in feature_spec.domains])
                parts.append(f"\n**Domains:** {domains}")
        else:
            parts.append(f"**Description:** {feature_description}")

        # Context report
        if context_report:
            parts.append("\n## Codebase Context")

            if context_report.repo_structure:
                rs = context_report.repo_structure
                parts.append(f"**Framework:** {rs.framework or 'Unknown'}")
                parts.append(f"**Language:** {rs.language or 'Unknown'}")
                if rs.entry_points:
                    parts.append(f"**Entry Points:** {', '.join(rs.entry_points)}")

            if context_report.relevant_files:
                parts.append("\n**Relevant Files:**")
                for rf in context_report.relevant_files[:10]:
                    parts.append(f"- {rf.path}: {rf.purpose}")

            if context_report.existing_patterns:
                parts.append("\n**Existing Patterns:**")
                for p in context_report.existing_patterns:
                    rec = " (recommended)" if p.recommended else ""
                    parts.append(f"- {p.name}: {p.description}{rec}")

            if context_report.mental_model:
                parts.append(f"\n**Integration Notes:** {context_report.mental_model}")

            if context_report.integration_risks:
                parts.append("\n**Known Risks:**")
                for risk in context_report.integration_risks:
                    parts.append(f"- {risk}")

        # Workplan tasks (from decomposition)
        if workplan and workplan.get("tasks"):
            tasks = workplan["tasks"]
            parts.append("\n## Implementation Tasks (ALL MUST BE ADDRESSED)")
            parts.append(f"The feature has been decomposed into {len(tasks)} tasks.")
            parts.append("Your design MUST cover ALL of these tasks:\n")
            for i, task in enumerate(tasks, 1):
                task_id = task.get("id", f"TASK-{i:03d}")
                title = task.get("title", "Untitled")
                desc = task.get("description", "")
                category = task.get("category", "")
                parts.append(f"**{task_id}: {title}** [{category}]")
                if desc:
                    parts.append(f"  {desc}")
                if task.get("target_files"):
                    parts.append(f"  Target files: {', '.join(task['target_files'])}")
                parts.append("")

            if workplan.get("execution_order"):
                parts.append(f"**Execution order:** {' → '.join(workplan['execution_order'])}")
                parts.append("")

        parts.append("\n## Task")
        parts.append("Create a detailed design proposal for implementing this feature.")
        if workplan and workplan.get("tasks"):
            parts.append(f"IMPORTANT: Your design must address ALL {len(workplan['tasks'])} tasks listed above.")
            parts.append("Include file_changes entries for EACH task in the workplan.")
        parts.append("Consider multiple approaches, recommend the best one, and provide a complete implementation plan.")
        parts.append("\nRespond with ONLY the JSON object.")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response JSON with repair."""
        import json

        result = parse_llm_json(response, default=None)

        if result is None:
            raise json.JSONDecodeError(
                "Could not parse JSON from response",
                response,
                0,
            )

        return result

    def _create_design_proposal(self, data: dict) -> DesignProposal:
        """Create validated DesignProposal from parsed data."""
        # Parse options
        options = []
        for opt in data.get("options", []):
            options.append(
                DesignOption(
                    name=opt.get("name", "Unnamed Option"),
                    description=opt.get("description", ""),
                    pros=opt.get("pros", []),
                    cons=opt.get("cons", []),
                    estimated_effort=opt.get("estimated_effort"),
                    recommended=opt.get("recommended", False),
                )
            )

        # Parse data flow
        data_flow = []
        for step in data.get("data_flow", []):
            data_flow.append(
                DataFlowStep(
                    order=step.get("order", len(data_flow) + 1),
                    component=step.get("component", "Unknown"),
                    action=step.get("action", ""),
                    data_in=step.get("data_in"),
                    data_out=step.get("data_out"),
                )
            )

        # Parse file changes
        file_changes = []
        for fc in data.get("file_changes", []):
            change_type_str = fc.get("change_type", "modify")
            try:
                change_type = ChangeType(change_type_str)
            except ValueError:
                change_type = ChangeType.MODIFY

            file_changes.append(
                FileChange(
                    path=fc.get("path", "unknown"),
                    change_type=change_type,
                    description=fc.get("description", ""),
                    estimated_lines=fc.get("estimated_lines"),
                )
            )

        # Parse dependencies
        new_dependencies = []
        for dep in data.get("new_dependencies", []):
            new_dependencies.append(
                Dependency(
                    name=dep.get("name", "unknown"),
                    version=dep.get("version"),
                    reason=dep.get("reason", ""),
                    alternatives=dep.get("alternatives", []),
                )
            )

        return DesignProposal(
            feature_id=data.get("feature_id", "FEAT-000"),
            summary=data.get("summary", "No summary provided"),
            architecture_sketch=data.get("architecture_sketch", "No diagram provided"),
            options=options,
            chosen_approach=data.get("chosen_approach", "Default approach"),
            rationale=data.get("rationale", "No rationale provided"),
            data_flow=data_flow,
            file_changes=file_changes,
            new_dependencies=new_dependencies,
            deprecation_notes=data.get("deprecation_notes", []),
            patterns_to_follow=data.get("patterns_to_follow", []),
            patterns_to_avoid=data.get("patterns_to_avoid", []),
            risks=data.get("risks", []),
            mitigations=data.get("mitigations", []),
            testing_strategy=data.get("testing_strategy", "Standard testing approach"),
            requires_approval=data.get("requires_approval", True),
            approval_notes=data.get("approval_notes"),
        )

    def generate_markdown_report(self, proposal: DesignProposal) -> str:
        """Generate a human-readable markdown report from the design proposal."""
        lines = [
            f"# Design Proposal: {proposal.feature_id}",
            "",
            "## Summary",
            "",
            proposal.summary,
            "",
            "## Architecture",
            "",
            "```",
            proposal.architecture_sketch,
            "```",
            "",
        ]

        # Options
        if proposal.options:
            lines.extend(["## Design Options", ""])
            for opt in proposal.options:
                rec = " **(Recommended)**" if opt.recommended else ""
                lines.append(f"### {opt.name}{rec}")
                lines.append("")
                lines.append(opt.description)
                lines.append("")
                if opt.pros:
                    lines.append("**Pros:**")
                    for pro in opt.pros:
                        lines.append(f"- {pro}")
                    lines.append("")
                if opt.cons:
                    lines.append("**Cons:**")
                    for con in opt.cons:
                        lines.append(f"- {con}")
                    lines.append("")
                if opt.estimated_effort:
                    lines.append(f"**Effort:** {opt.estimated_effort}")
                    lines.append("")

        # Chosen approach
        lines.extend([
            "## Chosen Approach",
            "",
            f"**{proposal.chosen_approach}**",
            "",
            proposal.rationale,
            "",
        ])

        # Data flow
        if proposal.data_flow:
            lines.extend(["## Data Flow", ""])
            for step in sorted(proposal.data_flow, key=lambda x: x.order):
                lines.append(f"{step.order}. **{step.component}**: {step.action}")
                if step.data_in:
                    lines.append(f"   - Input: {step.data_in}")
                if step.data_out:
                    lines.append(f"   - Output: {step.data_out}")
            lines.append("")

        # File changes
        if proposal.file_changes:
            lines.extend(["## File Changes", ""])
            lines.append("| File | Action | Description |")
            lines.append("|------|--------|-------------|")
            for fc in proposal.file_changes:
                lines.append(f"| `{fc.path}` | {fc.change_type.value} | {fc.description} |")
            lines.append("")

        # Dependencies
        if proposal.new_dependencies:
            lines.extend(["## New Dependencies", ""])
            for dep in proposal.new_dependencies:
                version = f" ({dep.version})" if dep.version else ""
                lines.append(f"- **{dep.name}**{version}: {dep.reason}")
            lines.append("")

        # Patterns
        if proposal.patterns_to_follow:
            lines.extend(["## Patterns to Follow", ""])
            for p in proposal.patterns_to_follow:
                lines.append(f"- {p}")
            lines.append("")

        if proposal.patterns_to_avoid:
            lines.extend(["## Anti-Patterns to Avoid", ""])
            for p in proposal.patterns_to_avoid:
                lines.append(f"- {p}")
            lines.append("")

        # Risks
        if proposal.risks:
            lines.extend(["## Risks & Mitigations", ""])
            for i, risk in enumerate(proposal.risks):
                mitigation = proposal.mitigations[i] if i < len(proposal.mitigations) else "No mitigation specified"
                lines.append(f"- **Risk:** {risk}")
                lines.append(f"  - **Mitigation:** {mitigation}")
            lines.append("")

        # Testing
        lines.extend([
            "## Testing Strategy",
            "",
            proposal.testing_strategy,
            "",
        ])

        # Approval
        if proposal.requires_approval:
            lines.extend([
                "---",
                "",
                "**This design requires approval before implementation.**",
                "",
            ])
            if proposal.approval_notes:
                lines.append(f"*Reviewer Notes:* {proposal.approval_notes}")
                lines.append("")

        return "\n".join(lines)

    def propose_design(
        self,
        feature_spec: FeatureSpec,
        context_report: ContextReport | None = None,
    ) -> DesignProposal | None:
        """Convenience method to propose a design.

        Args:
            feature_spec: The feature specification
            context_report: Optional context report

        Returns:
            DesignProposal if successful, None otherwise
        """
        input_data = AgentInput(
            context={
                "feature_spec": feature_spec.model_dump() if feature_spec else {},
                "context_report": context_report.model_dump() if context_report else {},
                "feature_description": feature_spec.original_description if feature_spec else "",
                "feature_id": feature_spec.id if feature_spec else "FEAT-000",
            }
        )

        output = self.run(input_data)

        if output.success:
            return DesignProposal(**output.data)
        return None
