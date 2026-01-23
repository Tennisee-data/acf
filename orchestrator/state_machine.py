"""State machine implementation for pipeline orchestration."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from schemas.pipeline_state import (
    PipelineState,
    RunStatus,
    Stage,
    StageResult,
    StageStatus,
)


@dataclass
class Transition:
    """Defines a valid state transition."""

    from_stage: Stage
    to_stage: Stage
    condition: Callable[[PipelineState], bool] | None = None
    on_transition: Callable[[PipelineState], None] | None = None


class StateMachine:
    """State machine for pipeline execution.

    Manages:
    - Valid stage transitions
    - State persistence
    - Checkpoint handling
    - Retry logic
    """

    # Define valid transitions
    TRANSITIONS: list[Transition] = [
        # Happy path
        Transition(Stage.INIT, Stage.SPEC),
        Transition(Stage.SPEC, Stage.CONTEXT),
        Transition(Stage.CONTEXT, Stage.DESIGN),
        Transition(Stage.DESIGN, Stage.DESIGN_APPROVAL),
        Transition(Stage.DESIGN_APPROVAL, Stage.API_CONTRACT),
        Transition(Stage.API_CONTRACT, Stage.IMPLEMENTATION),
        Transition(Stage.IMPLEMENTATION, Stage.TESTING),
        Transition(Stage.TESTING, Stage.COVERAGE),
        Transition(Stage.COVERAGE, Stage.SECRETS_SCAN),
        Transition(Stage.SECRETS_SCAN, Stage.DEPENDENCY_AUDIT),
        Transition(Stage.DEPENDENCY_AUDIT, Stage.DOCKER_BUILD),
        Transition(Stage.DOCKER_BUILD, Stage.ROLLBACK_STRATEGY),
        Transition(Stage.ROLLBACK_STRATEGY, Stage.OBSERVABILITY),
        Transition(Stage.OBSERVABILITY, Stage.CONFIG),
        Transition(Stage.CONFIG, Stage.DOCS),
        Transition(Stage.DOCS, Stage.CODE_REVIEW),
        Transition(Stage.CODE_REVIEW, Stage.POLICY),
        Transition(Stage.POLICY, Stage.VERIFICATION),
        Transition(Stage.VERIFICATION, Stage.PR_PACKAGE),
        Transition(Stage.PR_PACKAGE, Stage.FINAL_APPROVAL),
        Transition(Stage.FINAL_APPROVAL, Stage.DEPLOY),
        Transition(Stage.DEPLOY, Stage.DONE),
        # Skip API contract if not needed
        Transition(Stage.DESIGN_APPROVAL, Stage.IMPLEMENTATION),
        # Skip coverage if not needed
        Transition(Stage.TESTING, Stage.SECRETS_SCAN),
        Transition(Stage.TESTING, Stage.DOCKER_BUILD),
        # Skip secrets scan if not needed
        Transition(Stage.COVERAGE, Stage.DEPENDENCY_AUDIT),
        Transition(Stage.COVERAGE, Stage.DOCKER_BUILD),
        Transition(Stage.TESTING, Stage.DEPENDENCY_AUDIT),
        # Skip dependency audit if not needed
        Transition(Stage.SECRETS_SCAN, Stage.DOCKER_BUILD),
        # Skip Docker build if not needed
        Transition(Stage.DEPENDENCY_AUDIT, Stage.ROLLBACK_STRATEGY),
        Transition(Stage.SECRETS_SCAN, Stage.ROLLBACK_STRATEGY),
        Transition(Stage.COVERAGE, Stage.ROLLBACK_STRATEGY),
        Transition(Stage.TESTING, Stage.ROLLBACK_STRATEGY),
        # Skip rollback strategy if not needed
        Transition(Stage.DOCKER_BUILD, Stage.OBSERVABILITY),
        Transition(Stage.DEPENDENCY_AUDIT, Stage.OBSERVABILITY),
        Transition(Stage.SECRETS_SCAN, Stage.OBSERVABILITY),
        Transition(Stage.COVERAGE, Stage.OBSERVABILITY),
        Transition(Stage.TESTING, Stage.OBSERVABILITY),
        # Skip observability if not needed
        Transition(Stage.DOCKER_BUILD, Stage.CONFIG),
        Transition(Stage.DOCKER_BUILD, Stage.VERIFICATION),
        Transition(Stage.DEPENDENCY_AUDIT, Stage.CONFIG),
        Transition(Stage.DEPENDENCY_AUDIT, Stage.VERIFICATION),
        Transition(Stage.SECRETS_SCAN, Stage.CONFIG),
        Transition(Stage.SECRETS_SCAN, Stage.VERIFICATION),
        Transition(Stage.COVERAGE, Stage.CONFIG),
        Transition(Stage.COVERAGE, Stage.VERIFICATION),
        Transition(Stage.TESTING, Stage.CONFIG),
        Transition(Stage.TESTING, Stage.VERIFICATION),
        # Skip config if not needed
        Transition(Stage.OBSERVABILITY, Stage.DOCS),
        Transition(Stage.OBSERVABILITY, Stage.VERIFICATION),
        # Skip docs if not needed
        Transition(Stage.CONFIG, Stage.CODE_REVIEW),
        Transition(Stage.CONFIG, Stage.VERIFICATION),
        # Skip code review if not needed
        Transition(Stage.DOCS, Stage.POLICY),
        Transition(Stage.DOCS, Stage.VERIFICATION),
        # Skip policy if not needed
        Transition(Stage.CODE_REVIEW, Stage.VERIFICATION),
        # Skip PR package if not needed
        Transition(Stage.VERIFICATION, Stage.FINAL_APPROVAL),
        # Skip deploy for dry runs
        Transition(Stage.FINAL_APPROVAL, Stage.DONE),
        # Coverage iteration: loop back to testing
        Transition(Stage.COVERAGE, Stage.TESTING),
        # Retry failed stages
        Transition(Stage.SPEC, Stage.SPEC),
        Transition(Stage.CONTEXT, Stage.CONTEXT),
        Transition(Stage.DESIGN, Stage.DESIGN),
        Transition(Stage.API_CONTRACT, Stage.API_CONTRACT),
        Transition(Stage.IMPLEMENTATION, Stage.IMPLEMENTATION),
        Transition(Stage.TESTING, Stage.TESTING),
        Transition(Stage.COVERAGE, Stage.COVERAGE),
        Transition(Stage.SECRETS_SCAN, Stage.SECRETS_SCAN),
        Transition(Stage.DEPENDENCY_AUDIT, Stage.DEPENDENCY_AUDIT),
        Transition(Stage.ROLLBACK_STRATEGY, Stage.ROLLBACK_STRATEGY),
        Transition(Stage.OBSERVABILITY, Stage.OBSERVABILITY),
        Transition(Stage.CONFIG, Stage.CONFIG),
        Transition(Stage.DOCS, Stage.DOCS),
        Transition(Stage.CODE_REVIEW, Stage.CODE_REVIEW),
        Transition(Stage.POLICY, Stage.POLICY),
        Transition(Stage.PR_PACKAGE, Stage.PR_PACKAGE),
    ]

    # Stages that require human approval
    APPROVAL_STAGES = {Stage.DESIGN_APPROVAL, Stage.FINAL_APPROVAL}

    def __init__(self, state: PipelineState, artifacts_dir: Path) -> None:
        """Initialize state machine.

        Args:
            state: Initial pipeline state
            artifacts_dir: Directory to store state and artifacts
        """
        self.state = state
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Build transition map for quick lookup
        self._transition_map: dict[Stage, list[Stage]] = {}
        for t in self.TRANSITIONS:
            if t.from_stage not in self._transition_map:
                self._transition_map[t.from_stage] = []
            self._transition_map[t.from_stage].append(t.to_stage)

    def can_transition(self, to_stage: Stage) -> bool:
        """Check if transition to target stage is valid.

        Args:
            to_stage: Target stage

        Returns:
            True if transition is valid
        """
        current = self.state.current_stage
        valid_targets = self._transition_map.get(current, [])
        return to_stage in valid_targets

    def transition(self, to_stage: Stage) -> bool:
        """Attempt to transition to a new stage.

        Args:
            to_stage: Target stage

        Returns:
            True if transition succeeded
        """
        if not self.can_transition(to_stage):
            return False

        # Find the transition definition
        transition = None
        for t in self.TRANSITIONS:
            if t.from_stage == self.state.current_stage and t.to_stage == to_stage:
                transition = t
                break

        # Check condition if any
        if transition and transition.condition:
            if not transition.condition(self.state):
                return False

        # Execute transition callback if any
        if transition and transition.on_transition:
            transition.on_transition(self.state)

        # Update state
        self.state.current_stage = to_stage
        self.state.mark_stage_started(to_stage)

        # Note: Approval stages are handled by the handler, not here
        # The handler will call mark_awaiting_approval if needed

        # Persist state
        self.save_state()

        return True

    def complete_stage(
        self,
        artifact_path: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Mark current stage as completed.

        Args:
            artifact_path: Path to output artifact
            summary: Brief summary of stage output
        """
        self.state.mark_stage_completed(
            self.state.current_stage,
            artifact_path=artifact_path,
            summary=summary,
        )
        self.save_state()

    def fail_stage(self, error: str) -> None:
        """Mark current stage as failed.

        Args:
            error: Error message
        """
        self.state.mark_stage_failed(self.state.current_stage, error)
        self.state.status = RunStatus.FAILED
        self.save_state()

    def approve_stage(self, approved_by: str = "user", notes: str | None = None) -> None:
        """Approve current stage (for approval checkpoints).

        Args:
            approved_by: Who approved
            notes: Approval notes
        """
        stage_result = self.state.get_stage_result(self.state.current_stage)
        if stage_result:
            stage_result.status = StageStatus.APPROVED
            stage_result.approved_by = approved_by
            stage_result.approval_notes = notes
            stage_result.completed_at = datetime.now()

        self.state.status = RunStatus.RUNNING
        self.state.user_decisions[self.state.current_stage.value] = "approved"
        self.save_state()

    def reject_stage(self, reason: str, rejected_by: str = "user") -> None:
        """Reject current stage (for approval checkpoints).

        Args:
            reason: Rejection reason
            rejected_by: Who rejected
        """
        stage_result = self.state.get_stage_result(self.state.current_stage)
        if stage_result:
            stage_result.status = StageStatus.REJECTED
            stage_result.approved_by = rejected_by
            stage_result.approval_notes = reason
            stage_result.completed_at = datetime.now()

        self.state.status = RunStatus.CANCELLED
        self.state.user_decisions[self.state.current_stage.value] = f"rejected: {reason}"
        self.save_state()

    def get_valid_next_stages(self) -> list[Stage]:
        """Get list of valid next stages from current state.

        Returns:
            List of valid target stages
        """
        return self._transition_map.get(self.state.current_stage, [])

    def is_approval_stage(self) -> bool:
        """Check if current stage is an approval checkpoint.

        Returns:
            True if current stage requires approval
        """
        return self.state.current_stage in self.APPROVAL_STAGES

    def is_completed(self) -> bool:
        """Check if pipeline is completed.

        Returns:
            True if pipeline reached DONE stage
        """
        return self.state.current_stage == Stage.DONE

    def is_failed(self) -> bool:
        """Check if pipeline has failed.

        Returns:
            True if pipeline status is FAILED
        """
        return self.state.status == RunStatus.FAILED

    def save_state(self) -> Path:
        """Persist state to disk.

        Returns:
            Path to state file
        """
        state_file = self.artifacts_dir / "state.json"
        state_data = self.state.model_dump(mode="json")

        # Handle datetime serialization
        def serialize_datetime(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(v) for v in obj]
            return obj

        state_data = serialize_datetime(state_data)

        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        return state_file

    @classmethod
    def load_state(cls, artifacts_dir: Path) -> "StateMachine":
        """Load state machine from disk.

        Args:
            artifacts_dir: Directory containing state.json

        Returns:
            StateMachine instance with loaded state
        """
        state_file = artifacts_dir / "state.json"

        with open(state_file) as f:
            state_data = json.load(f)

        # Parse datetime strings back
        def parse_datetime(obj: Any) -> Any:
            if isinstance(obj, str):
                try:
                    return datetime.fromisoformat(obj)
                except ValueError:
                    return obj
            elif isinstance(obj, dict):
                return {k: parse_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [parse_datetime(v) for v in obj]
            return obj

        # Handle stage results reconstruction
        if "stages" in state_data:
            for stage_name, result_data in state_data["stages"].items():
                if isinstance(result_data, dict):
                    result_data["stage"] = Stage(result_data.get("stage", stage_name))
                    result_data["status"] = StageStatus(result_data.get("status", "pending"))

        state = PipelineState(**state_data)
        return cls(state, artifacts_dir)

    def get_progress_summary(self) -> dict[str, Any]:
        """Get a summary of pipeline progress.

        Returns:
            Progress summary dict
        """
        completed = sum(
            1
            for s in self.state.stages.values()
            if s.status in (StageStatus.COMPLETED, StageStatus.APPROVED)
        )
        total = len(Stage) - 2  # Exclude INIT and DONE

        return {
            "run_id": self.state.run_id,
            "status": self.state.status.value,
            "current_stage": self.state.current_stage.value,
            "progress": f"{completed}/{total}",
            "progress_percent": round(completed / total * 100) if total > 0 else 0,
            "stages": {
                name: result.status.value for name, result in self.state.stages.items()
            },
        }
