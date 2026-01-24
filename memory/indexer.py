"""Run indexer for extracting memories from completed runs.

Extracts and indexes:
- Feature descriptions and specifications
- Design decisions and rationale
- Implementation approaches
- Errors and their fixes
- Technical debt notes
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

try:
    from rag.embeddings import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None  # Semantic features not installed

from schemas.memory import (
    ErrorPattern,
    MemoryType,
    RunMemoryEntry,
    RunOutcome,
)
from schemas.pipeline_state import PipelineState, RunStatus, StageStatus

from .store import MemoryStore

logger = logging.getLogger(__name__)


class RunIndexer:
    """Index completed runs into the memory store.

    Extracts meaningful memories from run artifacts:
    - Feature specs → FEATURE memories
    - Design proposals → DESIGN_DECISION memories
    - Implementation notes → IMPLEMENTATION memories
    - Failed stages → ERROR_FIX memories
    - Code review issues → TECH_DEBT memories
    """

    def __init__(
        self,
        store: MemoryStore,
        embeddings: "OllamaEmbeddings | None" = None,
        embedding_model: str = "nomic-embed-text",
    ):
        """Initialize run indexer.

        Args:
            store: Memory store to index into
            embeddings: Embedding generator (creates one if not provided)
            embedding_model: Model name for embeddings
        """
        self.store = store
        if embeddings:
            self.embeddings = embeddings
        elif OllamaEmbeddings:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
        else:
            self.embeddings = None

    def _map_run_status_to_outcome(self, status: RunStatus) -> RunOutcome:
        """Map pipeline run status to memory outcome.

        Args:
            status: Pipeline run status

        Returns:
            Corresponding memory outcome
        """
        status_map = {
            RunStatus.COMPLETED: RunOutcome.SUCCESS,
            RunStatus.FAILED: RunOutcome.FAILED,
            RunStatus.CANCELLED: RunOutcome.CANCELLED,
            RunStatus.PAUSED: RunOutcome.PARTIAL,
            RunStatus.RUNNING: RunOutcome.PARTIAL,
            RunStatus.PENDING: RunOutcome.PARTIAL,
        }
        return status_map.get(status, RunOutcome.PARTIAL)

    def _generate_id(self) -> str:
        """Generate a unique memory ID.

        Returns:
            Unique ID string
        """
        return f"mem-{uuid.uuid4().hex[:12]}"

    def _read_json_artifact(self, path: Path) -> dict | None:
        """Read a JSON artifact file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON or None if error
        """
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to read JSON artifact %s: %s", path, e)
        return None

    def _read_text_artifact(self, path: Path) -> str | None:
        """Read a text artifact file.

        Args:
            path: Path to text file

        Returns:
            File content or None if error
        """
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read text artifact %s: %s", path, e)
        return None

    def _extract_feature_memory(
        self,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> RunMemoryEntry | None:
        """Extract feature description memory.

        Args:
            state: Pipeline state
            artifacts_dir: Path to artifacts directory

        Returns:
            Feature memory entry or None
        """
        # Try feature_spec.json first
        feature_spec = self._read_json_artifact(artifacts_dir / "feature_spec.json")
        if feature_spec:
            content_parts = [
                f"Feature: {feature_spec.get('title', state.feature_description)}",
            ]

            if feature_spec.get("user_story"):
                content_parts.append(f"User Story: {feature_spec['user_story']}")

            if feature_spec.get("domains"):
                content_parts.append(f"Domains: {', '.join(feature_spec['domains'])}")

            if feature_spec.get("estimated_complexity"):
                content_parts.append(f"Complexity: {feature_spec['estimated_complexity']}")

            content = "\n".join(content_parts)
        else:
            # Fall back to basic description
            content = f"Feature: {state.feature_description}"

        return RunMemoryEntry(
            id=self._generate_id(),
            run_id=state.run_id,
            memory_type=MemoryType.FEATURE,
            content=content,
            outcome=self._map_run_status_to_outcome(state.status),
            stage="spec",
            metadata={
                "feature_description": state.feature_description,
                "domains": feature_spec.get("domains", []) if feature_spec else [],
                "complexity": feature_spec.get("estimated_complexity") if feature_spec else None,
            },
        )

    def _extract_design_memory(
        self,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> RunMemoryEntry | None:
        """Extract design decision memory.

        Args:
            state: Pipeline state
            artifacts_dir: Path to artifacts directory

        Returns:
            Design memory entry or None
        """
        design_content = self._read_text_artifact(artifacts_dir / "design_proposal.md")
        if not design_content or len(design_content.strip()) < 50:
            return None

        # Clean up the content
        # Remove placeholder text
        if "TBD" in design_content and "to be determined" in design_content.lower():
            return None

        content = f"Design Decision for '{state.feature_description}':\n{design_content[:2000]}"

        return RunMemoryEntry(
            id=self._generate_id(),
            run_id=state.run_id,
            memory_type=MemoryType.DESIGN_DECISION,
            content=content,
            outcome=self._map_run_status_to_outcome(state.status),
            stage="design",
            metadata={
                "feature_description": state.feature_description,
            },
        )

    def _extract_implementation_memory(
        self,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> RunMemoryEntry | None:
        """Extract implementation approach memory.

        Args:
            state: Pipeline state
            artifacts_dir: Path to artifacts directory

        Returns:
            Implementation memory entry or None
        """
        impl_notes = self._read_text_artifact(artifacts_dir / "implementation_notes.md")
        diff_content = self._read_text_artifact(artifacts_dir / "diff.patch")

        if not impl_notes and not diff_content:
            return None

        content_parts = [f"Implementation of '{state.feature_description}':"]

        if impl_notes and len(impl_notes.strip()) > 20:
            content_parts.append(f"Notes: {impl_notes[:1000]}")

        if diff_content:
            # Extract file names from diff
            file_changes = re.findall(r"^(?:\+\+\+|---) [ab]/(.+)$", diff_content, re.MULTILINE)
            unique_files = list(set(file_changes))[:10]
            if unique_files:
                content_parts.append(f"Files changed: {', '.join(unique_files)}")

        content = "\n".join(content_parts)

        if len(content) < 50:
            return None

        return RunMemoryEntry(
            id=self._generate_id(),
            run_id=state.run_id,
            memory_type=MemoryType.IMPLEMENTATION,
            content=content,
            outcome=self._map_run_status_to_outcome(state.status),
            stage="implementation",
            metadata={
                "feature_description": state.feature_description,
                "files_changed": unique_files if diff_content else [],
            },
        )

    def _extract_error_memories(
        self,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> list[RunMemoryEntry]:
        """Extract error and fix memories from failed stages.

        Args:
            state: Pipeline state
            artifacts_dir: Path to artifacts directory

        Returns:
            List of error memory entries
        """
        memories = []

        for stage_name, stage_result in state.stages.items():
            if stage_result.status == StageStatus.FAILED and stage_result.error:
                content = (
                    f"Error in {stage_name} stage for '{state.feature_description}':\n"
                    f"{stage_result.error}"
                )

                memory = RunMemoryEntry(
                    id=self._generate_id(),
                    run_id=state.run_id,
                    memory_type=MemoryType.ERROR_FIX,
                    content=content,
                    outcome=RunOutcome.FAILED,
                    stage=stage_name,
                    metadata={
                        "feature_description": state.feature_description,
                        "error_message": stage_result.error,
                        "stage": stage_name,
                        "retry_count": stage_result.retry_count,
                    },
                )
                memories.append(memory)

        return memories

    def _extract_code_review_memory(
        self,
        state: PipelineState,
        artifacts_dir: Path,
    ) -> RunMemoryEntry | None:
        """Extract code review issues as tech debt.

        Args:
            state: Pipeline state
            artifacts_dir: Path to artifacts directory

        Returns:
            Tech debt memory entry or None
        """
        review_content = self._read_json_artifact(artifacts_dir / "code_review.json")
        if not review_content:
            review_md = self._read_text_artifact(artifacts_dir / "code_review_report.md")
            if not review_md or len(review_md.strip()) < 50:
                return None

            content = f"Code Review for '{state.feature_description}':\n{review_md[:2000]}"
        else:
            issues = review_content.get("issues", [])
            if not issues:
                return None

            issue_summaries = []
            for issue in issues[:5]:  # Top 5 issues
                issue_summaries.append(
                    f"- [{issue.get('severity', 'info')}] {issue.get('description', 'Unknown')}"
                )

            content = (
                f"Code Review Issues for '{state.feature_description}':\n"
                + "\n".join(issue_summaries)
            )

        return RunMemoryEntry(
            id=self._generate_id(),
            run_id=state.run_id,
            memory_type=MemoryType.TECH_DEBT,
            content=content,
            outcome=self._map_run_status_to_outcome(state.status),
            stage="code_review",
            metadata={
                "feature_description": state.feature_description,
            },
        )

    def index_run(
        self,
        run_dir: Path,
        force: bool = False,
    ) -> int:
        """Index a single run into the memory store.

        Args:
            run_dir: Path to run's artifacts directory
            force: Re-index even if already indexed

        Returns:
            Number of memories indexed
        """
        state_path = run_dir / "state.json"
        if not state_path.exists():
            logger.warning("No state.json found in %s", run_dir)
            return 0

        # Load state
        with open(state_path, encoding="utf-8") as f:
            state_data = json.load(f)

        state = PipelineState(**state_data)

        # Check if already indexed
        if not force and self.store.is_run_indexed(state.run_id):
            logger.info("Run %s already indexed, skipping", state.run_id)
            return 0

        # Only index completed or failed runs (not running/pending)
        if state.status in (RunStatus.PENDING, RunStatus.RUNNING):
            logger.info("Run %s is still %s, skipping", state.run_id, state.status.value)
            return 0

        logger.info("Indexing run %s (%s)", state.run_id, state.status.value)

        # If re-indexing, delete old memories first
        if force:
            self.store.delete_run(state.run_id)

        # Extract memories
        memories: list[RunMemoryEntry] = []

        # Feature memory
        feature_mem = self._extract_feature_memory(state, run_dir)
        if feature_mem:
            memories.append(feature_mem)

        # Design memory
        design_mem = self._extract_design_memory(state, run_dir)
        if design_mem:
            memories.append(design_mem)

        # Implementation memory
        impl_mem = self._extract_implementation_memory(state, run_dir)
        if impl_mem:
            memories.append(impl_mem)

        # Error memories
        error_mems = self._extract_error_memories(state, run_dir)
        memories.extend(error_mems)

        # Code review memory
        review_mem = self._extract_code_review_memory(state, run_dir)
        if review_mem:
            memories.append(review_mem)

        if not memories:
            logger.info("No memories extracted from run %s", state.run_id)
            return 0

        # Generate embeddings
        logger.info("Generating embeddings for %d memories", len(memories))
        texts = [m.content for m in memories]
        embeddings = self.embeddings.embed_batch(texts)

        # Add to store
        count = self.store.add(memories, embeddings)
        logger.info("Indexed %d memories from run %s", count, state.run_id)

        return count

    def index_all_runs(
        self,
        artifacts_dir: Path,
        force: bool = False,
    ) -> dict[str, int]:
        """Index all runs in an artifacts directory.

        Args:
            artifacts_dir: Path to artifacts directory
            force: Re-index even if already indexed

        Returns:
            Dict mapping run_id to memories indexed
        """
        results = {}

        if not artifacts_dir.exists():
            logger.warning("Artifacts directory does not exist: %s", artifacts_dir)
            return results

        # Find all run directories (format: YYYY-MM-DD-HHMMSS)
        run_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}$")

        for item in sorted(artifacts_dir.iterdir()):
            if item.is_dir() and run_pattern.match(item.name):
                count = self.index_run(item, force=force)
                results[item.name] = count

        # Save store after indexing all
        self.store.save()

        total = sum(results.values())
        logger.info(
            "Indexed %d runs, %d total memories",
            len([r for r in results.values() if r > 0]),
            total,
        )

        return results

    def extract_error_pattern(
        self,
        error_memory: RunMemoryEntry,
    ) -> ErrorPattern | None:
        """Extract an error pattern from an error memory.

        Args:
            error_memory: Error memory entry

        Returns:
            Error pattern or None
        """
        if error_memory.memory_type != MemoryType.ERROR_FIX:
            return None

        error_msg = error_memory.metadata.get("error_message", "")
        stage = error_memory.metadata.get("stage", "unknown")

        if not error_msg:
            return None

        # Create error signature (simplify the error message)
        signature = self._create_error_signature(error_msg)

        return ErrorPattern(
            id=f"err-{uuid.uuid4().hex[:12]}",
            error_signature=signature,
            source_runs=[error_memory.run_id],
            stage=stage,
            fix_description="",  # To be filled by PatternExtractor
            fix_code_snippet=None,
            prevention_hint="",
            occurrence_count=1,
            fix_success_rate=0.0,
        )

    def _create_error_signature(self, error_msg: str) -> str:
        """Create a simplified error signature for matching.

        Args:
            error_msg: Full error message

        Returns:
            Simplified signature
        """
        # Take first line or first 200 chars
        first_line = error_msg.split("\n")[0][:200]

        # Remove file-specific paths
        signature = re.sub(r"/[\w/.-]+\.(py|js|ts|go)", "<FILE>", first_line)

        # Remove line numbers
        signature = re.sub(r"line \d+", "line N", signature)

        # Remove memory addresses
        signature = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", signature)

        return signature.strip()
