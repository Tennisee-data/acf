"""Pattern extraction from multiple runs.

Uses LLM to identify recurring patterns from indexed memories
and consolidates them into reusable knowledge.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from llm_backend.base import LLMBackend
from schemas.memory import (
    ErrorPattern,
    ExtractedPattern,
    MemoryType,
    RunMemoryEntry,
    RunOutcome,
    TriagePattern,
)

from .store import MemoryStore

logger = logging.getLogger(__name__)

PATTERN_EXTRACTION_PROMPT = """Analyze these memory entries from past development runs and identify recurring patterns.

Memory entries:
{memories}

Identify patterns in the following categories:
- naming: Naming conventions for files, functions, variables
- architecture: Architectural patterns and structure
- testing: Testing approaches and patterns
- config: Configuration patterns
- error_handling: Error handling approaches
- api: API design patterns

For each pattern found, provide:
1. name: Short descriptive name
2. description: What the pattern is
3. pattern_type: Category from above
4. when_to_use: When to apply this pattern
5. when_to_avoid: When NOT to use this pattern
6. examples: 1-2 brief code/config examples

Respond with a JSON array of patterns. Only include patterns that appear in at least 2 different runs.
If no clear patterns are found, respond with an empty array: []

JSON format:
[
  {{
    "name": "Pattern Name",
    "description": "What this pattern is about",
    "pattern_type": "naming|architecture|testing|config|error_handling|api",
    "when_to_use": "When to apply this",
    "when_to_avoid": "When not to use this",
    "examples": ["example 1", "example 2"],
    "domains": ["backend", "api"]
  }}
]
"""

ERROR_PATTERN_PROMPT = """Analyze these error entries from past development runs and identify common error patterns with fixes.

Error entries:
{errors}

For each recurring error pattern, provide:
1. error_signature: Simplified error message pattern (remove file paths, line numbers)
2. stage: Pipeline stage where this occurs
3. fix_description: How to fix this error
4. prevention_hint: How to prevent this error in the future

Respond with a JSON array of error patterns. Only include errors that appear in multiple runs or have clear fixes.
If no clear patterns are found, respond with an empty array: []

JSON format:
[
  {{
    "error_signature": "Simplified error pattern",
    "stage": "stage_name",
    "fix_description": "How to fix this",
    "fix_code_snippet": "Optional code snippet",
    "prevention_hint": "How to prevent this"
  }}
]
"""

TRIAGE_PATTERN_PROMPT = """Analyze these triage decisions from past development runs and identify model routing patterns.

Triage decisions:
{triage_entries}

Each entry shows: prompt pattern, task size, recommended model tier (cheap/medium/premium), and success rate.

Identify routing rules that can be applied to similar future tasks. Look for:
1. Keywords or phrases that consistently indicate simple tasks (cheap model)
2. Keywords or phrases that consistently indicate complex tasks (premium model)
3. Domain patterns (e.g., "auth" tasks always need premium)
4. Size indicators in prompts (e.g., "simple", "basic" vs "complete", "full")

For each routing rule found, provide:
1. rule_name: Short descriptive name
2. description: What this rule detects
3. trigger_keywords: List of keywords that trigger this rule
4. trigger_domains: List of domains this applies to
5. recommended_tier: cheap, medium, or premium
6. confidence: How confident we are (based on success rate)

Respond with a JSON array of routing rules. Only include rules with clear patterns.
If no clear patterns are found, respond with an empty array: []

JSON format:
[
  {{
    "rule_name": "Simple Endpoint Addition",
    "description": "Adding basic API endpoints with minimal logic",
    "trigger_keywords": ["add", "endpoint", "simple", "basic"],
    "trigger_domains": ["api"],
    "recommended_tier": "cheap",
    "confidence": 0.9
  }}
]
"""


class PatternExtractor:
    """Extract patterns from indexed memories using LLM analysis.

    Identifies recurring patterns across multiple runs and
    consolidates them into reusable knowledge.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMBackend,
    ):
        """Initialize pattern extractor.

        Args:
            store: Memory store with indexed memories
            llm: LLM backend for pattern analysis
        """
        self.store = store
        self.llm = llm

    def extract_patterns(
        self,
        min_runs: int = 2,
        memory_types: list[MemoryType] | None = None,
    ) -> list[ExtractedPattern]:
        """Extract patterns from indexed memories.

        Args:
            min_runs: Minimum number of runs to consider
            memory_types: Filter by memory types

        Returns:
            List of extracted patterns
        """
        # Gather memories for analysis
        memories = self._gather_memories_for_analysis(min_runs, memory_types)

        if len(memories) < 3:
            logger.info("Not enough memories for pattern extraction")
            return []

        # Format memories for prompt
        memories_text = self._format_memories_for_prompt(memories)

        # Call LLM for pattern extraction
        prompt = PATTERN_EXTRACTION_PROMPT.format(memories=memories_text)

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a code pattern analyzer. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent output
            )

            patterns = self._parse_pattern_response(response, memories)
            logger.info("Extracted %d patterns from %d memories", len(patterns), len(memories))

            # Add patterns to store
            for pattern in patterns:
                self.store.add_pattern(pattern)

            return patterns

        except Exception as e:
            logger.error("Pattern extraction failed: %s", e)
            return []

    def extract_error_patterns(self) -> list[ErrorPattern]:
        """Extract error patterns from indexed error memories.

        Returns:
            List of extracted error patterns
        """
        # Get all error memories
        error_memories = [
            m for m in self.store.memories
            if m.memory_type == MemoryType.ERROR_FIX
        ]

        if len(error_memories) < 2:
            logger.info("Not enough error memories for pattern extraction")
            return []

        # Format errors for prompt
        errors_text = self._format_errors_for_prompt(error_memories)

        # Call LLM for error pattern extraction
        prompt = ERROR_PATTERN_PROMPT.format(errors=errors_text)

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are an error pattern analyzer. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            patterns = self._parse_error_response(response, error_memories)
            logger.info("Extracted %d error patterns", len(patterns))

            # Add error patterns to store
            for pattern in patterns:
                self.store.add_error_pattern(pattern)

            return patterns

        except Exception as e:
            logger.error("Error pattern extraction failed: %s", e)
            return []

    def extract_triage_patterns(
        self,
        min_success_rate: float = 0.6,
        min_entries: int = 3,
    ) -> list[dict]:
        """Extract routing patterns from triage decisions.

        Analyzes successful triage decisions to identify reusable
        routing rules that can skip LLM triage for known patterns.

        Args:
            min_success_rate: Minimum success rate to consider
            min_entries: Minimum triage entries required

        Returns:
            List of extracted routing rules (as dicts)
        """
        # Get triage entries with good success rates
        triage_entries = [
            t for t in self.store.triage
            if t.success_rate >= min_success_rate or t.success_count >= 2
        ]

        if len(triage_entries) < min_entries:
            logger.info(
                "Not enough triage entries for pattern extraction (%d < %d)",
                len(triage_entries), min_entries
            )
            return []

        # Format triage for prompt
        triage_text = self._format_triage_for_prompt(triage_entries)

        # Call LLM for triage pattern extraction
        prompt = TRIAGE_PATTERN_PROMPT.format(triage_entries=triage_text)

        try:
            response = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task complexity analyzer. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            rules = self._parse_triage_response(response, triage_entries)
            logger.info("Extracted %d triage routing rules", len(rules))

            return rules

        except Exception as e:
            logger.error("Triage pattern extraction failed: %s", e)
            return []

    def consolidate_triage_patterns(self) -> int:
        """Consolidate similar triage patterns to reduce redundancy.

        Merges triage patterns that have similar keywords and
        the same recommended tier.

        Returns:
            Number of patterns consolidated
        """
        if len(self.store.triage) < 2:
            return 0

        consolidated = 0
        patterns_to_remove = set()

        triage_list = list(self.store.triage)

        for i, t1 in enumerate(triage_list):
            if t1.id in patterns_to_remove:
                continue

            for j, t2 in enumerate(triage_list[i + 1:], i + 1):
                if t2.id in patterns_to_remove:
                    continue

                # Check if same tier and similar keywords
                if t1.recommended_tier != t2.recommended_tier:
                    continue

                # Compute keyword overlap
                k1 = set(t1.keywords)
                k2 = set(t2.keywords)
                if not k1 or not k2:
                    continue

                overlap = len(k1 & k2) / len(k1 | k2)
                if overlap >= 0.7:
                    # Merge t2 into t1
                    t1.keywords = list(k1 | k2)
                    t1.success_count += t2.success_count
                    t1.failure_count += t2.failure_count
                    t1.source_runs = list(set(t1.source_runs + t2.source_runs))
                    t1.updated_at = datetime.now()
                    patterns_to_remove.add(t2.id)
                    consolidated += 1

        # Remove merged patterns
        for pattern_id in patterns_to_remove:
            self.store.delete_triage_pattern(pattern_id)

        if consolidated > 0:
            self.store.save()
            logger.info("Consolidated %d similar triage patterns", consolidated)

        return consolidated

    def _format_triage_for_prompt(
        self,
        triage_entries: list[TriagePattern],
        max_length: int = 6000,
    ) -> str:
        """Format triage entries for LLM prompt.

        Args:
            triage_entries: Triage patterns to format
            max_length: Maximum total length

        Returns:
            Formatted string
        """
        formatted = []
        total_length = 0

        for triage in triage_entries:
            success_rate = triage.success_rate
            entry = (
                f"[Size: {triage.size}] [Tier: {triage.recommended_tier}] "
                f"[Success: {success_rate:.0%}]\n"
                f"Prompt: {triage.original_prompt[:200]}\n"
                f"Keywords: {', '.join(triage.keywords)}\n"
                f"Domains: {', '.join(triage.domains)}\n"
                "---"
            )

            if total_length + len(entry) > max_length:
                break

            formatted.append(entry)
            total_length += len(entry)

        return "\n".join(formatted)

    def _parse_triage_response(
        self,
        response: str,
        source_entries: list[TriagePattern],
    ) -> list[dict]:
        """Parse LLM response into triage routing rules.

        Args:
            response: LLM response text
            source_entries: Triage entries used for extraction

        Returns:
            List of routing rule dicts
        """
        rules = []

        # Extract JSON from response
        json_text = self._extract_json(response)
        if not json_text:
            return []

        try:
            data = json.loads(json_text)
            if not isinstance(data, list):
                return []

            for item in data:
                if not isinstance(item, dict):
                    continue

                rule = {
                    "rule_name": item.get("rule_name", "Unknown Rule"),
                    "description": item.get("description", ""),
                    "trigger_keywords": item.get("trigger_keywords", []),
                    "trigger_domains": item.get("trigger_domains", []),
                    "recommended_tier": item.get("recommended_tier", "medium"),
                    "confidence": item.get("confidence", 0.5),
                    "source_count": len(source_entries),
                    "extracted_at": datetime.now().isoformat(),
                }
                rules.append(rule)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse triage response: %s", e)

        return rules

    def _gather_memories_for_analysis(
        self,
        min_runs: int,
        memory_types: list[MemoryType] | None,
    ) -> list[RunMemoryEntry]:
        """Gather memories suitable for pattern analysis.

        Args:
            min_runs: Minimum number of unique runs required
            memory_types: Filter by types

        Returns:
            List of memories for analysis
        """
        memories = []
        run_ids = set()

        for memory in self.store.memories:
            # Filter by type if specified
            if memory_types and memory.memory_type not in memory_types:
                continue

            # Skip error memories (handled separately)
            if memory.memory_type == MemoryType.ERROR_FIX:
                continue

            # Prefer successful runs for pattern learning
            if memory.outcome in (RunOutcome.SUCCESS, RunOutcome.PARTIAL):
                memories.append(memory)
                run_ids.add(memory.run_id)

        # Check if we have enough runs
        if len(run_ids) < min_runs:
            return []

        return memories

    def _format_memories_for_prompt(
        self,
        memories: list[RunMemoryEntry],
        max_length: int = 8000,
    ) -> str:
        """Format memories for LLM prompt.

        Args:
            memories: Memories to format
            max_length: Maximum total length

        Returns:
            Formatted string
        """
        formatted = []
        total_length = 0

        for memory in memories:
            entry = (
                f"[Run: {memory.run_id}] [{memory.memory_type.value}]\n"
                f"{memory.content[:500]}\n"
                "---"
            )

            if total_length + len(entry) > max_length:
                break

            formatted.append(entry)
            total_length += len(entry)

        return "\n".join(formatted)

    def _format_errors_for_prompt(
        self,
        errors: list[RunMemoryEntry],
        max_length: int = 6000,
    ) -> str:
        """Format error memories for LLM prompt.

        Args:
            errors: Error memories to format
            max_length: Maximum total length

        Returns:
            Formatted string
        """
        formatted = []
        total_length = 0

        for error in errors:
            stage = error.metadata.get("stage", "unknown")
            error_msg = error.metadata.get("error_message", error.content)

            entry = (
                f"[Run: {error.run_id}] [Stage: {stage}]\n"
                f"Error: {error_msg[:400]}\n"
                "---"
            )

            if total_length + len(entry) > max_length:
                break

            formatted.append(entry)
            total_length += len(entry)

        return "\n".join(formatted)

    def _parse_pattern_response(
        self,
        response: str,
        source_memories: list[RunMemoryEntry],
    ) -> list[ExtractedPattern]:
        """Parse LLM response into patterns.

        Args:
            response: LLM response text
            source_memories: Memories used for extraction

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Extract JSON from response
        json_text = self._extract_json(response)
        if not json_text:
            return []

        try:
            data = json.loads(json_text)
            if not isinstance(data, list):
                return []

            # Get source run IDs
            source_runs = list(set(m.run_id for m in source_memories))

            for item in data:
                if not isinstance(item, dict):
                    continue

                pattern = ExtractedPattern(
                    id=f"pat-{uuid.uuid4().hex[:12]}",
                    name=item.get("name", "Unknown Pattern"),
                    description=item.get("description", ""),
                    source_runs=source_runs[:5],  # Top 5 runs
                    examples=item.get("examples", [])[:3],
                    pattern_type=item.get("pattern_type", "general"),
                    domains=item.get("domains", []),
                    occurrence_count=len(source_runs),
                    success_rate=1.0,  # Initial value
                    when_to_use=item.get("when_to_use", ""),
                    when_to_avoid=item.get("when_to_avoid", ""),
                )
                patterns.append(pattern)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse pattern response: %s", e)

        return patterns

    def _parse_error_response(
        self,
        response: str,
        source_errors: list[RunMemoryEntry],
    ) -> list[ErrorPattern]:
        """Parse LLM response into error patterns.

        Args:
            response: LLM response text
            source_errors: Error memories used for extraction

        Returns:
            List of extracted error patterns
        """
        patterns = []

        # Extract JSON from response
        json_text = self._extract_json(response)
        if not json_text:
            return []

        try:
            data = json.loads(json_text)
            if not isinstance(data, list):
                return []

            # Get source run IDs
            source_runs = list(set(e.run_id for e in source_errors))

            for item in data:
                if not isinstance(item, dict):
                    continue

                pattern = ErrorPattern(
                    id=f"err-{uuid.uuid4().hex[:12]}",
                    error_signature=item.get("error_signature", "Unknown error"),
                    source_runs=source_runs[:5],
                    stage=item.get("stage", "unknown"),
                    fix_description=item.get("fix_description", ""),
                    fix_code_snippet=item.get("fix_code_snippet"),
                    prevention_hint=item.get("prevention_hint", ""),
                    occurrence_count=len(source_errors),
                    fix_success_rate=0.5,  # Initial value
                )
                patterns.append(pattern)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse error response: %s", e)

        return patterns

    def _extract_json(self, text: str) -> str | None:
        """Extract JSON array from text response.

        Args:
            text: Response text that may contain JSON

        Returns:
            Extracted JSON string or None
        """
        # Try to find JSON array in response
        text = text.strip()

        # If it starts with [ and ends with ], it's likely JSON
        if text.startswith("[") and text.endswith("]"):
            return text

        # Try to extract JSON from markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if json_match:
            return json_match.group(1)

        # Try to find array anywhere in text
        array_match = re.search(r"(\[[\s\S]*\])", text)
        if array_match:
            return array_match.group(1)

        return None

    def merge_similar_patterns(
        self,
        similarity_threshold: float = 0.8,
    ) -> int:
        """Merge similar patterns to reduce duplication.

        Args:
            similarity_threshold: Minimum similarity to merge

        Returns:
            Number of patterns merged
        """
        merged_count = 0
        patterns_to_remove = set()

        patterns = self.store.patterns.copy()

        for i, pattern1 in enumerate(patterns):
            if pattern1.id in patterns_to_remove:
                continue

            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                if pattern2.id in patterns_to_remove:
                    continue

                # Simple name-based similarity
                similarity = self._compute_name_similarity(pattern1.name, pattern2.name)

                if similarity >= similarity_threshold:
                    # Merge pattern2 into pattern1
                    self._merge_patterns(pattern1, pattern2)
                    patterns_to_remove.add(pattern2.id)
                    merged_count += 1

        # Remove merged patterns
        for pattern_id in patterns_to_remove:
            self.store.delete_pattern(pattern_id)

        logger.info("Merged %d similar patterns", merged_count)
        return merged_count

    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """Compute similarity between two pattern names.

        Args:
            name1: First name
            name2: Second name

        Returns:
            Similarity score 0-1
        """
        # Simple word overlap similarity
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _merge_patterns(
        self,
        target: ExtractedPattern,
        source: ExtractedPattern,
    ) -> None:
        """Merge source pattern into target.

        Args:
            target: Pattern to merge into
            source: Pattern to merge from
        """
        # Merge source runs
        target.source_runs = list(set(target.source_runs + source.source_runs))

        # Merge examples (deduplicate)
        existing_examples = set(target.examples)
        for example in source.examples:
            if example not in existing_examples:
                target.examples.append(example)

        # Merge domains
        target.domains = list(set(target.domains + source.domains))

        # Update occurrence count
        target.occurrence_count += source.occurrence_count

        # Average success rates
        target.success_rate = (target.success_rate + source.success_rate) / 2

        # Update timestamp
        target.updated_at = datetime.now()

    def export_patterns_markdown(self, output_path: Path) -> Path:
        """Export patterns to markdown documentation.

        Args:
            output_path: Path to write markdown file

        Returns:
            Path to created file
        """
        lines = [
            "# Coding Patterns",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            f"Total patterns: {len(self.store.patterns)}",
            f"Total error patterns: {len(self.store.errors)}",
            f"Total triage patterns: {len(self.store.triage)}",
            "",
            "---",
            "",
        ]

        # Group patterns by type
        patterns_by_type: dict[str, list[ExtractedPattern]] = {}
        for pattern in self.store.patterns:
            pt = pattern.pattern_type
            if pt not in patterns_by_type:
                patterns_by_type[pt] = []
            patterns_by_type[pt].append(pattern)

        # Write patterns
        lines.append("## Patterns")
        lines.append("")

        for pattern_type, patterns in sorted(patterns_by_type.items()):
            lines.append(f"### {pattern_type.title()} Patterns")
            lines.append("")

            for pattern in patterns:
                lines.append(f"#### {pattern.name}")
                lines.append("")
                lines.append(pattern.description)
                lines.append("")

                if pattern.when_to_use:
                    lines.append(f"**When to use:** {pattern.when_to_use}")
                    lines.append("")

                if pattern.when_to_avoid:
                    lines.append(f"**When to avoid:** {pattern.when_to_avoid}")
                    lines.append("")

                if pattern.examples:
                    lines.append("**Examples:**")
                    for example in pattern.examples:
                        lines.append(f"- `{example}`")
                    lines.append("")

                lines.append(f"*Seen in {pattern.occurrence_count} runs*")
                lines.append("")

        # Write error patterns
        lines.append("## Error Patterns")
        lines.append("")

        for error in self.store.errors:
            lines.append(f"### {error.error_signature[:80]}...")
            lines.append("")
            lines.append(f"**Stage:** {error.stage}")
            lines.append("")
            lines.append(f"**Fix:** {error.fix_description}")
            lines.append("")

            if error.prevention_hint:
                lines.append(f"**Prevention:** {error.prevention_hint}")
                lines.append("")

            if error.fix_code_snippet:
                lines.append("**Code:**")
                lines.append("```")
                lines.append(error.fix_code_snippet)
                lines.append("```")
                lines.append("")

            lines.append(f"*Seen {error.occurrence_count} times*")
            lines.append("")

        # Write triage patterns
        if self.store.triage:
            lines.append("## Triage Patterns (Model Routing)")
            lines.append("")
            lines.append("Learned patterns for routing tasks to appropriate model tiers.")
            lines.append("")

            # Group by tier
            triage_by_tier: dict[str, list[TriagePattern]] = {}
            for triage in self.store.triage:
                tier = triage.recommended_tier
                if tier not in triage_by_tier:
                    triage_by_tier[tier] = []
                triage_by_tier[tier].append(triage)

            for tier in ["cheap", "medium", "premium"]:
                if tier not in triage_by_tier:
                    continue

                tier_label = {"cheap": "Cheap (Fast)", "medium": "Medium (Balanced)", "premium": "Premium (Quality)"}
                lines.append(f"### {tier_label.get(tier, tier.title())} Model")
                lines.append("")

                for triage in triage_by_tier[tier]:
                    success_rate = triage.success_rate
                    lines.append(f"#### {triage.original_prompt[:60]}...")
                    lines.append("")
                    lines.append(f"- **Size:** {triage.size}")
                    lines.append(f"- **Keywords:** {', '.join(triage.keywords) or 'none'}")
                    lines.append(f"- **Domains:** {', '.join(triage.domains) or 'none'}")
                    lines.append(f"- **Success Rate:** {success_rate:.0%} ({triage.success_count}/{triage.success_count + triage.failure_count})")
                    lines.append("")

        # Write to file
        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")

        logger.info("Exported patterns to %s", output_path)
        return output_path
