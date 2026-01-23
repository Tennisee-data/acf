"""Context Agent for analyzing repository and gathering relevant context."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from llm_backend import LLMBackend
from utils.json_repair import parse_llm_json
from schemas.context_report import (
    CodeSnippet,
    ContextReport,
    ExistingPattern,
    FileContext,
    RepoStructure,
)
from schemas.feature_spec import FeatureSpec
from tools import FilesystemTool, GitTool

from .base import AgentInput, AgentOutput, BaseAgent

if TYPE_CHECKING:
    from memory import MemoryRetriever
    from rag import CodeRetriever


SYSTEM_PROMPT = """Output a JSON object analyzing a codebase. Use EXACTLY this format - copy the structure and fill in values:

{"repo_structure":{"framework":null,"language":"Python","package_manager":"pip","test_framework":"pytest","entry_points":["main.py"],"key_directories":{"src":"source"}},"relevant_files":[{"path":"api/routes.py","purpose":"API endpoints","key_exports":["router"],"relevance":"where to add endpoint"}],"files_to_modify":["api/routes.py"],"files_to_create":[],"existing_patterns":[{"name":"REST endpoints","description":"FastAPI router pattern","examples":["api/routes.py"],"recommended":true}],"mental_model":"Add endpoint to existing router following the pattern in routes.py","external_dependencies":["fastapi"],"potential_new_dependencies":[],"integration_risks":["ensure consistent error handling"]}

Replace the example values with actual analysis. Output ONLY the JSON object, nothing else."""


# Common file patterns to look for
FILE_PATTERNS = {
    "python": {
        "entry_points": ["main.py", "app.py", "__main__.py", "cli.py", "api.py"],
        "config": ["config.py", "settings.py", "config.toml", "pyproject.toml"],
        "tests": ["test_*.py", "*_test.py", "conftest.py"],
        "models": ["models.py", "schemas.py", "entities.py"],
        "routes": ["routes.py", "views.py", "endpoints.py", "api/*.py"],
    },
    "javascript": {
        "entry_points": ["index.js", "app.js", "main.js", "server.js"],
        "config": ["package.json", "tsconfig.json", ".env"],
        "tests": ["*.test.js", "*.spec.js", "__tests__/*"],
        "components": ["components/*.jsx", "components/*.tsx"],
    },
}

# Frameworks detection patterns
FRAMEWORK_PATTERNS = {
    "fastapi": ["from fastapi", "FastAPI()", "@app.get", "@app.post"],
    "django": ["from django", "INSTALLED_APPS", "urlpatterns"],
    "flask": ["from flask", "Flask(__name__)", "@app.route"],
    "express": ["require('express')", "express()", "app.get(", "app.post("],
    "react": ["from 'react'", "import React", "useState", "useEffect"],
    "vue": ["from 'vue'", "createApp", "defineComponent"],
}


class ContextAgent(BaseAgent):
    """Agent for analyzing repository and gathering context.

    Scans the repository to identify:
    - Relevant files and modules
    - Existing patterns and conventions
    - Entry points and integration points
    - Dependencies and potential new requirements

    When a RAG retriever is provided, uses semantic search to find
    relevant code chunks in addition to keyword-based search.
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo_path: Path | str | None = None,
        system_prompt: str | None = None,
        retriever: "CodeRetriever | None" = None,
        memory_retriever: "MemoryRetriever | None" = None,
    ) -> None:
        """Initialize ContextAgent.

        Args:
            llm: LLM backend for analysis
            repo_path: Path to repository (default: current directory)
            system_prompt: Override default system prompt
            retriever: Optional RAG retriever for semantic code search
            memory_retriever: Optional memory retriever for historical context
        """
        super().__init__(llm, system_prompt)
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.fs_tool = FilesystemTool(base_path=self.repo_path)
        self.git_tool = GitTool(repo_path=self.repo_path)
        self.retriever = retriever
        self.memory_retriever = memory_retriever

    def default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return SYSTEM_PROMPT

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Analyze repository and generate context report.

        Args:
            input_data: Must contain 'feature_spec' in context (dict or FeatureSpec)

        Returns:
            AgentOutput with ContextReport data
        """
        # Extract feature spec
        feature_spec_data = input_data.context.get("feature_spec", {})
        if isinstance(feature_spec_data, FeatureSpec):
            feature_spec = feature_spec_data
        else:
            try:
                feature_spec = FeatureSpec(**feature_spec_data)
            except Exception:
                # Minimal spec if parsing fails
                feature_spec = None

        feature_id = feature_spec.id if feature_spec else input_data.context.get("run_id", "unknown")
        feature_description = (
            feature_spec.original_description
            if feature_spec
            else input_data.context.get("feature_description", "")
        )

        try:
            # Step 1: Scan repository structure
            repo_info = self._scan_repository()

            # Step 2: Detect framework and language
            framework, language = self._detect_stack(repo_info)

            # Step 3: Find relevant files based on feature
            relevant_files = self._find_relevant_files(feature_description, repo_info)

            # Step 4: Extract code snippets from relevant files
            snippets = self._extract_snippets(relevant_files, feature_description)

            # Step 5: Use LLM to analyze and generate mental model
            llm_analysis = self._llm_analyze(
                feature_spec=feature_spec,
                feature_description=feature_description,
                repo_info=repo_info,
                relevant_files=relevant_files,
                snippets=snippets,
            )

            # Step 6: Build context report
            context_report = self._build_report(
                feature_id=feature_id,
                repo_info=repo_info,
                framework=framework,
                language=language,
                relevant_files=relevant_files,
                snippets=snippets,
                llm_analysis=llm_analysis,
            )

            return AgentOutput(
                success=True,
                data=context_report.model_dump(),
                artifacts=["context_report.md", "context_snippets.json"],
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                errors=[f"ContextAgent error: {str(e)}"],
            )

    def _scan_repository(self) -> dict:
        """Scan repository structure."""
        info = {
            "files": [],
            "directories": [],
            "tree": "",
            "config_files": [],
            "source_files": [],
        }

        # Get directory tree
        tree_result = self.fs_tool.execute("tree", path=".", max_depth=3, max_items=200)
        if tree_result.success:
            info["tree"] = tree_result.output

        # Find all files
        for ext in ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.rs", "*.java"]:
            glob_result = self.fs_tool.execute("glob", pattern=f"**/{ext}")
            if glob_result.success:
                info["source_files"].extend(glob_result.output)

        # Find config files
        for config in ["*.toml", "*.json", "*.yaml", "*.yml", "*.env*", "Dockerfile*", "*.md"]:
            glob_result = self.fs_tool.execute("glob", pattern=config)
            if glob_result.success:
                info["config_files"].extend(glob_result.output)

        # Deduplicate
        info["source_files"] = list(set(info["source_files"]))
        info["config_files"] = list(set(info["config_files"]))

        return info

    def _detect_stack(self, repo_info: dict) -> tuple[str | None, str]:
        """Detect framework and primary language."""
        language = "unknown"
        framework = None

        # Detect language by file count
        py_count = sum(1 for f in repo_info["source_files"] if f.endswith(".py"))
        js_count = sum(1 for f in repo_info["source_files"] if f.endswith((".js", ".jsx", ".ts", ".tsx")))
        go_count = sum(1 for f in repo_info["source_files"] if f.endswith(".go"))

        if py_count > js_count and py_count > go_count:
            language = "Python"
        elif js_count > py_count and js_count > go_count:
            language = "JavaScript/TypeScript"
        elif go_count > 0:
            language = "Go"

        # Detect framework by scanning key files
        files_to_check = repo_info["source_files"][:20]  # Check first 20 files
        for file_path in files_to_check:
            read_result = self.fs_tool.execute("read", path=file_path)
            if read_result.success:
                content = read_result.output
                for fw, patterns in FRAMEWORK_PATTERNS.items():
                    if any(p in content for p in patterns):
                        framework = fw
                        break
            if framework:
                break

        return framework, language

    def _find_relevant_files(self, feature_description: str, repo_info: dict) -> list[dict]:
        """Find files relevant to the feature."""
        relevant = []
        seen_paths = set()

        # Use RAG semantic search if available
        if self.retriever and self.retriever.is_indexed():
            rag_results = self._rag_search(feature_description)
            for result in rag_results:
                path = result["path"]
                if path not in seen_paths:
                    seen_paths.add(path)
                    relevant.append(result)

        # Keyword-based search
        keywords = self._extract_keywords(feature_description)

        # Search for keyword matches in files
        for keyword in keywords[:5]:  # Limit keyword searches
            search_result = self.fs_tool.execute(
                "search",
                pattern=keyword,
                file_pattern="*.py",  # Start with Python
                max_results=10,
            )
            if search_result.success:
                for match in search_result.output:
                    file_path = match.get("file", "")
                    if file_path and file_path not in seen_paths:
                        seen_paths.add(file_path)
                        relevant.append({
                            "path": file_path,
                            "match_keyword": keyword,
                            "match_line": match.get("line", 0),
                            "match_content": match.get("content", ""),
                        })

        # Also check common entry points
        for pattern_type, patterns in FILE_PATTERNS.get("python", {}).items():
            for pattern in patterns:
                glob_result = self.fs_tool.execute("glob", pattern=f"**/{pattern}")
                if glob_result.success:
                    for path in glob_result.output[:3]:
                        if path not in seen_paths:
                            seen_paths.add(path)
                            relevant.append({
                                "path": path,
                                "match_keyword": pattern_type,
                                "match_line": 0,
                                "match_content": f"Entry point: {pattern_type}",
                            })

        return relevant[:15]  # Limit to 15 most relevant

    def _rag_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for relevant code using RAG.

        Args:
            query: Search query (feature description)
            top_k: Number of results to return

        Returns:
            List of relevant file info dicts
        """
        if not self.retriever:
            return []

        try:
            results = self.retriever.search(query, top_k=top_k)
            relevant = []

            for result in results:
                chunk = result.chunk
                relevant.append({
                    "path": chunk.file_path,
                    "match_keyword": f"semantic:{chunk.chunk_type}",
                    "match_line": chunk.start_line,
                    "match_content": f"[RAG score: {result.score:.2f}] {chunk.name or chunk.chunk_type}",
                    "rag_score": result.score,
                    "rag_chunk": chunk,
                })

            return relevant
        except Exception:
            # RAG search failed, continue without it
            return []

    def _get_rag_context(self, query: str, max_tokens: int = 4000) -> str:
        """Get RAG context for the LLM prompt.

        Args:
            query: The feature description to search for
            max_tokens: Maximum tokens to include in context

        Returns:
            Formatted context string from RAG search
        """
        if not self.retriever or not self.retriever.is_indexed():
            return ""

        try:
            return self.retriever.get_context(query, max_tokens=max_tokens)
        except Exception:
            return ""

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from feature description."""
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "own", "same", "than", "too",
            "very", "just", "add", "implement", "create", "make", "new",
            "feature", "want", "using", "use",
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique

    def _extract_snippets(self, relevant_files: list[dict], feature_description: str) -> list[dict]:
        """Extract relevant code snippets from files."""
        snippets = []

        for file_info in relevant_files[:10]:
            file_path = file_info["path"]
            read_result = self.fs_tool.execute("read", path=file_path)

            if not read_result.success:
                continue

            content = read_result.output
            lines = content.split("\n")

            # Find the most relevant section
            match_line = file_info.get("match_line", 0)
            if match_line > 0:
                # Extract context around the match
                start = max(0, match_line - 5)
                end = min(len(lines), match_line + 15)
                snippet_lines = lines[start:end]

                snippets.append({
                    "file_path": file_path,
                    "start_line": start + 1,
                    "end_line": end,
                    "content": "\n".join(snippet_lines),
                    "relevance": file_info.get("match_content", ""),
                    "language": self._detect_language(file_path),
                })
            elif len(lines) < 100:
                # Small file - include header
                snippets.append({
                    "file_path": file_path,
                    "start_line": 1,
                    "end_line": min(50, len(lines)),
                    "content": "\n".join(lines[:50]),
                    "relevance": f"Entry point: {file_info.get('match_keyword', 'unknown')}",
                    "language": self._detect_language(file_path),
                })

        return snippets[:8]  # Limit snippets

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
        }
        for ext, lang in ext_map.items():
            if file_path.endswith(ext):
                return lang
        return "text"

    def _llm_analyze(
        self,
        feature_spec: FeatureSpec | None,
        feature_description: str,
        repo_info: dict,
        relevant_files: list[dict],
        snippets: list[dict],
    ) -> dict:
        """Use LLM to analyze repository and generate insights."""
        # Build context for LLM
        context_parts = []

        # Feature info
        if feature_spec:
            context_parts.append(f"## Feature Specification\n")
            context_parts.append(f"Title: {feature_spec.title}")
            context_parts.append(f"Description: {feature_spec.original_description}")
            context_parts.append(f"User Story: {feature_spec.user_story}")
            if feature_spec.domains:
                context_parts.append(f"Domains: {', '.join([d.value for d in feature_spec.domains])}")
        else:
            context_parts.append(f"## Feature\n{feature_description}")

        # Add historical context from memory if available
        if self.memory_retriever:
            domains = [d.value for d in feature_spec.domains] if feature_spec and feature_spec.domains else None
            historical = self.memory_retriever.get_historical_context(
                feature_description=feature_description,
                stage="context",
                domains=domains,
            )
            if historical.summary and historical.summary != "No relevant historical context found.":
                context_parts.append(f"\n## Historical Context (from past runs)")
                context_parts.append(historical.summary)

        # Add RAG context if available (semantic search results)
        rag_context = self._get_rag_context(feature_description)
        if rag_context:
            context_parts.append(f"\n## Semantically Related Code (via RAG)")
            context_parts.append(rag_context)

        # Repository structure
        context_parts.append(f"\n## Repository Structure\n```\n{repo_info.get('tree', 'N/A')}\n```")

        # Source files summary
        source_files = repo_info.get("source_files", [])
        if source_files:
            context_parts.append(f"\n## Source Files ({len(source_files)} total)")
            for f in source_files[:20]:
                context_parts.append(f"- {f}")

        # Relevant files
        if relevant_files:
            context_parts.append(f"\n## Potentially Relevant Files")
            for rf in relevant_files:
                context_parts.append(f"- {rf['path']}: {rf.get('match_content', '')}")

        # Code snippets
        if snippets:
            context_parts.append(f"\n## Code Snippets")
            for snippet in snippets[:5]:
                context_parts.append(f"\n### {snippet['file_path']} (lines {snippet['start_line']}-{snippet['end_line']})")
                context_parts.append(f"Relevance: {snippet['relevance']}")
                context_parts.append(f"```{snippet['language']}\n{snippet['content'][:1000]}\n```")

        user_message = "\n".join(context_parts)
        user_message += "\n\nAnalyze this repository and provide your assessment as JSON."

        try:
            response = self._chat(user_message, temperature=0.3)
            return self._parse_llm_response(response)
        except Exception as e:
            # Return minimal analysis on error
            return {
                "repo_structure": {
                    "framework": None,
                    "language": "unknown",
                    "entry_points": [],
                    "key_directories": {},
                },
                "relevant_files": [],
                "files_to_modify": [],
                "files_to_create": [],
                "existing_patterns": [],
                "mental_model": f"Analysis failed: {str(e)}",
                "external_dependencies": [],
                "potential_new_dependencies": [],
                "integration_risks": [f"LLM analysis failed: {str(e)}"],
            }

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response JSON with repair.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed even after repair
        """
        import json

        result = parse_llm_json(response, default=None)

        if result is None:
            raise json.JSONDecodeError(
                "Could not parse JSON from response",
                response,
                0,
            )

        return result

    def _build_report(
        self,
        feature_id: str,
        repo_info: dict,
        framework: str | None,
        language: str,
        relevant_files: list[dict],
        snippets: list[dict],
        llm_analysis: dict,
    ) -> ContextReport:
        """Build the final ContextReport."""
        # Helper to ensure dict[str, str] for key_directories
        def ensure_str_dict(d: dict) -> dict[str, str]:
            result = {}
            for k, v in d.items():
                if isinstance(v, str):
                    result[str(k)] = v
                elif isinstance(v, dict):
                    # Extract purpose/description from nested dict
                    result[str(k)] = v.get("purpose", v.get("description", v.get("text", str(v))))
                else:
                    result[str(k)] = str(v) if v is not None else ""
            return result

        # Build RepoStructure
        llm_repo = llm_analysis.get("repo_structure", {})
        repo_structure = RepoStructure(
            framework=llm_repo.get("framework") or framework,
            language=llm_repo.get("language") or language,
            package_manager=llm_repo.get("package_manager"),
            test_framework=llm_repo.get("test_framework"),
            entry_points=llm_repo.get("entry_points", []),
            key_directories=ensure_str_dict(llm_repo.get("key_directories", {})),
        )

        # Build FileContext list
        file_contexts = []
        for rf in llm_analysis.get("relevant_files", []):
            if isinstance(rf, dict):
                file_contexts.append(
                    FileContext(
                        path=rf.get("path", ""),
                        purpose=rf.get("purpose", rf.get("relevance", "")),
                        key_exports=rf.get("key_exports", []),
                        dependencies=rf.get("dependencies", []),
                        dependents=rf.get("dependents", []),
                    )
                )

        # Build CodeSnippet list
        code_snippets = []
        for snippet in snippets:
            code_snippets.append(
                CodeSnippet(
                    file_path=snippet["file_path"],
                    start_line=snippet["start_line"],
                    end_line=snippet["end_line"],
                    content=snippet["content"],
                    relevance=snippet["relevance"],
                    language=snippet.get("language"),
                )
            )

        # Build ExistingPattern list
        patterns = []
        for p in llm_analysis.get("existing_patterns", []):
            if isinstance(p, dict):
                patterns.append(
                    ExistingPattern(
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        examples=p.get("examples", []),
                        recommended=p.get("recommended", True),
                    )
                )

        # Helper to ensure list of strings
        def ensure_str_list(items: list) -> list[str]:
            result = []
            for item in items:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    # Extract path or name from dict
                    result.append(item.get("path", item.get("name", str(item))))
                else:
                    result.append(str(item))
            return result

        # Helper to ensure string value
        def ensure_str(value, default: str = "") -> str:
            if isinstance(value, str):
                return value
            elif isinstance(value, dict):
                # Try common keys for text content
                return value.get("text", value.get("description", value.get("summary", value.get("purpose", str(value)))))
            elif value is None:
                return default
            else:
                return str(value)

        return ContextReport(
            feature_id=feature_id,
            repo_structure=repo_structure,
            relevant_files=file_contexts,
            files_to_modify=ensure_str_list(llm_analysis.get("files_to_modify", [])),
            files_to_create=ensure_str_list(llm_analysis.get("files_to_create", [])),
            snippets=code_snippets,
            existing_patterns=patterns,
            mental_model=ensure_str(llm_analysis.get("mental_model"), "No analysis available"),
            external_dependencies=ensure_str_list(llm_analysis.get("external_dependencies", [])),
            potential_new_dependencies=ensure_str_list(llm_analysis.get("potential_new_dependencies", [])),
            integration_risks=ensure_str_list(llm_analysis.get("integration_risks", [])),
        )

    def generate_markdown_report(self, context_report: ContextReport) -> str:
        """Generate markdown version of the context report."""
        lines = []
        lines.append(f"# Context Report: {context_report.feature_id}")
        lines.append("")

        # Mental Model
        lines.append("## Mental Model")
        lines.append(context_report.mental_model)
        lines.append("")

        # Repository Structure
        lines.append("## Repository Structure")
        rs = context_report.repo_structure
        lines.append(f"- **Framework:** {rs.framework or 'Unknown'}")
        lines.append(f"- **Language:** {rs.language}")
        lines.append(f"- **Package Manager:** {rs.package_manager or 'Unknown'}")
        lines.append(f"- **Test Framework:** {rs.test_framework or 'Unknown'}")
        if rs.entry_points:
            lines.append(f"- **Entry Points:** {', '.join(rs.entry_points)}")
        lines.append("")

        # Relevant Files
        if context_report.relevant_files:
            lines.append("## Relevant Files")
            for f in context_report.relevant_files:
                lines.append(f"### `{f.path}`")
                lines.append(f"{f.purpose}")
                if f.key_exports:
                    lines.append(f"- Exports: {', '.join(f.key_exports)}")
                lines.append("")

        # Files to Modify
        if context_report.files_to_modify:
            lines.append("## Files to Modify")
            for f in context_report.files_to_modify:
                lines.append(f"- `{f}`")
            lines.append("")

        # Files to Create
        if context_report.files_to_create:
            lines.append("## Files to Create")
            for f in context_report.files_to_create:
                lines.append(f"- `{f}`")
            lines.append("")

        # Existing Patterns
        if context_report.existing_patterns:
            lines.append("## Existing Patterns")
            for p in context_report.existing_patterns:
                rec = "✓ Recommended" if p.recommended else "⚠ Not recommended"
                lines.append(f"### {p.name} ({rec})")
                lines.append(p.description)
                if p.examples:
                    lines.append(f"Examples: {', '.join(p.examples)}")
                lines.append("")

        # Dependencies
        if context_report.external_dependencies:
            lines.append("## External Dependencies")
            for d in context_report.external_dependencies:
                lines.append(f"- {d}")
            lines.append("")

        if context_report.potential_new_dependencies:
            lines.append("## Potential New Dependencies")
            for d in context_report.potential_new_dependencies:
                lines.append(f"- {d}")
            lines.append("")

        # Integration Risks
        if context_report.integration_risks:
            lines.append("## Integration Risks")
            for r in context_report.integration_risks:
                lines.append(f"- ⚠ {r}")
            lines.append("")

        return "\n".join(lines)
