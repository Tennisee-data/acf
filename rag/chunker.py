"""Code and document chunking for RAG indexing."""

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator

# Optional PDF support
try:
    import fitz  # pymupdf

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "module", "block"
    name: str | None = None  # Function/class name if applicable

    @property
    def id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_text(self) -> str:
        """Convert to text for embedding."""
        header = f"# File: {self.file_path} (lines {self.start_line}-{self.end_line})"
        if self.name:
            header += f"\n# {self.chunk_type}: {self.name}"
        return f"{header}\n{self.content}"


class CodeChunker:
    """Split code files into meaningful chunks for RAG.

    Strategies:
    - Split by functions/classes (semantic)
    - Fall back to line-based chunking
    - Preserve context with overlap
    """

    def __init__(
        self,
        chunk_size: int = 1500,  # Target chunk size in characters
        chunk_overlap: int = 200,  # Overlap between chunks
        min_chunk_size: int = 100,  # Minimum chunk size
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum size to create a chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Patterns for different languages
        self.patterns = {
            "python": {
                "function": r"^(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:",
                "class": r"^(\s*)class\s+(\w+)(?:\([^)]*\))?:",
            },
            "javascript": {
                "function": r"^(\s*)(async\s+)?function\s+(\w+)\s*\([^)]*\)",
                "class": r"^(\s*)class\s+(\w+)",
                "arrow": r"^(\s*)(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
            },
            "typescript": {
                "function": r"^(\s*)(async\s+)?function\s+(\w+)\s*[<(]",
                "class": r"^(\s*)(?:export\s+)?class\s+(\w+)",
                "interface": r"^(\s*)(?:export\s+)?interface\s+(\w+)",
            },
        }

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """Chunk a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of code chunks
        """
        suffix = file_path.suffix.lower()

        # Handle PDF files
        if suffix == ".pdf":
            return self._chunk_pdf(file_path)

        # Handle Jupyter notebooks
        if suffix == ".ipynb":
            return self._chunk_notebook(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        # Determine language
        language = self._detect_language(suffix)

        if language and language in self.patterns:
            # Try semantic chunking first
            chunks = self._semantic_chunk(content, str(file_path), language)
            if chunks:
                return chunks

        # Fall back to line-based chunking
        return self._line_chunk(content, str(file_path))

    def chunk_directory(
        self,
        directory: Path,
        extensions: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
    ) -> Iterator[CodeChunk]:
        """Chunk all files in a directory.

        Args:
            directory: Directory to process
            extensions: File extensions to include (e.g., [".py", ".js"])
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", "node_modules"])

        Yields:
            Code chunks from all files
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".md", ".txt", ".pdf", ".ipynb"]

        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__",
                "node_modules",
                ".git",
                ".venv",
                "venv",
                "dist",
                "build",
                ".egg-info",
            ]

        for file_path in directory.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Check extension
            if file_path.suffix.lower() not in extensions:
                continue

            # Check exclude patterns
            path_str = str(file_path)
            if any(pattern in path_str for pattern in exclude_patterns):
                continue

            # Chunk the file
            for chunk in self.chunk_file(file_path):
                yield chunk

    def _detect_language(self, suffix: str) -> str | None:
        """Detect language from file suffix."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }
        return mapping.get(suffix)

    def _semantic_chunk(
        self,
        content: str,
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
        """Chunk code by semantic units (functions, classes)."""
        lines = content.split("\n")
        chunks = []
        patterns = self.patterns[language]

        # Find all definitions
        definitions = []
        for i, line in enumerate(lines):
            for def_type, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    indent = len(match.group(1))
                    name = match.group(3) if len(match.groups()) >= 3 else match.group(2)
                    definitions.append({
                        "line": i,
                        "type": def_type,
                        "name": name,
                        "indent": indent,
                    })
                    break

        if not definitions:
            return []

        # Extract chunks based on definitions
        for i, defn in enumerate(definitions):
            start = defn["line"]

            # Find end: next definition at same or lower indent, or end of file
            end = len(lines)
            for j in range(i + 1, len(definitions)):
                if definitions[j]["indent"] <= defn["indent"]:
                    end = definitions[j]["line"]
                    break

            # Extract content
            chunk_lines = lines[start:end]
            chunk_content = "\n".join(chunk_lines).rstrip()

            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=start + len(chunk_lines),
                    chunk_type=defn["type"],
                    name=defn["name"],
                ))

        # If chunks are too large, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.chunk_size * 2:
                # Split large chunks
                final_chunks.extend(self._split_large_chunk(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _line_chunk(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk code by lines with overlap."""
        lines = content.split("\n")
        chunks = []

        # Calculate lines per chunk (approximate)
        avg_line_len = len(content) / max(len(lines), 1)
        lines_per_chunk = max(10, int(self.chunk_size / avg_line_len))

        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = "\n".join(chunk_lines).rstrip()

            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end,
                    chunk_type="block",
                ))

            # Move forward with overlap
            overlap_lines = int(self.chunk_overlap / avg_line_len)
            i = end - overlap_lines if end < len(lines) else len(lines)

        return chunks

    def _split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split a large chunk into smaller pieces."""
        lines = chunk.content.split("\n")
        sub_chunks = []

        avg_line_len = len(chunk.content) / max(len(lines), 1)
        lines_per_chunk = max(10, int(self.chunk_size / avg_line_len))

        i = 0
        part = 1
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = "\n".join(chunk_lines).rstrip()

            if len(chunk_content) >= self.min_chunk_size:
                sub_chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line + i,
                    end_line=chunk.start_line + end - 1,
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name} (part {part})" if chunk.name else None,
                ))
                part += 1

            i = end

        return sub_chunks

    def _chunk_pdf(self, file_path: Path) -> List[CodeChunk]:
        """Chunk a PDF file by extracting text.

        Note: Images, tables, charts, and other embedded objects are NOT extracted.
        Only plain text content is indexed.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of chunks from the PDF
        """
        if not PDF_SUPPORT:
            warnings.warn(
                f"PDF support not available. Install with: pip install coding-factory[pdf]\n"
                f"Skipping: {file_path}"
            )
            return []

        chunks = []

        try:
            doc = fitz.open(file_path)

            # Warn about limitations on first PDF
            if not hasattr(self, "_pdf_warning_shown"):
                warnings.warn(
                    "PDF indexing extracts TEXT ONLY. Images, tables, charts, diagrams, "
                    "and other embedded objects are NOT supported and will be skipped."
                )
                self._pdf_warning_shown = True

            full_text = []
            page_boundaries = []  # Track where each page starts

            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    page_boundaries.append((len(full_text), page_num + 1))
                    full_text.append(page_text)

            doc.close()

            if not full_text:
                return []

            # Join all text and chunk by paragraphs
            combined_text = "\n\n".join(full_text)
            paragraphs = self._split_into_paragraphs(combined_text)

            # Group paragraphs into chunks
            current_chunk = []
            current_size = 0
            chunk_start_para = 0

            for i, para in enumerate(paragraphs):
                para_size = len(para)

                if current_size + para_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_content = "\n\n".join(current_chunk)
                    if len(chunk_content) >= self.min_chunk_size:
                        chunks.append(CodeChunk(
                            content=chunk_content,
                            file_path=str(file_path),
                            start_line=chunk_start_para + 1,
                            end_line=i,
                            chunk_type="document",
                            name=f"Section {len(chunks) + 1}",
                        ))

                    # Start new chunk with overlap
                    overlap_paras = []
                    overlap_size = 0
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) > self.chunk_overlap:
                            break
                        overlap_paras.insert(0, p)
                        overlap_size += len(p)

                    current_chunk = overlap_paras + [para]
                    current_size = overlap_size + para_size
                    chunk_start_para = i - len(overlap_paras)
                else:
                    current_chunk.append(para)
                    current_size += para_size

            # Don't forget the last chunk
            if current_chunk:
                chunk_content = "\n\n".join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=chunk_start_para + 1,
                        end_line=len(paragraphs),
                        chunk_type="document",
                        name=f"Section {len(chunks) + 1}",
                    ))

        except Exception as e:
            warnings.warn(f"Failed to process PDF {file_path}: {e}")
            return []

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Full text content

        Returns:
            List of paragraphs
        """
        # Split on double newlines or more
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Clean up and filter
        paragraphs = []
        for para in raw_paragraphs:
            cleaned = para.strip()
            # Skip very short paragraphs (likely headers or noise)
            if len(cleaned) > 20:
                paragraphs.append(cleaned)

        return paragraphs

    def _chunk_notebook(self, file_path: Path) -> List[CodeChunk]:
        """Chunk a Jupyter notebook by extracting cells.

        Args:
            file_path: Path to the .ipynb file

        Returns:
            List of chunks from the notebook
        """
        chunks = []

        try:
            content = file_path.read_text(encoding="utf-8")
            notebook = json.loads(content)
        except Exception as e:
            warnings.warn(f"Failed to parse notebook {file_path}: {e}")
            return []

        cells = notebook.get("cells", [])
        if not cells:
            return []

        # Process each cell
        current_chunk_content = []
        current_chunk_start = 0
        current_size = 0

        for cell_idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", [])

            # Handle source as list or string
            if isinstance(source, list):
                cell_content = "".join(source)
            else:
                cell_content = source

            cell_content = cell_content.strip()
            if not cell_content:
                continue

            # Add cell type prefix for context
            if cell_type == "markdown":
                formatted = f"[Markdown]\n{cell_content}"
            elif cell_type == "code":
                formatted = f"[Code Cell {cell_idx + 1}]\n{cell_content}"
            else:
                formatted = cell_content

            cell_size = len(formatted)

            # Check if adding this cell would exceed chunk size
            if current_size + cell_size > self.chunk_size and current_chunk_content:
                # Save current chunk
                chunk_content = "\n\n".join(current_chunk_content)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=current_chunk_start + 1,
                        end_line=cell_idx,
                        chunk_type="notebook",
                        name=f"Cells {current_chunk_start + 1}-{cell_idx}",
                    ))

                # Start new chunk
                current_chunk_content = [formatted]
                current_chunk_start = cell_idx
                current_size = cell_size
            else:
                current_chunk_content.append(formatted)
                current_size += cell_size

        # Don't forget the last chunk
        if current_chunk_content:
            chunk_content = "\n\n".join(current_chunk_content)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=current_chunk_start + 1,
                    end_line=len(cells),
                    chunk_type="notebook",
                    name=f"Cells {current_chunk_start + 1}-{len(cells)}",
                ))

        return chunks
