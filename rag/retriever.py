"""Code retriever combining chunking, embedding, and search."""

from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .embeddings import OllamaEmbeddings
from .chunker import CodeChunker, CodeChunk
from .store import VectorStore, SearchResult


console = Console()


class CodeRetriever:
    """High-level interface for code RAG.

    Combines:
    - Code chunking
    - Embedding generation
    - Vector storage and search
    - Incremental updates
    - Multi-repository support

    Usage:
        retriever = CodeRetriever(repo_path)
        retriever.index()  # Full index
        retriever.incremental_index()  # Update only changed files
        results = retriever.search("authentication logic")
    """

    def __init__(
        self,
        repo_path: Path,
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        store_name: str = ".coding-factory-index",
        repo_id: str | None = None,
    ):
        """Initialize retriever.

        Args:
            repo_path: Path to the repository
            embedding_model: Ollama embedding model to use
            ollama_url: Ollama API URL
            store_name: Name of the index directory
            repo_id: Repository identifier for multi-repo support
        """
        self.repo_path = Path(repo_path)
        self.store_path = self.repo_path / store_name
        self.repo_id = repo_id or self.repo_path.name

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url,
        )
        self.chunker = CodeChunker()
        self.store = VectorStore(store_path=self.store_path, repo_id=self.repo_id)

    def index(
        self,
        extensions: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
        force: bool = False,
        incremental: bool = False,
    ) -> dict:
        """Index the repository.

        Args:
            extensions: File extensions to index
            exclude_patterns: Patterns to exclude
            force: Re-index even if index exists
            incremental: Only index changed files

        Returns:
            Indexing statistics
        """
        if incremental and self.store_path.exists():
            return self.incremental_index(extensions, exclude_patterns)

        if self.store_path.exists() and not force:
            console.print("[yellow]Index already exists. Use --force to re-index or --incremental for updates.[/yellow]")
            return self.store.stats()

        # Ensure embedding model is available
        console.print(f"[dim]Checking embedding model: {self.embeddings.model}[/dim]")
        if not self.embeddings.ensure_model():
            raise RuntimeError(f"Could not load embedding model: {self.embeddings.model}")

        # Clear existing store
        self.store.clear()

        # Register this repo
        self.store.register_repo(self.repo_id, str(self.repo_path))

        # Collect all files first (for change tracking)
        console.print("[bold]Scanning files...[/bold]")
        all_files = list(self._collect_files(extensions, exclude_patterns))
        console.print(f"[dim]Found {len(all_files)} files to index[/dim]")

        # Collect chunks and track file metadata
        console.print("[bold]Chunking code files...[/bold]")
        chunks = []
        file_chunks: dict[str, list] = {}  # Track chunks per file

        for file_path in all_files:
            file_chunks_list = self.chunker.chunk_file(file_path)
            if file_chunks_list:
                rel_path = str(file_path.relative_to(self.repo_path))
                file_chunks[rel_path] = file_chunks_list
                chunks.extend(file_chunks_list)

                # Track file metadata
                file_hash = VectorStore.compute_file_hash(file_path)
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                self.store.set_file_metadata(
                    rel_path, file_hash, mod_time, len(file_chunks_list)
                )

        console.print(f"[green]Found {len(chunks)} chunks from {len(file_chunks)} files[/green]")

        if not chunks:
            console.print("[yellow]No code files found to index.[/yellow]")
            self.store.save()
            return {"total_chunks": 0, "total_files": 0}

        # Generate embeddings
        console.print("[bold]Generating embeddings...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(chunks))

            batch_size = 10
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.to_text() for c in batch]
                embeddings = self.embeddings.embed_batch(texts)
                all_embeddings.extend(embeddings)
                progress.update(task, advance=len(batch))

        # Store embeddings
        self.store.add(chunks, all_embeddings)
        self.store.update_repo_stats(self.repo_id, len(file_chunks))
        self.store.save()

        stats = self.store.stats()
        console.print(f"[green]Indexed {stats['total_chunks']} chunks from {stats['total_files']} files[/green]")

        return stats

    def _collect_files(
        self,
        extensions: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
    ) -> List[Path]:
        """Collect all indexable files from the repository.

        Args:
            extensions: File extensions to include
            exclude_patterns: Patterns to exclude

        Returns:
            List of file paths
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".md", ".txt", ".pdf", ".ipynb"]

        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", "node_modules", ".git", ".venv", "venv",
                "dist", "build", ".egg-info", ".coding-factory-index",
            ]

        files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_dir():
                continue

            if file_path.suffix.lower() not in extensions:
                continue

            path_str = str(file_path)
            if any(pattern in path_str for pattern in exclude_patterns):
                continue

            files.append(file_path)

        return files

    def incremental_index(
        self,
        extensions: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
    ) -> dict:
        """Incrementally update the index with only changed files.

        Args:
            extensions: File extensions to index
            exclude_patterns: Patterns to exclude

        Returns:
            Indexing statistics including update info
        """
        if not self.store_path.exists():
            console.print("[yellow]No existing index. Running full index...[/yellow]")
            return self.index(extensions, exclude_patterns, force=True)

        # Ensure embedding model is available
        console.print(f"[dim]Checking embedding model: {self.embeddings.model}[/dim]")
        if not self.embeddings.ensure_model():
            raise RuntimeError(f"Could not load embedding model: {self.embeddings.model}")

        # Collect all current files
        console.print("[bold]Scanning for changes...[/bold]")
        all_files = list(self._collect_files(extensions, exclude_patterns))

        # Find changed and deleted files
        changed_files, deleted_files = self.store.get_changed_files(
            self.repo_path, all_files
        )

        if not changed_files and not deleted_files:
            console.print("[green]Index is up to date. No changes detected.[/green]")
            return {
                **self.store.stats(),
                "files_added": 0,
                "files_updated": 0,
                "files_deleted": 0,
            }

        console.print(f"[dim]Found {len(changed_files)} changed files, {len(deleted_files)} deleted files[/dim]")

        # Remove deleted files from index
        for rel_path in deleted_files:
            self.store.delete_file(rel_path)
            console.print(f"[red]- Removed: {rel_path}[/red]")

        # Process changed files
        if changed_files:
            console.print("[bold]Processing changed files...[/bold]")

            total_chunks = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Indexing...", total=len(changed_files))

                for file_path in changed_files:
                    rel_path = str(file_path.relative_to(self.repo_path))

                    # Remove old chunks for this file
                    self.store.delete_file(rel_path)

                    # Chunk the file
                    file_chunks = self.chunker.chunk_file(file_path)
                    if not file_chunks:
                        progress.update(task, advance=1)
                        continue

                    # Generate embeddings
                    texts = [c.to_text() for c in file_chunks]
                    embeddings = self.embeddings.embed_batch(texts)

                    # Add to store
                    self.store.add(file_chunks, embeddings)
                    total_chunks += len(file_chunks)

                    # Update file metadata
                    file_hash = VectorStore.compute_file_hash(file_path)
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    self.store.set_file_metadata(
                        rel_path, file_hash, mod_time, len(file_chunks)
                    )

                    progress.update(task, advance=1)

            console.print(f"[green]Added {total_chunks} chunks from {len(changed_files)} files[/green]")

        self.store.save()

        stats = self.store.stats()
        stats.update({
            "files_added": len([f for f in changed_files if self.store.get_file_metadata(
                str(f.relative_to(self.repo_path))
            )]),
            "files_updated": len(changed_files),
            "files_deleted": len(deleted_files),
        })

        console.print(f"[green]Index updated: {stats['total_chunks']} total chunks[/green]")
        return stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[SearchResult]:
        """Search for relevant code.

        Args:
            query: Natural language query
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of search results
        """
        if not self.store.chunks:
            # Try loading from disk
            if self.store_path.exists():
                self.store.load()
            else:
                console.print("[yellow]No index found. Run 'coding-factory index' first.[/yellow]")
                return []

        # Generate query embedding
        query_embedding = self.embeddings.embed(query)

        # Search
        results = self.store.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 8000,
        top_k: int = 20,
    ) -> str:
        """Get context for an LLM prompt.

        Args:
            query: The query or task description
            max_tokens: Maximum tokens for context
            top_k: Number of chunks to consider

        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return ""

        # Build context, respecting token limit
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate: 4 chars per token

        for result in results:
            chunk_text = result.chunk.to_text()
            if total_chars + len(chunk_text) > max_chars:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def update_file(self, file_path: Path) -> int:
        """Update index for a single file.

        Args:
            file_path: Path to the changed file

        Returns:
            Number of chunks added
        """
        # Remove old chunks for this file
        rel_path = str(file_path.relative_to(self.repo_path))
        self.store.delete_file(rel_path)

        # Re-chunk the file
        chunks = self.chunker.chunk_file(file_path)
        if not chunks:
            self.store.save()
            return 0

        # Generate embeddings
        texts = [c.to_text() for c in chunks]
        embeddings = self.embeddings.embed_batch(texts)

        # Add to store
        self.store.add(chunks, embeddings)
        self.store.save()

        return len(chunks)

    def stats(self) -> dict:
        """Get index statistics."""
        if not self.store.chunks and self.store_path.exists():
            self.store.load()
        return self.store.stats()

    def is_indexed(self) -> bool:
        """Check if repository is indexed."""
        return self.store_path.exists() and (self.store_path / "metadata.json").exists()

    def list_repos(self) -> List[dict]:
        """List all indexed repositories.

        Returns:
            List of repository info dicts
        """
        if not self.store.chunks and self.store_path.exists():
            self.store.load()
        return self.store.list_repos()

    def add_repo(
        self,
        repo_path: Path,
        repo_id: str | None = None,
        extensions: List[str] | None = None,
        exclude_patterns: List[str] | None = None,
    ) -> dict:
        """Add another repository to the index.

        Args:
            repo_path: Path to the repository to add
            repo_id: Optional identifier (defaults to directory name)
            extensions: File extensions to index
            exclude_patterns: Patterns to exclude

        Returns:
            Indexing statistics
        """
        repo_path = Path(repo_path)
        repo_id = repo_id or repo_path.name

        # Check if repo already exists
        existing_repos = self.store.list_repos()
        if any(r["repo_id"] == repo_id for r in existing_repos):
            console.print(f"[yellow]Repository '{repo_id}' already indexed. Use remove_repo first.[/yellow]")
            return self.store.stats()

        # Ensure embedding model is available
        console.print(f"[dim]Checking embedding model: {self.embeddings.model}[/dim]")
        if not self.embeddings.ensure_model():
            raise RuntimeError(f"Could not load embedding model: {self.embeddings.model}")

        # Register this repo
        self.store.register_repo(repo_id, str(repo_path))

        # Use a temporary chunker for the new repo
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".md", ".txt", ".pdf", ".ipynb"]

        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", "node_modules", ".git", ".venv", "venv",
                "dist", "build", ".egg-info", ".coding-factory-index",
            ]

        # Collect files from the new repo
        console.print(f"[bold]Indexing repository: {repo_id}[/bold]")
        files = []
        for file_path in repo_path.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in extensions:
                continue
            path_str = str(file_path)
            if any(pattern in path_str for pattern in exclude_patterns):
                continue
            files.append(file_path)

        console.print(f"[dim]Found {len(files)} files[/dim]")

        # Chunk and embed
        chunks = []
        file_count = 0
        for file_path in files:
            file_chunks = self.chunker.chunk_file(file_path)
            if file_chunks:
                # Use full path as file_path to distinguish repos
                for chunk in file_chunks:
                    chunk.file_path = str(file_path)
                chunks.extend(file_chunks)

                # Track file metadata with repo prefix
                rel_path = f"{repo_id}/{file_path.relative_to(repo_path)}"
                file_hash = VectorStore.compute_file_hash(file_path)
                from datetime import datetime
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                self.store.set_file_metadata(rel_path, file_hash, mod_time, len(file_chunks))
                file_count += 1

        if not chunks:
            console.print("[yellow]No code files found to index.[/yellow]")
            self.store.save()
            return {"total_chunks": 0, "total_files": 0}

        # Generate embeddings
        console.print("[bold]Generating embeddings...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(chunks))
            batch_size = 10
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.to_text() for c in batch]
                embeddings = self.embeddings.embed_batch(texts)
                all_embeddings.extend(embeddings)
                progress.update(task, advance=len(batch))

        # Add to store
        self.store.add(chunks, all_embeddings)
        self.store.update_repo_stats(repo_id, file_count)
        self.store.save()

        console.print(f"[green]Added {len(chunks)} chunks from {file_count} files in '{repo_id}'[/green]")
        return self.store.stats()

    def remove_repo(self, repo_id: str) -> int:
        """Remove a repository from the index.

        Args:
            repo_id: Repository identifier

        Returns:
            Number of chunks removed
        """
        deleted = self.store.delete_repo(repo_id)
        if deleted > 0:
            self.store.save()
            console.print(f"[green]Removed {deleted} chunks from '{repo_id}'[/green]")
        else:
            console.print(f"[yellow]Repository '{repo_id}' not found[/yellow]")
        return deleted
