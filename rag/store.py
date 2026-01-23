"""Vector store for code embeddings."""

import hashlib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple


from .chunker import CodeChunk


@dataclass
class FileMetadata:
    """Metadata for an indexed file."""

    file_path: str
    content_hash: str  # MD5 hash of file content
    modified_time: str  # ISO format timestamp
    chunk_count: int
    indexed_at: str  # When this file was indexed


@dataclass
class SearchResult:
    """A search result with score."""

    chunk: CodeChunk
    score: float  # Similarity score (0-1, higher is better)


class VectorStore:
    """Simple vector store using NumPy.

    Stores embeddings and metadata for code chunks.
    Uses cosine similarity for search.
    Supports incremental updates and multi-repository indexing.

    For production, consider:
    - ChromaDB
    - FAISS
    - Qdrant
    - Pinecone
    """

    def __init__(self, store_path: Path | None = None, repo_id: str | None = None):
        """Initialize vector store.

        Args:
            store_path: Path to persist the store (optional)
            repo_id: Repository identifier for multi-repo support
        """
        self.store_path = store_path
        self.embeddings: np.ndarray | None = None
        self.chunks: List[CodeChunk] = []
        self.metadata: dict = {
            "version": 2,
            "count": 0,
            "dimensions": 0,
            "repo_id": repo_id,
            "repos": {},  # repo_id -> {"path": str, "indexed_at": str, "file_count": int}
            "files": {},  # file_path -> FileMetadata as dict
            "last_indexed": None,
        }

        # Load existing store if available
        if store_path and store_path.exists():
            self.load()
            # Update repo_id if provided
            if repo_id and self.metadata.get("repo_id") != repo_id:
                self.metadata["repo_id"] = repo_id

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file's content.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hex digest
        """
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def get_file_metadata(self, file_path: str) -> FileMetadata | None:
        """Get metadata for an indexed file.

        Args:
            file_path: Relative path to the file

        Returns:
            FileMetadata or None if not indexed
        """
        file_data = self.metadata.get("files", {}).get(file_path)
        if file_data:
            return FileMetadata(**file_data)
        return None

    def set_file_metadata(
        self,
        file_path: str,
        content_hash: str,
        modified_time: str,
        chunk_count: int,
    ) -> None:
        """Set metadata for a file.

        Args:
            file_path: Relative path to the file
            content_hash: MD5 hash of content
            modified_time: File modification time (ISO format)
            chunk_count: Number of chunks from this file
        """
        if "files" not in self.metadata:
            self.metadata["files"] = {}

        self.metadata["files"][file_path] = {
            "file_path": file_path,
            "content_hash": content_hash,
            "modified_time": modified_time,
            "chunk_count": chunk_count,
            "indexed_at": datetime.now().isoformat(),
        }

    def is_file_changed(self, file_path: Path, rel_path: str) -> bool:
        """Check if a file has changed since last indexing.

        Args:
            file_path: Absolute path to the file
            rel_path: Relative path (used as key in metadata)

        Returns:
            True if file is new or has changed
        """
        existing = self.get_file_metadata(rel_path)
        if not existing:
            return True

        current_hash = self.compute_file_hash(file_path)
        return current_hash != existing.content_hash

    def get_changed_files(
        self,
        directory: Path,
        file_paths: List[Path],
    ) -> tuple[List[Path], List[str]]:
        """Get list of files that have changed since last index.

        Args:
            directory: Base directory for relative paths
            file_paths: List of file paths to check

        Returns:
            Tuple of (changed_files, deleted_files)
        """
        changed = []
        indexed_files = set(self.metadata.get("files", {}).keys())
        current_files = set()

        for file_path in file_paths:
            try:
                rel_path = str(file_path.relative_to(directory))
            except ValueError:
                rel_path = str(file_path)

            current_files.add(rel_path)

            if self.is_file_changed(file_path, rel_path):
                changed.append(file_path)

        # Files that were indexed but no longer exist
        deleted = list(indexed_files - current_files)

        return changed, deleted

    def add(self, chunks: List[CodeChunk], embeddings: List[List[float]]) -> None:
        """Add chunks with their embeddings.

        Args:
            chunks: Code chunks to add
            embeddings: Corresponding embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.chunks = chunks
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.chunks.extend(chunks)

        self.metadata["count"] = len(self.chunks)
        self.metadata["dimensions"] = new_embeddings.shape[1]
        self.metadata["last_indexed"] = datetime.now().isoformat()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        repo_id: str | None = None,
    ) -> List[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            repo_id: Filter by repository ID (for multi-repo support)

        Returns:
            List of search results sorted by score
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        embeddings_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        )

        # Compute cosine similarity
        similarities = embeddings_norm @ query_norm

        # Filter by repo_id if specified
        if repo_id:
            # Create mask for chunks belonging to this repo
            # Chunks store file_path, and repos store file patterns
            repo_info = self.metadata.get("repos", {}).get(repo_id)
            if repo_info:
                repo_path = repo_info.get("path", "")
                valid_indices = [
                    i for i, c in enumerate(self.chunks)
                    if c.file_path.startswith(repo_path) or repo_path == ""
                ]
                if valid_indices:
                    # Only consider valid indices
                    valid_set = set(valid_indices)
                    top_indices = [
                        i for i in np.argsort(similarities)[::-1]
                        if i in valid_set
                    ][:top_k]
                else:
                    return []
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                ))

        return results

    def search_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks from a specific file.

        Args:
            file_path: Path to search for

        Returns:
            List of chunks from that file
        """
        return [c for c in self.chunks if c.file_path == file_path]

    def delete_file(self, file_path: str) -> int:
        """Delete all chunks from a file.

        Args:
            file_path: Path to delete

        Returns:
            Number of chunks deleted
        """
        indices_to_keep = [
            i for i, c in enumerate(self.chunks)
            if c.file_path != file_path
        ]

        if len(indices_to_keep) == len(self.chunks):
            return 0

        deleted = len(self.chunks) - len(indices_to_keep)

        self.chunks = [self.chunks[i] for i in indices_to_keep]
        if self.embeddings is not None and indices_to_keep:
            self.embeddings = self.embeddings[indices_to_keep]
        elif not indices_to_keep:
            self.embeddings = None

        self.metadata["count"] = len(self.chunks)

        # Remove file metadata
        if "files" in self.metadata and file_path in self.metadata["files"]:
            del self.metadata["files"][file_path]

        return deleted

    def clear(self, keep_repos: bool = False) -> None:
        """Clear all data from the store.

        Args:
            keep_repos: If True, keep repo metadata but clear chunks
        """
        self.embeddings = None
        self.chunks = []
        self.metadata["count"] = 0
        self.metadata["files"] = {}
        self.metadata["last_indexed"] = None
        if not keep_repos:
            self.metadata["repos"] = {}

    def register_repo(self, repo_id: str, repo_path: str) -> None:
        """Register a repository for multi-repo support.

        Args:
            repo_id: Unique identifier for the repository
            repo_path: Base path prefix for files in this repo
        """
        if "repos" not in self.metadata:
            self.metadata["repos"] = {}

        self.metadata["repos"][repo_id] = {
            "path": repo_path,
            "indexed_at": datetime.now().isoformat(),
            "file_count": 0,
        }

    def update_repo_stats(self, repo_id: str, file_count: int) -> None:
        """Update statistics for a repository.

        Args:
            repo_id: Repository identifier
            file_count: Number of files indexed
        """
        if "repos" in self.metadata and repo_id in self.metadata["repos"]:
            self.metadata["repos"][repo_id]["file_count"] = file_count
            self.metadata["repos"][repo_id]["indexed_at"] = datetime.now().isoformat()

    def list_repos(self) -> List[dict]:
        """List all registered repositories.

        Returns:
            List of repo info dicts
        """
        repos = self.metadata.get("repos", {})
        return [
            {"repo_id": rid, **info}
            for rid, info in repos.items()
        ]

    def delete_repo(self, repo_id: str) -> int:
        """Delete all chunks from a repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Number of chunks deleted
        """
        if "repos" not in self.metadata or repo_id not in self.metadata["repos"]:
            return 0

        repo_info = self.metadata["repos"][repo_id]
        repo_path = repo_info.get("path", "")

        # Find and delete chunks belonging to this repo
        indices_to_keep = [
            i for i, c in enumerate(self.chunks)
            if not c.file_path.startswith(repo_path)
        ]

        deleted = len(self.chunks) - len(indices_to_keep)

        if deleted > 0:
            self.chunks = [self.chunks[i] for i in indices_to_keep]
            if self.embeddings is not None and indices_to_keep:
                self.embeddings = self.embeddings[indices_to_keep]
            elif not indices_to_keep:
                self.embeddings = None

            self.metadata["count"] = len(self.chunks)

        # Remove file metadata for this repo
        if "files" in self.metadata:
            files_to_delete = [
                f for f in self.metadata["files"]
                if f.startswith(repo_path)
            ]
            for f in files_to_delete:
                del self.metadata["files"][f]

        # Remove repo registration
        del self.metadata["repos"][repo_id]

        return deleted

    def save(self) -> None:
        """Save store to disk."""
        if self.store_path is None:
            raise ValueError("No store path configured")

        self.store_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self.embeddings is not None:
            np.save(self.store_path / "embeddings.npy", self.embeddings)

        # Save chunks as JSON
        chunks_data = [
            {
                "content": c.content,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_type": c.chunk_type,
                "name": c.name,
            }
            for c in self.chunks
        ]
        with open(self.store_path / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        # Save metadata
        with open(self.store_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self) -> None:
        """Load store from disk."""
        if self.store_path is None:
            raise ValueError("No store path configured")

        # Load metadata
        metadata_path = self.store_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                loaded_metadata = json.load(f)

            # Migrate from v1 to v2 if needed
            if loaded_metadata.get("version", 1) < 2:
                loaded_metadata["version"] = 2
                loaded_metadata.setdefault("repo_id", None)
                loaded_metadata.setdefault("repos", {})
                loaded_metadata.setdefault("files", {})
                loaded_metadata.setdefault("last_indexed", None)

            self.metadata = loaded_metadata

        # Load embeddings
        embeddings_path = self.store_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # Load chunks
        chunks_path = self.store_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks_data = json.load(f)
                self.chunks = [
                    CodeChunk(**c) for c in chunks_data
                ]

    def stats(self) -> dict:
        """Get store statistics."""
        files = set(c.file_path for c in self.chunks)
        repos = self.metadata.get("repos", {})

        return {
            "total_chunks": len(self.chunks),
            "total_files": len(files),
            "dimensions": self.metadata.get("dimensions", 0),
            "files": list(files)[:20],  # First 20 files
            "last_indexed": self.metadata.get("last_indexed"),
            "version": self.metadata.get("version", 1),
            "repos": list(repos.keys()),
            "tracked_files": len(self.metadata.get("files", {})),
        }
