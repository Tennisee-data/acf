"""BM25 implementation for lexical search.

BM25 (Best Matching 25) is a ranking function used for lexical retrieval.
It extends TF-IDF with document length normalization.

Formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

Where:
    - f(qi,D) = term frequency of qi in document D
    - |D| = length of document D
    - avgdl = average document length
    - k1 = term frequency saturation parameter (default: 1.5)
    - b = length normalization parameter (default: 0.75)
"""

import math
import re
from collections import Counter


class BM25:
    """BM25 ranking for lexical search.

    Maintains an inverted index for fast retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25.

        Args:
            k1: Term frequency saturation (higher = more weight to TF)
            b: Length normalization (0 = no normalization, 1 = full)
        """
        self.k1 = k1
        self.b = b

        # Index structures
        self.documents: list[list[str]] = []  # Tokenized documents
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0

        # Inverted index: term -> {doc_id -> frequency}
        self.inverted_index: dict[str, dict[int, int]] = {}

        # Document frequencies: term -> number of docs containing term
        self.doc_freqs: dict[str, int] = {}

        # IDF cache
        self._idf_cache: dict[str, float] = {}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Input text

        Returns:
            List of lowercase tokens
        """
        # Lowercase and split on non-alphanumeric (keep underscores for code)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Document ID (index in documents list)
            text: Document text
        """
        tokens = self.tokenize(text)

        # Ensure documents list is large enough
        while len(self.documents) <= doc_id:
            self.documents.append([])
            self.doc_lengths.append(0)

        self.documents[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)

        # Update inverted index
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
                self.doc_freqs[term] = 0

            # Only increment doc_freq if this doc didn't have this term before
            if doc_id not in self.inverted_index[term]:
                self.doc_freqs[term] += 1

            self.inverted_index[term][doc_id] = count

        # Update stats
        self.doc_count = len([d for d in self.documents if d])
        total_length = sum(self.doc_lengths)
        self.avgdl = total_length / self.doc_count if self.doc_count > 0 else 0

        # Clear IDF cache (doc freqs changed)
        self._idf_cache.clear()

    def remove_document(self, doc_id: int) -> None:
        """Remove a document from the index.

        Args:
            doc_id: Document ID to remove
        """
        if doc_id >= len(self.documents) or not self.documents[doc_id]:
            return

        tokens = self.documents[doc_id]
        term_counts = Counter(tokens)

        for term in term_counts:
            if term in self.inverted_index and doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]
                self.doc_freqs[term] -= 1

                # Clean up empty entries
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
                    del self.doc_freqs[term]

        self.documents[doc_id] = []
        self.doc_lengths[doc_id] = 0

        # Update stats
        self.doc_count = len([d for d in self.documents if d])
        total_length = sum(self.doc_lengths)
        self.avgdl = total_length / self.doc_count if self.doc_count > 0 else 0

        self._idf_cache.clear()

    def _idf(self, term: str) -> float:
        """Calculate IDF for a term.

        Args:
            term: Query term

        Returns:
            IDF score
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        df = self.doc_freqs.get(term, 0)
        if df == 0:
            idf = 0.0
        else:
            # Standard BM25 IDF formula
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

        self._idf_cache[term] = idf
        return idf

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search for documents matching query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if self.doc_count == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Calculate scores for all candidate documents
        scores: dict[int, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self._idf(term)

            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0) + score

        # Sort by score and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_score(self, query: str, doc_id: int) -> float:
        """Get BM25 score for a specific document.

        Args:
            query: Search query
            doc_id: Document ID

        Returns:
            BM25 score
        """
        if doc_id >= len(self.documents) or not self.documents[doc_id]:
            return 0.0

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return 0.0

        score = 0.0
        doc_len = self.doc_lengths[doc_id]

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            if doc_id not in self.inverted_index[term]:
                continue

            tf = self.inverted_index[term][doc_id]
            idf = self._idf(term)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def clear(self) -> None:
        """Clear the index."""
        self.documents = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.doc_count = 0
        self.inverted_index = {}
        self.doc_freqs = {}
        self._idf_cache = {}

    def to_dict(self) -> dict:
        """Serialize index to dict for persistence.

        Returns:
            Dict representation
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {
                term: dict(postings)
                for term, postings in self.inverted_index.items()
            },
            "doc_freqs": self.doc_freqs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25":
        """Deserialize index from dict.

        Args:
            data: Dict representation

        Returns:
            BM25 instance
        """
        bm25 = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        bm25.documents = data.get("documents", [])
        bm25.doc_lengths = data.get("doc_lengths", [])
        bm25.inverted_index = {
            term: dict(postings)
            for term, postings in data.get("inverted_index", {}).items()
        }
        bm25.doc_freqs = data.get("doc_freqs", {})

        # Recalculate stats
        bm25.doc_count = len([d for d in bm25.documents if d])
        total_length = sum(bm25.doc_lengths)
        bm25.avgdl = total_length / bm25.doc_count if bm25.doc_count > 0 else 0

        return bm25
