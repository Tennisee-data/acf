"""Test RAG Retriever - Verifies marketplace RAG extension integration."""

from typing import Any, List, Tuple


class TestRAGRetriever:
    """Simple test retriever to verify extension loading works.

    This retriever returns a fixed response to prove the extension
    system properly loads and uses marketplace RAG kits.
    """

    def __init__(self):
        """Initialize the test retriever."""
        print("\n" + "=" * 60)
        print("TEST RAG EXTENSION LOADED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        """Retrieve test content.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of test documents
        """
        print(f"\n[TEST RAG] retrieve() called with query: {query[:50]}...")
        return [
            {"content": f"Test result for: {query}", "score": 0.95},
        ]

    def retrieve_with_budget(
        self,
        query: str,
        token_budget: int,
        top_k: int = 50,
    ) -> Tuple[List[dict], str]:
        """Retrieve content within token budget.

        Args:
            query: Search query
            token_budget: Maximum tokens
            top_k: Maximum results

        Returns:
            Tuple of (results, formatted_content)
        """
        print(f"\n[TEST RAG] retrieve_with_budget() called!")
        print(f"[TEST RAG] Query: {query[:50]}...")
        print(f"[TEST RAG] Token budget: {token_budget}")

        # Return test content that proves the extension was used
        test_content = f"""
## Test RAG Extension Active

This content is from the **test-rag** marketplace extension.
If you see this, the RAG extension integration is working correctly!

Query received: {query[:100]}...
Token budget: {token_budget}
"""

        results = [{"content": test_content, "score": 1.0}]

        return results, test_content
