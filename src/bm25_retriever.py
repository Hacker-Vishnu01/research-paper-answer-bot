"""
BM25 Retriever module for the Research Paper Answer Bot.

Provides sparse keyword-based lexical retrieval using the BM25 algorithm.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.chunking import DocumentChunk


class BM25RetrieverWrapper:
    """Wrapper around BM25 for lexical document search."""

    def __init__(self, chunks: List[DocumentChunk]):
        """
        Initialize the BM25 retriever index.

        Args:
            chunks: List of DocumentChunk instances to index.
        """
        if not chunks:
            raise ValueError("chunks cannot be empty.")

        self.chunks = chunks

        # Tokenize every chunk and build the BM25 index.
        tokenized_documents = [
            self._tokenize(chunk.content)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 retrieval.

        Keeps words, numbers, and common technical tokens.
        """
        if not text or not text.strip():
            return []

        return re.findall(r"[A-Za-z0-9_./+-]+", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k document chunks matching the query.

        Args:
            query: User's search query string.
            top_k: Number of highest-scoring chunks to return.

        Returns:
            List of tuples:
                (DocumentChunk, BM25 score)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Sort by score descending while preserving chunk index.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: List[Tuple[DocumentChunk, float]] = []

        for index in ranked_indices[:top_k]:
            results.append(
                (
                    self.chunks[index],
                    float(scores[index]),
                )
            )

        return results