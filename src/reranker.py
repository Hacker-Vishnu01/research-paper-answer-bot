"""
Reranker module for the Research Paper Answer Bot.

Applies a Cross-Encoder model to score and re-order candidate passages
retrieved by dense or hybrid search algorithms.
"""

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.chunking import DocumentChunk


class CrossEncoderReranker:
    """Reranks candidate document chunks using a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the Cross-Encoder model.

        Args:
            model_name: Hugging Face model identifier for the Cross-Encoder.
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Score and rerank candidate passages for a query.

        Args:
            query: User's query string.
            candidates: Candidate document chunks.
            top_k: Number of reranked candidates to return.

        Returns:
            List of (DocumentChunk, score) sorted by descending relevance.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        top_k = min(top_k, len(candidates))

        pairs = [
            (query, candidate.content)
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        scored_candidates = [
            (candidate, float(score))
            for candidate, score in zip(candidates, scores)
        ]

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_candidates[:top_k]
