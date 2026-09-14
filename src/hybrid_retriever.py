"""
Hybrid Retriever module for the Research Paper Answer Bot.

Combines dense semantic retrieval and BM25 lexical retrieval
using Reciprocal Rank Fusion (RRF).
"""

from typing import Any, Dict, List, Tuple

from src.chunking import DocumentChunk


class HybridRetriever:
    """Combines dense vector retrieval and BM25 retrieval."""

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            dense_retriever:
                DenseRetriever instance.

            bm25_retriever:
                BM25RetrieverWrapper instance.

            alpha:
                Weight given to dense retrieval.
                BM25 receives (1 - alpha).

            rrf_k:
                RRF constant used to reduce the impact of rank.
        """
        if dense_retriever is None:
            raise ValueError("dense_retriever cannot be None.")

        if bm25_retriever is None:
            raise ValueError("bm25_retriever cannot be None.")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve documents using hybrid dense + BM25 retrieval.

        Reciprocal Rank Fusion is used because dense and BM25
        retrieval scores are on different scales.

        Args:
            query:
                User's search query.

            top_k:
                Number of final hybrid results.

        Returns:
            List of (DocumentChunk, hybrid_score) tuples,
            ordered from highest to lowest score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Retrieve extra candidates from both systems.
        # This gives the fusion stage more candidates to work with.
        candidate_k = max(top_k * 2, 10)

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # Store chunks and their RRF scores by unique chunk ID.
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_rrf_scores: Dict[str, float] = {}
        bm25_rrf_scores: Dict[str, float] = {}

        # ---------------------------------------------------------
        # Dense RRF scores
        # ---------------------------------------------------------
        for rank, result in enumerate(dense_results, start=1):
            chunk = result.document

            chunk_id = str(
                chunk.metadata.get("chunk_id", result.chunk_id)
            )

            if not chunk_id:
                continue

            chunk_map[chunk_id] = self._document_to_chunk(chunk, result)

            dense_rrf_scores[chunk_id] = (
                1.0 / (self.rrf_k + rank)
            )

        # ---------------------------------------------------------
        # BM25 RRF scores
        # ---------------------------------------------------------
        for rank, (chunk, _bm25_score) in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = str(chunk.metadata.chunk_id)

            if not chunk_id:
                continue

            chunk_map[chunk_id] = chunk

            bm25_rrf_scores[chunk_id] = (
                1.0 / (self.rrf_k + rank)
            )

        # ---------------------------------------------------------
        # Combine dense + BM25 scores
        # ---------------------------------------------------------
        all_chunk_ids = set(chunk_map)

        fused_results = []

        for chunk_id in all_chunk_ids:
            dense_score = dense_rrf_scores.get(
                chunk_id,
                0.0,
            )

            bm25_score = bm25_rrf_scores.get(
                chunk_id,
                0.0,
            )

            hybrid_score = (
                self.alpha * dense_score
                + (1.0 - self.alpha) * bm25_score
            )

            fused_results.append(
                (
                    chunk_map[chunk_id],
                    hybrid_score,
                )
            )

        # Highest hybrid score first.
        fused_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return fused_results[:top_k]

    @staticmethod
    def _document_to_chunk(
        document: Any,
        dense_result: Any,
    ) -> DocumentChunk:
        """
        Convert a LangChain Document returned by the dense
        retriever into the project's DocumentChunk structure.

        The original DocumentChunk is not stored in Chroma, so
        its content and metadata are reconstructed here.
        """
        from src.chunking import ChunkMetadata

        metadata = document.metadata

        chunk_metadata = ChunkMetadata(
            chunk_id=str(metadata.get("chunk_id", "")),
            paper_id=str(metadata.get("paper_id", "")),
            paper_title=str(metadata.get("paper_title", "")),
            authors=str(metadata.get("authors", "")),
            year=int(metadata.get("year", 0)),
            topic=str(metadata.get("topic", "")),
            source_filename=str(
                metadata.get("source_filename", "")
            ),
            source_url=str(metadata.get("source_url", "")),
            pdf_url=str(metadata.get("pdf_url", "")),
            page_number=int(
                metadata.get("page_number", 0)
            ),
            chunk_size=int(
                metadata.get("chunk_size", 0)
            ),
            chunk_overlap=int(
                metadata.get("chunk_overlap", 0)
            ),
        )

        return DocumentChunk(
            content=document.page_content,
            metadata=chunk_metadata,
        )