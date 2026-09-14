"""
Dense retrieval module for the Research Paper Answer Bot.

Provides semantic retrieval using the persistent ChromaDB vector store.
"""

import time
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma


@dataclass
class RetrievalResult:
    """Represents one retrieved document chunk."""

    document: Document
    chunk_id: str
    paper_id: str
    paper_title: str
    page_number: int
    score: float


class DenseRetriever:
    """
    Dense semantic retriever backed by ChromaDB.

    The retriever uses the embedding function configured when the
    Chroma vector store was loaded.
    """

    def __init__(self, vectorstore: Chroma):
        """
        Initialize the dense retriever.

        Args:
            vectorstore:
                Loaded Chroma vector store.
        """
        if vectorstore is None:
            raise ValueError("vectorstore cannot be None.")

        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most semantically relevant chunks.

        Args:
            query:
                User's search question.

            top_k:
                Number of results to return.

        Returns:
            List of RetrievalResult objects ordered by relevance.

        Raises:
            ValueError:
                If the query is empty or top_k is invalid.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results_with_scores = self.vectorstore.similarity_search_with_score(
            query.strip(),
            k=top_k,
        )

        results: List[RetrievalResult] = []

        for document, score in results_with_scores:
            metadata = document.metadata

            results.append(
                RetrievalResult(
                    document=document,
                    chunk_id=str(metadata.get("chunk_id", "")),
                    paper_id=str(metadata.get("paper_id", "")),
                    paper_title=str(metadata.get("paper_title", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    score=float(score),
                )
            )

        return results

    def retrieve_with_latency(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[List[RetrievalResult], float]:
        """
        Retrieve documents and measure retrieval latency.

        Args:
            query:
                User's search question.

            top_k:
                Number of results to return.

        Returns:
            Tuple containing:
            - List of RetrievalResult objects
            - latency in seconds
        """
        start_time = time.perf_counter()

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        latency = time.perf_counter() - start_time

        return results, latency