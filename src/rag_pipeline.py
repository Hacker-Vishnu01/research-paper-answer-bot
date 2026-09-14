"""
RAG Pipeline Orchestrator for the Research Paper Answer Bot.

Coordinates:
    1. Hybrid/dense/BM25 retrieval
    2. Optional Cross-Encoder reranking
    3. Grounded prompt construction
    4. LLM answer generation
    5. Top-3 academic citation extraction

The pipeline is designed to prevent unsupported answers by requiring
the LLM to use only the retrieved research passages.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_context_passages,
)


# ============================================================
# CONSTANTS
# ============================================================

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The provided research papers do not contain sufficient evidence "
    "to answer this question."
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Citation:
    """Represents a verifiable academic citation supporting an answer."""

    paper_title: str
    page_number: int
    passage: str


@dataclass
class RAGResponse:
    """Standardized output structure for RAG question answering."""

    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Any]
    latency_seconds: float
    retrieval_strategy: str


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Orchestrates end-to-end question answering over academic papers.

    Pipeline:

        Question
            ↓
        Retriever
            ↓
        Candidate chunks
            ↓
        Optional Cross-Encoder reranking
            ↓
        Top context passages
            ↓
        Grounded prompt
            ↓
        LLM
            ↓
        Answer + citations
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Optional[Any] = None,
        llm: Optional[Any] = None,
        retrieval_top_k: int = 10,
        context_top_k: int = 3,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever:
                Dense, BM25, or HybridRetriever instance.

            reranker:
                Optional CrossEncoderReranker instance.

            llm:
                LangChain-compatible LLM instance.

            retrieval_top_k:
                Number of candidate chunks retrieved before
                optional reranking.

            context_top_k:
                Number of final passages supplied to the LLM.
        """

        if retriever is None:
            raise ValueError(
                "retriever cannot be None."
            )

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if context_top_k <= 0:
            raise ValueError(
                "context_top_k must be greater than zero."
            )

        if context_top_k > retrieval_top_k:
            raise ValueError(
                "context_top_k cannot be greater than "
                "retrieval_top_k."
            )

        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    # ========================================================
    # MAIN ANSWER METHOD
    # ========================================================

    def answer(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Process a user question and return a grounded answer.

        Args:
            question:
                User's academic research question.

        Returns:
            RAGResponse containing:
                - generated answer
                - top supporting citations
                - retrieved chunks
                - latency
                - retrieval strategy
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if self.llm is None:
            raise RuntimeError(
                "No LLM has been configured for the RAG pipeline. "
                "Provide a LangChain-compatible LLM instance."
            )

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # STEP 1: RETRIEVE CANDIDATE CHUNKS
        # ----------------------------------------------------

        retrieved_results = self.retriever.retrieve(
            query=question,
            top_k=self.retrieval_top_k,
        )

        if not retrieved_results:
            latency = (
                time.perf_counter()
                - start_time
            )

            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                latency_seconds=latency,
                retrieval_strategy=self._get_retrieval_strategy(),
            )

        # ----------------------------------------------------
        # STEP 2: NORMALIZE RETRIEVED CHUNKS
        # ----------------------------------------------------

        candidate_chunks = self._extract_chunks(
            retrieved_results
        )

        if not candidate_chunks:
            latency = (
                time.perf_counter()
                - start_time
            )

            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                latency_seconds=latency,
                retrieval_strategy=self._get_retrieval_strategy(),
            )

        # ----------------------------------------------------
        # STEP 3: OPTIONAL CROSS-ENCODER RERANKING
        # ----------------------------------------------------

        if self.reranker is not None:

            reranked_results = self.reranker.rerank(
                query=question,
                candidates=candidate_chunks,
                top_k=min(
                    self.retrieval_top_k,
                    len(candidate_chunks),
                ),
            )

            candidate_chunks = [
                chunk
                for chunk, score in reranked_results
            ]

        # ----------------------------------------------------
        # STEP 4: SELECT FINAL CONTEXT
        # ----------------------------------------------------

        context_chunks = candidate_chunks[
            :self.context_top_k
        ]

        if not context_chunks:
            latency = (
                time.perf_counter()
                - start_time
            )

            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                latency_seconds=latency,
                retrieval_strategy=self._get_retrieval_strategy(),
            )

        # ----------------------------------------------------
        # STEP 5: FORMAT CONTEXT
        # ----------------------------------------------------

        passages = self._chunks_to_passages(
            context_chunks
        )

        context = format_context_passages(
            passages
        )

        # ----------------------------------------------------
        # STEP 6: BUILD GROUNDED PROMPT
        # ----------------------------------------------------

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # ----------------------------------------------------
        # STEP 7: CALL LLM
        # ----------------------------------------------------

        answer_text = self._invoke_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # ----------------------------------------------------
        # STEP 8: NORMALIZE ANSWER
        # ----------------------------------------------------

        answer_text = answer_text.strip()

        # ----------------------------------------------------
        # STEP 9: CREATE CITATIONS
        # ----------------------------------------------------

        # If the LLM determines that the provided context does
        # not contain enough evidence, do not display the
        # retrieved passages as supporting citations.
        if self._is_insufficient_evidence(
            answer_text
        ):
            citations = []
        else:
            citations = self._create_citations(
                context_chunks
            )

        # ----------------------------------------------------
        # STEP 10: CALCULATE LATENCY
        # ----------------------------------------------------

        latency = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # STEP 11: RETURN STANDARDIZED RESPONSE
        # ----------------------------------------------------

        return RAGResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=context_chunks,
            latency_seconds=latency,
            retrieval_strategy=self._get_retrieval_strategy(),
        )

    # ========================================================
    # CHECK INSUFFICIENT EVIDENCE
    # ========================================================

    @staticmethod
    def _is_insufficient_evidence(
        answer: str,
    ) -> bool:
        """
        Determine whether the generated answer indicates
        insufficient evidence.

        The LLM may include a short explanation before or after
        the configured insufficient-evidence message, so detection
        uses both exact matching and robust marker matching.
        """

        if not answer:
            return True

        normalized_answer = " ".join(
            answer.strip().split()
        ).lower()

        normalized_expected = " ".join(
            INSUFFICIENT_EVIDENCE_MESSAGE.split()
        ).lower()

        if normalized_answer == normalized_expected:
            return True

        insufficient_markers = [
            "do not contain sufficient evidence to answer",
            "does not contain sufficient evidence to answer",
            "does not provide sufficient evidence to answer",
            "not enough evidence to answer",
            "insufficient evidence to answer",
            "insufficient evidence",
            "not sufficient evidence to answer",
        ]

        return any(
            marker in normalized_answer
            for marker in insufficient_markers
        )

    # ========================================================
    # EXTRACT CHUNKS FROM RETRIEVER RESULTS
    # ========================================================

    @staticmethod
    def _extract_chunks(
        retrieved_results: List[Any],
    ) -> List[Any]:
        """
        Convert different retriever result formats into
        a list of document chunks.

        Supported formats:

            DenseRetriever:
                RetrievalResult

            BM25Retriever:
                (DocumentChunk, score)

            HybridRetriever:
                (DocumentChunk, score)
        """

        chunks = []

        for result in retrieved_results:

            # ------------------------------------------------
            # Tuple format:
            # (DocumentChunk, score)
            # ------------------------------------------------

            if isinstance(
                result,
                tuple,
            ):
                if len(result) >= 1:
                    chunks.append(
                        result[0]
                    )

                continue

            # ------------------------------------------------
            # Dense RetrievalResult format:
            # object.document
            # ------------------------------------------------

            if hasattr(
                result,
                "document",
            ):
                chunks.append(
                    result.document
                )

                continue

            # ------------------------------------------------
            # Direct DocumentChunk
            # ------------------------------------------------

            chunks.append(result)

        return chunks

    # ========================================================
    # CONVERT CHUNKS TO PROMPT PASSAGES
    # ========================================================

    @staticmethod
    def _chunks_to_passages(
        chunks: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert DocumentChunk objects into dictionaries
        required by format_context_passages().
        """

        passages = []

        for chunk in chunks:

            metadata = getattr(
                chunk,
                "metadata",
                None,
            )

            if metadata is None:
                continue

            content = getattr(
                chunk,
                "content",
                "",
            )

            passages.append(
                {
                    "paper_title": getattr(
                        metadata,
                        "paper_title",
                        "Unknown Title",
                    ),
                    "page_number": getattr(
                        metadata,
                        "page_number",
                        0,
                    ),
                    "content": content,
                }
            )

        return passages

    # ========================================================
    # LLM INVOCATION
    # ========================================================

    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Invoke a LangChain-compatible LLM.

        Supports modern LangChain chat models that expose
        the invoke() method.
        """

        try:

            from langchain_core.messages import (
                HumanMessage,
                SystemMessage,
            )

            messages = [
                SystemMessage(
                    content=system_prompt
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]

            response = self.llm.invoke(
                messages
            )

        except Exception as exc:

            raise RuntimeError(
                "LLM invocation failed: "
                f"{exc}"
            ) from exc

        # ----------------------------------------------------
        # LangChain AIMessage response
        # ----------------------------------------------------

        if hasattr(
            response,
            "content",
        ):

            content = response.content

            if isinstance(
                content,
                str,
            ):
                return content.strip()

            return str(content).strip()

        # ----------------------------------------------------
        # Plain string response
        # ----------------------------------------------------

        if isinstance(
            response,
            str,
        ):
            return response.strip()

        return str(response).strip()

    # ========================================================
    # CREATE CITATIONS
    # ========================================================

    @staticmethod
    def _create_citations(
        chunks: List[Any],
    ) -> List[Citation]:
        """
        Create top supporting citations from the final
        context chunks.

        At most three citations are returned.
        """

        citations = []

        for chunk in chunks[:3]:

            metadata = getattr(
                chunk,
                "metadata",
                None,
            )

            if metadata is None:
                continue

            paper_title = getattr(
                metadata,
                "paper_title",
                "Unknown Title",
            )

            page_number = getattr(
                metadata,
                "page_number",
                0,
            )

            passage = getattr(
                chunk,
                "content",
                "",
            )

            # Keep citation passages concise.
            passage = passage.strip()

            if len(passage) > 500:
                passage = (
                    passage[:500].rstrip()
                    + "..."
                )

            citations.append(
                Citation(
                    paper_title=paper_title,
                    page_number=int(
                        page_number
                    ),
                    passage=passage,
                )
            )

        return citations

    # ========================================================
    # RETRIEVAL STRATEGY DESCRIPTION
    # ========================================================

    def _get_retrieval_strategy(self) -> str:
        """
        Return a human-readable description of the
        configured retrieval pipeline.
        """

        retriever_name = (
            self.retriever.__class__.__name__
        )

        if self.reranker is not None:

            reranker_name = (
                self.reranker.__class__.__name__
            )

            return (
                f"{retriever_name} + "
                f"{reranker_name}"
            )

        return retriever_name


