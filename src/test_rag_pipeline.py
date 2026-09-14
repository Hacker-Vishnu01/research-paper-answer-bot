"""
End-to-end performance test for the Research Paper Answer Bot RAG pipeline.

Measures:
    - Paper loading time
    - Chunking time
    - Vector store loading time
    - Hybrid retriever build time
    - LLM initialization time
    - RAG execution time
    - Context size
    - Total test time

Pipeline:
    Research papers
        ↓
    MPNet vector store
        ↓
    BM25
        ↓
    Hybrid retrieval
        ↓
    Llama 3.2 via Ollama
        ↓
    Grounded answer + citations
"""

import time

from src.bm25_retriever import BM25RetrieverWrapper
from src.chunking import chunk_document_pages
from src.config import config
from src.document_loader import load_all_papers
from src.dense_retriever import DenseRetriever
from src.embeddings import MPNET_MODEL_NAME, get_embedding_model
from src.hybrid_retriever import HybridRetriever
from src.llm import get_llm
from src.rag_pipeline import RAGPipeline
from src.vectorstore import load_vectorstore


def main():
    print("=" * 70)
    print("RESEARCH PAPER ANSWER BOT - RAG PERFORMANCE TEST")
    print("=" * 70)

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Load research papers
    # ---------------------------------------------------------
    print("\n[1/6] Loading research papers...")

    start = time.perf_counter()

    pages = load_all_papers(config.data_dir)

    paper_loading_time = time.perf_counter() - start

    print(f"Pages loaded: {len(pages)}")
    print(
        f"Paper loading time: "
        f"{paper_loading_time:.3f} seconds"
    )

    # ---------------------------------------------------------
    # 2. Create chunks for BM25
    # ---------------------------------------------------------
    print("\n[2/6] Creating document chunks...")

    start = time.perf_counter()

    chunks = chunk_document_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunking_time = time.perf_counter() - start

    print(f"Chunks created: {len(chunks)}")
    print(
        f"Chunking time: "
        f"{chunking_time:.3f} seconds"
    )

    # ---------------------------------------------------------
    # 3. Load MPNet vector store
    # ---------------------------------------------------------
    print("\n[3/6] Loading MPNet vector store...")

    start = time.perf_counter()

    embeddings = get_embedding_model(
        MPNET_MODEL_NAME
    )

    vectorstore = load_vectorstore(
        embeddings,
        collection_name="research_papers_mpnet",
    )

    vectorstore_loading_time = time.perf_counter() - start

    print(
        "Vector store documents:",
        vectorstore._collection.count(),
    )

    print(
        f"Vector store loading time: "
        f"{vectorstore_loading_time:.3f} seconds"
    )

    dense_retriever = DenseRetriever(
        vectorstore
    )

    # ---------------------------------------------------------
    # 4. Build BM25 + Hybrid retriever
    # ---------------------------------------------------------
    print("\n[4/6] Building hybrid retriever...")

    start = time.perf_counter()

    bm25_retriever = BM25RetrieverWrapper(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,
        rrf_k=60,
    )

    hybrid_build_time = time.perf_counter() - start

    print("Hybrid retriever: READY")

    print(
        f"Hybrid retriever build time: "
        f"{hybrid_build_time:.3f} seconds"
    )

    # ---------------------------------------------------------
    # 5. Create local LLM
    # ---------------------------------------------------------
    print("\n[5/6] Loading local LLM...")

    start = time.perf_counter()

    llm = get_llm(
        model_name="llama3.2:3b",
        temperature=0.0,
    )

    llm_loading_time = time.perf_counter() - start

    print("LLM: llama3.2:3b")

    print(
        f"LLM initialization time: "
        f"{llm_loading_time:.3f} seconds"
    )

    # ---------------------------------------------------------
    # 6. Run RAG pipeline
    # ---------------------------------------------------------
    print("\n[6/6] Running RAG pipeline...")

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=None,
        llm=llm,
        retrieval_top_k=10,
        context_top_k=3,
    )

    question = (
        "What is scaled dot-product attention, "
        "and what is the formula used to calculate it?"
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerating answer...")
    print(
        "Please wait; the local LLM may take some time."
    )

    rag_start = time.perf_counter()

    response = pipeline.answer(
        question
    )

    rag_time = time.perf_counter() - rag_start

    # ---------------------------------------------------------
    # Display answer
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(response.answer)

    # ---------------------------------------------------------
    # Display citations
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP SUPPORTING CITATIONS")
    print("=" * 70)

    for index, citation in enumerate(
        response.citations,
        start=1,
    ):
        print(f"\n[{index}]")
        print(
            f"Paper: {citation.paper_title}"
        )
        print(
            f"Page: {citation.page_number}"
        )
        print(
            f"Passage: {citation.passage}"
        )

    # ---------------------------------------------------------
    # Context analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONTEXT ANALYSIS")
    print("=" * 70)

    context_chunks = response.retrieved_chunks

    total_context_chars = 0

    for index, chunk in enumerate(
        context_chunks,
        start=1,
    ):
        content = getattr(
            chunk,
            "content",
            "",
        )

        chunk_chars = len(content)

        total_context_chars += chunk_chars

        print(
            f"Context chunk {index}: "
            f"{chunk_chars} characters"
        )

        print(
            f"  Paper: "
            f"{chunk.metadata.paper_title}"
        )

        print(
            f"  Page: "
            f"{chunk.metadata.page_number}"
        )

    estimated_tokens = total_context_chars / 4

    print(
        f"\nTotal context characters: "
        f"{total_context_chars}"
    )

    print(
        f"Estimated context tokens: "
        f"{estimated_tokens:.0f}"
    )

    # ---------------------------------------------------------
    # Pipeline information
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE INFORMATION")
    print("=" * 70)

    print(
        f"Retrieval strategy: "
        f"{response.retrieval_strategy}"
    )

    print(
        f"Final context chunks: "
        f"{len(response.retrieved_chunks)}"
    )

    print(
        f"RAG pipeline execution time: "
        f"{rag_time:.3f} seconds"
    )

    print(
        f"Response latency recorded by pipeline: "
        f"{response.latency_seconds:.3f} seconds"
    )

    # ---------------------------------------------------------
    # Performance summary
    # ---------------------------------------------------------
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        f"Paper loading:          "
        f"{paper_loading_time:.3f} sec"
    )

    print(
        f"Chunking:               "
        f"{chunking_time:.3f} sec"
    )

    print(
        f"Vector store loading:   "
        f"{vectorstore_loading_time:.3f} sec"
    )

    print(
        f"Hybrid build:           "
        f"{hybrid_build_time:.3f} sec"
    )

    print(
        f"LLM initialization:     "
        f"{llm_loading_time:.3f} sec"
    )

    print(
        f"RAG execution:          "
        f"{rag_time:.3f} sec"
    )

    print(
        f"Total test time:        "
        f"{total_time:.3f} sec"
    )

    print(
        f"\nContext characters:     "
        f"{total_context_chars}"
    )

    print(
        f"Estimated context tokens:"
        f" {estimated_tokens:.0f}"
    )

    print("\n" + "=" * 70)
    print("RAG PERFORMANCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()