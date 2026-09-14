"""
Multi-question end-to-end RAG test.

Tests the Research Paper Answer Bot using:
    MPNet + BM25 Hybrid Retrieval
    +
    Llama 3.2 3B via Ollama

This is an integration test for generated answers and citations.
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


QUESTIONS = [
    {
        "id": "Q001",
        "question": (
            "What is scaled dot-product attention, "
            "and what is the formula used to calculate it?"
        ),
    },
    {
        "id": "Q003",
        "question": (
            "What are the two main pre-training objectives "
            "used by BERT?"
        ),
    },
    {
        "id": "Q005",
        "question": (
            "What is the difference between RAG-Sequence "
            "and RAG-Token?"
        ),
    },
    {
        "id": "Q012",
        "question": (
            "What is LoRA and how does it use low-rank "
            "decomposition for parameter-efficient fine-tuning?"
        ),
    },
    {
        "id": "Q019",
        "question": (
            "What is the Lost in the Middle phenomenon "
            "in long-context language models?"
        ),
    },
]


def build_pipeline():
    """Build the complete hybrid RAG pipeline."""

    print("[1/5] Loading research papers...")

    pages = load_all_papers(config.data_dir)

    print(f"Pages loaded: {len(pages)}")

    print("\n[2/5] Creating chunks...")

    chunks = chunk_document_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    print(f"Chunks created: {len(chunks)}")

    print("\n[3/5] Loading MPNet vector store...")

    embeddings = get_embedding_model(
        MPNET_MODEL_NAME
    )

    vectorstore = load_vectorstore(
        embeddings,
        collection_name="research_papers_mpnet",
    )

    print(
        "Vector store documents:",
        vectorstore._collection.count(),
    )

    dense_retriever = DenseRetriever(
        vectorstore
    )

    print("\n[4/5] Building hybrid retriever...")

    bm25_retriever = BM25RetrieverWrapper(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,
        rrf_k=60,
    )

    print("Hybrid retriever: READY")

    print("\n[5/5] Loading Llama 3.2...")

    llm = get_llm(
        model_name="llama3.2:3b",
        temperature=0.0,
    )

    print("LLM: llama3.2:3b")

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=None,
        llm=llm,
        retrieval_top_k=10,
        context_top_k=3,
    )

    return pipeline


def main():
    """Run the multi-question RAG test."""

    print("=" * 70)
    print("MULTI-QUESTION RAG ANSWER TEST")
    print("=" * 70)

    pipeline = build_pipeline()

    total_start = time.perf_counter()

    results = []

    for index, item in enumerate(
        QUESTIONS,
        start=1,
    ):
        question_id = item["id"]
        question = item["question"]

        print("\n")
        print("=" * 70)
        print(
            f"QUESTION {index}/{len(QUESTIONS)} - {question_id}"
        )
        print("=" * 70)

        print("\nQuestion:")
        print(question)

        print("\nGenerating answer...")
        print("Local LLM generation may take some time.")

        start = time.perf_counter()

        response = pipeline.answer(
            question
        )

        elapsed = time.perf_counter() - start

        print("\nANSWER")
        print("-" * 70)
        print(response.answer)

        print("\nCITATIONS")
        print("-" * 70)

        for citation_index, citation in enumerate(
            response.citations,
            start=1,
        ):
            print(
                f"\n[{citation_index}] "
                f"{citation.paper_title} "
                f"(Page {citation.page_number})"
            )

            print(
                f"Passage: {citation.passage}"
            )

        print("\nMETRICS")
        print("-" * 70)
        print(
            f"Retrieval strategy: "
            f"{response.retrieval_strategy}"
        )
        print(
            f"Context chunks: "
            f"{len(response.retrieved_chunks)}"
        )
        print(
            f"Latency: "
            f"{elapsed:.3f} seconds"
        )

        results.append(
            {
                "question_id": question_id,
                "latency": elapsed,
                "citations": len(
                    response.citations
                ),
                "answer_length": len(
                    response.answer
                ),
            }
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print("\n")
    print("=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    for result in results:
        print(
            f"{result['question_id']}: "
            f"{result['latency']:.3f}s | "
            f"citations={result['citations']} | "
            f"answer_chars={result['answer_length']}"
        )

    average_latency = (
        sum(
            result["latency"]
            for result in results
        )
        / len(results)
    )

    print("\n")
    print(
        f"Average answer latency: "
        f"{average_latency:.3f} seconds"
    )

    print(
        f"Total test time: "
        f"{total_elapsed:.3f} seconds"
    )

    print("\n" + "=" * 70)
    print("MULTI-QUESTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()