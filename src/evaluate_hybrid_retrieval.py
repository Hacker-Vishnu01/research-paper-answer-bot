"""
Benchmark hybrid dense + BM25 retrieval on the 20-question
evaluation dataset.

Uses Reciprocal Rank Fusion (RRF) and evaluates results at
the paper level for direct comparison with dense MPNet and BM25.
"""

import json
import time
from pathlib import Path

import pandas as pd

from src.bm25_retriever import BM25RetrieverWrapper
from src.chunking import chunk_document_pages
from src.config import config
from src.dense_retriever import DenseRetriever
from src.document_loader import load_all_papers
from src.embeddings import get_embedding_model
from src.evaluation import (
    compute_hit_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from src.hybrid_retriever import HybridRetriever
from src.vectorstore import load_vectorstore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "hybrid_retrieval_results.csv"


MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "research_papers_mpnet"


def load_questions():
    """Load the verified evaluation questions."""
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 60)
    print("HYBRID RETRIEVAL BENCHMARK")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load documents and create the same chunks used by
    # the MPNet and BM25 baselines.
    # ---------------------------------------------------------
    print("\nLoading research papers...")

    pages = load_all_papers(config.data_dir)

    print(f"Pages loaded: {len(pages)}")

    print("\nCreating chunks...")

    chunks = chunk_document_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    print(f"Chunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # Load MPNet embedding model and persistent Chroma store.
    # ---------------------------------------------------------
    print("\nLoading MPNet embedding model...")

    embedding_model = get_embedding_model(MPNET_MODEL_NAME)

    vectorstore = load_vectorstore(
        embedding_model,
        collection_name=COLLECTION_NAME,
    )

    # ---------------------------------------------------------
    # Create dense and BM25 retrievers.
    # ---------------------------------------------------------
    print("\nCreating retrievers...")

    dense_retriever = DenseRetriever(vectorstore)

    bm25_retriever = BM25RetrieverWrapper(chunks)

    # ---------------------------------------------------------
    # Create hybrid retriever.
    # alpha=0.5 means equal weighting.
    # ---------------------------------------------------------
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,
        rrf_k=60,
    )

    print("Hybrid configuration:")
    print("  Dense weight: 0.5")
    print("  BM25 weight:  0.5")
    print("  RRF k:        60")

    # ---------------------------------------------------------
    # Load evaluation dataset.
    # ---------------------------------------------------------
    questions = load_questions()

    print(f"\nEvaluation questions: {len(questions)}")

    results = []

    total_latency = 0.0

    # ---------------------------------------------------------
    # Evaluate every question.
    # ---------------------------------------------------------
    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        expected_paper_ids = item["expected_paper_ids"]

        ground_truth_ids = set(expected_paper_ids)

        start_time = time.perf_counter()

        retrieved = hybrid_retriever.retrieve(
            query=question,
            top_k=10,
        )

        latency = time.perf_counter() - start_time
        total_latency += latency

        # Convert chunks to paper IDs.
        retrieved_paper_ids = [
            chunk.metadata.paper_id
            for chunk, score in retrieved
        ]

        # Deduplicate paper IDs while preserving rank.
        unique_retrieved_paper_ids = list(
            dict.fromkeys(retrieved_paper_ids)
        )

        hit_at_3 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=3,
        )

        hit_at_5 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=5,
        )

        hit_at_10 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=10,
        )

        mrr = compute_reciprocal_rank(
            unique_retrieved_paper_ids,
            ground_truth_ids,
        )

        precision_at_3 = compute_precision_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=3,
        )

        recall_at_3 = compute_recall_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=3,
        )

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_paper_ids": ",".join(expected_paper_ids),
                "top_1_paper": (
                    unique_retrieved_paper_ids[0]
                    if unique_retrieved_paper_ids
                    else ""
                ),
                "retrieved_paper_ids": ",".join(
                    unique_retrieved_paper_ids
                ),
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
                "mrr": mrr,
                "precision_at_3": precision_at_3,
                "recall_at_3": recall_at_3,
                "latency_seconds": latency,
            }
        )

        print(
            f"{question_id}: "
            f"Top-1="
            f"{unique_retrieved_paper_ids[0] if unique_retrieved_paper_ids else 'N/A'} "
            f"| MRR={mrr:.4f} "
            f"| Latency={latency:.4f}s"
        )

    # ---------------------------------------------------------
    # Aggregate metrics.
    # ---------------------------------------------------------
    dataframe = pd.DataFrame(results)

    average_hit_at_3 = dataframe["hit_at_3"].mean()
    average_hit_at_5 = dataframe["hit_at_5"].mean()
    average_hit_at_10 = dataframe["hit_at_10"].mean()
    average_mrr = dataframe["mrr"].mean()
    average_precision_at_3 = dataframe["precision_at_3"].mean()
    average_recall_at_3 = dataframe["recall_at_3"].mean()
    average_latency = dataframe["latency_seconds"].mean()

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    print(f"Hit@3:          {average_hit_at_3:.4f}")
    print(f"Hit@5:          {average_hit_at_5:.4f}")
    print(f"Hit@10:         {average_hit_at_10:.4f}")
    print(f"MRR:            {average_mrr:.4f}")
    print(f"Precision@3:    {average_precision_at_3:.4f}")
    print(f"Recall@3:       {average_recall_at_3:.4f}")
    print(f"Avg latency:    {average_latency:.4f}s")

    # ---------------------------------------------------------
    # Save benchmark results.
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nHybrid retrieval benchmark completed successfully.")


if __name__ == "__main__":
    main()