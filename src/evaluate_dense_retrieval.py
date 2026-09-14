"""
Benchmark dense retrieval on the curated evaluation questions.
"""

import json
import time
from pathlib import Path

import pandas as pd

from src.embeddings import get_embedding_model, MPNET_MODEL_NAME
from src.vectorstore import load_vectorstore
from src.dense_retriever import DenseRetriever
from src.evaluation import (
    compute_hit_at_k,
    compute_reciprocal_rank,
    compute_precision_at_k,
    compute_recall_at_k,
)


def load_questions():
    """Load evaluation questions from evaluation/questions.json."""

    questions_path = Path("evaluation/questions.json")

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found: {questions_path}"
        )

    with open(questions_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "evaluation/questions.json must contain a JSON list."
        )

    return questions


def main():
    print("=" * 60)
    print("DENSE RETRIEVAL BENCHMARK")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load evaluation questions
    # ---------------------------------------------------------

    questions = load_questions()

    print(f"Evaluation questions: {len(questions)}")

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print("\nLoading MPNet embedding model...")

    embeddings = get_embedding_model(MPNET_MODEL_NAME)

    # ---------------------------------------------------------
    # Load persistent vector store
    # ---------------------------------------------------------

    print("Loading persistent Chroma vector store...")

    vectorstore = load_vectorstore(
        embeddings,
        collection_name="research_papers_mpnet",
    )

    print("Vector store loaded.")

    # ---------------------------------------------------------
    # Create dense retriever
    # ---------------------------------------------------------

    retriever = DenseRetriever(vectorstore)

    evaluation_rows = []

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------

    for index, question_data in enumerate(questions, start=1):

        question_id = question_data["question_id"]
        question = question_data["question"]

        expected_paper_ids = question_data["expected_paper_ids"]

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        start_time = time.perf_counter()

        results = retriever.retrieve(
            query=question,
            top_k=10,
        )

        latency = time.perf_counter() - start_time

        retrieved_paper_ids = [
            result.paper_id
            for result in results
        ]

        # Remove duplicate paper IDs while preserving ranking order.
        unique_retrieved_paper_ids = list(
            dict.fromkeys(retrieved_paper_ids)
        )

        # -----------------------------------------------------
        # Determine relevance
        # -----------------------------------------------------
        # A retrieved chunk is relevant if its paper ID matches
        # any of the expected paper IDs for this question.
        # -----------------------------------------------------


        # -----------------------------------------------------
        # Retrieval metrics
        # -----------------------------------------------------

        ground_truth_ids = set(expected_paper_ids)

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

        # -----------------------------------------------------
        # Store evaluation row
        # -----------------------------------------------------

        evaluation_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "retriever_type": "Dense-MPNet",
                "top_k": 10,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
                "precision_at_3": precision_at_3,
                "recall_at_3": recall_at_3,
                "mrr": mrr,
                "latency_seconds": latency,
                "expected_paper_ids": "|".join(expected_paper_ids),
                "top_1_paper_id": (
                    retrieved_paper_ids[0]
                    if retrieved_paper_ids
                    else ""
                ),
                "top_1_chunk_id": (
                    results[0].chunk_id
                    if results
                    else ""
                ),
            }
        )

        print(
            f"  Expected: {', '.join(expected_paper_ids)}"
        )

        print(
            f"  Top-1: "
            f"{retrieved_paper_ids[0] if retrieved_paper_ids else 'NONE'}"
        )

        print(
            f"  Hit@3={hit_at_3} | "
            f"Hit@5={hit_at_5} | "
            f"Hit@10={hit_at_10} | "
            f"MRR={mrr:.4f} | "
            f"Latency={latency:.4f}s"
        )

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------

    results_df = pd.DataFrame(evaluation_rows)

    # ---------------------------------------------------------
    # Calculate aggregate metrics
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    print(
        f"Hit@3:          "
        f"{results_df['hit_at_3'].mean():.4f}"
    )

    print(
        f"Hit@5:          "
        f"{results_df['hit_at_5'].mean():.4f}"
    )

    print(
        f"Hit@10:         "
        f"{results_df['hit_at_10'].mean():.4f}"
    )

    print(
        f"MRR:            "
        f"{results_df['mrr'].mean():.4f}"
    )

    print(
        f"Precision@3:    "
        f"{results_df['precision_at_3'].mean():.4f}"
    )

    print(
        f"Recall@3:       "
        f"{results_df['recall_at_3'].mean():.4f}"
    )

    print(
        f"Avg latency:    "
        f"{results_df['latency_seconds'].mean():.4f}s"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = Path("evaluation/retrieval_results.csv")

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nResults saved to:")
    print(output_path.resolve())

    print("\nDense retrieval benchmark completed successfully.")


if __name__ == "__main__":
    main()