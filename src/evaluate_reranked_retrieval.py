"""
Benchmark Hybrid + Cross-Encoder reranking on the 20-question
evaluation dataset.

Pipeline:

    Dense MPNet + BM25
            |
            v
       Hybrid RRF
            |
            v
      Top-10 candidates
            |
            v
     Cross-Encoder
            |
            v
       Reranked results

Results are evaluated at the paper level for comparison with:

1. Dense MPNet
2. BM25
3. Hybrid RRF
4. Hybrid + Cross-Encoder
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
from src.reranker import CrossEncoderReranker
from src.vectorstore import load_vectorstore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reranked_retrieval_results.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MPNET_MODEL_NAME = (
    "sentence-transformers/all-mpnet-base-v2"
)

COLLECTION_NAME = "research_papers_mpnet"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_questions():
    """
    Load the verified 20-question evaluation dataset.
    """

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID + CROSS-ENCODER RERANKING BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: LOAD RESEARCH PAPERS
    # --------------------------------------------------------

    print("\n[1/7] Loading research papers...")

    pages = load_all_papers(
        config.data_dir
    )

    print(
        f"Pages loaded: {len(pages)}"
    )

    # --------------------------------------------------------
    # STEP 2: CREATE CHUNKS
    # --------------------------------------------------------

    print("\n[2/7] Creating document chunks...")

    chunks = chunk_document_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------------
    # STEP 3: LOAD MPNet + CHROMA
    # --------------------------------------------------------

    print(
        "\n[3/7] Loading MPNet embedding model..."
    )

    embedding_model = get_embedding_model(
        MPNET_MODEL_NAME
    )

    print(
        "MPNet embedding model loaded."
    )

    print(
        "\nLoading existing Chroma vector store..."
    )

    vectorstore = load_vectorstore(
        embedding_model,
        collection_name=COLLECTION_NAME,
    )

    print(
        "Chroma vector store loaded."
    )

    # --------------------------------------------------------
    # STEP 4: CREATE DENSE + BM25 + HYBRID RETRIEVERS
    # --------------------------------------------------------

    print(
        "\n[4/7] Creating retrieval components..."
    )

    dense_retriever = DenseRetriever(
        vectorstore
    )

    print(
        "Dense retriever: OK"
    )

    bm25_retriever = BM25RetrieverWrapper(
        chunks
    )

    print(
        "BM25 retriever: OK"
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,
        rrf_k=60,
    )

    print(
        "Hybrid RRF retriever: OK"
    )

    # --------------------------------------------------------
    # STEP 5: LOAD CROSS-ENCODER
    # --------------------------------------------------------

    print(
        "\n[5/7] Loading Cross-Encoder reranker..."
    )

    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL_NAME
    )

    print(
        f"Cross-Encoder loaded: "
        f"{RERANKER_MODEL_NAME}"
    )

    # --------------------------------------------------------
    # STEP 6: LOAD EVALUATION DATASET
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    print(
        "\n[6/7] Running benchmark..."
    )

    # --------------------------------------------------------
    # RESULT STORAGE
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # EVALUATE EVERY QUESTION
    # --------------------------------------------------------

    for item in questions:

        question_id = item[
            "question_id"
        ]

        question = item[
            "question"
        ]

        expected_paper_ids = item[
            "expected_paper_ids"
        ]

        ground_truth_ids = set(
            expected_paper_ids
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Question: {question_id}"
        )

        print(
            f"Query: {question}"
        )

        # ----------------------------------------------------
        # START TIMER
        # ----------------------------------------------------

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # HYBRID RETRIEVAL
        #
        # Retrieve 10 candidate chunks.
        # ----------------------------------------------------

        hybrid_results = (
            hybrid_retriever.retrieve(
                query=question,
                top_k=10,
            )
        )

        print(
            f"Hybrid candidates: "
            f"{len(hybrid_results)}"
        )

        # ----------------------------------------------------
        # EXTRACT DOCUMENT CHUNKS
        # ----------------------------------------------------

        candidates = [
            chunk
            for chunk, score
            in hybrid_results
        ]

        # ----------------------------------------------------
        # CROSS-ENCODER RERANKING
        #
        # Rerank all 10 candidates.
        # ----------------------------------------------------

        reranked_results = (
            reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=10,
            )
        )

        # ----------------------------------------------------
        # END TIMER
        # ----------------------------------------------------

        latency = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # CONVERT RESULTS TO PAPER IDs
        # ----------------------------------------------------

        retrieved_paper_ids = [
            chunk.metadata.paper_id
            for chunk, score
            in reranked_results
        ]

        # ----------------------------------------------------
        # REMOVE DUPLICATE PAPER IDs
        #
        # Keep the original ranking order.
        # ----------------------------------------------------

        unique_retrieved_paper_ids = list(
            dict.fromkeys(
                retrieved_paper_ids
            )
        )

        # ----------------------------------------------------
        # CALCULATE HIT@3
        # ----------------------------------------------------

        hit_at_3 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=3,
        )

        # ----------------------------------------------------
        # CALCULATE HIT@5
        # ----------------------------------------------------

        hit_at_5 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=5,
        )

        # ----------------------------------------------------
        # CALCULATE HIT@10
        # ----------------------------------------------------

        hit_at_10 = compute_hit_at_k(
            unique_retrieved_paper_ids,
            ground_truth_ids,
            k=10,
        )

        # ----------------------------------------------------
        # CALCULATE MRR
        # ----------------------------------------------------

        mrr = compute_reciprocal_rank(
            unique_retrieved_paper_ids,
            ground_truth_ids,
        )

        # ----------------------------------------------------
        # CALCULATE PRECISION@3
        # ----------------------------------------------------

        precision_at_3 = (
            compute_precision_at_k(
                unique_retrieved_paper_ids,
                ground_truth_ids,
                k=3,
            )
        )

        # ----------------------------------------------------
        # CALCULATE RECALL@3
        # ----------------------------------------------------

        recall_at_3 = (
            compute_recall_at_k(
                unique_retrieved_paper_ids,
                ground_truth_ids,
                k=3,
            )
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        results.append(
            {
                "question_id": question_id,

                "question": question,

                "expected_paper_ids": ",".join(
                    expected_paper_ids
                ),

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

        # ----------------------------------------------------
        # PRINT QUESTION RESULT
        # ----------------------------------------------------

        top_1 = (
            unique_retrieved_paper_ids[0]
            if unique_retrieved_paper_ids
            else "N/A"
        )

        print(
            f"Top-1: {top_1}"
        )

        print(
            f"Expected: "
            f"{','.join(expected_paper_ids)}"
        )

        print(
            f"MRR: {mrr:.4f}"
        )

        print(
            f"Hit@3: {hit_at_3:.4f}"
        )

        print(
            f"Recall@3: {recall_at_3:.4f}"
        )

        print(
            f"Latency: {latency:.4f}s"
        )

    # ========================================================
    # STEP 7: CALCULATE AGGREGATE RESULTS
    # ========================================================

    print(
        "\n[7/7] Calculating aggregate metrics..."
    )

    dataframe = pd.DataFrame(
        results
    )

    average_hit_at_3 = (
        dataframe["hit_at_3"].mean()
    )

    average_hit_at_5 = (
        dataframe["hit_at_5"].mean()
    )

    average_hit_at_10 = (
        dataframe["hit_at_10"].mean()
    )

    average_mrr = (
        dataframe["mrr"].mean()
    )

    average_precision_at_3 = (
        dataframe[
            "precision_at_3"
        ].mean()
    )

    average_recall_at_3 = (
        dataframe[
            "recall_at_3"
        ].mean()
    )

    average_latency = (
        dataframe[
            "latency_seconds"
        ].mean()
    )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HYBRID + CROSS-ENCODER RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Hit@3:          "
        f"{average_hit_at_3:.4f}"
    )

    print(
        f"Hit@5:          "
        f"{average_hit_at_5:.4f}"
    )

    print(
        f"Hit@10:         "
        f"{average_hit_at_10:.4f}"
    )

    print(
        f"MRR:            "
        f"{average_mrr:.4f}"
    )

    print(
        f"Precision@3:    "
        f"{average_precision_at_3:.4f}"
    )

    print(
        f"Recall@3:       "
        f"{average_recall_at_3:.4f}"
    )

    print(
        f"Avg latency:    "
        f"{average_latency:.4f}s"
    )

    # ========================================================
    # SAVE RESULTS TO CSV
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BENCHMARK COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()