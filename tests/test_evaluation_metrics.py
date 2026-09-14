"""
Tests for evaluation metric calculations.
"""

from src.evaluation import (
    compute_hit_at_k,
    compute_reciprocal_rank,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval_batch,
)


def test_compute_hit_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    ground_truth = {"doc3"}

    assert compute_hit_at_k(retrieved, ground_truth, k=1) == 0.0
    assert compute_hit_at_k(retrieved, ground_truth, k=2) == 0.0
    assert compute_hit_at_k(retrieved, ground_truth, k=3) == 1.0
    assert compute_hit_at_k(retrieved, ground_truth, k=5) == 1.0


def test_compute_reciprocal_rank():
    retrieved = ["doc1", "doc2", "doc3"]
    assert compute_reciprocal_rank(retrieved, {"doc1"}) == 1.0
    assert compute_reciprocal_rank(retrieved, {"doc2"}) == 0.5
    assert compute_reciprocal_rank(retrieved, {"doc3"}) == 1.0 / 3.0
    assert compute_reciprocal_rank(retrieved, {"doc4"}) == 0.0


def test_compute_precision_and_recall():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    ground_truth = {"doc2", "doc4", "doc5"}

    # In top 2: doc1 (no), doc2 (yes) -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, ground_truth, k=2) == 0.5
    # In top 4: doc2, doc4 (2 relevant) -> 2/4 = 0.5
    assert compute_precision_at_k(retrieved, ground_truth, k=4) == 0.5

    # Recall at k=4: 2 found out of 3 total relevant -> 2/3
    assert abs(compute_recall_at_k(retrieved, ground_truth, k=4) - (2.0 / 3.0)) < 1e-6


def test_evaluate_retrieval_batch():
    predictions = [
        ["doc1", "doc2", "doc3"],
        ["docA", "docB", "docC"],
    ]
    ground_truths = [
        {"doc1"},
        {"docB"},
    ]
    results = evaluate_retrieval_batch(predictions, ground_truths, k_values=[1, 3])
    # Query 1 RR: 1.0; Query 2 RR: 0.5 -> Mean MRR = 0.75
    assert abs(results["MRR"] - 0.75) < 1e-6
    # Hit@1: Query 1 = 1, Query 2 = 0 -> Mean = 0.5
    assert abs(results["Hit@1"] - 0.5) < 1e-6
    # Hit@3: Query 1 = 1, Query 2 = 1 -> Mean = 1.0
    assert abs(results["Hit@3"] - 1.0) < 1e-6
