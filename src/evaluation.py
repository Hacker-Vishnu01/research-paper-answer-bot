"""
Evaluation module for the Research Paper Answer Bot.

Provides calculation functions for standard IR retrieval metrics (Hit@K, Precision@K,
Recall@K, MRR) and placeholders for answer faithfulness and citation verification.
"""

from typing import Any, Dict, List, Set


def compute_hit_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Hit@K metric: 1.0 if any ground-truth item is in top-k, else 0.0.
    
    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        ground_truth_ids: Set of relevant ground-truth IDs.
        k: Cutoff rank.
        
    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    top_k = retrieved_ids[:k]
    for item_id in top_k:
        if item_id in ground_truth_ids:
            return 1.0
    return 0.0


def compute_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """
    Computes Reciprocal Rank (RR): 1/rank of the first relevant retrieved item.
    
    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        ground_truth_ids: Set of relevant ground-truth IDs.
        
    Returns:
        Reciprocal rank float (0.0 if no relevant items retrieved).
    """
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Precision@K: Fraction of top-k retrieved items that are relevant.
    
    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        ground_truth_ids: Set of relevant ground-truth IDs.
        k: Cutoff rank.
        
    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for item_id in top_k if item_id in ground_truth_ids)
    return relevant_count / k


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K: Proportion of relevant items that are retrieved in top-k.
    
    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        ground_truth_ids: Set of relevant ground-truth IDs.
        k: Cutoff rank.
        
    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for item_id in top_k if item_id in ground_truth_ids)
    return relevant_count / len(ground_truth_ids)


def evaluate_retrieval_batch(
    predictions: List[List[str]],
    ground_truths: List[Set[str]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Computes aggregated retrieval metrics over a benchmark dataset.
    
    Args:
        predictions: List of retrieved ID lists per query.
        ground_truths: List of relevant ID sets per query.
        k_values: List of cutoff ranks to evaluate.
        
    Returns:
        Dictionary of mean metric scores.
    """
    if not predictions or len(predictions) != len(ground_truths):
        return {}

    n = len(predictions)
    mrr_total = sum(
        compute_reciprocal_rank(preds, gt)
        for preds, gt in zip(predictions, ground_truths)
    )

    results = {"MRR": mrr_total / n}
    for k in k_values:
        hit_k = sum(compute_hit_at_k(p, gt, k) for p, gt in zip(predictions, ground_truths)) / n
        prec_k = sum(compute_precision_at_k(p, gt, k) for p, gt in zip(predictions, ground_truths)) / n
        rec_k = sum(compute_recall_at_k(p, gt, k) for p, gt in zip(predictions, ground_truths)) / n
        results[f"Hit@{k}"] = hit_k
        results[f"Precision@{k}"] = prec_k
        results[f"Recall@{k}"] = rec_k

    return results
