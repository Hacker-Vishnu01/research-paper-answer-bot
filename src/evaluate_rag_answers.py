"""
End-to-End RAG Answer Evaluation.

Runs the production Research Paper Answer Bot pipeline against
the verified 20-question evaluation dataset.

The script records:
    - Question ID
    - Dataset split
    - Topic
    - Expected paper IDs
    - Generated answer
    - Retrieved/cited paper IDs
    - Citation correctness
    - Citation count
    - Insufficient-evidence behavior
    - Response latency

This script does NOT modify the existing retrieval benchmark files.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.bm25_retriever import BM25RetrieverWrapper
from src.chunking import chunk_document_pages
from src.config import config
from src.dense_retriever import DenseRetriever
from src.document_loader import load_all_papers
from src.embeddings import (
    MPNET_MODEL_NAME,
    get_embedding_model,
)
from src.hybrid_retriever import HybridRetriever
from src.llm import OLLAMA_MODEL, get_llm
from src.rag_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RAGPipeline,
)
from src.vectorstore import load_vectorstore


# ============================================================
# PATHS
# ============================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "answer_evaluation.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

VECTORSTORE_COLLECTION = "research_papers_mpnet"

RETRIEVAL_TOP_K = 10
CONTEXT_TOP_K = 9

RRF_ALPHA = 0.5
RRF_K = 60


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Load the verified evaluation questions from JSON.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        
    ) as file:
        questions = json.load(file)

    if not isinstance(
        questions,
        list,
    ):
        raise ValueError(
            "questions.json must contain a JSON list."
        )

    if not questions:
        raise ValueError(
            "questions.json contains no questions."
        )

    return questions


# ============================================================
# BUILD PRODUCTION PIPELINE
# ============================================================

def build_pipeline() -> RAGPipeline:
    """
    Build the same production RAG pipeline used by Streamlit.

    Architecture:

        Papers
          â†“
        1000/150 chunks
          â†“
        MPNet ChromaDB
          +
        BM25
          â†“
        Hybrid RRF
          â†“
        Llama 3.2 3B
    """

    print("=" * 70)
    print("BUILDING PRODUCTION RAG PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load papers
    # --------------------------------------------------------

    print("\n[1/6] Loading research papers...")

    pages = load_all_papers(
        config.data_dir
    )

    if not pages:
        raise RuntimeError(
            "No research paper pages were loaded."
        )

    print(
        f"Pages loaded: {len(pages)}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print(
        "\n[2/6] Creating document chunks..."
    )

    chunks = chunk_document_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise RuntimeError(
            "No document chunks were created."
        )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print(
        "\n[3/6] Loading MPNet embeddings..."
    )

    embeddings = get_embedding_model(
        MPNET_MODEL_NAME
    )

    print(
        "Embedding model: "
        f"{MPNET_MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------------

    print(
        "\n[4/6] Loading ChromaDB..."
    )

    vectorstore = load_vectorstore(
        embedding_function=embeddings,
        persist_directory=(
            config.chroma_persist_directory
        ),
        collection_name=(
            VECTORSTORE_COLLECTION
        ),
    )

    print(
        "ChromaDB collection: "
        f"{VECTORSTORE_COLLECTION}"
    )

    # --------------------------------------------------------
    # Build retrievers
    # --------------------------------------------------------

    print(
        "\n[5/6] Building hybrid retriever..."
    )

    dense_retriever = DenseRetriever(
        vectorstore=vectorstore
    )

    bm25_retriever = BM25RetrieverWrapper(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=RRF_ALPHA,
        rrf_k=RRF_K,
    )

    print(
        "Hybrid retriever: READY"
    )

    # --------------------------------------------------------
    # Load local LLM
    # --------------------------------------------------------

    print(
        "\n[6/6] Loading local LLM..."
    )

    llm = get_llm(
        model_name=OLLAMA_MODEL,
        temperature=0.0,
    )

    print(
        f"LLM: {OLLAMA_MODEL}"
    )

    # --------------------------------------------------------
    # Create RAG pipeline
    # --------------------------------------------------------

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=None,
        llm=llm,
        retrieval_top_k=RETRIEVAL_TOP_K,
        context_top_k=CONTEXT_TOP_K,
    )

    print(
        "\nProduction RAG pipeline: READY"
    )

    return pipeline


# ============================================================
# EXTRACT CITED PAPER IDS
# ============================================================

def get_cited_paper_ids(
    pipeline_response: Any,
    chunks: List[Any],
) -> List[str]:
    """
    Determine the paper IDs associated with the final
    supporting citations.

    Citation objects contain title/page/passage, while
    paper IDs are available in retrieved chunk metadata.
    """

    cited_ids = []

    citations = getattr(
        pipeline_response,
        "citations",
        [],
    )

    if not citations:
        return cited_ids

    for citation in citations:

        citation_title = getattr(
            citation,
            "paper_title",
            "",
        )

        citation_page = getattr(
            citation,
            "page_number",
            None,
        )

        for chunk in chunks:

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
                "",
            )

            page_number = getattr(
                metadata,
                "page_number",
                None,
            )

            paper_id = getattr(
                metadata,
                "paper_id",
                "",
            )

            if (
                paper_title == citation_title
                and page_number == citation_page
                and paper_id
            ):
                if paper_id not in cited_ids:
                    cited_ids.append(
                        paper_id
                    )

                break

    return cited_ids


# ============================================================
# CITATION CORRECTNESS
# ============================================================

def compute_citation_correctness(
    cited_paper_ids: List[str],
    expected_paper_ids: Set[str],
) -> str:
    """
    Determine whether at least one supporting citation
    belongs to an expected paper.

    Returns:
        "correct"
        "incorrect"
        "not_applicable"
    """

    if not cited_paper_ids:
        return "not_applicable"

    if any(
        paper_id in expected_paper_ids
        for paper_id in cited_paper_ids
    ):
        return "correct"

    return "incorrect"


# ============================================================
# INSUFFICIENT-EVIDENCE CHECK
# ============================================================

def is_insufficient_evidence(
    answer: str,
) -> bool:
    """
    Detect whether the generated answer indicates insufficient
    evidence, even when the LLM adds an explanation.
    """

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

def evaluate_question(
    pipeline: RAGPipeline,
    item: Dict[str, Any],
    question_number: int,
    total_questions: int,
) -> Dict[str, Any]:
    """
    Run one evaluation question through the production
    RAG pipeline and return a structured result.
    """

    question_id = item.get(
        "question_id",
        "",
    )

    question = item.get(
        "question",
        "",
    )

    expected_paper_ids = set(
        item.get(
            "expected_paper_ids",
            [],
        )
    )

    dataset_split = item.get(
        "dataset_split",
        "",
    )

    topic = item.get(
        "topic",
        "",
    )

    print("\n" + "=" * 70)

    print(
        f"QUESTION {question_number}/{total_questions}"
    )

    print(
        f"ID: {question_id}"
    )

    print(
        f"Split: {dataset_split}"
    )

    print(
        f"Topic: {topic}"
    )

    print(
        f"Question: {question}"
    )

    start_time = time.perf_counter()

    response = pipeline.answer(
        question
    )

    measured_latency = (
        time.perf_counter()
        - start_time
    )

    answer = response.answer.strip()

    citations = response.citations

    cited_paper_ids = get_cited_paper_ids(
        response,
        response.retrieved_chunks,
    )

    citation_correctness = (
        compute_citation_correctness(
            cited_paper_ids,
            expected_paper_ids,
        )
    )

    insufficient_evidence = (
        is_insufficient_evidence(
            answer
        )
    )

    expected_paper_ids_text = (
        ";".join(
            sorted(expected_paper_ids)
        )
    )

    cited_paper_ids_text = (
        ";".join(
            cited_paper_ids
        )
    )

    citation_count = len(
        citations
    )

    result = {
        "question_id": question_id,
        "dataset_split": dataset_split,
        "topic": topic,
        "question": question,
        "expected_paper_ids": (
            expected_paper_ids_text
        ),
        "cited_paper_ids": (
            cited_paper_ids_text
        ),
        "citation_count": citation_count,
        "citation_correctness": (
            citation_correctness
        ),
        "insufficient_evidence": (
            insufficient_evidence
        ),
        "answer": answer,
        "latency_seconds": round(
            response.latency_seconds,
            3,
        ),
        "measured_latency_seconds": round(
            measured_latency,
            3,
        ),
        "retrieval_strategy": (
            response.retrieval_strategy
        ),
    }

    print(
        "\nGenerated Answer:"
    )

    print(answer)

    print(
        "\nExpected Paper IDs: "
        f"{expected_paper_ids_text}"
    )

    print(
        "Cited Paper IDs: "
        f"{cited_paper_ids_text or 'None'}"
    )

    print(
        "Citation Count: "
        f"{citation_count}"
    )

    print(
        "Citation Correctness: "
        f"{citation_correctness}"
    )

    print(
        "Insufficient Evidence: "
        f"{insufficient_evidence}"
    )

    print(
        "Latency: "
        f"{response.latency_seconds:.3f}s"
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save evaluation results to CSV.
    """

    if not results:
        raise ValueError(
            "No evaluation results to save."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        results[0].keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        "\nResults saved to:"
    )

    print(output_path)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    results: List[Dict[str, Any]],
) -> None:
    """
    Print aggregate end-to-end evaluation statistics.
    """

    total = len(results)

    if total == 0:
        return

    citation_correct = sum(
        1
        for result in results
        if result[
            "citation_correctness"
        ]
        == "correct"
    )

    citation_applicable = sum(
        1
        for result in results
        if result[
            "citation_correctness"
        ]
        != "not_applicable"
    )

    insufficient_count = sum(
        1
        for result in results
        if result[
            "insufficient_evidence"
        ]
    )

    latencies = [
        float(
            result[
                "latency_seconds"
            ]
        )
        for result in results
    ]

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    print("\n")
    print("=" * 70)
    print("END-TO-END RAG EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total questions: {total}"
    )

    print(
        "Citation-correct responses: "
        f"{citation_correct}"
    )

    print(
        "Citation correctness "
        "among applicable responses: "
        f"{citation_correct}/{citation_applicable}"
    )

    print(
        "Insufficient-evidence responses: "
        f"{insufficient_count}"
    )

    print(
        "Average response latency: "
        f"{average_latency:.3f}s"
    )

    print(
        "\nNOTE:"
    )

    print(
        "Citation correctness checks whether the cited "
        "paper matches the expected paper."
    )

    print(
        "It does NOT automatically determine whether "
        "the generated answer is semantically correct."
    )

    print(
        "Semantic answer correctness and faithfulness "
        "should be manually reviewed for the final report."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the end-to-end evaluation.

    By default, all questions are evaluated.

    Optional:
        --question-id Q001

    evaluates only the specified question.
    """

    print(
        "Research Paper Answer Bot"
    )

    print(
        "End-to-End Answer Evaluation"
    )

    print(
        f"\nDataset: {QUESTIONS_PATH}"
    )

    questions = load_questions(
        QUESTIONS_PATH
    )

    print(
        f"Evaluation questions available: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Optional single-question mode
    # --------------------------------------------------------

    question_id_filter = None

    if len(sys.argv) >= 3:

        if sys.argv[1] != "--question-id":
            raise ValueError(
                "Unknown argument. Use: "
                "--question-id Q001"
            )

        question_id_filter = sys.argv[2].strip()

        if not question_id_filter:
            raise ValueError(
                "Question ID cannot be empty."
            )

        filtered_questions = [
            item
            for item in questions
            if item.get("question_id") == question_id_filter
        ]

        if not filtered_questions:
            raise ValueError(
                f"Question ID not found: {question_id_filter}"
            )

        questions = filtered_questions

        print(
            f"Single-question mode: "
            f"{question_id_filter}"
        )

    elif len(sys.argv) > 1:

        raise ValueError(
            "Usage: "
            "python src\\evaluate_rag_answers.py "
            "[--question-id Q001]"
        )

    print(
        f"Questions to evaluate: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline()

    results = []

    total_questions = len(
        questions
    )

    overall_start = time.perf_counter()

    # --------------------------------------------------------
    # Evaluate questions
    # --------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        try:

            result = evaluate_question(
                pipeline=pipeline,
                item=item,
                question_number=index,
                total_questions=(
                    total_questions
                ),
            )

            results.append(
                result
            )

        except Exception as exc:

            question_id = item.get(
                "question_id",
                f"Q{index:03d}",
            )

            print(
                "\nERROR evaluating "
                f"{question_id}: {exc}"
            )

            results.append(
                {
                    "question_id": question_id,
                    "dataset_split": item.get(
                        "dataset_split",
                        "",
                    ),
                    "topic": item.get(
                        "topic",
                        "",
                    ),
                    "question": item.get(
                        "question",
                        "",
                    ),
                    "expected_paper_ids": ";".join(
                        item.get(
                            "expected_paper_ids",
                            [],
                        )
                    ),
                    "cited_paper_ids": "",
                    "citation_count": 0,
                    "citation_correctness": (
                        "error"
                    ),
                    "insufficient_evidence": False,
                    "answer": (
                        f"ERROR: {exc}"
                    ),
                    "latency_seconds": 0.0,
                    "measured_latency_seconds": 0.0,
                    "retrieval_strategy": "",
                }
            )

    overall_time = (
        time.perf_counter()
        - overall_start
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        results,
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    print(
        "\nTotal evaluation runtime: "
        f"{overall_time:.2f}s"
    )

    print(
        "\nEvaluation completed."
    )


if __name__ == "__main__":
    main()



