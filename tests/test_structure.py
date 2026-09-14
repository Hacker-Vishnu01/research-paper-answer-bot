"""
Tests to verify the initial project file and directory structure.
"""

from pathlib import Path


def test_required_directories_exist():
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw_papers",
        "data/processed",
        "src",
        "notebooks",
        "evaluation",
        "experiments",
        "screenshots",
        "tests",
        "docs",
    ]
    for d in expected_dirs:
        dir_path = base_dir / d
        assert dir_path.exists(), f"Directory missing: {d}"
        assert dir_path.is_dir(), f"Expected directory, but found file: {d}"


def test_required_files_exist():
    base_dir = Path(__file__).resolve().parent.parent
    expected_files = [
        "AGENTS.md",
        "README.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "app.py",
        "data/metadata.csv",
        "data/dataset_statistics.csv",
        "src/__init__.py",
        "src/config.py",
        "src/document_loader.py",
        "src/chunking.py",
        "src/embeddings.py",
        "src/vectorstore.py",
        "src/bm25_retriever.py",
        "src/hybrid_retriever.py",
        "src/reranker.py",
        "src/prompts.py",
        "src/rag_pipeline.py",
        "src/evaluation.py",
        "src/utils.py",
        "notebooks/capstone_experiments.ipynb",
        "evaluation/questions.json",
        "evaluation/retrieval_results.csv",
        "evaluation/final_results.csv",
        "experiments/chunking_results.csv",
        "experiments/embedding_results.csv",
        "experiments/retrieval_results.csv",
        "docs/architecture.md",
        "docs/methodology.md",
        "docs/evaluation.md",
        "docs/limitations.md",
        "docs/dataset.md",
    ]
    for f in expected_files:
        file_path = base_dir / f
        assert file_path.exists(), f"File missing: {f}"
        assert file_path.is_file(), f"Expected file, but found directory: {f}"
