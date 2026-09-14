"""
Tests for research paper dataset integrity, metadata catalog, and page-aware document loader.
"""

from pathlib import Path
import pandas as pd
import pypdf
from src.document_loader import load_metadata_catalog, load_all_papers, DocumentPage


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
STATS_PATH = DATA_DIR / "dataset_statistics.csv"


def test_metadata_file_exists():
    assert METADATA_PATH.exists(), f"Metadata CSV not found at {METADATA_PATH}"
    assert METADATA_PATH.stat().st_size > 0, "Metadata CSV is empty"


def test_required_metadata_columns():
    df = pd.read_csv(METADATA_PATH, encoding="utf-8")
    expected_columns = {
        "paper_id",
        "title",
        "authors",
        "year",
        "topic",
        "source_url",
        "pdf_url",
        "local_filename",
    }
    assert expected_columns.issubset(set(df.columns)), (
        f"Missing columns. Expected: {expected_columns}, Found: {set(df.columns)}"
    )
    assert len(df) == 12, f"Expected 12 papers in dataset, found {len(df)}"
    # Check no nulls
    assert df.isnull().sum().sum() == 0, "Found null values in metadata.csv"


def test_paper_ids_and_titles_unique():
    df = pd.read_csv(METADATA_PATH, encoding="utf-8")
    assert df["paper_id"].is_unique, "Paper IDs must be strictly unique"
    assert df["title"].is_unique, "Paper titles must be strictly unique"
    assert df["local_filename"].is_unique, "Local filenames must be strictly unique"


def test_pdf_files_exist_and_readable():
    df = pd.read_csv(METADATA_PATH, encoding="utf-8")
    for _, row in df.iterrows():
        pdf_path = RAW_PAPERS_DIR / row["local_filename"]
        assert pdf_path.exists(), f"PDF file not found: {pdf_path}"
        assert pdf_path.stat().st_size > 10000, f"PDF file too small: {pdf_path}"
        
        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) > 0, f"Zero pages in {pdf_path}"
        first_page_text = reader.pages[0].extract_text()
        assert first_page_text and len(first_page_text.strip()) > 50, (
            f"Extracted text too short in {pdf_path}"
        )


def test_page_metadata_preservation():
    catalog = load_metadata_catalog(METADATA_PATH)
    assert len(catalog) == 12

    pages = load_all_papers(DATA_DIR)
    assert len(pages) == 259, f"Expected exactly 259 total pages, got {len(pages)}"

    for page in pages:
        assert isinstance(page, DocumentPage)
        assert page.paper_id in catalog
        assert page.page_number >= 1
        assert len(page.paper_title) > 0
        assert len(page.source_filename) > 0
        assert page.source_url.startswith("http")
        assert page.pdf_url.startswith("http")


def test_dataset_statistics():
    assert STATS_PATH.exists(), f"Dataset statistics file missing at {STATS_PATH}"
    df_stats = pd.read_csv(STATS_PATH, encoding="utf-8")
    assert len(df_stats) == 1
    assert int(df_stats.iloc[0]["paper_count"]) == 12
    assert int(df_stats.iloc[0]["total_pages"]) == 259
