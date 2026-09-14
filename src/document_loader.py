"""
Document loader module for the Research Paper Answer Bot.

Handles PDF ingestion, text extraction, page-level tracking, and metadata preservation.
Ensures every page preserves complete provenance: paper ID, title, authors, year,
topic, source filename, 1-indexed page number, source URL, and PDF URL.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pypdf


@dataclass
class PaperMetadata:
    """Metadata schema for each indexed research paper."""
    paper_id: str
    title: str
    authors: str
    year: int
    topic: str
    source_url: str
    pdf_url: str
    local_filename: str


@dataclass
class DocumentPage:
    """Represents a single extracted page preserving full provenance metadata."""
    paper_id: str
    paper_title: str
    authors: str
    year: int
    topic: str
    source_filename: str
    page_number: int  # 1-indexed page number
    source_url: str
    pdf_url: str
    text: str


def load_metadata_catalog(catalog_path: Path) -> Dict[str, PaperMetadata]:
    """
    Loads paper metadata catalog from metadata.csv.
    
    Args:
        catalog_path: Path to metadata.csv.
        
    Returns:
        Dictionary mapping paper_id to PaperMetadata instance.
    """
    if not catalog_path.exists():
        return {}
    df = pd.read_csv(catalog_path, encoding="utf-8")
    catalog = {}
    for _, row in df.iterrows():
        paper_id = str(row["paper_id"]).strip()
        catalog[paper_id] = PaperMetadata(
            paper_id=paper_id,
            title=str(row.get("title", "")).strip(),
            authors=str(row.get("authors", "")).strip(),
            year=int(row.get("year", 0)),
            topic=str(row.get("topic", "")).strip(),
            source_url=str(row.get("source_url", "")).strip(),
            pdf_url=str(row.get("pdf_url", "")).strip(),
            local_filename=str(row.get("local_filename", "")).strip(),
        )
    return catalog


def load_pdf_pages(pdf_path: Path, paper_metadata: PaperMetadata) -> List[DocumentPage]:
    """
    Extracts text from each page of a research paper PDF preserving page numbers.
    
    Args:
        pdf_path: Path to the PDF file.
        paper_metadata: Metadata to attach to extracted pages.
        
    Returns:
        List of DocumentPage objects with 1-indexed page numbers.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    pages: List[DocumentPage] = []

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        doc_page = DocumentPage(
            paper_id=paper_metadata.paper_id,
            paper_title=paper_metadata.title,
            authors=paper_metadata.authors,
            year=paper_metadata.year,
            topic=paper_metadata.topic,
            source_filename=paper_metadata.local_filename,
            page_number=idx + 1,  # 1-indexed page number
            source_url=paper_metadata.source_url,
            pdf_url=paper_metadata.pdf_url,
            text=page_text,
        )
        pages.append(doc_page)

    return pages


def load_all_papers(data_dir: Path) -> List[DocumentPage]:
    """
    Loads all research papers declared in metadata.csv and parses all pages.
    
    Args:
        data_dir: Path to the project data directory containing metadata.csv and raw_papers/.
        
    Returns:
        List of all DocumentPage objects across all papers.
    """
    metadata_csv = data_dir / "metadata.csv"
    raw_papers_dir = data_dir / "raw_papers"

    catalog = load_metadata_catalog(metadata_csv)
    all_pages: List[DocumentPage] = []

    for paper_id, meta in catalog.items():
        pdf_path = raw_papers_dir / meta.local_filename
        if pdf_path.exists():
            pages = load_pdf_pages(pdf_path, meta)
            all_pages.extend(pages)
        else:
            raise FileNotFoundError(f"PDF for {paper_id} not found at {pdf_path}")

    return all_pages
