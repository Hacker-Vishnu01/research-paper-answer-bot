"""
Chunking and text preprocessing module for the Research Paper Answer Bot.

Provides conservative scientific text cleaning and page-aware text splitting
using RecursiveCharacterTextSplitter. Preserves complete provenance metadata:
paper_id, paper_title, authors, year, topic, source_filename, source_url,
pdf_url, page_number, chunk_id, chunk_size, and chunk_overlap.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader import DocumentPage


def clean_text(text: str) -> str:
    """
    Conservative scientific text cleaning.
    
    Normalizes line breaks and whitespace while preserving mathematical
    symbols, equations, and section headings. Handles common hyphenation
    at line endings without aggressively deleting content.
    
    Args:
        text: Raw extracted page text.
        
    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    # Replace form feeds and control characters except newlines/tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Rejoin hyphenated words split across lines (e.g. "trans-\nformer" -> "transformer")
    # Only if the hyphen is preceded and followed by lowercase letters
    cleaned = re.sub(r"([a-z])-[\r\n]+([a-z])", r"\1\2", cleaned)

    # Normalize Windows and Mac line breaks to Unix
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace (spaces, tabs) into single spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse 3 or more consecutive newlines into 2 (preserving paragraph structure)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


@dataclass
class ChunkMetadata:
    """Complete provenance metadata preserved for every chunk."""
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: str
    year: int
    topic: str
    source_filename: str
    source_url: str
    pdf_url: str
    page_number: int
    chunk_size: int
    chunk_overlap: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "authors": self.authors,
            "year": self.year,
            "topic": self.topic,
            "source_filename": self.source_filename,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
            "page_number": self.page_number,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


@dataclass
class DocumentChunk:
    """A text chunk with complete provenance metadata."""
    content: str
    metadata: ChunkMetadata


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[DocumentChunk]:
    """
    Splits document pages into overlapping text chunks using RecursiveCharacterTextSplitter.
    
    Every generated chunk strictly preserves the page number and document provenance.
    
    Args:
        pages: List of DocumentPage instances.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlap in characters between adjacent chunks.
        
    Returns:
        List of DocumentChunk instances with unique chunk_ids.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[DocumentChunk] = []

    for page in pages:
        cleaned_content = clean_text(page.text)
        if not cleaned_content:
            continue

        raw_splits = splitter.split_text(cleaned_content)
        for c_idx, split_text in enumerate(raw_splits):
            chunk_id = f"{page.paper_id}_p{page.page_number}_c{c_idx}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                paper_id=page.paper_id,
                paper_title=page.paper_title,
                authors=page.authors,
                year=page.year,
                topic=page.topic,
                source_filename=page.source_filename,
                source_url=page.source_url,
                pdf_url=page.pdf_url,
                page_number=page.page_number,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks.append(DocumentChunk(content=split_text, metadata=meta))

    return chunks


def evaluate_chunking_configuration(
    pages: List[DocumentPage],
    config_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    """
    Evaluates a specific chunking configuration empirically over the corpus.
    
    Args:
        pages: Complete list of DocumentPage objects.
        config_name: Name/identifier of the configuration (e.g. 'Config A').
        chunk_size: Target chunk size.
        chunk_overlap: Overlap size.
        
    Returns:
        Dictionary of empirical metrics.
    """
    start_time = time.perf_counter()
    chunks = chunk_document_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elapsed = time.perf_counter() - start_time

    lengths = [len(c.content) for c in chunks] if chunks else [0]
    num_pages = len(pages)
    num_papers = len(set(p.paper_id for p in pages))

    # Verify metadata preservation across all chunks
    metadata_ok = all(
        c.metadata.chunk_id and c.metadata.paper_id and c.metadata.page_number >= 1
        for c in chunks
    )

    return {
        "config_name": config_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "avg_chunks_per_paper": round(len(chunks) / num_papers, 1) if num_papers else 0,
        "avg_chunks_per_page": round(len(chunks) / num_pages, 2) if num_pages else 0,
        "metadata_preserved": metadata_ok,
        "processing_time_seconds": round(elapsed, 4),
    }
