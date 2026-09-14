"""
Persistent ChromaDB vector store for the Research Paper Answer Bot.

This module converts DocumentChunk objects into LangChain Documents,
preserves complete provenance metadata, and stores/retrieves them
using a persistent Chroma vector database.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .chunking import DocumentChunk
from .config import config


def document_chunks_to_documents(
    chunks: List[DocumentChunk],
) -> List[Document]:
    """
    Convert project DocumentChunk objects into LangChain Documents.

    All required provenance metadata is preserved.

    Args:
        chunks: List of DocumentChunk objects.

    Returns:
        List of LangChain Document objects.
    """
    if not chunks:
        raise ValueError("No document chunks were provided.")

    documents: List[Document] = []

    for chunk in chunks:
        if not chunk.content or not chunk.content.strip():
            continue

        metadata = chunk.metadata.to_dict()

        documents.append(
            Document(
                page_content=chunk.content,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No valid document chunks were available after "
            "removing empty chunks."
        )

    return documents


def create_vectorstore(
    chunks: List[DocumentChunk],
    embedding_function,
    persist_directory: Optional[Path] = None,
    collection_name: str = "research_papers",
    overwrite: bool = False,
) -> Chroma:
    """
    Create and persist a Chroma vector store from DocumentChunk objects.

    Args:
        chunks:
            Document chunks produced by src.chunking.py.

        embedding_function:
            LangChain-compatible embedding function.

        persist_directory:
            Directory where Chroma will persist its database.
            Defaults to config.chroma_persist_directory.

        collection_name:
            Name of the Chroma collection.

        overwrite:
            If False, refuse to overwrite an existing Chroma database.
            If True, delete the existing collection before rebuilding.

    Returns:
        Persistent Chroma vector store.

    Raises:
        ValueError:
            If chunks, embedding function, or collection name are invalid.

        FileExistsError:
            If an existing Chroma database is detected and overwrite=False.
    """
    if not chunks:
        raise ValueError(
            "Cannot create vector store from an empty chunk list."
        )

    if embedding_function is None:
        raise ValueError("An embedding function must be provided.")

    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty.")

    persist_path = Path(
        persist_directory
        if persist_directory is not None
        else config.chroma_persist_directory
    )

    persist_path.mkdir(parents=True, exist_ok=True)

    documents = document_chunks_to_documents(chunks)

    document_ids = [
        document.metadata["chunk_id"]
        for document in documents
    ]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError(
            "Duplicate chunk_id values detected. "
            "Every chunk must have a unique chunk_id."
        )

    chroma_database_marker = persist_path / "chroma.sqlite3"

    if chroma_database_marker.exists() and not overwrite:
        raise FileExistsError(
            f"An existing Chroma database was found at:\n"
            f"{persist_path}\n\n"
            f"Refusing to overwrite it. "
            f"Use load_vectorstore() to reuse the existing database, "
            f"or explicitly set overwrite=True to rebuild it."
        )

    if overwrite and chroma_database_marker.exists():
        existing_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_path),
        )

        existing_store.delete_collection()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_path),
    )

    vectorstore.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return vectorstore


def load_vectorstore(
    embedding_function,
    persist_directory: Optional[Path] = None,
    collection_name: str = "research_papers",
) -> Chroma:
    """
    Load an existing persistent Chroma vector store.

    Args:
        embedding_function:
            LangChain-compatible embedding function.

        persist_directory:
            Directory containing the Chroma database.
            Defaults to config.chroma_persist_directory.

        collection_name:
            Name of the Chroma collection.

    Returns:
        Loaded Chroma vector store.

    Raises:
        FileNotFoundError:
            If the Chroma database does not exist.

        ValueError:
            If the embedding function or collection name is invalid.
    """
    if embedding_function is None:
        raise ValueError("An embedding function must be provided.")

    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty.")

    persist_path = Path(
        persist_directory
        if persist_directory is not None
        else config.chroma_persist_directory
    )

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persistence directory does not exist:\n"
            f"{persist_path}"
        )

    chroma_database_marker = persist_path / "chroma.sqlite3"

    if not chroma_database_marker.exists():
        raise FileNotFoundError(
            f"No Chroma database was found in:\n"
            f"{persist_path}"
        )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_path),
    )

    return vectorstore


def get_vectorstore_count(vectorstore: Chroma) -> int:
    """
    Return the number of documents stored in a Chroma collection.

    Args:
        vectorstore:
            Chroma vector store instance.

    Returns:
        Number of stored documents.
    """
    if vectorstore is None:
        raise ValueError("vectorstore cannot be None.")

    return vectorstore._collection.count()