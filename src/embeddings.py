"""
Embeddings module for the Research Paper Answer Bot.

Provides model loading and embedding generation for:
1. sentence-transformers/all-mpnet-base-v2 (Open Source)
2. BAAI/bge-base-en-v1.5 (Open Source)
3. OpenAI text-embedding-3-small (Commercial API)

Local Hugging Face models are configured to use cached files only,
avoiding unnecessary network requests during application startup.
"""

import logging
import time
from typing import Any, List, Tuple

import numpy as np

from src.config import config


logger = logging.getLogger(__name__)


# ============================================================
# Supported model identifiers
# ============================================================

MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
OPENAI_MODEL_NAME = "text-embedding-3-small"


# ============================================================
# Hugging Face model configuration
# ============================================================

LOCAL_MODEL_KWARGS = {
    "device": "cpu",
    "local_files_only": True,
}

LOCAL_ENCODE_KWARGS = {
    "normalize_embeddings": True,
}


# ============================================================
# Embedding model factory
# ============================================================

def get_embedding_model(model_name: str) -> Any:
    """
    Factory function returning a LangChain-compatible embedding model.

    Hugging Face models are loaded from the local cache only.
    This avoids repeated internet checks when the models are already
    downloaded.

    Args:
        model_name:
            Identifier of the embedding model.

    Returns:
        LangChain-compatible embedding model instance.

    Raises:
        ValueError:
            If the model is unsupported or the OpenAI API key is missing.
    """

    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    # --------------------------------------------------------
    # MPNet
    # --------------------------------------------------------
    if model_name == MPNET_MODEL_NAME:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading MPNet embedding model from local cache.")

        return HuggingFaceEmbeddings(
            model_name=MPNET_MODEL_NAME,
            model_kwargs=LOCAL_MODEL_KWARGS.copy(),
            encode_kwargs=LOCAL_ENCODE_KWARGS.copy(),
        )

    # --------------------------------------------------------
    # BGE
    # --------------------------------------------------------
    elif model_name == BGE_MODEL_NAME:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading BGE embedding model from local cache.")

        return HuggingFaceEmbeddings(
            model_name=BGE_MODEL_NAME,
            model_kwargs=LOCAL_MODEL_KWARGS.copy(),
            encode_kwargs=LOCAL_ENCODE_KWARGS.copy(),
        )

    # --------------------------------------------------------
    # OpenAI
    # --------------------------------------------------------
    elif model_name == OPENAI_MODEL_NAME:

        if not config.has_openai_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "OpenAI embedding experiment cannot be executed."
            )

        from langchain_openai import OpenAIEmbeddings

        logger.info("Loading OpenAI embedding model.")

        return OpenAIEmbeddings(
            model=OPENAI_MODEL_NAME,
            api_key=config.openai_api_key,
        )

    # --------------------------------------------------------
    # Unsupported model
    # --------------------------------------------------------
    else:
        raise ValueError(
            f"Unsupported embedding model: '{model_name}'"
        )


# ============================================================
# Document embedding
# ============================================================

def embed_documents_with_timing(
    model: Any,
    texts: List[str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, float]:
    """
    Embed a list of documents and measure encoding time.

    Args:
        model:
            Initialized LangChain-compatible embedding model.

        texts:
            List of document strings.

        batch_size:
            Intended batch size for embedding operations.

    Returns:
        Tuple containing:
            - embeddings matrix as float32 NumPy array
            - elapsed encoding time in seconds
    """

    if not texts:
        raise ValueError("texts cannot be empty.")

    start_time = time.perf_counter()

    raw_embeddings = model.embed_documents(texts)

    elapsed = time.perf_counter() - start_time

    embeddings_matrix = np.array(
        raw_embeddings,
        dtype=np.float32,
    )

    return embeddings_matrix, elapsed


# ============================================================
# Query embedding
# ============================================================

def embed_query(
    model: Any,
    query: str,
) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        model:
            Initialized LangChain-compatible embedding model.

        query:
            Query text.

    Returns:
        1D NumPy array containing the query embedding.
    """

    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    vec = model.embed_query(query)

    return np.array(
        vec,
        dtype=np.float32,
    )