"""
Configuration module for the Research Paper Answer Bot.

Loads environment variables from .env file and provides structured access
to all pipeline parameters, models, and file paths.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """Application configuration parameters."""

    # Project directories
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_papers_dir: Path = BASE_DIR / "data" / "raw_papers"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    metadata_path: Path = BASE_DIR / "data" / "metadata.csv"

    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", "")

    # Embedding & LLM Models
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Local open-source embedding models for benchmarking
    mpnet_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    bge_embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Vector store configuration
    chroma_persist_directory: Path = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "processed" / "chroma_db"))
    )

    # Document Chunking Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Configuration
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())


# Global configuration instance
config = Config()
