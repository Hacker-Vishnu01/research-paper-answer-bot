"""
Tests for configuration loading and validation.
"""

from pathlib import Path
from src.config import config, Config


def test_config_instance():
    assert isinstance(config, Config)
    assert config.base_dir.exists()
    assert config.data_dir.exists()
    assert isinstance(config.chunk_size, int)
    assert config.chunk_size > 0
    assert isinstance(config.chunk_overlap, int)
    assert config.chunk_overlap >= 0
    assert config.chunk_overlap < config.chunk_size
    assert isinstance(config.retrieval_top_k, int)
    assert config.retrieval_top_k > 0


def test_config_defaults():
    assert config.openai_embedding_model == "text-embedding-3-small"
    assert config.llm_model == "gpt-4o-mini"
    assert config.mpnet_embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert config.bge_embedding_model == "BAAI/bge-base-en-v1.5"
