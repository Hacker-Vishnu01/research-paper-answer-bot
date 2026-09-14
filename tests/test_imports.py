"""
Tests to verify that all core project dependencies can be imported cleanly.
"""

import pytest


def test_import_dotenv():
    import dotenv
    assert hasattr(dotenv, "load_dotenv")


def test_import_pandas_numpy_matplotlib():
    import pandas as pd
    import numpy as np
    import matplotlib
    assert pd is not None
    assert np is not None
    assert matplotlib is not None


def test_import_pypdf():
    import pypdf
    assert hasattr(pypdf, "PdfReader")


def test_import_langchain():
    import langchain
    import langchain_core
    import langchain_community
    assert langchain is not None
    assert langchain_core is not None
    assert langchain_community is not None


def test_import_chromadb():
    import chromadb
    assert chromadb is not None


def test_import_sentence_transformers():
    import sentence_transformers
    from sentence_transformers import SentenceTransformer, CrossEncoder
    assert SentenceTransformer is not None
    assert CrossEncoder is not None


def test_import_bm25():
    import rank_bm25
    from rank_bm25 import BM25Okapi
    assert BM25Okapi is not None


def test_import_streamlit():
    import streamlit as st
    assert st is not None


def test_import_project_src():
    import src
    from src import config, document_loader, chunking, embeddings, vectorstore
    from src import bm25_retriever, hybrid_retriever, reranker, prompts, rag_pipeline, evaluation, utils
    assert src is not None
