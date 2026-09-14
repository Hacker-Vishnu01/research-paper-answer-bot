"""
Research Paper Answer Bot - Streamlit Application.

Provides a web interface for querying indexed academic research papers
using the production RAG pipeline.

Architecture:
    Research Papers
        ↓
    MPNet + ChromaDB
        +
    BM25
        ↓
    Hybrid Retrieval
        ↓
    Top Context Passages
        ↓
    Llama 3.2 3B via Ollama
        ↓
    Grounded Answer + Supporting Citations
"""

import os
import sys
import time

import streamlit as st


# ============================================================
# Ensure project root is available
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Project imports
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
from src.rag_pipeline import RAGPipeline
from src.vectorstore import load_vectorstore


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Application header
# ============================================================

st.title("📚 Research Paper Answer Bot")

st.caption(
    "Grounded Academic Question Answering with "
    "Hybrid Retrieval and Verifiable Citations"
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Initialize RAG pipeline
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    """
    Initialize and cache the complete RAG pipeline.

    The expensive resources are created only once per Streamlit
    session/cache lifecycle.
    """

    # --------------------------------------------------------
    # 1. Load research papers
    # --------------------------------------------------------

    pages = load_all_papers(
        config.data_dir
    )

    if not pages:
        raise RuntimeError(
            "No research paper pages were found."
        )

    # --------------------------------------------------------
    # 2. Create document chunks
    # --------------------------------------------------------

    chunks = chunk_document_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
    )

    if not chunks:
        raise RuntimeError(
            "No document chunks were created."
        )

    # --------------------------------------------------------
    # 3. Load MPNet embedding model
    # --------------------------------------------------------

    embeddings = get_embedding_model(
        MPNET_MODEL_NAME
    )

    # --------------------------------------------------------
    # 4. Load existing ChromaDB vector store
    # --------------------------------------------------------

    vectorstore = load_vectorstore(
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_directory,
        collection_name="research_papers_mpnet",
    )

    # --------------------------------------------------------
    # 5. Create dense retriever
    # --------------------------------------------------------

    dense_retriever = DenseRetriever(
        vectorstore=vectorstore
    )

    # --------------------------------------------------------
    # 6. Create BM25 retriever
    # --------------------------------------------------------

    bm25_retriever = BM25RetrieverWrapper(
        chunks
    )

    # --------------------------------------------------------
    # 7. Create hybrid retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5,
        rrf_k=60,
    )

    # --------------------------------------------------------
    # 8. Load local Llama 3.2 3B
    # --------------------------------------------------------

    llm = get_llm(
        model_name=OLLAMA_MODEL,
        temperature=0.0,
    )

    # --------------------------------------------------------
    # 9. Create RAG pipeline
    # --------------------------------------------------------

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=None,
        llm=llm,
        retrieval_top_k=10,
        context_top_k=3,
    )

    return pipeline


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.subheader("🤖 Models")

    st.markdown(
        "**Embedding Model**"
    )

    st.code(
        "all-mpnet-base-v2",
        language="text",
    )

    st.markdown(
        "**Language Model**"
    )

    st.code(
        OLLAMA_MODEL,
        language="text",
    )

    st.divider()

    st.subheader("🔎 Retrieval")

    st.markdown(
        "**Strategy:** Hybrid Retrieval"
    )

    st.markdown(
        "**Dense:** MPNet"
    )

    st.markdown(
        "**Sparse:** BM25"
    )

    st.markdown(
        "**Fusion:** Reciprocal Rank Fusion"
    )

    st.markdown(
        "**RRF k:** `60`"
    )

    st.markdown(
        "**Final Context:** `3` passages"
    )

    st.divider()

    st.subheader("📚 Research Corpus")

    st.metric(
        "Research Papers",
        "12",
    )

    st.metric(
        "Pages",
        "259",
    )

    st.metric(
        "Chunks",
        "1172",
    )

    st.divider()

    st.subheader("✂️ Chunking")

    st.markdown(
        "**Chunk Size:** `1000`"
    )

    st.markdown(
        "**Chunk Overlap:** `150`"
    )

    st.divider()

    st.subheader("💻 Runtime")

    st.success(
        "Local RAG Pipeline"
    )

    st.caption(
        "Llama 3.2 3B runs locally through Ollama. "
        "No OpenAI API key is required."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Initialize backend
# ============================================================

pipeline = None

with st.spinner(
    "Initializing the Research Paper Answer Bot..."
):

    try:

        initialization_start = time.perf_counter()

        pipeline = initialize_rag_pipeline()

        initialization_time = (
            time.perf_counter()
            - initialization_start
        )

    except Exception as exc:

        st.error(
            "Failed to initialize the RAG pipeline."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# System information
# ============================================================

with st.expander(
    "ℹ️ System Information",
    expanded=False,
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Papers",
            "12",
        )

    with col2:
        st.metric(
            "Pages",
            "259",
        )

    with col3:
        st.metric(
            "Chunks",
            "1172",
        )

    with col4:
        st.metric(
            "Retrieval",
            "Hybrid",
        )

    st.caption(
        "MPNet embeddings + BM25 lexical retrieval + "
        "Reciprocal Rank Fusion + Llama 3.2 3B"
    )

    st.caption(
        f"Initialization time: "
        f"{initialization_time:.2f} seconds"
    )


# ============================================================
# Display conversation history
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(role):

        st.markdown(content)

        citations = message.get(
            "citations",
            [],
        )

        if citations:

            st.markdown(
                "### 📑 Supporting Sources"
            )

            for index, citation in enumerate(
                citations,
                start=1,
            ):

                paper_title = citation.paper_title
                page_number = citation.page_number
                passage = citation.passage

                with st.expander(
                    f"{index}. {paper_title} — Page {page_number}"
                ):

                    st.markdown(
                        "**Supporting Passage**"
                    )

                    st.markdown(
                        f"> {passage}"
                    )


# ============================================================
# User question
# ============================================================

user_question = st.chat_input(
    "Ask a question about the indexed research papers..."
)


if user_question:

    user_question = user_question.strip()

    if not user_question:
        st.warning(
            "Please enter a question."
        )
        st.stop()

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status = st.empty()

        try:

            status.info(
                "🔎 Searching the research papers..."
            )

            start_time = time.perf_counter()

            response = pipeline.answer(
                user_question
            )

            request_time = (
                time.perf_counter()
                - start_time
            )

            status.empty()

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.markdown(
                response.answer
            )

            # ------------------------------------------------
            # Supporting sources
            # ------------------------------------------------

            citations = response.citations

            if citations:
                st.markdown("### 📑 Supporting Sources")

                for index, citation in enumerate(citations, start=1):
                    paper_title = citation.paper_title
                    page_number = citation.page_number
                    passage = citation.passage

                    with st.expander(
                        f"{index}. {paper_title} — Page {page_number}"
                    ):
                        st.markdown("**Supporting Passage**")
                        st.markdown(f"> {passage}")
            # ------------------------------------------------
            # Response information
            # ------------------------------------------------

            with st.expander(
                "⚡ Response Information",
                expanded=False,
            ):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Response Time",
                        f"{response.latency_seconds:.2f}s",
                    )

                with col2:
                    st.metric(
                        "Context Chunks",
                        len(
                            response.retrieved_chunks
                        ),
                    )

                with col3:
                    st.metric(
                        "Retrieval",
                        "Hybrid",
                    )

                st.caption(
                    f"Request time: "
                    f"{request_time:.2f} seconds"
                )

                st.caption(
                    f"Model: {OLLAMA_MODEL}"
                )

                st.caption(
                    "Reranker: Disabled "
                    "(Hybrid RRF selected as primary strategy)"
                )

            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "citations": citations,
                    "latency_seconds": response.latency_seconds,
                    "retrieval_strategy": response.retrieval_strategy,
                }
            )

        except Exception as exc:

            status.empty()

            st.error(
                "An error occurred while generating the answer."
            )

            st.exception(exc)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I could not generate an answer because "
                        "an error occurred in the RAG pipeline."
                    ),
                }
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Research Paper Answer Bot | "
    "Hybrid Retrieval (MPNet + BM25) | "
    "ChromaDB | "
    "Llama 3.2 3B via Ollama"
)