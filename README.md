# Research Paper Answer Bot

An academic question-answering system powered by Retrieval-Augmented Generation (RAG) that answers complex questions using curated peer-reviewed research papers and provides precise, verifiable citations.

---

## 1. Project Title

**Research Paper Answer Bot: Grounded Academic QA System with Verifiable Citations**

---

## 2. Project Overview

Research literature in AI, Machine Learning, and Computer Science is expanding at an unprecedented rate. Navigating hundreds of dense papers to find specific factual findings or methodology nuances is time-consuming and prone to human oversight. The **Research Paper Answer Bot** is a production-quality academic assistant designed to index, retrieve, and synthesize answers from research papers while strictly attributing claims to source documents.

Every generated answer is accompanied by supporting citations containing:
- **Paper title**
- **Page number**
- **Retrieved passage extract**

---

## 3. Problem Statement

General-purpose Large Language Models (LLMs) suffer from hallucinations, lack access to recent scientific literature, and do not provide verifiable citations pointing to specific pages or passages. In academic research, ungrounded claims and hallucinated references are unacceptable. This project addresses these challenges by implementing an end-to-end RAG architecture with dense and sparse hybrid retrieval, cross-encoder reranking, and explicit citation attribution.

---

## 4. Project Objectives

- **Grounded Information Retrieval**: Accurately retrieve passages relevant to complex academic queries across AI/LLM research domains.
- **Strict Hallucination Prevention**: Anchor LLM synthesis strictly to retrieved passages, explicitly stating when available evidence is insufficient.
- **Verifiable Multi-Level Citations**: Preserve complete document metadata (paper ID, title, authors, year, page number, chunk ID) across every stage of ingestion and retrieval.
- **Empirical Rigor**: Systematically benchmark chunking strategies, embedding models, and retrieval techniques on a dedicated evaluation dataset without fabricating experimental numbers.
- **Intuitive Academic Interface**: Provide researchers with a clean Streamlit interface showing responses, top-3 cited passages, and transparency into retrieval metrics.

---

## 5. Planned Architecture

The planned RAG pipeline processes research papers through the following pipeline:

```
Research Paper PDFs
       │
       ▼
[PDF Extraction (PyPDF)]
       │
       ▼
[Text Cleaning & Metadata Enrichment]
       │
       ▼
[Chunking & Chunk ID Tracking]
       │
       ▼
[Embedding Generation & Vector Database (ChromaDB)]
       │
       ├─────────────────────────┐
       ▼                         ▼
[Dense Semantic Search]    [Sparse Keyword Search (BM25)]
       │                         │
       └───────────┬─────────────┘
                   ▼
       [Hybrid Fusion (RRF / Weighted)]
                   │
                   ▼
       [Cross-Encoder Reranking]
                   │
                   ▼
       [Prompt Construction with Top Contexts]
                   │
                   ▼
       [LLM Synthesis (Grounded Answer)]
                   │
                   ▼
       [Answer + Top 3 Supporting Citations]
```

---

## 6. Technology Stack

- **Language & Runtime**: Python 3.11 / Python 3.13
- **Orchestration Framework**: LangChain (`langchain`, `langchain-core`, `langchain-community`)
- **Vector Database**: ChromaDB (`langchain-chroma`, `chromadb`)
- **Embeddings**: Sentence Transformers (`all-mpnet-base-v2`, `bge-base-en-v1.5`), OpenAI (`text-embedding-3-small`)
- **Lexical Retrieval**: BM25 (`rank-bm25`)
- **Reranking**: HuggingFace Cross-Encoder models
- **LLM Integrations**: OpenAI (`gpt-4o-mini`, `gpt-4o`) via `langchain-openai`
- **PDF Extraction**: PyPDF
- **Data & Evaluation**: Pandas, NumPy, Matplotlib
- **Web Interface**: Streamlit
- **Configuration & Environment**: python-dotenv

---

## 7. Planned Embedding Experiments

The following embedding models will be benchmarked on academic document chunks:

| Model Name | Provider / Framework | Dimension | Status | Hit@K | MRR |
|---|---|---|---|---|---|
| `sentence-transformers/all-mpnet-base-v2` | Hugging Face / Open-source | 768 | **To be evaluated** | To be evaluated | To be evaluated |
| `BAAI/bge-base-en-v1.5` | Hugging Face / Open-source | 768 | **To be evaluated** | To be evaluated | To be evaluated |
| `text-embedding-3-small` | OpenAI API | 1536 | **To be evaluated** | To be evaluated | To be evaluated |

*Note: All experimental results will be derived from actual empirical runs; no fabricated numbers are recorded.*

---

## 8. Planned Retrieval Experiments

We will implement, benchmark, and compare five distinct retrieval strategies:

| Strategy | Description | Status | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **1. Dense Similarity** | Standard cosine similarity in vector space | **To be evaluated** | To be evaluated | To be evaluated | To be evaluated |
| **2. MMR (Maximal Marginal Relevance)** | Balances query relevance with diversity | **To be evaluated** | To be evaluated | To be evaluated | To be evaluated |
| **3. BM25** | Exact keyword/lexical frequency scoring | **To be evaluated** | To be evaluated | To be evaluated | To be evaluated |
| **4. Hybrid (BM25 + Dense)** | Reciprocal Rank Fusion / linear score combination | **To be evaluated** | To be evaluated | To be evaluated | To be evaluated |
| **5. Hybrid + Reranking** | Top candidates reranked by Cross-Encoder | **To be evaluated** | To be evaluated | To be evaluated | To be evaluated |

---

## 9. Planned Evaluation Metrics

The retrieval and generation components will be evaluated using standard quantitative metrics:

- **Retrieval Metrics**:
  - **Hit@K**: Proportion of queries where at least one ground-truth passage appears in top-K results.
  - **Precision@K**: Fraction of retrieved chunks in top-K that are relevant.
  - **Recall@K**: Proportion of all relevant chunks retrieved within top-K.
  - **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of the first relevant document.
- **Generation & Citation Metrics**:
  - **Faithfulness / Groundedness**: Verification that claims in the answer are substantiated by retrieved passages.
  - **Answer Relevance**: Semantic alignment between the user question and generated answer.
  - **Citation Correctness**: Accuracy of attributed paper title, page number, and text extract.
  - **Retrieval & Inference Latency**: End-to-end execution time in seconds.

---

## 10. Planned Streamlit Interface

The web user interface will provide:
- **Interactive Question Input**: Query bar with sample prompts from the research paper domain.
- **Synthesized Answer Display**: Grounded answer formatted in clean Markdown.
- **Top 3 Supporting Citations**: Expandable citation cards showing:
  - Paper title
  - Exact page number
  - Verifiable retrieved passage snippet
- **Conversation History**: Multi-turn dialogue management with context memory.
- **Clear Conversation Action**: Instant reset of session state.
- **System & Retrieval Metadata**: Sidebar showing active embedding model, retriever mode, and similarity scores.

---

## 11. Project Structure

```
research-paper-answer-bot/
│
├── AGENTS.md                   # Project requirements, rules, and development guidelines
├── README.md                   # Project overview, architecture, and instructions
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore definitions
│
├── app.py                      # Streamlit application entrypoint
│
├── data/
│   ├── raw_papers/             # Original research paper PDFs
│   ├── processed/              # Processed artifacts & Chroma persistence
│   └── metadata.csv            # Paper metadata catalog
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Centralized configuration & environment loader
│   ├── document_loader.py      # PDF extraction & document structure loader
│   ├── chunking.py             # Chunking strategies & overlap experiments
│   ├── embeddings.py           # Embedding model interfaces
│   ├── vectorstore.py          # ChromaDB integration & persistence
│   ├── bm25_retriever.py       # BM25 sparse keyword retriever
│   ├── hybrid_retriever.py     # Hybrid dense-sparse fusion retriever
│   ├── reranker.py             # Cross-Encoder reranker
│   ├── prompts.py              # Prompt templates & citation constraints
│   ├── rag_pipeline.py         # End-to-end RAG workflow orchestration
│   ├── evaluation.py           # Retrieval and generation evaluation utilities
│   └── utils.py                # Logging, formatting, and helper utilities
│
├── notebooks/
│   └── capstone_experiments.ipynb  # Interactive experiment notebooks
│
├── evaluation/
│   ├── questions.json          # Curated evaluation questions & ground truth
│   ├── retrieval_results.csv   # Retrieval benchmark scores (To be evaluated)
│   └── final_results.csv       # Overall system evaluation scores (To be evaluated)
│
├── experiments/
│   ├── chunking_results.csv    # Chunk size & overlap benchmark data (To be evaluated)
│   ├── embedding_results.csv   # Embedding model comparisons (To be evaluated)
│   └── retrieval_results.csv   # Retrieval algorithm comparisons (To be evaluated)
│
├── screenshots/                # UI and architecture screenshots
│
├── tests/                      # Automated unit and integration tests
│
└── docs/                       # Project design and research documentation
    ├── architecture.md         # Detailed pipeline architecture
    ├── methodology.md          # Research paper selection & experimental methodology
    ├── evaluation.md           # Metric formulas & evaluation protocols
    └── limitations.md          # Known limitations & ethical considerations
```

---

## 12. Setup Instructions

### Prerequisites
- Python 3.11+ installed on the host machine.
- Git installed.

### Installation
1. Clone the repository or navigate to the project root directory:
   ```bash
   cd research-paper-answer-bot
   ```
2. Create and activate a virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy environment template and configure secrets:
   ```bash
   cp .env.example .env
   ```
5. Run the test suite:
   ```bash
   pytest tests/
   ```

---

## 13. Environment Variables

Configure your credentials in `.env` (never commit this file to version control):

| Variable | Description | Default / Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API Secret Key | `sk-...` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model for OpenAI experiments | `text-embedding-3-small` |
| `LLM_MODEL` | Synthesis LLM model | `gpt-4o-mini` |
| `CHROMA_PERSIST_DIRECTORY` | Local persistence directory for ChromaDB | `./data/processed/chroma_db` |
| `CHUNK_SIZE` | Default text chunk size in tokens/characters | `1000` |
| `CHUNK_OVERLAP` | Default chunk overlap | `200` |
| `RETRIEVAL_TOP_K` | Number of context passages to retrieve | `5` |

---

## 14. Future Improvements

- **Multi-Modal Document Processing**: Support for table parsing, LaTeX equations, and figure captions using vision-language models.
- **Hierarchical Indexing**: Parent-child document retrieval to preserve broader section context.
- **Active Query Reformulation**: HyDE (Hypothetical Document Embeddings) and query decomposition for multi-part academic questions.
- **User Paper Upload**: Dynamic UI ingestion allowing users to upload custom PDFs and immediately query them.
