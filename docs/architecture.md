# Research Paper Answer Bot - Architecture Documentation

## 1. Architecture Overview

The Research Paper Answer Bot is a multi-stage **Retrieval-Augmented Generation (RAG)** system designed to answer questions using a curated corpus of 12 research papers.

The architecture separates the system into two major stages:

1. **Offline document ingestion and indexing**
2. **Online retrieval and grounded answer generation**

The production architecture combines:

* Page-aware PDF text extraction
* Conservative text preprocessing
* Recursive character-based chunking
* MPNet dense embeddings
* ChromaDB vector storage
* BM25 lexical retrieval
* Hybrid Reciprocal Rank Fusion (RRF)
* Strict context-grounded prompting
* Local Llama 3.2 3B generation through Ollama
* Application-generated citations based on retrieved passages

The architecture is designed to preserve document provenance throughout the complete pipeline.

---

## 2. High-Level Production Pipeline

```text
                    OFFLINE INGESTION
                           |
                           v
              +-------------------------+
              |   12 Research Papers    |
              |       259 pages         |
              +------------+------------+
                           |
                           v
              +-------------------------+
              |   PDF Text Extraction   |
              |         pypdf            |
              |    Page-by-page text    |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Conservative Cleaning   |
              |                         |
              | * Control characters    |
              | * Hyphenation repair    |
              | * Line normalization    |
              | * Whitespace cleanup    |
              | * Paragraph preservation|
              +------------+------------+
                           |
                           v
              +-------------------------+
              |       Chunking          |
              | RecursiveCharacter      |
              | TextSplitter            |
              |                         |
              | Size = 1000             |
              | Overlap = 150           |
              +------------+------------+
                           |
                           v
                    1172 Production
                         Chunks
                           |
                 +---------+---------+
                 |                   |
                 v                   v
       +------------------+  +------------------+
       |  MPNet Embeddings|  |   BM25 Index     |
       |  768 dimensions  |  |  Lexical Search  |
       +--------+---------+  +--------+---------+
                |                     |
                v                     |
       +------------------+            |
       |    ChromaDB      |            |
       | Persistent Vector|            |
       |      Store       |            |
       +--------+---------+            |
                |                     |
                +----------+----------+
                           |
                           | ONLINE QUERY
                           v
                  +------------------+
                  |   User Question  |
                  +--------+---------+
                           |
                           v
              +-------------------------+
              | Hybrid Retrieval (RRF) |
              |                         |
              | Dense + BM25            |
              | alpha = 0.5             |
              | RRF k = 60              |
              +------------+------------+
                           |
                           v
                  Top 10 Hybrid Chunks
                           |
                           v
                  Top 9 Context Chunks
                           |
                           v
              +-------------------------+
              |  Grounded Prompt       |
              |                         |
              | * Use provided context |
              | * No outside knowledge |
              | * Support factual      |
              |   claims with evidence |
              | * Abstain when evidence|
              |   is insufficient      |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Ollama                  |
              | Llama 3.2 3B            |
              | Temperature = 0.0       |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Generated Answer       |
              | + Up to 3 Application- |
              |   Generated Citations  |
              +-------------------------+
```

---

## 3. Offline Document Ingestion

The offline stage prepares the research corpus for retrieval.

### 3.1 PDF Loading

The project uses `pypdf` to read the 12 research-paper PDFs.

Documents are processed **page by page** rather than as a single large text block.

For each page, the system preserves:

* Paper ID
* Paper title
* Authors
* Publication year
* Topic
* Source filename
* Source URL
* PDF URL
* 1-indexed page number

The implementation is located in:

```text
src/document_loader.py
```

---

## 4. Text Preprocessing

The extracted text undergoes conservative cleaning before chunking.

The preprocessing implementation is located in:

```text
src/chunking.py
```

The cleaning process:

* Removes control characters while preserving newline and tab characters.
* Rejoins lowercase-letter hyphenation across line breaks.
* Normalizes line endings.
* Collapses unnecessary horizontal whitespace.
* Reduces excessive blank lines.
* Preserves mathematical symbols.
* Preserves headings and paragraph boundaries.

The system does **not** intentionally remove document headers and footers.

This conservative approach is used to minimize information loss from academic documents.

---

## 5. Chunking

The project uses:

```text
RecursiveCharacterTextSplitter
```

with the following production configuration:

| Parameter     |                        Production Value |
| ------------- | --------------------------------------: |
| Chunk size    |                         1000 characters |
| Chunk overlap |                          150 characters |
| Total chunks  |                                    1172 |
| Separators    | `\n\n`, `\n`, `. `, space, empty string |

Each page is chunked independently.

Every generated chunk retains provenance metadata so that the retrieved text can be traced back to its source paper and page.

The implementation is located in:

```text
src/chunking.py
```

---

## 6. Embedding and Vector Storage

The production dense embedding model is:

```text
sentence-transformers/all-mpnet-base-v2
```

Configuration:

* **Embedding dimension:** 768
* **Production chunks:** 1172
* **Normalization:** enabled
* **Execution:** local CPU environment

The vectors are stored persistently in ChromaDB.

Production collection:

```text
research_papers_mpnet
```

The relevant implementation files are:

```text
src/embeddings.py
src/vectorstore.py
```

Other embedding models are supported experimentally, but MPNet is the model used by the production evaluation pipeline.

---

## 7. BM25 Lexical Retrieval

In addition to dense semantic retrieval, the system maintains a BM25 lexical retrieval path.

Implementation:

```text
rank_bm25.BM25Okapi
```

The BM25 tokenizer retains technical tokens such as:

* words
* numbers
* underscores
* periods
* slashes
* plus signs
* hyphenated terms

This is useful for research questions containing exact technical terminology, model names, acronyms, or specialized terms.

Implementation:

```text
src/bm25_retriever.py
```

---

## 8. Hybrid Retrieval

The production retrieval strategy combines dense MPNet retrieval and BM25 retrieval using **Reciprocal Rank Fusion (RRF)**.

The production configuration is:

| Parameter                  |                  Value |
| -------------------------- | ---------------------: |
| Dense retrieval            |                  MPNet |
| Lexical retrieval          |                   BM25 |
| Fusion method              | Reciprocal Rank Fusion |
| Alpha                      |                    0.5 |
| RRF constant `k`           |                     60 |
| Final retrieval candidates |                     10 |

The hybrid score is calculated from the dense and BM25 reciprocal-rank scores.

With `alpha = 0.5`, the production system gives equal weighting to the two retrieval sources.

Implementation:

```text
src/hybrid_retriever.py
```

---

## 9. Retrieval Flow

For each user question:

1. The question is passed to the dense retriever.
2. The question is passed to the BM25 retriever.
3. Each retriever produces ranked candidate chunks.
4. The rankings are combined using RRF.
5. The top **10 hybrid chunks** are retained.
6. The first **9 chunks** are passed to the generation stage.

The production retrieval path does **not** use cross-encoder reranking.

---

## 10. Cross-Encoder Reranking

A cross-encoder reranking implementation exists as an experimental evaluation path.

It was evaluated separately from the production pipeline.

The benchmark showed:

* Hybrid RRF MRR: **1.000**
* Reranked MRR: **1.000**
* Hybrid average retrieval latency: **0.0750 seconds**
* Reranked average retrieval latency: **0.9231 seconds**

Therefore, cross-encoder reranking does not provide a measured MRR improvement over Hybrid RRF in the current benchmark, while introducing substantially higher retrieval latency.

For this reason, reranking is **experimental and not part of the production answer pipeline**.

Implementation:

```text
src/reranker.py
src/evaluate_reranked_retrieval.py
```

---

## 11. Context Assembly

The production RAG pipeline retrieves 10 hybrid candidates and uses the first 9 as generation context.

Each context passage includes:

* Paper title
* Page number
* Retrieved passage text

The context is assembled by:

```text
src/rag_pipeline.py
```

This provides the language model with a compact set of evidence passages while preserving source provenance.

---

## 12. Grounded Prompting

The generation prompt is designed to prevent unsupported answers.

The system instructs the language model to:

* Use only the supplied research passages.
* Avoid outside knowledge.
* Support factual claims using the provided evidence.
* Answer all supported parts of the question.
* Synthesize multiple passages when the relationship is directly supported by the evidence.
* Identify unsupported parts when only partial evidence is available.
* Return an explicit insufficient-evidence response when the papers do not contain enough information.

The exact fallback response is:

```text
The provided research papers do not contain sufficient evidence to answer this question.
```

Prompt implementation:

```text
src/prompts.py
```

---

## 13. Local Language Model

The production generation model is:

```text
Llama 3.2 3B
```

served locally through:

```text
Ollama
```

Production generation configuration:

| Parameter   | Value         |
| ----------- | ------------- |
| Model       | `llama3.2:3b` |
| Temperature | `0.0`         |
| Execution   | Local Ollama  |

The zero-temperature configuration is used to make generation behavior more deterministic during evaluation.

Implementation:

```text
src/llm.py
```

---

## 14. Answer Generation and Citations

After retrieving context, the RAG pipeline sends the grounded prompt and evidence passages to the local language model.

The generated answer may be accompanied by up to **three application-generated citations**.

The application selects citation information from the final context passages rather than allowing the model to invent arbitrary source references.

Citation information is derived from retrieved chunk metadata, including:

* Paper title
* Page number
* Source passage

The application logic is implemented in:

```text
src/rag_pipeline.py
```

The current automated evaluation primarily verifies whether the cited **paper ID matches the expected paper ID** for the question. It does not constitute a complete independent verification of every quoted passage.

---

## 15. Provenance Flow

Document provenance is preserved throughout the system:

```text
PDF
 v
Page number
 v
Extracted page text
 v
Chunk
 v
Chunk metadata
 v
Embedding / BM25 index
 v
Retrieved chunk
 v
Context passage
 v
Generated citation
```

This allows retrieved evidence to be traced back to the original research paper and page.

---

## 16. Production Components

| Component          | Technology                     | Purpose                            |
| ------------------ | ------------------------------ | ---------------------------------- |
| PDF extraction     | `pypdf`                        | Page-level text extraction         |
| Text preprocessing | Python                         | Conservative text cleaning         |
| Chunking           | RecursiveCharacterTextSplitter | Create retrieval chunks            |
| Dense embeddings   | `all-mpnet-base-v2`            | Semantic representation            |
| Vector database    | ChromaDB                       | Dense vector storage               |
| Lexical retrieval  | BM25Okapi                      | Keyword-based retrieval            |
| Hybrid fusion      | RRF                            | Combine dense and lexical rankings |
| LLM serving        | Ollama                         | Local model inference              |
| Generation model   | Llama 3.2 3B                   | Grounded answer generation         |

---

## 17. Production Configuration

The evaluated production pipeline uses:

```text
Papers:                  12
Pages:                   259
Production chunks:       1172

Chunk size:              1000
Chunk overlap:            150

Embedding model:         sentence-transformers/all-mpnet-base-v2
Embedding dimension:      768

Dense retrieval:          MPNet
Lexical retrieval:        BM25

Hybrid method:            RRF
Hybrid alpha:             0.5
RRF k:                    60
Retrieval top-k:          10
Context top-k:             9

Reranker:                  None

LLM:                       llama3.2:3b
LLM provider:              Ollama
Temperature:               0.0
```

---

## 18. Offline and Online Separation

The architecture separates expensive document processing from query-time operations.

### Offline Stage

The following operations are performed before answering user questions:

```text
PDF loading
    v
Text extraction
    v
Cleaning
    v
Chunking
    v
Embedding generation
    v
ChromaDB indexing
    v
BM25 index construction
```

### Online Stage

The following operations occur when a user asks a question:

```text
User question
    v
Dense retrieval
    +
BM25 retrieval
    v
RRF fusion
    v
Top 10 candidates
    v
Top 9 context passages
    v
Grounded prompt
    v
Llama 3.2 3B
    v
Answer + application-generated citations
```

This separation avoids repeating the complete document-ingestion process for every user query.

---

## 19. Architecture Design Principles

The architecture follows several core principles:

### 19.1 Grounded Generation

The language model is instructed to answer using the retrieved research passages rather than relying on external knowledge.

### 19.2 Provenance Preservation

Paper and page metadata are preserved from document ingestion through retrieval and answer generation.

### 19.3 Hybrid Retrieval

Dense semantic retrieval and lexical BM25 retrieval complement each other and are combined using RRF.

### 19.4 Conservative Preprocessing

The preprocessing stage avoids aggressive transformations that could remove useful scientific information.

### 19.5 Explicit Abstention

When sufficient evidence is unavailable, the system can return an explicit insufficient-evidence response instead of fabricating an answer.

### 19.6 Reproducible Local Inference

The evaluated generation pipeline uses a locally hosted Llama 3.2 3B model through Ollama with temperature set to 0.0.

---

## 20. Architecture Limitations

The current architecture has several practical limitations:

1. The corpus contains only 12 research papers.
2. PDF text extraction may lose some visual layout information.
3. Tables and figures are not represented as multimodal data.
4. Local LLM generation introduces substantial end-to-end latency.
5. Cross-encoder reranking is currently too expensive for the selected production configuration.
6. The current citation evaluation primarily verifies expected paper IDs rather than performing complete independent quote verification.
7. The system has not been evaluated on a large-scale external academic corpus.

---

## 21. Source Code Mapping

The major architecture components map to the following source files:

```text
src/
+-- document_loader.py
|   +-- PDF loading and page-level metadata
|
+-- chunking.py
|   +-- Text cleaning and recursive chunking
|
+-- embeddings.py
|   +-- Embedding model configuration
|
+-- vectorstore.py
|   +-- ChromaDB vector storage
|
+-- dense_retriever.py
|   +-- Dense similarity retrieval
|
+-- bm25_retriever.py
|   +-- BM25 lexical retrieval
|
+-- hybrid_retriever.py
|   +-- RRF hybrid retrieval
|
+-- reranker.py
|   +-- Experimental cross-encoder reranking
|
+-- prompts.py
|   +-- Grounded generation instructions
|
+-- llm.py
|   +-- Ollama / Llama 3.2 3B integration
|
+-- rag_pipeline.py
    +-- Retrieval-to-generation orchestration
```

---

## 22. Architecture Summary

The final production architecture is:

```text
12 Research Papers
        v
259 Pages
        v
Page-Level PDF Extraction
        v
Conservative Cleaning
        v
1000 / 150 Chunking
        v
1172 Chunks
        v
 +---------------+---------------+
 |               |               |
 v               v               |
MPNet          BM25              |
Dense          Lexical           |
Retrieval      Retrieval         |
 |               |               |
 +-------+-------+               |
         v                       |
    Hybrid RRF                   |
    alpha = 0.5                 |
    k = 60                       |
         v                       |
    Top 10 Chunks                |
         v                       |
    Top 9 Context                |
         v                       |
   Grounded Prompt               |
         v                       |
   Llama 3.2 3B                 |
   via Ollama                    |
         v                       |
    Generated Answer             |
         v                       |
 Up to 3 Application-Generated   |
       Citations                 |
```

This architecture provides a traceable, hybrid-retrieval RAG pipeline that combines semantic and lexical retrieval with locally hosted grounded generation.

