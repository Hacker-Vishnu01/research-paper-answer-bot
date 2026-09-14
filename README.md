# Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) system for answering academic questions from a curated collection of research papers with grounded, paper-level evidence.

The system combines **dense semantic retrieval**, **BM25 lexical retrieval**, **Reciprocal Rank Fusion (RRF)**, and a **local Llama 3.2 3B language model** running through Ollama.

---

## 1. Project Overview

Research papers contain large amounts of technical information, mathematical expressions, terminology, and supporting evidence. Finding reliable answers manually across multiple papers can be time-consuming.

The Research Paper Answer Bot addresses this problem by:

* Ingesting a curated collection of research papers.
* Extracting and processing PDF text.
* Splitting documents into overlapping chunks.
* Generating dense semantic embeddings.
* Building a sparse BM25 retrieval index.
* Combining dense and sparse rankings using Reciprocal Rank Fusion.
* Selecting the most relevant passages.
* Generating grounded answers using a local LLM.
* Evaluating answers against a predefined question set.
* Checking citation correctness and evidence sufficiency.

The system is designed to keep generated answers bounded by the indexed research-paper corpus.

---

## 2. Problem Statement

Traditional keyword-based search can miss relevant information when a user's question uses terminology different from the wording in a research paper.

Pure semantic retrieval can also miss important exact terms, technical keywords, equations, and named concepts.

This project therefore combines:

1. **Dense retrieval** for semantic similarity.
2. **BM25 retrieval** for lexical matching.
3. **Reciprocal Rank Fusion** for combining the two retrieval rankings.
4. **Grounded generation** using retrieved evidence.

The goal is to improve retrieval coverage while reducing unsupported answers.

---

## 3. Objectives

The main objectives are:

* Build a reproducible RAG pipeline for academic research papers.
* Support semantic and keyword-based retrieval.
* Preserve useful mathematical expressions during PDF extraction.
* Provide grounded answers using retrieved evidence.
* Prevent unsupported information from being presented as fact.
* Evaluate retrieval and answer-generation behavior using a fixed question set.
* Provide a simple Streamlit interface for user interaction.
* Maintain a clean and reproducible project structure.

---

## 4. System Architecture

```text
                    Research Paper PDFs
                           |
                           v
                  PDF Document Loader
                           |
                           v
              Conservative Text Extraction
                           |
                           v
                Document Chunking
             (1000 chars / 150 overlap)
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Dense Embeddings               BM25 Index
       all-mpnet-base-v2              Lexical Search
             |                           |
             +-------------+-------------+
                           |
                           v
              Reciprocal Rank Fusion
                    RRF (k=60)
                           |
                           v
                 Top 10 Candidates
                           |
                           v
                  Top 3 Context
                           |
                           v
                  Grounded LLM
              Llama 3.2 3B / Ollama
                           |
                           v
                  Grounded Answer
                           |
                           v
                 Streamlit Interface
```

---

## 5. Production Configuration

The production pipeline uses the following configuration:

| Component            | Production configuration                  |
| -------------------- | ----------------------------------------- |
| Corpus               | 12 research papers                        |
| Pages                | 259                                       |
| Indexed chunks       | 1,172                                     |
| Chunk size           | 1,000 characters                          |
| Chunk overlap        | 150 characters                            |
| Dense embedding      | `sentence-transformers/all-mpnet-base-v2` |
| Sparse retrieval     | BM25                                      |
| Hybrid fusion        | Reciprocal Rank Fusion (RRF)              |
| RRF constant         | `k = 60`                                  |
| Retrieval candidates | Top 10                                    |
| Final context        | Top 3 passages                            |
| LLM                  | `llama3.2:3b`                             |
| LLM runtime          | Ollama                                    |
| Temperature          | `0.0`                                     |
| Production reranker  | None                                      |

The production configuration favors a lightweight local architecture combining semantic retrieval, lexical retrieval, and grounded answer generation.

---

## 6. Document Processing

The document-processing pipeline extracts text from research-paper PDFs and converts the extracted content into searchable chunks.

### PDF Extraction

Normal `pypdf` extraction is used by default.

A conservative fallback detects a small class of suspicious mathematical extraction patterns and uses layout extraction only for affected pages.

This targeted approach avoids applying layout extraction globally, because global layout extraction can produce incomplete output for rotated or specially formatted PDF text.

The processing pipeline preserves metadata such as:

* Paper ID
* Paper title
* Page number
* Source filename
* Source URL
* PDF URL

### Mathematical Text Handling

Mathematical expressions require special handling because PDF text extraction can split formulas across multiple lines.

For example, the Scaled Dot-Product Attention expression from the Transformer paper was validated after targeted extraction handling:

```text
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

The answer-generation prompt also instructs the model to preserve mathematical notation and avoid changing the mathematical meaning of formulas.

---

## 7. Chunking

The production chunking configuration is:

* **Chunk size:** 1,000 characters
* **Chunk overlap:** 150 characters

Each chunk contains metadata used for retrieval, evaluation, and citation:

```text
chunk_id
paper_id
paper_title
page_number
source_filename
source_url
pdf_url
chunk_size
chunk_overlap
```

The production corpus contains **1,172 indexed chunks** from the 12 research papers.

---

## 8. Hybrid Retrieval

The system combines dense semantic retrieval with sparse lexical retrieval.

### Dense Retrieval

Dense embeddings are generated using:

```text
sentence-transformers/all-mpnet-base-v2
```

Dense retrieval allows the system to identify passages based on semantic similarity even when the query and source passage do not use exactly the same wording.

### Sparse Retrieval

BM25 provides lexical retrieval.

This is useful for:

* Technical terminology
* Exact keywords
* Named concepts
* Equations
* Abbreviations
* Paper-specific terminology

### Reciprocal Rank Fusion

The dense and BM25 rankings are combined using Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

where:

* `d` is a document or passage.
* `rank(d)` is its rank in an individual retrieval result.
* `k = 60`.

The production retriever considers the top **10 retrieval candidates** and passes the top **3 passages** to the answer-generation stage.

---

## 9. Answer Generation

The retrieved passages are provided to a local language model through LangChain and Ollama.

### Production Model

```text
Llama 3.2 3B
```

### Runtime

```text
Ollama
```

### Temperature

```text
0.0
```

The generation prompt is designed to keep answers grounded in the supplied retrieval context.

The prompt rules include:

* Answer only from the supplied context.
* Do not invent unsupported facts.
* Report insufficient evidence when the context does not support an answer.
* Preserve mathematical notation.
* Reconstruct formulas when PDF extraction has separated mathematical components.
* Preserve mathematical operators, fractions, superscripts, subscripts, parentheses, square roots, and transpose notation.
* Do not assign meanings to mathematical symbols unless the meaning is supported by the context.
* Use clear mathematical notation when appropriate.
* Keep responses concise but complete.
* Do not invent citation information.

---

## 10. Citation and Grounding

The system is designed to connect generated answers to evidence retrieved from the research-paper corpus.

Citation-related evaluation checks whether generated responses identify the expected paper evidence and whether the cited paper IDs are correct.

The answer-generation prompt also explicitly prevents the model from inventing citation information.

This creates a grounding boundary: the system is expected to answer from the indexed research-paper corpus rather than relying on unsupported external knowledge.

---

## 11. Evaluation

The project includes a fixed evaluation set of **20 research questions**.

The evaluation pipeline measures:

* Citation correctness
* Insufficient-evidence behavior
* Response latency
* Expected paper IDs
* Cited paper IDs
* Citation counts

### Final Evaluation Results

The final 20-question evaluation produced:

| Metric                          |          Result |
| ------------------------------- | --------------: |
| Total questions                 |              20 |
| Citation-correct responses      |         20 / 20 |
| Citation correctness            |            100% |
| Insufficient-evidence responses |          0 / 20 |
| Average response latency        | 161.763 seconds |
| Total evaluation runtime        | 3235.32 seconds |
| Total runtime                   |   ~53.9 minutes |

The evaluation results are stored in:

```text
evaluation/answer_evaluation.csv
```

### Important Interpretation

The 100% citation-correctness result applies to this specific **20-question evaluation set**. It should not be interpreted as proof that the system will achieve the same performance on arbitrary questions.

The relatively high latency is primarily associated with local LLM answer generation and should be considered when evaluating the system for interactive use.

---

## 12. Evaluation Case Studies

### Q001 — Mathematical Formula Extraction

The Transformer paper contains the Scaled Dot-Product Attention formula.

Normal PDF extraction initially separated portions of the mathematical expression across lines.

A targeted extraction fallback was implemented rather than enabling layout extraction globally.

After rebuilding the processed corpus, the formula was correctly present in the relevant chunks.

The final generated response preserved the mathematical structure:

```text
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

The evaluation result for Q001 identified the expected paper and passed citation correctness.

### Q013 — LoRA Inference Latency Verification

The original generated response incorrectly generalized LoRA as providing an inference-latency reduction.

The source paper was manually verified.

The paper states that LoRA can reduce GPU memory requirements and that it introduces **no additional inference latency** when the adapted weight matrix is explicitly computed and stored.

The evaluation answer was therefore corrected to reflect the source evidence:

```text
LoRA provides storage/memory and inference-latency advantages compared to traditional full model fine-tuning. For GPT-3 175B fine-tuned with Adam, LoRA reduces the GPU memory requirement by 3 times. For inference, LoRA introduces no additional inference latency because the adapted weight matrix can be explicitly computed and stored as W = W0 + BA, after which inference is performed as usual.
```

This case demonstrates the importance of verifying generated answers against the original research-paper evidence.

---

## 13. Experimental Components

The repository also contains experimental components used during development and evaluation.

These include:

* Alternative embedding models.
* Different chunk sizes and overlaps.
* Cross-encoder reranking experiments.
* Retrieval experiments.
* Embedding experiments.
* Chunking experiments.
* Evaluation scripts.

The **production pipeline does not use cross-encoder reranking**.

The reranker-related code is retained for experimentation and future comparison.

---

## 14. Technology Stack

### Programming Language

* Python 3.11+

### User Interface

* Streamlit

### Retrieval

* Sentence Transformers
* BM25
* Reciprocal Rank Fusion

### Embeddings

```text
sentence-transformers/all-mpnet-base-v2
```

### Vector Database

* ChromaDB

### LLM

```text
llama3.2:3b
```

### LLM Runtime

* Ollama

### LLM Framework

* LangChain
* LangChain Ollama integration

### PDF Processing

* pypdf

### Evaluation

* Python-based evaluation scripts
* CSV result storage
* Predefined evaluation questions

### Development

* Git
* GitHub
* VS Code
* PowerShell

---

## 15. Project Structure

```text
research-paper-answer-bot/
│
├── data/
│   ├── raw_papers/
│   │   └── *.pdf
│   │
│   └── processed/
│       └── chroma_db/
│
├── docs/
│
├── evaluation/
│   ├── questions.json
│   └── answer_evaluation.csv
│
├── experiments/
│   ├── chunking_results.csv
│   ├── embedding_results.csv
│   ├── embedding_retrieval_results.csv
│   └── retrieval_results.csv
│
├── notebooks/
│
├── screenshots/
│
├── src/
│   ├── bm25_retriever.py
│   ├── chunking.py
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── evaluate_rag_answers.py
│   ├── llm.py
│   ├── prompts.py
│   └── rag_pipeline.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── AGENTS.md
├── app.py
├── create_final_results.py
├── README.md
└── requirements.txt
```

Generated files such as Python cache directories, the virtual environment, ChromaDB persistence data, and raw research-paper PDFs are intentionally excluded from version control where appropriate.

---

## 16. Requirements

Before running the project, install:

* Python 3.11 or newer
* Git
* Ollama
* Llama 3.2 3B model
* Internet access for initial dependency and model downloads

The project was developed and tested using Python 3.11.

---

## 17. Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd research-paper-answer-bot
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the Environment File

```powershell
Copy-Item .env.example .env
```

The environment example contains the project's current configuration values.

---

## 18. Ollama Setup

The production application uses a local Llama 3.2 3B model through Ollama.

Install Ollama and then pull the model:

```powershell
ollama pull llama3.2:3b
```

Verify that the model is available:

```powershell
ollama list
```

The model should appear in the available Ollama models.

The application uses:

```text
LLM_MODEL=llama3.2:3b
```

---

## 19. Research Paper Data

The project uses a curated collection of **12 research papers** containing a total of **259 pages**.

The raw PDF files are stored locally under:

```text
data/raw_papers/
```

The raw research papers are intentionally excluded from Git because of repository size and source-distribution considerations.

To reproduce the complete indexed corpus, place the required PDFs in the expected `data/raw_papers/` directory before running the document-processing/indexing workflow.

The generated ChromaDB persistence directory is stored under:

```text
data/processed/chroma_db/
```

ChromaDB data is also excluded from Git because it is generated data.

---

## 20. Running the Application

After completing the environment setup and ensuring the required processed data and Ollama model are available, start the Streamlit application:

```powershell
streamlit run app.py
```

Streamlit will display the local application URL in the terminal.

The application provides an interface for submitting research-paper questions and viewing generated answers based on retrieved evidence.

---

## 21. Running Tests

Run the project's test suite with:

```powershell
pytest tests/
```

To verify that the source files compile successfully:

```powershell
python -m compileall -q .\src
```

A successful compile command produces no output.

---

## 22. Running the Full Evaluation

The complete 20-question evaluation can be executed with:

```powershell
python .\src\evaluate_rag_answers.py
```

The resulting evaluation file is:

```text
evaluation/answer_evaluation.csv
```

The full evaluation invokes the local LLM for each question and can therefore take a significant amount of time.

---

## 23. Running an Individual Evaluation Question

For targeted testing, an individual evaluation question can be run:

```powershell
python .\src\evaluate_rag_answers.py --question-id Q001
```

This is useful when validating:

* Prompt changes
* Mathematical extraction fixes
* Retrieval changes
* Citation behavior
* Individual answer quality

Running individual questions is generally faster than rerunning the complete evaluation.

---

## 24. Reproducibility

The project aims to keep the main pipeline reproducible through explicit configuration and version-controlled source code.

Important production parameters are documented in:

```text
.env.example
src/rag_pipeline.py
src/document_loader.py
src/prompts.py
src/llm.py
```

The production retrieval configuration is:

```text
Chunk size:       1000
Chunk overlap:    150
Embedding:        all-mpnet-base-v2
Sparse retrieval: BM25
Fusion:           RRF
RRF k:            60
Candidates:       10
Final context:    3
LLM:              llama3.2:3b
Temperature:      0.0
```

The generated ChromaDB index is not committed to Git, so reproducibility requires rebuilding the index from the source research papers when necessary.

---

## 25. Limitations

### 1. Limited Corpus

The system is currently bounded by the 12 research papers included in the project corpus.

Questions requiring information outside these papers may not be answerable.

### 2. PDF Extraction

PDF text extraction can be imperfect, particularly for:

* Mathematical formulas
* Tables
* Multi-column layouts
* Special typography
* Rotated text
* Figures containing text

A targeted mathematical extraction fallback has been implemented, but it does not eliminate every possible PDF extraction issue.

### 3. Local LLM Latency

The application uses a local 3B-parameter model through Ollama.

Answer generation can therefore be relatively slow, especially without GPU acceleration.

The final 20-question evaluation averaged approximately **161.763 seconds per response**.

### 4. Small Evaluation Set

The current evaluation contains only 20 questions.

A larger and more diverse benchmark would provide stronger evidence about system performance.

### 5. No Production Cross-Encoder Reranking

Although reranking components exist for experimentation, the production retrieval pipeline currently does not use a cross-encoder reranker.

### 6. Corpus Updates Require Reprocessing

Adding or changing research papers requires the document-processing and indexing workflow to be rerun so that the retrieval index reflects the updated corpus.

---

## 26. Future Improvements

Potential future improvements include:

* Expand the evaluation dataset beyond 20 questions.
* Add more research papers.
* Improve mathematical-expression extraction.
* Improve table and figure extraction.
* Add multimodal document processing.
* Introduce hierarchical retrieval.
* Experiment with query decomposition.
* Evaluate HyDE-based retrieval.
* Add citation verification.
* Add stronger groundedness evaluation.
* Add answer faithfulness metrics.
* Investigate GPU acceleration for embeddings and LLM inference.
* Evaluate larger local language models.
* Compare different embedding models systematically.
* Compare different chunking strategies.
* Evaluate production cross-encoder reranking.
* Support user-uploaded research papers.
* Explore scalable vector-database deployments.
* Improve retrieval observability and debugging tools.

---

## 27. Project Status

The current implementation includes:

* [x] Research-paper PDF ingestion
* [x] Conservative PDF text extraction
* [x] Targeted mathematical extraction fallback
* [x] Document chunking
* [x] Dense embeddings
* [x] BM25 retrieval
* [x] Hybrid retrieval
* [x] Reciprocal Rank Fusion
* [x] Local Llama 3.2 3B generation
* [x] Grounded prompting
* [x] Citation evaluation
* [x] Insufficient-evidence evaluation
* [x] 20-question evaluation dataset
* [x] Streamlit interface
* [x] Reproducible environment configuration
* [x] Git version control

---

## 28. License

This project is intended as an academic/capstone project.

If this repository is published publicly, add the appropriate license and ensure that redistribution of the research-paper PDFs complies with the terms of their respective sources.

---

## 29. Acknowledgements

This project builds upon open-source technologies and research including:

* Retrieval-Augmented Generation
* Transformer architectures
* Sentence Transformers
* BM25 information retrieval
* ChromaDB
* LangChain
* Ollama
* Llama
* Streamlit

The research papers used by the system remain the intellectual property of their respective authors and publishers.
