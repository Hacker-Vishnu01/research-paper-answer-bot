# Research Paper Answer Bot - Methodology

## 1. Dataset Selection

The Research Paper Answer Bot uses a corpus of 12 selected research papers covering core topics in modern information retrieval, natural language processing, and large language models.

The corpus contains 259 pages in total and includes papers covering:

* Retrieval-Augmented Generation (RAG)
* Transformers and attention
* BERT and sentence embeddings
* Dense passage retrieval
* Contrastive representation learning
* Parameter-efficient fine-tuning
* Prompting and reasoning
* Retrieval evaluation and analysis

The papers are stored locally in `data/raw_papers/`. Paper metadata is maintained in `data/metadata.csv`, including:

* Paper ID
* Title
* Authors
* Publication year
* Topic
* Source URL
* PDF URL
* Local filename

The system uses the metadata catalog as the source of document provenance and does not fabricate citation metadata.

---

## 2. Text Extraction and Preprocessing

PDF documents are loaded using `pypdf`.

Text is extracted on a page-by-page basis rather than treating each PDF as a single document. This preserves the original page boundaries and allows retrieved passages to be associated with their source page.

Each extracted page retains:

* Paper ID
* Paper title
* Authors
* Publication year
* Topic
* Source filename
* 1-indexed page number
* Source URL
* PDF URL
* Extracted text

A conservative preprocessing stage is applied before chunking. The cleaning process:

* Removes non-printable control characters while preserving newlines and tabs.
* Rejoins simple words split by line-ending hyphenation.
* Normalizes Windows and Mac line endings.
* Collapses repeated horizontal whitespace.
* Reduces excessive consecutive blank lines.
* Preserves mathematical symbols, equations, headings, and paragraph structure.

The preprocessing stage does not intentionally remove document headers or footers.

---

## 3. Chunking Strategy

The cleaned page text is divided into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

The splitter uses the following separator hierarchy:

1. Paragraph breaks
2. Line breaks
3. Sentence boundaries
4. Spaces
5. Character-level splitting as a final fallback

Character count is used as the chunk length function.

Chunking is performed independently for each page so that every chunk retains its source page information.

### 3.1 Chunking Experiments

Four chunk-size and overlap configurations were evaluated:

| Configuration | Chunk Size | Overlap | Total Chunks |
| ------------- | ---------: | ------: | -----------: |
| Config A      |        500 |      50 |        2,182 |
| Config B      |        750 |     100 |        1,543 |
| Config C      |      1,000 |     150 |        1,172 |
| Config D      |      1,500 |     200 |          804 |

The chunking experiment records:

* Total number of chunks
* Average chunk length
* Minimum chunk length
* Maximum chunk length
* Average chunks per paper
* Average chunks per page
* Metadata preservation
* Chunking processing time

The experiment does not directly measure information loss. Downstream retrieval quality is evaluated separately using the retrieval benchmark.

### 3.2 Selected Production Configuration

Config C was selected for the production pipeline:

* Chunk size: **1,000 characters**
* Chunk overlap: **150 characters**
* Total chunks: **1,172**

This configuration provides a balance between chunk granularity and context size while maintaining complete provenance metadata.

Each chunk stores:

* `chunk_id`
* `paper_id`
* `paper_title`
* `authors`
* `year`
* `topic`
* `source_filename`
* `source_url`
* `pdf_url`
* `page_number`
* `chunk_size`
* `chunk_overlap`

---

## 4. Embedding Strategy

Three embedding models were supported for experimentation:

1. `sentence-transformers/all-mpnet-base-v2`
2. `BAAI/bge-base-en-v1.5`
3. `text-embedding-3-small`

The Hugging Face models are configured to use local cached files and CPU execution. Their embeddings are normalized before use.

### 4.1 Embedding Experiment Results

The MPNet model successfully generated embeddings for all 1,172 production chunks.

| Model                                     | Dimension | Chunks | Status                            |
| ----------------------------------------- | --------: | -----: | --------------------------------- |
| `sentence-transformers/all-mpnet-base-v2` |       768 |  1,172 | Successfully executed             |
| `BAAI/bge-base-en-v1.5`                   |       768 |      0 | Could not be loaded               |
| `text-embedding-3-small`                  |     1,536 |  1,172 | Not executed; API key unavailable |

The production embedding model is:

`sentence-transformers/all-mpnet-base-v2`

It was selected because it was successfully executed in the project environment and generated embeddings for the complete production corpus.

The experiment does not claim that MPNet outperformed BGE or OpenAI embeddings because the alternative models were not successfully benchmarked under the same experimental conditions.

---

## 5. Vector Store

The production MPNet embeddings are stored in a persistent ChromaDB vector store.

The production collection is:

`research_papers_mpnet`

The vector store maintains the chunk content and provenance metadata required for semantic retrieval and source attribution.

---

## 6. Retrieval Strategies

The system evaluates multiple retrieval approaches:

### 6.1 Dense Semantic Retrieval

Dense retrieval uses MPNet embeddings stored in ChromaDB.

For a user question, the question is embedded using the same embedding model and ChromaDB performs semantic similarity search against the indexed document chunks.

This approach is intended to retrieve passages that are semantically related to the question even when the exact query terminology does not occur in the document.

### 6.2 BM25 Lexical Retrieval

BM25 provides sparse keyword-based retrieval using `rank_bm25.BM25Okapi`.

Document chunks are tokenized using a regular expression that preserves:

* Words
* Numbers
* Underscores
* Periods
* Slashes
* Plus and minus signs

This allows technical terminology, acronyms, identifiers, and other exact lexical patterns to contribute to retrieval.

### 6.3 Hybrid Retrieval

The production system combines dense semantic retrieval and BM25 lexical retrieval using Reciprocal Rank Fusion (RRF).

For each query:

1. Dense retrieval produces candidate chunks.
2. BM25 produces candidate chunks.
3. Candidate rankings are converted into reciprocal-rank scores.
4. Dense and BM25 scores are combined.
5. Results are sorted by the fused score.
6. The top 10 chunks are returned.

The production configuration uses:

* Dense weight (`alpha`): **0.5**
* BM25 weight: **0.5**
* RRF constant (`k`): **60**
* Final hybrid results: **Top 10**

The equal weighting allows semantic similarity and exact lexical matching to contribute equally.

### 6.4 Cross-Encoder Reranking

A cross-encoder reranker was evaluated as an experimental retrieval strategy.

It is not part of the production answer-generation pipeline.

The retrieval benchmark showed that both Hybrid and Reranked retrieval achieved an MRR of 1.0 on the evaluation set, while reranking introduced substantially higher retrieval latency. Therefore, the production pipeline uses hybrid retrieval without cross-encoder reranking.

MMR is not used in the verified production implementation.

---

## 7. Retrieval Evaluation

Retrieval quality is evaluated using a curated set of 20 questions with expected relevant paper identifiers.

The evaluation reports:

* Hit@3
* Hit@5
* Hit@10
* Mean Reciprocal Rank (MRR)
* Precision@3
* Recall@3
* Average retrieval latency

The measured retrieval results were:

| Retriever   | Hit@3 | Hit@5 | Hit@10 |   MRR | Precision@3 | Recall@3 | Avg. Latency |
| ----------- | ----: | ----: | -----: | ----: | ----------: | -------: | -----------: |
| Dense MPNet |  1.00 |  1.00 |   1.00 | 0.925 |       0.333 |     1.00 |     0.1498 s |
| BM25        |  1.00 |  1.00 |   1.00 | 0.975 |       0.333 |     1.00 |     0.0125 s |
| Hybrid      |  1.00 |  1.00 |   1.00 | 1.000 |       0.333 |     1.00 |     0.0750 s |
| Reranked    |  1.00 |  1.00 |   1.00 | 1.000 |       0.333 |     1.00 |     0.9231 s |

Hybrid retrieval was selected for production because it achieved perfect Hit@3, Hit@5, Hit@10, and MRR on the evaluation set while maintaining substantially lower latency than the reranked configuration.

---

## 8. Retrieval-to-Generation Pipeline

The production answer pipeline uses the following configuration:

1. Load the 12 research papers.
2. Extract text page by page.
3. Apply conservative text preprocessing.
4. Generate 1,172 chunks using 1,000-character chunks with 150-character overlap.
5. Generate 768-dimensional MPNet embeddings.
6. Store embeddings in ChromaDB.
7. Build a BM25 lexical index over the chunks.
8. Retrieve candidates using dense MPNet retrieval and BM25.
9. Fuse the rankings using RRF with `alpha=0.5` and `k=60`.
10. Select the top 10 hybrid retrieval results.
11. Use the top 9 results as the generation context.
12. Generate the final answer using the local Llama 3.2 3B model through Ollama.

The production generation configuration uses:

* Model: `llama3.2:3b`
* Runtime: Ollama
* Temperature: `0.0`
* Retrieval top-k: `10`
* Generation context top-k: `9`

---

## 9. Grounded Answer Generation

The generation prompt instructs the language model to answer using only the retrieved research passages.

The system requires that:

* Factual claims must be supported by the supplied passages.
* The model must not introduce unsupported outside knowledge.
* Multiple retrieved passages may be synthesized when their relationship is directly supported by the evidence.
* Supported parts of partially answerable questions should be answered.
* The system should identify unsupported portions rather than inventing information.
* If the research corpus genuinely does not contain sufficient evidence, the system returns:

> The provided research papers do not contain sufficient evidence to answer this question.

This design is intended to reduce unsupported generation and maintain traceability between answers and source passages.

---

## 10. Citation and Provenance

Each retrieved chunk retains its paper and page provenance throughout the pipeline.

Generated answers can be associated with up to three source citations based on the final context passages.

Citation metadata is derived from the indexed chunk metadata rather than generated independently by the language model.

The evaluation checks whether the cited paper identifiers correspond to the expected relevant papers for each evaluation question.

---

## 11. End-to-End Answer Evaluation

The complete RAG pipeline was evaluated using 20 curated questions.

The evaluation measures:

* Faithfulness
* Answer relevance
* Semantic correctness
* Citation correctness
* Insufficient-evidence behavior
* End-to-end response latency

The final evaluation results were:

| Metric                          |            Result |
| ------------------------------- | ----------------: |
| Faithfulness                    |  1.90 / 2 (95.0%) |
| Answer Relevance                | 2.00 / 2 (100.0%) |
| Semantic Correctness            |  1.75 / 2 (87.5%) |
| Citation Correctness            |             95.0% |
| Insufficient-Evidence Responses |     1 / 20 (5.0%) |
| Average Response Latency        |          180.97 s |

The evaluation indicates strong retrieval and answer relevance performance, while semantic correctness and generation latency remain areas for future improvement.

---

## 12. Reproducibility

The project stores experiment outputs and evaluation results in structured CSV files.

Important experiment artifacts include:

* `experiments/chunking_results.csv`
* `experiments/embedding_results.csv`
* `experiments/retrieval_results.csv`
* `evaluation/retrieval_results.csv`
* `evaluation/bm25_retrieval_results.csv`
* `evaluation/hybrid_retrieval_results.csv`
* `evaluation/reranked_retrieval_results.csv`
* `evaluation/answer_evaluation.csv`
* `evaluation/final_results.csv`

The production configuration is therefore reproducible from the stored corpus, metadata catalog, source code, experiment configurations, and evaluation results.
