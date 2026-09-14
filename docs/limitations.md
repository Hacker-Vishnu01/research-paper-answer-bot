# Research Paper Answer Bot - Limitations & Boundary Analysis

## 1. Domain Coverage

- The bot's knowledge is bounded by the 12 research papers indexed in the system.
- If a query refers to information outside the ingested corpus, the system is designed to report insufficient evidence rather than rely on outside knowledge.
- The evaluation dataset contains 20 curated questions, so the reported answer quality may not represent performance on all possible research questions.

---

## 2. Document Modality Limitations

- **Complex Tables**: Plain text extraction may flatten multi-row and multi-column tables, leading to loss of relational context.
- **Figures & Visual Diagrams**: Visual content and graphic schematics are not parsed by the plain-text PDF extraction pipeline.
- **Mathematical Formulations**: Complex multi-line mathematical expressions may experience character or formatting loss during text extraction.
- The current ingestion pipeline is primarily designed for text-based research content rather than multimodal document understanding.

---

## 3. Chunking Constraints

- The production configuration uses a chunk size of 1000 characters with 150-character overlap.
- Information spanning chunk boundaries may be partially separated despite the overlap.
- The chunking experiment compares chunk statistics and processing time, but does not by itself prove that Config C provides the best downstream retrieval accuracy.

---

## 4. Retrieval Constraints

- **Vocabulary Mismatch**: Dense retrieval may miss rare scientific identifiers, while BM25 may be less effective for paraphrased semantic queries.
- The production system mitigates these limitations through hybrid dense + BM25 retrieval using Reciprocal Rank Fusion (RRF), but retrieval errors can still occur.
- The retrieval benchmark achieved Hit@3, Hit@5, and Hit@10 of 1.0000 for all evaluated retrievers, but the benchmark is based on the available evaluation questions and therefore does not guarantee perfect retrieval for unseen queries.

---

## 5. LLM Generation Limitations

- The system uses a locally hosted Llama 3.2 3B model with a grounded prompt and temperature 0.0.
- The prompt restricts generation to the supplied research passages and provides an explicit insufficient-evidence response when supporting evidence is unavailable.
- These controls reduce unsupported generation but cannot guarantee that every generated statement will always be perfectly faithful to the retrieved evidence.
- The 20-question evaluation produced one insufficient-evidence response, showing that evidence-based answer generation can still be limited even when relevant information may be available in the retrieved context.

---

## 6. Compute & Latency Trade-offs

- End-to-end answer generation is the major latency component of the current system.
- The 20-question evaluation recorded an average response latency of approximately 180.97 seconds per question.
- Cross-encoder reranking is implemented as an experimental retrieval configuration and is not used in the production pipeline.
- In the retrieval benchmark, hybrid retrieval achieved MRR 1.0000 with an average latency of 0.0750 seconds, while reranked retrieval also achieved MRR 1.0000 but required 0.9231 seconds on average.
- Therefore, reranking maintained the measured ranking quality but introduced substantial additional retrieval latency.

---

## 7. Embedding Model Constraints

- The production system uses `sentence-transformers/all-mpnet-base-v2` with 768-dimensional embeddings.
- MPNet was selected because it successfully generated embeddings for all 1172 production chunks.
- `BAAI/bge-base-en-v1.5` could not be loaded in the experiment environment.
- `text-embedding-3-small` was not executed because an API key was unavailable.
- Therefore, the embedding experiment does not constitute a complete performance comparison across all candidate embedding models.

---

## 8. Evaluation Limitations

- The answer evaluation uses 20 curated questions and is therefore limited in breadth.
- Citation correctness primarily verifies whether the cited paper ID matches the expected paper ID; it does not constitute full verification of every cited statement against the original page text.
- Retrieval evaluation uses the available ground-truth paper mappings and should not be interpreted as a guarantee of performance on unseen datasets.
- Manual answer scoring is based on faithfulness, relevance, and semantic correctness and may contain evaluator subjectivity.

---

## 9. Reproducibility & Environment Constraints

- Embedding and LLM processing times depend on the local hardware and software environment.
- The current production generation setup uses a local Ollama-hosted Llama 3.2 3B model, so latency and answer quality may differ on other hardware or model configurations.
- External embedding APIs were not included in the production benchmark because the required API key was unavailable during experimentation.

---

## 10. Boundary of the Current System

The current system is intended as a research-paper question-answering prototype rather than a general-purpose search engine or unrestricted knowledge assistant. Its answers should be interpreted within the boundaries of the ingested corpus, extraction pipeline, retrieval benchmark, generation model, and evaluation methodology described in the project documentation.
