# Research Paper Answer Bot - Evaluation Framework

## 1. Evaluation Overview

The Research Paper Answer Bot is evaluated at two levels:

1. **Retrieval evaluation** - measures whether relevant research-paper chunks are retrieved and how highly they are ranked.
2. **End-to-end answer evaluation** - measures the quality, grounding, citation correctness, and latency of generated answers.

The evaluation dataset contains **20 curated research questions** covering the 12-paper research corpus.

The questions are stored in:

```text
evaluation/questions.json
```

The end-to-end results are stored in:

```text
evaluation/answer_evaluation.csv
evaluation/final_results.csv
```

Retrieval benchmark results are stored in:

```text
evaluation/retrieval_results.csv
evaluation/bm25_retrieval_results.csv
evaluation/hybrid_retrieval_results.csv
evaluation/reranked_retrieval_results.csv
```

---

## 2. Retrieval Metrics

Retrieval performance is measured using standard information-retrieval metrics.

### 2.1 Hit@K

Hit@K measures whether at least one relevant ground-truth item appears in the top-K retrieved results.

$$
Hit@K =
\begin{cases}
1 & \text{if any relevant item appears in top-}K \\
0 & \text{otherwise}
\end{cases}
$$

The reported benchmark uses:

* Hit@3
* Hit@5
* Hit@10

---

### 2.2 Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result is ranked.

$$
MRR = \frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\frac{1}{rank_i}
$$

where $rank_i$ is the position of the first relevant retrieved item for query $i$.

A value of 1.0 indicates that the first retrieved result is relevant for every query.

---

### 2.3 Precision@K

Precision@K measures the fraction of the top-K retrieved items that are relevant.

$$
Precision@K =
\frac{|\text{relevant items in top-}K|}{K}
$$

---

### 2.4 Recall@K

Recall@K measures the proportion of all ground-truth relevant items that appear in the top-K results.

$$
Recall@K =
\frac{|\text{relevant items in top-}K|}
{|\text{total relevant items}|}
$$

The implementation returns 0.0 when no ground-truth items are defined.

---

## 3. Retrieval Benchmark

The retrieval benchmark compares four retrieval configurations:

* Dense MPNet
* BM25
* Hybrid RRF
* Cross-encoder reranking

The production retrieval strategy is **Hybrid RRF**. Cross-encoder reranking is evaluated as an experimental comparison and is not part of the production pipeline.

| Retriever              | Hit@3 | Hit@5 | Hit@10 |       MRR | Precision@3 | Recall@3 | Avg. Latency |
| ---------------------- | ----: | ----: | -----: | --------: | ----------: | -------: | -----------: |
| Dense MPNet            | 1.000 | 1.000 |  1.000 |     0.925 |       0.333 |    1.000 |     0.1498 s |
| BM25                   | 1.000 | 1.000 |  1.000 |     0.975 |       0.333 |    1.000 |     0.0125 s |
| Hybrid RRF             | 1.000 | 1.000 |  1.000 | **1.000** |       0.333 |    1.000 | **0.0750 s** |
| Cross-encoder Reranked | 1.000 | 1.000 |  1.000 | **1.000** |       0.333 |    1.000 |     0.9231 s |

### Retrieval interpretation

The Hybrid RRF retriever provides the best practical balance between ranking quality and latency.

* It achieves perfect Hit@3, Hit@5, and Hit@10.
* It achieves an MRR of 1.000.
* Its average retrieval latency is 0.075 seconds.
* BM25 is the fastest individual retriever at 0.0125 seconds.
* Dense MPNet has an MRR of 0.925 because some relevant papers are not ranked first.
* Cross-encoder reranking also achieves MRR 1.000, but its average latency of 0.9231 seconds is substantially higher than Hybrid RRF.
* Therefore, cross-encoder reranking is retained as an experimental comparison rather than the production retrieval strategy.

---

## 4. Retrieval Evaluation Dataset

The retrieval benchmark contains 20 questions associated with expected research-paper IDs.

Each question records:

* Question ID
* Question text
* Retriever type
* Retrieval top-K
* Expected paper ID(s)
* Top-1 paper ID
* Top-1 chunk ID
* Hit@3
* Hit@5
* Hit@10
* Precision@3
* Recall@3
* MRR
* Retrieval latency

The benchmark uses the same 20 research questions across the retrieval configurations to support direct comparison.

---

## 5. End-to-End Answer Evaluation

The complete RAG pipeline is evaluated using the 20-question benchmark.

For each question, the system:

1. Performs hybrid retrieval.
2. Retrieves the top 10 candidate chunks.
3. Uses the first 9 retrieved chunks as generation context.
4. Generates an answer using Llama 3.2 3B through Ollama.
5. Extracts up to three application-generated citations.
6. Records answer quality, citation information, insufficient-evidence status, and latency.

The production generation configuration uses:

* Hybrid RRF retrieval
* Alpha = 0.5
* RRF constant $k = 60$
* Retrieval top-K = 10
* Context top-K = 9
* Llama 3.2 3B
* Temperature = 0.0
* No cross-encoder reranker in the production pipeline

---

## 6. Answer Quality Metrics

### 6.1 Faithfulness / Groundedness

Faithfulness measures whether the generated answer is supported by the retrieved research passages.

The manual scoring scale is:

* **2** - fully supported by the retrieved evidence
* **1** - partially supported or contains a minor grounding limitation
* **0** - unsupported or substantially inconsistent with the evidence

The final average faithfulness score was:

**1.90 / 2.00 (95.0%)**

---

### 6.2 Answer Relevance

Answer relevance measures whether the generated response directly addresses the question.

The manual scoring scale is:

* **2** - directly answers the question
* **1** - partially answers the question
* **0** - does not adequately answer the question

The final average answer relevance score was:

**2.00 / 2.00 (100.0%)**

---

### 6.3 Semantic Correctness

Semantic correctness measures whether the generated answer accurately represents the meaning of the underlying research evidence.

The manual scoring scale is:

* **2** - semantically correct
* **1** - partially correct
* **0** - incorrect

The final average semantic correctness score was:

**1.75 / 2.00 (87.5%)**

---

### 6.4 Citation Correctness

Citation correctness checks whether the paper identified by the generated citation matches the expected paper ID for the evaluation question.

The final results were:

* Applicable questions: **19**
* Correct citation responses: **19**
* Citation correctness: **100% of applicable responses**
* Overall citation-correct response rate across all 20 questions: **95%**

Question Q015 was marked `not_applicable` because the system intentionally returned an insufficient-evidence response and therefore did not produce a citation.

---

### 6.5 Insufficient-Evidence Handling

The system is instructed not to fabricate information when the retrieved research passages do not provide sufficient evidence.

The exact fallback response is:

```text
The provided research papers do not contain sufficient evidence to answer this question.
```

In the 20-question evaluation:

* Insufficient-evidence responses: **1**
* Rate: **5.0%**

Q015 triggered this behavior. The generated response correctly identified that the retrieved evidence supported the individual roles of Double Quantization and Paged Optimizers but did not sufficiently establish the complete interaction requested by the question.

This demonstrates the system's ability to abstain rather than invent unsupported evidence.

---

## 7. End-to-End Evaluation Results

The final manually reviewed evaluation produced the following results:

| Metric                             |            Result |
| ---------------------------------- | ----------------: |
| Number of questions                |                20 |
| Faithfulness                       |       1.90 / 2.00 |
| Faithfulness percentage            |             95.0% |
| Answer relevance                   |       2.00 / 2.00 |
| Answer relevance percentage        |            100.0% |
| Semantic correctness               |       1.75 / 2.00 |
| Semantic correctness percentage    |             87.5% |
| Citation correctness               |  19/19 applicable |
| Citation correctness percentage    | 100.0% applicable |
| Citation-correct responses overall |             19/20 |
| Insufficient-evidence responses    |              1/20 |
| Insufficient-evidence rate         |              5.0% |
| Average end-to-end latency         |          180.97 s |

---

## 8. Latency Evaluation

Latency is measured during retrieval and end-to-end answer generation.

### Retrieval latency

| Retriever              | Average latency |
| ---------------------- | --------------: |
| Dense MPNet            |        0.1498 s |
| BM25                   |        0.0125 s |
| Hybrid RRF             |        0.0750 s |
| Cross-encoder Reranked |        0.9231 s |

### End-to-end latency

The complete RAG pipeline has an average response latency of approximately:

**180.97 seconds per question**

This includes retrieval, context preparation, local LLM generation, and answer processing.

The relatively high end-to-end latency is primarily associated with local LLM generation using Llama 3.2 3B on the development environment rather than retrieval itself.

---

## 9. Dataset Split Information

The answer evaluation CSV records each question as either `dev` or `test`.

The current 20-question evaluation therefore contains both development and test questions. The `dev` questions were available during iterative development and prompt/system refinement, while the `test` questions provide a separate subset for checking generalization.

Because development questions were used during system refinement, the reported 20-question results should be interpreted as a **curated project benchmark**, not as a fully blind independent test-set evaluation.

---

## 10. Evaluation Limitations

The evaluation has several limitations:

1. The benchmark contains only 20 questions.
2. The question set covers a limited 12-paper research corpus.
3. The answer-quality metrics include manual scoring and therefore involve evaluator judgment.
4. Citation correctness is evaluated primarily using expected paper IDs in the current automated evaluation rather than a fully independent verification of every cited passage.
5. End-to-end latency depends strongly on the local hardware and Ollama configuration.
6. The BGE embedding model was not successfully evaluated because the model could not be loaded in the development environment.
7. OpenAI `text-embedding-3-small` was not evaluated because an API key was unavailable.
8. Cross-encoder reranking was evaluated as an experimental retrieval comparison but is not used in the production pipeline.
9. The retrieval benchmark uses ground-truth paper/chunk associations prepared for the project and therefore may not represent arbitrary real-world queries.
10. The current benchmark does not establish statistical significance across a large evaluation population.

---

## 11. Reproducibility

The following files contain the evaluation implementation and results:

```text
src/evaluation.py
src/evaluate_rag_answers.py
evaluation/questions.json
evaluation/retrieval_results.csv
evaluation/bm25_retrieval_results.csv
evaluation/hybrid_retrieval_results.csv
evaluation/reranked_retrieval_results.csv
evaluation/answer_evaluation.csv
evaluation/final_results.csv
```

The evaluation can be reproduced using the configured Python virtual environment, the local ChromaDB vector store, the MPNet embedding model, and the Ollama Llama 3.2 3B model.

---

## 12. Summary of Evaluation Findings

The evaluation shows that the Hybrid RRF retrieval strategy provides strong retrieval performance across the 20-question benchmark, achieving perfect Hit@3, Hit@5, Hit@10, and MRR while maintaining substantially lower latency than cross-encoder reranking.

The end-to-end system also demonstrates strong grounded-answer behavior, with 95.0% faithfulness, 100.0% answer relevance, and 87.5% semantic correctness under the project's manual scoring criteria.

The system produced an explicit insufficient-evidence response for one question instead of fabricating unsupported information. This behavior is consistent with the grounded-generation design.

The main practical limitation is end-to-end response latency, which is approximately 181 seconds per question in the local development environment. Retrieval itself is comparatively fast, with Hybrid RRF averaging approximately 0.075 seconds.
