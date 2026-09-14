# Research Paper Answer Bot - Dataset Documentation

## 1. Dataset Overview

The Research Paper Answer Bot uses a curated corpus of **12 research papers** covering Natural Language Processing, Machine Learning, dense retrieval, parameter-efficient fine-tuning, reasoning, and Retrieval-Augmented Generation (RAG).

The original papers are stored as PDF files in:

```text
data/raw_papers/
```

Bibliographic and source metadata are maintained in:

```text
data/metadata.csv
```

The document-processing pipeline extracts text **page by page** so that retrieved passages retain their original document page number. This page-level provenance supports traceability and application-generated citations.

The corpus contains **12 papers spanning 259 pages**.

---

## 2. Number of Papers

* **Total research papers:** 12
* **Total pages:** 259
* **Corpus format:** Original PDF documents with page-aware text extraction
* **Metadata catalog:** `data/metadata.csv`
* **PDF storage:** `data/raw_papers/`

The corpus was selected to provide coverage of the main concepts required by the Research Paper Answer Bot, including transformers, language models, dense retrieval, embeddings, efficient fine-tuning, reasoning, and RAG.

---

## 3. Topics and Thematic Categories

The 12 papers are organized into the following thematic categories:

| Topic / Category                         | Count | Representative Papers                   |
| ---------------------------------------- | ----: | --------------------------------------- |
| **Transformers & Language Models**       |     2 | Attention Is All You Need, BERT         |
| **Retrieval-Augmented Generation (RAG)** |     1 | Retrieval-Augmented Generation          |
| **Embeddings & Dense Representation**    |     2 | Sentence-BERT, Contriever               |
| **Dense Retrieval for QA**               |     1 | Dense Passage Retrieval                 |
| **Efficient LLMs & Fine-Tuning**         |     2 | LoRA, QLoRA                             |
| **Prompt Engineering & Reasoning**       |     1 | Chain-of-Thought                        |
| **Advanced RAG & Self-Reflection**       |     1 | Self-RAG                                |
| **Context Limitations & Long Contexts**  |     1 | Lost in the Middle                      |
| **RAG Literature Surveys**               |     1 | RAG for Large Language Models: A Survey |

---

## 4. Paper Inventory

| ID       | Title                                                                            | Authors                                                                                                                                                                                    | Year | Topic                             |
| -------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---: | --------------------------------- |
| **P001** | Attention Is All You Need                                                        | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin                                                                   | 2017 | Transformers                      |
| **P002** | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                                                                                                                               | 2018 | BERT / Language Models            |
| **P003** | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks                 | Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tomáš Rocktäschel, Sebastian Riedel, Douwe Kiela | 2020 | Retrieval-Augmented Generation    |
| **P004** | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks                   | Nils Reimers, Iryna Gurevych                                                                                                                                                               | 2019 | Sentence Embeddings               |
| **P005** | Dense Passage Retrieval for Open-Domain Question Answering                       | Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih                                                                               | 2020 | Dense Retrieval                   |
| **P006** | Unsupervised Dense Information Retrieval with Contrastive Learning               | Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave                                                                          | 2021 | Unsupervised Dense Retrieval      |
| **P007** | LoRA: Low-Rank Adaptation of Large Language Models                               | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen                                                                                  | 2021 | Efficient LLMs / PEFT             |
| **P008** | QLoRA: Efficient Finetuning of Quantized LLMs                                    | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer                                                                                                                             | 2023 | Efficient LLMs / Quantization     |
| **P009** | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models            | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou                                                                              | 2022 | Prompt Engineering & Reasoning    |
| **P010** | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection   | Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi                                                                                                                        | 2023 | Advanced RAG / Self-Reflection    |
| **P011** | Lost in the Middle: How Language Models Use Long Contexts                        | Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang                                                                                    | 2023 | Context Windows / RAG Limitations |
| **P012** | Retrieval-Augmented Generation for Large Language Models: A Survey               | Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang                                                                           | 2023 | RAG Survey                        |

---

## 5. Source Information

The papers were collected from arXiv and are referenced through their corresponding arXiv abstract and PDF URLs.

| ID       | arXiv Identifier | Source Abstract URL              | Official PDF URL                     | Local Filename                            |
| -------- | ---------------- | -------------------------------- | ------------------------------------ | ----------------------------------------- |
| **P001** | `1706.03762`     | https://arxiv.org/abs/1706.03762 | https://arxiv.org/pdf/1706.03762.pdf | `P001_attention_is_all_you_need.pdf`      |
| **P002** | `1810.04805`     | https://arxiv.org/abs/1810.04805 | https://arxiv.org/pdf/1810.04805.pdf | `P002_bert.pdf`                           |
| **P003** | `2005.11401`     | https://arxiv.org/abs/2005.11401 | https://arxiv.org/pdf/2005.11401.pdf | `P003_retrieval_augmented_generation.pdf` |
| **P004** | `1908.10084`     | https://arxiv.org/abs/1908.10084 | https://arxiv.org/pdf/1908.10084.pdf | `P004_sentence_bert.pdf`                  |
| **P005** | `2004.04906`     | https://arxiv.org/abs/2004.04906 | https://arxiv.org/pdf/2004.04906.pdf | `P005_dense_passage_retrieval.pdf`        |
| **P006** | `2112.09118`     | https://arxiv.org/abs/2112.09118 | https://arxiv.org/pdf/2112.09118.pdf | `P006_contriever.pdf`                     |
| **P007** | `2106.09685`     | https://arxiv.org/abs/2106.09685 | https://arxiv.org/pdf/2106.09685.pdf | `P007_lora.pdf`                           |
| **P008** | `2305.14314`     | https://arxiv.org/abs/2305.14314 | https://arxiv.org/pdf/2305.14314.pdf | `P008_qlora.pdf`                          |
| **P009** | `2201.11903`     | https://arxiv.org/abs/2201.11903 | https://arxiv.org/pdf/2201.11903.pdf | `P009_chain_of_thought.pdf`               |
| **P010** | `2310.11511`     | https://arxiv.org/abs/2310.11511 | https://arxiv.org/pdf/2310.11511.pdf | `P010_self_rag.pdf`                       |
| **P011** | `2307.03172`     | https://arxiv.org/abs/2307.03172 | https://arxiv.org/pdf/2307.03172.pdf | `P011_lost_in_the_middle.pdf`             |
| **P012** | `2312.10997`     | https://arxiv.org/abs/2312.10997 | https://arxiv.org/pdf/2312.10997.pdf | `P012_rag_survey.pdf`                     |

---

## 6. Page Counts per Paper

| ID       | Paper Filename                            | Page Count |
| -------- | ----------------------------------------- | ---------: |
| **P001** | `P001_attention_is_all_you_need.pdf`      |         15 |
| **P002** | `P002_bert.pdf`                           |         16 |
| **P003** | `P003_retrieval_augmented_generation.pdf` |         19 |
| **P004** | `P004_sentence_bert.pdf`                  |         11 |
| **P005** | `P005_dense_passage_retrieval.pdf`        |         13 |
| **P006** | `P006_contriever.pdf`                     |         21 |
| **P007** | `P007_lora.pdf`                           |         26 |
| **P008** | `P008_qlora.pdf`                          |         26 |
| **P009** | `P009_chain_of_thought.pdf`               |         43 |
| **P010** | `P010_self_rag.pdf`                       |         30 |
| **P011** | `P011_lost_in_the_middle.pdf`             |         18 |
| **P012** | `P012_rag_survey.pdf`                     |         21 |

**Total:** 259 pages.

---

## 7. Corpus Statistics

The corpus contains:

* **12 research papers**
* **259 total pages**
* **934,952 extracted characters**
* **21.6 average pages per paper**
* **77,912.7 average extracted characters per paper**

These statistics describe the source corpus before chunking and embedding.

---

## 8. Text Extraction

PDF text is extracted using the `pypdf` library.

The document loader processes each PDF page independently and preserves the original **1-indexed page number**.

The extraction metadata includes:

* `paper_id`
* `title`
* `authors`
* `year`
* `topic`
* `source_filename`
* `source_url`
* `pdf_url`
* `page_number`

The document loader itself performs text extraction and metadata assignment. Conservative text cleaning is performed subsequently by the chunking pipeline.

---

## 9. Text Preprocessing

The project applies conservative text cleaning before chunking.

The preprocessing stage:

* removes control characters while preserving newline and tab characters
* rejoins lowercase-letter hyphenation across line breaks
* normalizes line endings
* collapses unnecessary horizontal whitespace
* reduces excessive blank lines
* preserves mathematical symbols
* preserves headings and paragraph boundaries
* does not intentionally remove headers or footers

The objective is to improve chunk consistency while minimizing information loss from the original papers.

---

## 10. Chunked Dataset

After preprocessing, the production configuration uses:

* **Chunking method:** Recursive Character Text Splitter
* **Chunk size:** 1000 characters
* **Chunk overlap:** 150 characters
* **Total production chunks:** 1172

The production chunk metadata preserves the original document provenance, including:

| Metadata Field    | Purpose                   |
| ----------------- | ------------------------- |
| `chunk_id`        | Unique chunk identifier   |
| `paper_id`        | Source paper identifier   |
| `paper_title`     | Source paper title        |
| `authors`         | Paper authors             |
| `year`            | Publication/preprint year |
| `topic`           | Paper topic               |
| `source_filename` | Original PDF filename     |
| `source_url`      | Source abstract URL       |
| `pdf_url`         | Source PDF URL            |
| `page_number`     | Original page number      |
| `chunk_size`      | Configured chunk size     |
| `chunk_overlap`   | Configured overlap        |

The 1172 chunks form the production retrieval corpus stored in ChromaDB.

---

## 11. Chunking Experiment

Four chunk configurations were evaluated during development:

| Configuration | Chunk Size | Overlap | Total Chunks | Avg. Chunk Length | Avg. Chunks/Page |
| ------------- | ---------: | ------: | -----------: | ----------------: | ---------------: |
| Config A      |        500 |      50 |         2182 |             436.4 |             8.42 |
| Config B      |        750 |     100 |         1543 |             656.4 |             5.96 |
| **Config C**  |   **1000** | **150** |     **1172** |         **876.5** |         **4.53** |
| Config D      |       1500 |     200 |          804 |            1268.1 |             3.10 |

All four configurations preserved the required metadata.

Config C was selected as the production configuration based on the project design and resulting corpus characteristics.

The chunking experiment measured chunk statistics and processing behavior. It should not be interpreted as a standalone proof that Config C has the best possible retrieval accuracy.

Detailed chunking results are stored in:

```text
experiments/chunking_results.csv
```

---

## 12. Embedding Dataset

The production retrieval corpus is embedded using:

```text
sentence-transformers/all-mpnet-base-v2
```

Configuration:

* **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
* **Embedding dimension:** 768
* **Embedded chunks:** 1172
* **Vector store:** ChromaDB
* **Collection:** `research_papers_mpnet`

The embedding experiment also included BGE and OpenAI embedding configurations. MPNet was selected for production because it successfully generated embeddings for all 1172 production chunks in the available local environment.

The BGE model could not be loaded in the local environment, and the OpenAI embedding configuration was not executed because an API key was unavailable. Therefore, the experiment does **not** establish that MPNet is universally superior to those alternatives.

Embedding experiment results are stored in:

```text
experiments/embedding_results.csv
```

---

## 13. Dataset Storage Structure

The relevant dataset directories are:

```text
data/
├── metadata.csv
├── dataset_statistics.csv
├── raw_papers/
│   ├── P001_attention_is_all_you_need.pdf
│   ├── P002_bert.pdf
│   ├── P003_retrieval_augmented_generation.pdf
│   ├── P004_sentence_bert.pdf
│   ├── P005_dense_passage_retrieval.pdf
│   ├── P006_contriever.pdf
│   ├── P007_lora.pdf
│   ├── P008_qlora.pdf
│   ├── P009_chain_of_thought.pdf
│   ├── P010_self_rag.pdf
│   ├── P011_lost_in_the_middle.pdf
│   └── P012_rag_survey.pdf
└── processed/
    └── chroma_db/
```

The original PDFs remain under `data/raw_papers/`, while processed vector-store data is maintained under `data/processed/chroma_db/`.

---

## 14. Metadata Schema

The primary paper-level metadata catalog is:

```text
data/metadata.csv
```

| Column           | Type    | Description                             |
| ---------------- | ------- | --------------------------------------- |
| `paper_id`       | String  | Unique paper identifier (`P001`–`P012`) |
| `title`          | String  | Paper title                             |
| `authors`        | String  | Paper authors                           |
| `year`           | Integer | Publication/preprint year               |
| `topic`          | String  | Thematic category                       |
| `source_url`     | String  | arXiv abstract landing-page URL         |
| `pdf_url`        | String  | Direct PDF URL                          |
| `local_filename` | String  | Local PDF filename                      |

---

## 15. Data-Quality Validation

The dataset preparation process validates the following properties:

* [x] **12 papers present** in the corpus
* [x] **Unique paper IDs**
* [x] **Unique paper titles**
* [x] **Unique local filenames**
* [x] **Required metadata fields populated**
* [x] **PDF documents readable through `pypdf`**
* [x] **Page-level extraction available**
* [x] **Page numbers preserved as provenance metadata**
* [x] **Required source and PDF URLs retained**

The page-aware processing allows retrieved chunks to be traced back to the source paper and page.

---

## 16. Known Dataset Limitations

### 16.1 PDF Text Extraction

Academic papers commonly use multi-column layouts. Standard PDF text extraction may occasionally produce ordering artifacts, such as text from columns, headers, footers, or references appearing in an unexpected sequence.

The project therefore uses conservative preprocessing rather than aggressive document restructuring.

### 16.2 Equations and Mathematical Symbols

Complex mathematical expressions may not be represented perfectly by plain PDF text extraction. Some formatting information from the original PDF may be lost.

### 16.3 Tables and Figures

Tables, diagrams, and figures are not converted into structured multimodal data.

Text references such as "Figure 1" or "Table 2" may remain in the extracted text, but the visual contents themselves are not independently represented in the retrieval corpus.

### 16.4 Layout Information

The retrieval system primarily operates on extracted text and metadata. Original PDF visual layout, typography, and spatial relationships are not preserved as structured retrieval information.

### 16.5 Corpus Size

The corpus contains only 12 research papers. It is therefore suitable for demonstrating and evaluating the project architecture but is not representative of the scale of a production academic search system containing thousands or millions of documents.

### 16.6 Domain Coverage

The corpus is focused primarily on NLP, language models, dense retrieval, reasoning, and RAG. Results may not generalize to unrelated scientific domains.

---

## 17. Reproducibility

The main dataset-related artifacts are:

```text
data/metadata.csv
data/dataset_statistics.csv
data/raw_papers/
data/processed/chroma_db/
experiments/chunking_results.csv
experiments/embedding_results.csv
```

The processing implementation is primarily contained in:

```text
src/document_loader.py
src/chunking.py
src/embeddings.py
src/vectorstore.py
```

Together, these components provide the source-document loading, preprocessing, chunking, embedding, and vector-storage stages used by the Research Paper Answer Bot.

---

## 18. Dataset Summary

The final corpus consists of **12 research papers and 259 pages**, producing **1172 production chunks** using a 1000-character chunk size and 150-character overlap.

Each production chunk retains paper-level and page-level provenance metadata. The final retrieval corpus is embedded using `sentence-transformers/all-mpnet-base-v2` and stored in ChromaDB.

This dataset design provides a compact, traceable research corpus suitable for evaluating dense retrieval, BM25 retrieval, hybrid RRF retrieval, grounded answer generation, and citation-aware responses.
