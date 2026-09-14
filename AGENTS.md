# Research Paper Answer Bot - Project Instructions

## Project Goal

Build a production-quality academic Research Paper Answer Bot using Retrieval-Augmented Generation (RAG).

The system will answer questions using a collection of research papers and provide supporting citations containing:

- Paper title
- Page number
- Retrieved passage

## Core Technologies

- Python 3.11
- LangChain
- ChromaDB
- Sentence Transformers
- BM25
- Cross-Encoder reranking
- Streamlit
- Pandas
- NumPy
- Matplotlib
- PyPDF
- python-dotenv

## Main Architecture

Research Paper PDFs
→ PDF extraction
→ text cleaning
→ chunking
→ embeddings
→ vector database
→ retrieval
→ reranking
→ prompt construction
→ LLM
→ answer + citations

## Embedding Experiments

Compare at least:

1. sentence-transformers/all-mpnet-base-v2
2. BAAI/bge-base-en-v1.5
3. OpenAI text-embedding-3-small

Do not fabricate experimental results.

All experimental results must come from actual experiments.

## Retrieval Experiments

Implement and compare:

1. Dense similarity retrieval
2. MMR retrieval
3. BM25 retrieval
4. Hybrid BM25 + dense retrieval
5. Hybrid retrieval + reranking

## Dataset

Use approximately 10-12 publicly accessible research papers related to:

- Retrieval-Augmented Generation
- Transformers
- Large Language Models
- Embeddings
- Efficient LLMs
- Prompting and reasoning

Prefer:

- arXiv
- ACL Anthology
- official research repositories

Maintain metadata for every paper:

- paper ID
- title
- authors
- year
- source URL
- PDF URL
- local filename

Do not fabricate metadata.

## Document Metadata

Every page and chunk must preserve:

- paper ID
- paper title
- page number
- source filename
- chunk ID

This metadata must be available when generating citations.

## Chunking Experiments

Experiment with:

- chunk size
- chunk overlap

Do not choose values without evaluating them.

## RAG Requirements

The final system must:

- retrieve relevant research passages
- generate answers using retrieved context
- avoid unsupported claims
- explicitly state when evidence is insufficient
- show the top 3 supporting sources
- show paper title
- show page number
- show supporting passages

## Hallucination Prevention

The LLM must answer using the retrieved research context.

If the retrieved context is insufficient, the system must clearly state that the available papers do not contain enough evidence.

Never invent citations.

## Evaluation

Evaluate retrieval using:

- Hit@K
- Precision@K
- Recall@K
- MRR

Evaluate generated answers using appropriate groundedness/faithfulness and relevance checks.

Also measure:

- latency
- citation correctness

Never invent evaluation scores.

## Evaluation Dataset

Create a manually verified question dataset.

Separate development questions from final evaluation questions.

Do not repeatedly tune the system on final test questions.

## Application

Create a Streamlit interface with:

- question input
- generated answer
- top 3 supporting sources
- paper title
- page number
- supporting passages
- conversation history
- clear conversation button
- model/retrieval information

## Security

Never hard-code API keys.

Use .env for secrets.

Never commit .env to Git.

Validate uploaded files.

Do not execute arbitrary content from research PDFs.

## Code Quality

Use modular Python files.

Do not put the entire project in app.py.

Use meaningful function names.

Use type hints where practical.

Use logging.

Add tests for important components.

## Documentation

Maintain:

- README.md
- architecture documentation
- methodology
- experiment results
- evaluation methodology
- limitations
- future improvements

## Academic Integrity

Never fabricate:

- datasets
- papers
- metadata
- citations
- experimental results
- evaluation metrics

All reported results must be generated from actual experiments.

## Development Strategy

Build the project incrementally.

After every major component:

1. Run tests.
2. Check errors.
3. Verify output.
4. Document the result.
5. Do not proceed if the current component is broken.

For now, ONLY create AGENTS.md.

Do not create Python files.
Do not install packages.
Do not download datasets.
Do not build the RAG pipeline.
