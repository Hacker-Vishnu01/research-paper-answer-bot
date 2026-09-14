from src.evaluate_rag_answers import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MPNET_MODEL_NAME,
    RRF_ALPHA,
    RRF_K,
    VECTORSTORE_COLLECTION,
    RETRIEVAL_TOP_K,
    CONTEXT_TOP_K,
    load_all_papers,
    chunk_document_pages,
    get_embedding_model,
    load_vectorstore,
    DenseRetriever,
    BM25RetrieverWrapper,
    HybridRetriever,
)

from src.config import config

QUESTION = (
    "What percentage of tokens are masked in BERT's "
    "Masked Language Model (MLM), and how are the chosen "
    "tokens replaced?"
)

print("\n[1/5] Loading research papers...")

pages = load_all_papers(
    config.data_dir
)

if not pages:
    raise RuntimeError(
        "No research paper pages were loaded."
    )

print(f"Pages loaded: {len(pages)}")

print("\n[2/5] Creating document chunks...")

chunks = chunk_document_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

if not chunks:
    raise RuntimeError(
        "No document chunks were created."
    )

print(f"Chunks created: {len(chunks)}")

print("\n[3/5] Loading MPNet embeddings...")

embeddings = get_embedding_model(
    MPNET_MODEL_NAME
)

print(
    f"Embedding model: {MPNET_MODEL_NAME}"
)

print("\n[4/5] Loading ChromaDB...")

vectorstore = load_vectorstore(
    embedding_function=embeddings,
    persist_directory=config.chroma_persist_directory,
    collection_name=VECTORSTORE_COLLECTION,
)

print(
    f"ChromaDB collection: {VECTORSTORE_COLLECTION}"
)

print("\n[5/5] Building hybrid retriever...")

dense_retriever = DenseRetriever(
    vectorstore=vectorstore
)

bm25_retriever = BM25RetrieverWrapper(
    chunks
)

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever,
    alpha=RRF_ALPHA,
    rrf_k=RRF_K,
)

print("Hybrid retriever: READY")

print("\n" + "=" * 80)
print("Q004 RETRIEVAL INSPECTION")
print("=" * 80)

results = hybrid_retriever.retrieve(
    query=QUESTION,
    top_k=RETRIEVAL_TOP_K,
)

print(f"Retrieved candidates: {len(results)}")
print(f"Final context size: {CONTEXT_TOP_K}")

for index, result in enumerate(
    results[:CONTEXT_TOP_K],
    start=1,
):
    print("\n" + "-" * 80)
    print(f"CONTEXT CHUNK {index}")
    print("-" * 80)

    if isinstance(result, tuple):
        chunk = result[0]
        score = result[1]
        print("Score:", score)
    else:
        chunk = result

    metadata = getattr(
        chunk,
        "metadata",
        None,
    )

    content = getattr(
        chunk,
        "content",
        "",
    )

    if metadata is not None:
        print(
            "Paper ID:",
            getattr(
                metadata,
                "paper_id",
                "N/A",
            ),
        )

        print(
            "Paper title:",
            getattr(
                metadata,
                "paper_title",
                "N/A",
            ),
        )

        print(
            "Page:",
            getattr(
                metadata,
                "page_number",
                "N/A",
            ),
        )

    print("\nPASSAGE:")
    print(content)

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
