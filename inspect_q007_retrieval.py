from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "Why is Sentence-BERT computationally faster than cross-encoder BERT "
    "for large-scale semantic similarity search and clustering?"
)

pipeline = build_pipeline()

results = pipeline.retriever.retrieve(
    query=QUESTION,
    top_k=10
)

print("\n" + "=" * 80)
print("Q007 TOP-10 RETRIEVAL RESULTS")
print("=" * 80)

for rank, item in enumerate(results, start=1):
    chunk, score = item

    print(f"\n--- Rank {rank} | Score: {score:.6f} ---")
    print(f"Chunk type: {type(chunk).__name__}")

    if hasattr(chunk, "metadata"):
        print(f"Metadata: {chunk.metadata}")

    if hasattr(chunk, "text"):
        print("\nContent:")
        print(chunk.text[:2000])
    elif hasattr(chunk, "page_content"):
        print("\nContent:")
        print(chunk.page_content[:2000])
    else:
        print("\nChunk:")
        print(chunk)

