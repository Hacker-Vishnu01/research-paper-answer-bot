from src.evaluate_rag_answers import build_pipeline

QUESTION = "Why is Sentence-BERT computationally faster than cross-encoder BERT for large-scale semantic similarity search and clustering?"

pipeline = build_pipeline()
response = pipeline.answer(QUESTION)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)
print(response.answer)

print("\n" + "=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)

for i, chunk in enumerate(response.retrieved_chunks, start=1):
    metadata = getattr(chunk, "metadata", None)

    print(f"\n--- CHUNK {i} ---")

    if metadata is not None:
        print(f"Paper: {getattr(metadata, 'paper_id', '')}")
        print(f"Title: {getattr(metadata, 'paper_title', '')}")
        print(f"Page: {getattr(metadata, 'page_number', '')}")

    print("\nContent:")
    print(getattr(chunk, "page_content", str(chunk)))

print("\n" + "=" * 80)
print("CITATIONS")
print("=" * 80)

for citation in response.citations:
    print(citation)
