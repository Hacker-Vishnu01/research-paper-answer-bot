from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "What percentage of tokens are masked in BERT's "
    "Masked Language Model (MLM), and how are the chosen "
    "tokens replaced?"
)

pipeline = build_pipeline()

results = pipeline.retriever.retrieve(
    query=QUESTION,
    top_k=10,
)

chunks = pipeline._extract_chunks(results)

print("\n" + "=" * 80)
print("Q004 - TOP 10 RETRIEVAL")
print("=" * 80)

for i, chunk in enumerate(chunks, start=1):
    metadata = getattr(chunk, "metadata", None)
    content = getattr(chunk, "content", "")

    print("\n" + "-" * 80)
    print(f"RANK {i}")
    print("-" * 80)

    if metadata:
        print("Paper ID:", getattr(metadata, "paper_id", "N/A"))
        print("Page:", getattr(metadata, "page_number", "N/A"))

    print(content[:2500])

print("\n" + "=" * 80)
