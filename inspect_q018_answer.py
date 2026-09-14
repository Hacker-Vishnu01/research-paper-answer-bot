from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "How does Self-RAG assess whether retrieved passages are relevant "
    "using the ISREL reflection token?"
)

pipeline = build_pipeline()

response = pipeline.answer(QUESTION)

print("\n" + "=" * 80)
print("Q018 DIRECT ANSWER")
print("=" * 80)

print("\nANSWER:")
print(response.answer)

print("\nCITATIONS:")
for citation in response.citations:
    print(citation)

print("\nRETRIEVED CONTEXT:")
print("=" * 80)

for i, chunk in enumerate(response.retrieved_chunks, start=1):
    metadata = getattr(chunk, "metadata", None)
    content = getattr(chunk, "content", "")

    print("\n" + "-" * 80)
    print(f"CONTEXT {i}")
    print("-" * 80)

    if metadata:
        print("Paper ID:", getattr(metadata, "paper_id", "N/A"))
        print("Page:", getattr(metadata, "page_number", "N/A"))

    print(content[:3000])
