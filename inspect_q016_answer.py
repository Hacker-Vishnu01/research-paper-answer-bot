from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "What are the main components of the Self-RAG framework and how "
    "does it improve the factuality and quality of generated responses?"
)

pipeline = build_pipeline()

response = pipeline.answer(QUESTION)

print("\n" + "=" * 80)
print("Q016 DIRECT ANSWER")
print("=" * 80)

print("\nANSWER:")
print(response.answer)

print("\nCITATIONS:")
for citation in response.citations:
    print(citation)

print("\nINSUFFICIENT EVIDENCE DETECTED:")
print(pipeline._is_insufficient_evidence(response.answer))
