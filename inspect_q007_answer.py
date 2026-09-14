from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "Why is Sentence-BERT computationally faster than cross-encoder BERT "
    "for large-scale semantic similarity search and clustering?"
)

pipeline = build_pipeline()
response = pipeline.answer(QUESTION)

print("\n" + "=" * 80)
print("Q007 DIRECT ANSWER")
print("=" * 80)

print("\nANSWER:")
print(response.answer)

print("\nCITATIONS:")
for citation in response.citations:
    print(citation)

print("\nINSUFFICIENT EVIDENCE:")
print(pipeline._is_insufficient_evidence(response.answer))
