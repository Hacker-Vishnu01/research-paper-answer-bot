from src.evaluate_rag_answers import build_pipeline

QUESTION = (
    "How does LoRA modify the original weight matrix, and what are "
    "the dimensions and rank constraint of the trainable matrices?"
)

pipeline = build_pipeline()

results = pipeline.retriever.retrieve(
    query=QUESTION,
    top_k=10
)

print("\n" + "=" * 80)
print("Q012 TOP 10 RETRIEVAL RESULTS")
print("=" * 80)

for i, result in enumerate(results, start=1):
    print(f"\n{'-' * 80}")
    print(f"RANK {i}")
    print(f"{'-' * 80}")
    print(f"TYPE: {type(result)}")
    print(f"RAW RESULT:\n{result}")
