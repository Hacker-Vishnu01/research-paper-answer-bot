from src.evaluate_rag_answers import build_pipeline

QUESTIONS = {
    "Q006": "How does Retrieval-Augmented Generation combine parametric memory with non-parametric retrieval memory?",
    "Q012": "How does Low-Rank Adaptation (LoRA) decompose weight updates in transformer layers using low-rank matrices?",
    "Q015": "How do Double Quantization and Paged Optimizers in QLoRA reduce GPU memory footprint during model fine-tuning?",
    "Q019": "What is the U-shaped performance degradation observed in Lost in the Middle when relevant information is positioned in the center of long input contexts?",
}

pipeline = build_pipeline()

for question_id, question in QUESTIONS.items():

    print("\n" + "=" * 100)
    print(question_id)
    print("=" * 100)
    print("QUESTION:")
    print(question)

    response = pipeline.answer(question)

    print("\nANSWER:")
    print(response.answer)

    print("\nRETRIEVED CHUNKS:")

    for i, chunk in enumerate(response.retrieved_chunks, start=1):
        metadata = getattr(chunk, "metadata", None)

        print(f"\n--- CHUNK {i} ---")

        if metadata is not None:
            print(f"Paper: {getattr(metadata, 'paper_id', '')}")
            print(f"Title: {getattr(metadata, 'paper_title', '')}")
            print(f"Page: {getattr(metadata, 'page_number', '')}")

        print("\nContent:")
        print(getattr(chunk, "page_content", str(chunk)))
