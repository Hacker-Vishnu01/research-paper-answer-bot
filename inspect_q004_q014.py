from src.evaluate_rag_answers import build_pipeline

QUESTIONS = {
    "Q004": "How does BERT's Masked Language Model objective select and replace tokens during pretraining?",
    "Q014": "What is NormalFloat (NF4) quantization and why is it useful for QLoRA?",
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
