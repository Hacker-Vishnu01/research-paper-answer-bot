import csv
from pathlib import Path

INPUT_FILE = Path("evaluation/answer_evaluation.csv")
OUTPUT_FILE = Path("evaluation/final_results.csv")

# Manual evaluation
# Faithfulness, relevance and semantic correctness are scored 0-2.
SCORES = {
    "Q001": (2, 2, 2),
    "Q002": (2, 2, 2),
    "Q003": (2, 2, 2),
    "Q004": (2, 2, 2),
    "Q005": (2, 2, 2),
    "Q006": (2, 2, 2),
    "Q007": (2, 2, 2),
    "Q008": (2, 2, 2),
    "Q009": (2, 2, 2),
    "Q010": (2, 2, 2),
    "Q011": (2, 2, 2),
    "Q012": (1, 2, 1),
    "Q013": (2, 2, 1),
    "Q014": (2, 2, 1),
    "Q015": (1, 2, 1),
    "Q016": (2, 2, 2),
    "Q017": (2, 2, 2),
    "Q018": (2, 2, 2),
    "Q019": (2, 2, 1),
    "Q020": (2, 2, 2),
}

with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

final_rows = []

for row in rows:
    qid = row["question_id"]

    if qid not in SCORES:
        print(f"WARNING: No manual score for {qid}")
        continue

    faithfulness, relevance, semantic = SCORES[qid]

    final_rows.append({
        "query_id": qid,
        "query": row["question"],
        "model": "llama3.2:3b",
        "retrieval_strategy": row["retrieval_strategy"],
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "semantic_correctness": semantic,
        "citation_correctness": row["citation_correctness"],
        "insufficient_evidence": row["insufficient_evidence"],
        "latency_seconds": row["measured_latency_seconds"],
    })

fieldnames = [
    "query_id",
    "query",
    "model",
    "retrieval_strategy",
    "faithfulness",
    "answer_relevance",
    "semantic_correctness",
    "citation_correctness",
    "insufficient_evidence",
    "latency_seconds",
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print("=" * 70)
print("FINAL EVALUATION CREATED")
print("=" * 70)
print(f"Questions: {len(final_rows)}")
print(f"Output: {OUTPUT_FILE}")
print()

# Summary
n = len(final_rows)

faithfulness_avg = sum(int(r["faithfulness"]) for r in final_rows) / n
relevance_avg = sum(int(r["answer_relevance"]) for r in final_rows) / n
semantic_avg = sum(int(r["semantic_correctness"]) for r in final_rows) / n
citation_avg = sum(1 if str(r["citation_correctness"]).lower() == "correct" else 0 for r in final_rows) / n
insufficient_count = sum(
    1 for r in final_rows
    if str(r["insufficient_evidence"]).lower() == "true"
)
latency_avg = sum(float(r["latency_seconds"]) for r in final_rows) / n

print(f"Faithfulness:          {faithfulness_avg:.2f}/2 ({faithfulness_avg/2*100:.1f}%)")
print(f"Answer relevance:      {relevance_avg:.2f}/2 ({relevance_avg/2*100:.1f}%)")
print(f"Semantic correctness:  {semantic_avg:.2f}/2 ({semantic_avg/2*100:.1f}%)")
print(f"Citation correctness:  {citation_avg*100:.1f}%")
print(f"Insufficient evidence:  {insufficient_count}/{n} ({insufficient_count/n*100:.1f}%)")
print(f"Average latency:       {latency_avg:.2f}s")
