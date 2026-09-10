import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

# Run from backend/ or adjust this import path as needed.
from agents.graph import build_graph

DATASET = Path(__file__).parent / "test.json"
GRAPH = build_graph()

def norm(value):
    return " ".join(str(value or "").strip().lower().split())

def key(entity):
    return (str(entity.get("type", "")).upper(), norm(entity.get("value", "")))

def run_pipeline(text):
    result = GRAPH.invoke({
        "text": text,
        "regex_entities": [],
        "nlp_entities": [],
        "contextual_entities": [],
        "preliminary_entities": [],
        "final_entities": []
    })
    return result.get("final_entities", [])

def evaluate():
    data = json.loads(DATASET.read_text(encoding="utf-8"))

    tp = 0
    fp = 0
    fn = 0

    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    source_counts = Counter()
    fallback_docs = 0
    latencies = []

    import time

    for item in data:
        start = time.perf_counter()
        predicted = run_pipeline(item["text"])
        latencies.append(time.perf_counter() - start)

        gt = { (str(x["type"]).upper(), norm(x["value"])) for x in item["ground_truth"] }
        pred = { key(x) for x in predicted }

        matched = gt & pred
        false_pos = pred - gt
        missed = gt - pred

        tp += len(matched)
        fp += len(false_pos)
        fn += len(missed)

        for typ, value in matched:
            per_type[typ]["tp"] += 1

        for typ, value in false_pos:
            per_type[typ]["fp"] += 1

        for typ, value in missed:
            per_type[typ]["fn"] += 1

        for entity in predicted:
            source_counts[str(entity.get("source", "unknown"))] += 1

        # This detects LLM participation from final entities.
        if any("llm" in str(x.get("source", "")).lower() or
               "context" in str(x.get("source", "")).lower()
               for x in predicted):
            fallback_docs += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\n=== OVERALL ENTITY-LEVEL RESULTS ===")
    print(f"Documents evaluated : {len(data)}")
    print(f"True positives      : {tp}")
    print(f"False positives     : {fp}")
    print(f"False negatives     : {fn}")
    print(f"Precision           : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall              : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1                  : {f1:.4f} ({f1*100:.2f}%)")

    print("\n=== PER-ENTITY-TYPE RESULTS ===")
    print(f"{'TYPE':20} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    for typ in sorted(per_type):
        x = per_type[typ]
        p = x["tp"] / (x["tp"] + x["fp"]) if x["tp"] + x["fp"] else 0.0
        r = x["tp"] / (x["tp"] + x["fn"]) if x["tp"] + x["fn"] else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        print(f"{typ:20} {p:8.4f} {r:8.4f} {f:8.4f} {x['tp']:6} {x['fp']:6} {x['fn']:6}")

    print("\n=== SOURCE DISTRIBUTION ===")
    total_pred = sum(source_counts.values())
    for source, count in source_counts.most_common():
        pct = count / total_pred * 100 if total_pred else 0
        print(f"{source:20} {count:6} ({pct:5.1f}%)")

    print("\n=== LLM FALLBACK ===")
    print(f"Documents invoking fallback : {fallback_docs}")
    print(f"Fallback rate               : {fallback_docs/len(data)*100:.2f}%")

    latencies_sorted = sorted(latencies)
    avg = sum(latencies) / len(latencies)
    median = latencies_sorted[len(latencies)//2]
    p95 = latencies_sorted[min(len(latencies)-1, int(len(latencies)*0.95))]
    print("\n=== LATENCY ===")
    print(f"Average : {avg:.4f}s/document")
    print(f"Median  : {median:.4f}s/document")
    print(f"P95     : {p95:.4f}s/document")

if __name__ == "__main__":
    evaluate()
