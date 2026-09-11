import json
import time
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(BACKEND_DIR)
)

from agents.graph import build_graph


DATASET = Path(__file__).parent / "test.json"

GRAPH = build_graph()


def normalize_text(value):
    return " ".join(
        str(value or "").strip().lower().split()
    )


def compact(value):

    return re.sub(
        r"[\s-]+",
        "",
        normalize_text(value)
    )


def normalize_entity_value(entity_type, value):
    entity_type = str(
        entity_type or ""
    ).upper()

    value = str(
        value or ""
    ).strip()

    if entity_type == "EMAIL":
        value = re.sub(
            r"\[([^\]]+)\]\(mailto:([^)]+)\)",
            r"\2",
            value,
            flags=re.IGNORECASE
        )

        return normalize_text(value)

    if entity_type in {
        "AADHAAR",
        "DRIVING_LICENCE",
        "VOTER_ID",
        "PHONE",
    }:
        return compact(value).lower()

    if entity_type == "PAN":
        return compact(value).upper()

    if entity_type == "DOB":
        value = " ".join(
            value.split()
        )

        value = value.replace(
            " ",
            ""
        )

        value = value.replace(
            "-",
            "/"
        ).replace(
            ".",
            "/"
        )

        return value.lower()

    return normalize_text(value)


def entity_key(entity):
    entity_type = str(
        entity.get(
            "type",
            ""
        )
    ).upper()

    return (
        entity_type,
        normalize_entity_value(
            entity_type,
            entity.get(
                "value",
                ""
            )
        )
    )


def run_pipeline(text):
    result = GRAPH.invoke({
        "text": text,
        "regex_entities": [],
        "nlp_entities": [],
        "contextual_entities": [],
        "preliminary_entities": [],
        "final_entities": []
    })

    return result.get(
        "final_entities",
        []
    )


def calculate_metrics(stats):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def update_stats(
    stats,
    matched,
    false_positive,
    missed
):
    stats["tp"] += len(matched)
    stats["fp"] += len(false_positive)
    stats["fn"] += len(missed)


def evaluate():

    data = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    total = {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }

    per_type = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    per_document_type = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    source_counts = Counter()

    fallback_docs = 0

    latencies = []

    error_examples = []

    for item in data:

        start = time.perf_counter()

        predicted = run_pipeline(
            item["text"]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

        gt = {
            entity_key(entity)
            for entity in item.get(
                "ground_truth",
                []
            )
        }

        pred = {
            entity_key(entity)
            for entity in predicted
        }

        matched = gt & pred
        false_positive = pred - gt
        missed = gt - pred

        update_stats(
            total,
            matched,
            false_positive,
            missed
        )

        document_type = str(
            item.get(
                "document_type",
                "unknown"
            )
        ).lower()

        update_stats(
            per_document_type[
                document_type
            ],
            matched,
            false_positive,
            missed
        )

        for entity_type, value in matched:

            per_type[
                entity_type
            ]["tp"] += 1

        for entity_type, value in false_positive:

            per_type[
                entity_type
            ]["fp"] += 1

            if len(error_examples) < 30:

                error_examples.append({
                    "document_id":
                        item.get(
                            "document_id"
                        ),
                    "document_type":
                        document_type,
                    "error":
                        "false_positive",
                    "type":
                        entity_type,
                    "value":
                        value
                })

        for entity_type, value in missed:

            per_type[
                entity_type
            ]["fn"] += 1

            if len(error_examples) < 30:

                error_examples.append({
                    "document_id":
                        item.get(
                            "document_id"
                        ),
                    "document_type":
                        document_type,
                    "error":
                        "false_negative",
                    "type":
                        entity_type,
                    "value":
                        value
                })

        for entity in predicted:

            source_counts[
                str(
                    entity.get(
                        "source",
                        "unknown"
                    )
                )
            ] += 1

        if any(
            (
                "llm"
                in str(
                    entity.get(
                        "source",
                        ""
                    )
                ).lower()
            )
            or
            (
                "context"
                in str(
                    entity.get(
                        "source",
                        ""
                    )
                ).lower()
            )
            for entity in predicted
        ):
            fallback_docs += 1

    precision, recall, f1 = calculate_metrics(
        total
    )

    print(
        "\n=== OVERALL ENTITY-LEVEL RESULTS ==="
    )

    print(
        f"Documents evaluated : {len(data)}"
    )

    print(
        f"True positives      : {total['tp']}"
    )

    print(
        f"False positives     : {total['fp']}"
    )

    print(
        f"False negatives     : {total['fn']}"
    )

    print(
        f"Precision           : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall              : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1                  : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        "\n=== PER-ENTITY-TYPE RESULTS ==="
    )

    print(
        f"{'TYPE':20} "
        f"{'P':>8} "
        f"{'R':>8} "
        f"{'F1':>8} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )

    for entity_type in sorted(
        per_type
    ):

        stats = per_type[
            entity_type
        ]

        p, r, f = calculate_metrics(
            stats
        )

        print(
            f"{entity_type:20} "
            f"{p:8.4f} "
            f"{r:8.4f} "
            f"{f:8.4f} "
            f"{stats['tp']:6} "
            f"{stats['fp']:6} "
            f"{stats['fn']:6}"
        )

    print(
        "\n=== PER-DOCUMENT-TYPE RESULTS ==="
    )

    print(
        f"{'DOCUMENT':20} "
        f"{'P':>8} "
        f"{'R':>8} "
        f"{'F1':>8} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )

    for document_type in sorted(
        per_document_type
    ):

        stats = per_document_type[
            document_type
        ]

        p, r, f = calculate_metrics(
            stats
        )

        print(
            f"{document_type:20} "
            f"{p:8.4f} "
            f"{r:8.4f} "
            f"{f:8.4f} "
            f"{stats['tp']:6} "
            f"{stats['fp']:6} "
            f"{stats['fn']:6}"
        )

    print(
        "\n=== SOURCE DISTRIBUTION ==="
    )

    total_predicted = sum(
        source_counts.values()
    )

    for source, count in (
        source_counts.most_common()
    ):

        percentage = (
            count
            / total_predicted
            * 100
            if total_predicted
            else 0
        )

        print(
            f"{source:20} "
            f"{count:6} "
            f"({percentage:5.1f}%)"
        )

    print(
        "\n=== LLM FALLBACK ==="
    )

    fallback_rate = (
        fallback_docs
        / len(data)
        * 100
        if data
        else 0
    )

    print(
        f"Documents invoking fallback : "
        f"{fallback_docs}"
    )

    print(
        f"Fallback rate               : "
        f"{fallback_rate:.2f}%"
    )

    if latencies:

        latencies_sorted = sorted(
            latencies
        )

        average = (
            sum(latencies)
            / len(latencies)
        )

        median = latencies_sorted[
            len(latencies) // 2
        ]

        p95_index = min(
            len(latencies) - 1,
            int(
                len(latencies) * 0.95
            )
        )

        p95 = latencies_sorted[
            p95_index
        ]

        print(
            "\n=== LATENCY ==="
        )

        print(
            f"Average : "
            f"{average:.4f}s/document"
        )

        print(
            f"Median  : "
            f"{median:.4f}s/document"
        )

        print(
            f"P95     : "
            f"{p95:.4f}s/document"
        )

    print(
        "\n=== SAMPLE ERRORS ==="
    )

    for error in error_examples:

        print(
            f"{error['document_id']} | "
            f"{error['document_type']} | "
            f"{error['error']} | "
            f"{error['type']} | "
            f"{error['value']}"
        )


if __name__ == "__main__":
    evaluate()