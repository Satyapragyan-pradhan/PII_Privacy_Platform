import json
from pathlib import Path

from extraction.regex import extract_regex_entities


DATASET_PATH = Path(
    __file__
).parent / "dataset.json"


def normalize(value):
    return (
        value
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )


def evaluate():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for sample in dataset:

        predictions = extract_regex_entities(
            sample["text"]
        )

        predicted_set = {
            (
                entity["type"],
                normalize(
                    entity["value"]
                )
            )
            for entity in predictions
        }

        actual_set = {
            (
                entity["type"],
                normalize(
                    entity["value"]
                )
            )
            for entity in sample["entities"]
        }

        true_positive += len(
            predicted_set & actual_set
        )

        false_positive += len(
            predicted_set - actual_set
        )

        false_negative += len(
            actual_set - predicted_set
        )

    precision = (
        true_positive /
        (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall
        else 0
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall: {recall:.3f}"
    )

    print(
        f"F1 Score: {f1:.3f}"
    )


if __name__ == "__main__":
    evaluate()