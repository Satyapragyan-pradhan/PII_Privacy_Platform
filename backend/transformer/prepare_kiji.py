import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent / "data" / "kiji"


LABEL_MAP = {
    "FIRSTNAME": "NAME",
    "SURNAME": "NAME",

    "DATEOFBIRTH": "DOB",

    "PHONENUMBER": "PHONE",
    "EMAIL": "EMAIL",

    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "BUILDNUM": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "PROVINCE": "ADDRESS",
    "ZIP": "ADDRESS",
    "COLONY": "ADDRESS",
    "LOCATION": "ADDRESS",
    "COUNTY": "ADDRESS",
}


def convert_example(example):

    text = example["text"]

    entities = []

    for item in example["privacy_mask"]:

        original_label = item["label"]

        if original_label not in LABEL_MAP:
            continue

        label = LABEL_MAP[original_label]

        entities.append({
            "start": item["start"],
            "end": item["end"],
            "label": label,
        })

    return {
        "text": text,
        "entities": entities,
    }


def save_split(dataset, filename):

    output = []

    for example in dataset:
        converted = convert_example(example)

        if converted["entities"]:
            output.append(converted)

    output_path = OUTPUT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"{filename}: {len(output)} examples"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Kiji...")

    dataset = load_dataset(
        "DataikuNLP/kiji-pii-training-data"
    )

    save_split(
        dataset["train"],
        "train.json",
    )

    save_split(
        dataset["test"],
        "validation.json",
    )


if __name__ == "__main__":
    main()