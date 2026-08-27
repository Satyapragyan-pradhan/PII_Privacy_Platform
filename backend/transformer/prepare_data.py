import json
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "indian_pii.csv"

TRAIN_PATH = DATA_DIR / "train.json"
VALIDATION_PATH = DATA_DIR / "validation.json"


def add_entity(text, entities, value, label):

    if not value or pd.isna(value):
        return

    value = str(value).strip()

    if not value:
        return

    start = text.find(value)

    if start == -1:
        return

    entities.append({
        "start": start,
        "end": start + len(value),
        "label": label
    })


def convert_row(row):

    text = (
        f"Name: {row['Name']}. "
        f"Email: {row['Email']}. "
        f"Phone: {row['Phone']}. "
        f"Address: {row['Address']}. "
        f"Aadhaar: {row['Aadhaar']}. "
        f"PAN: {row['PAN']}. "
        f"DOB: {row['DOB']}."
    )

    entities = []

    add_entity(
        text,
        entities,
        row["Name"],
        "NAME"
    )

    add_entity(
        text,
        entities,
        row["Email"],
        "EMAIL"
    )

    add_entity(
        text,
        entities,
        row["Phone"],
        "PHONE"
    )

    add_entity(
        text,
        entities,
        row["Address"],
        "ADDRESS"
    )

    add_entity(
        text,
        entities,
        row["Aadhaar"],
        "AADHAAR"
    )

    add_entity(
        text,
        entities,
        row["PAN"],
        "PAN"
    )

    add_entity(
        text,
        entities,
        row["DOB"],
        "DOB"
    )

    return {
        "text": text,
        "entities": entities
    }


def main():

    df = pd.read_csv(CSV_PATH)

    examples = []

    for _, row in df.iterrows():
        examples.append(convert_row(row))

    random.seed(42)
    random.shuffle(examples)

    split = int(len(examples) * 0.9)

    train = examples[:split]
    validation = examples[split:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")


if __name__ == "__main__":
    main()