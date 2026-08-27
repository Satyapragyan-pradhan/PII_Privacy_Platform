LABEL_TYPES = [
    "NAME",
    "PAN",
    "AADHAAR",
    "EMAIL",
    "PHONE",
    "DOB",
    "ADDRESS",
]

LABELS = ["O"]

for label in LABEL_TYPES:
    LABELS.append(f"B-{label}")
    LABELS.append(f"I-{label}")


LABEL2ID = {
    label: idx
    for idx, label in enumerate(LABELS)
}


ID2LABEL = {
    idx: label
    for label, idx in LABEL2ID.items()
}


def create_bio_labels(text, entities):
    """
    Create character-level BIO labels.

    Entity format:

    {
        "start": 11,
        "end": 23,
        "label": "NAME"
    }

    End index is exclusive.
    """

    labels = ["O"] * len(text)

    for entity in entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        # Ignore malformed entities
        if (
            start < 0
            or end > len(text)
            or start >= end
        ):
            continue

        # Ignore unsupported entity types
        if label not in LABEL_TYPES:
            continue

        labels[start] = f"B-{label}"

        for i in range(start + 1, end):
            labels[i] = f"I-{label}"

    return labels


def tokenize_and_align_labels(example, tokenizer):
    """
    Tokenize text and align character-level BIO
    entities with Transformer tokens.
    """

    text = example["text"]
    entities = example["entities"]

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    token_labels = []

    entity_token_started = set()

    for token_start, token_end in tokenized["offset_mapping"]:

        # [CLS], [SEP], etc.
        if token_start == token_end:
            token_labels.append(-100)
            continue

        assigned_label = "O"

        for entity_index, entity in enumerate(entities):

            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_type = entity["label"]

            if entity_type not in LABEL_TYPES:
                continue

            overlaps = (
                token_start < entity_end
                and token_end > entity_start
            )

            if not overlaps:
                continue

            if entity_index not in entity_token_started:
                assigned_label = f"B-{entity_type}"
                entity_token_started.add(entity_index)
            else:
                assigned_label = f"I-{entity_type}"

            break

        token_labels.append(
            LABEL2ID[assigned_label]
        )

    tokenized["labels"] = token_labels

    del tokenized["offset_mapping"]

    return tokenized