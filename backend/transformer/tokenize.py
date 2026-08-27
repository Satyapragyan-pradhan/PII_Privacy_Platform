from transformers import AutoTokenizer

from transformer.labels import (
    LABEL2ID,
    create_bio_labels,
)


MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_example(example):
    text = example["text"]
    entities = example["entities"]

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    character_labels = create_bio_labels(
        text,
        entities,
    )

    token_labels = []

    for token_start, token_end in tokenized["offset_mapping"]:

        # [CLS], [SEP], etc.
        if token_start == token_end:
            token_labels.append(-100)
            continue

        assigned_label = "O"

        for entity in entities:

            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_type = entity["label"]

            # Token overlaps entity
            if (
                token_start < entity_end
                and token_end > entity_start
            ):

                # First token overlapping entity
                if token_start <= entity_start:
                    assigned_label = f"B-{entity_type}"
                else:
                    assigned_label = f"I-{entity_type}"

                break

        token_labels.append(
            LABEL2ID[assigned_label]
        )

    tokenized["labels"] = token_labels

    del tokenized["offset_mapping"]

    return tokenized