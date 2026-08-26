from datasets import load_dataset


def load_pii_dataset():

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.json",
            "validation": "data/validation.json",
        }
    )

    return dataset