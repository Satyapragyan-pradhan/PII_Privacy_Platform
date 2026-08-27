from pathlib import Path

from datasets import load_dataset, concatenate_datasets,DatasetDict


DATA_DIR = Path(__file__).parent / "data"


def load_indian_dataset():
    return load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.json"),
            "validation": str(DATA_DIR / "validation.json"),
        },
    )


def load_kiji_dataset():
    return load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "kiji" / "train.json"),
            "validation": str(DATA_DIR / "kiji" / "validation.json"),
        },
    )


def load_pii_dataset():

    indian = load_indian_dataset()
    kiji = load_kiji_dataset()

    train = concatenate_datasets([
        indian["train"],
        kiji["train"],
    ])

    validation = concatenate_datasets([
        indian["validation"],
        kiji["validation"],
    ])

    return DatasetDict({
    "train": train,
    "validation": validation,
   })