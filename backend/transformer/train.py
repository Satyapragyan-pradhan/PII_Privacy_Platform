import numpy as np
import torch

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from transformer.dataset import load_pii_dataset
from transformer.tokenize import tokenize_example, tokenizer

from transformer.labels import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
)


MODEL_NAME = "microsoft/deberta-v3-base"

OUTPUT_DIR = "transformer/model/deberta-pii"


def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []
        current_labels = []

        for p, l in zip(
            prediction,
            label
        ):

            if l == -100:
                continue

            current_predictions.append(
                ID2LABEL[p]
            )

            current_labels.append(
                ID2LABEL[l]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions
        ),
        "recall": recall_score(
            true_labels,
            true_predictions
        ),
        "f1": f1_score(
            true_labels,
            true_predictions
        ),
    }


def main():

    print("Loading dataset...")

    dataset = load_pii_dataset()

    print(dataset)

    print("Tokenizing dataset...")

    tokenized = dataset.map(
        tokenize_example
    )

    print(tokenized)

    print("Loading DeBERTa-v3-base...")

    model=AutoModelForTokenClassification.from_pretrained(
     MODEL_NAME,
     num_labels=len(LABELS),
     id2label=ID2LABEL,
     label2id=LABEL2ID,
     use_safetensors=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=False,
    bf16=torch.cuda.is_available(),

    report_to="none",
)

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    print("Starting training...")

    trainer.train()

    print("Training complete.")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(
        f"Model saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()