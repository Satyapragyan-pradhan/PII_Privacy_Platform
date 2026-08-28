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

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    total_true_entities = 0
    total_pred_entities = 0

    for prediction, label in zip(predictions, labels):

        current_predictions = []
        current_labels = []

        for p, l in zip(prediction, label):

            if l == -100:
                continue

            pred_label = ID2LABEL[p]
            true_label = ID2LABEL[l]

            current_predictions.append(pred_label)
            current_labels.append(true_label)

            if true_label != "O":
                total_true_entities += 1

            if pred_label != "O":
                total_pred_entities += 1

        true_predictions.append(current_predictions)
        true_labels.append(current_labels)

    precision = precision_score(
        true_labels,
        true_predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        true_predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        true_predictions,
        zero_division=0
    )

    print("\n========== ENTITY DIAGNOSTICS ==========")
    print("True entity tokens     :", total_true_entities)
    print("Predicted entity tokens:", total_pred_entities)
    print("Precision              :", precision)
    print("Recall                 :", recall)
    print("F1                     :", f1)
    print("========================================\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    print("=" * 60)
    print("PII DeBERTa-v3 Token Classification Training")
    print("=" * 60)

    print("\nLoading dataset...")

    dataset = load_pii_dataset()

    print(dataset)

    print("\nTokenizing dataset...")

    tokenized = dataset.map(
        tokenize_example
    )

    print(tokenized)

    print("\nLoading DeBERTa-v3-base...")

    # IMPORTANT:
    # Force FP32 model weights.
    # This avoids the FP16 + AdamW NaN issue observed locally.
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=True,
        dtype=torch.float32,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        # Evaluate/save periodically rather than every epoch
        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,

        # Optimization
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,

        # Actual training
        num_train_epochs=3,

        # Logging
        logging_steps=100,

        # Best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Keep FP16 disabled because it caused NaN locally.
        fp16=False,
        bf16=False,

        # No external logging
        report_to="none",

        # Better memory behavior
        dataloader_pin_memory=True,

        # Reproducibility
        seed=42,
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

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print("Device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)
    print("Train batch size:", training_args.per_device_train_batch_size)
    print("Gradient accumulation:", training_args.gradient_accumulation_steps)
    print(
        "Effective batch size:",
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    print("Learning rate:", training_args.learning_rate)
    print("Epochs:", training_args.num_train_epochs)
    print("FP16:", training_args.fp16)
    print("BF16:", training_args.bf16)

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))

        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(
            "GPU memory:",
            round(total_memory / 1024**3, 2),
            "GB"
        )

    print("=" * 60)

    print("\nStarting training...\n")

    trainer.train()

    print("\nTraining complete.")

    print("\nSaving final model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()