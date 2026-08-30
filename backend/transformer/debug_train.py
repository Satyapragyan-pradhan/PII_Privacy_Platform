
import torch
import numpy as np

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from transformer.dataset import load_pii_dataset
from transformer.tokenize import tokenize_example, tokenizer

from transformer.labels import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
)


MODEL_NAME = "microsoft/deberta-v3-base"


def check_tensor(name, tensor):
    """
    Check whether a tensor contains NaN or Inf values.
    """

    if tensor is None:
        return True

    finite = torch.isfinite(tensor).all().item()

    if not finite:
        print(f"❌ NON-FINITE TENSOR: {name}")

        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()

        print(f"   NaN count : {nan_count}")
        print(f"   Inf count : {inf_count}")

        finite_values = tensor[torch.isfinite(tensor)]

        if finite_values.numel() > 0:
            print(
                f"   finite min: {finite_values.min().item()}"
            )
            print(
                f"   finite max: {finite_values.max().item()}"
            )

        return False

    return True


def check_model_parameters(model, stage):

    for name, param in model.named_parameters():

        if not torch.isfinite(param).all():

            print()
            print("========================================")
            print("❌ MODEL PARAMETERS BECAME NaN / INF")
            print("Stage:", stage)
            print("Parameter:", name)
            print("========================================")

            print(
                "NaN:",
                torch.isnan(param).sum().item()
            )

            print(
                "Inf:",
                torch.isinf(param).sum().item()
            )

            return False

    return True


def check_gradients(model, stage):

    max_grad = 0.0
    max_grad_name = None

    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        grad = param.grad

        if not torch.isfinite(grad).all():

            print()
            print("========================================")
            print("❌ GRADIENT BECAME NaN / INF")
            print("Stage:", stage)
            print("Parameter:", name)
            print("========================================")

            print(
                "NaN:",
                torch.isnan(grad).sum().item()
            )

            print(
                "Inf:",
                torch.isinf(grad).sum().item()
            )

            return False, max_grad, max_grad_name

        current_max = grad.abs().max().item()

        if current_max > max_grad:
            max_grad = current_max
            max_grad_name = name

    return True, max_grad, max_grad_name


def main():

    print("Loading dataset...")

    dataset = load_pii_dataset()

    print("Tokenizing...")

    tokenized = dataset.map(
        tokenize_example
    )

    print("Loading model...")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    model.train()

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    print()
    print("========================================")
    print("MODEL / GPU")
    print("========================================")

    print("Device:", device)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("========================================")
    print("MODEL INFORMATION")
    print("========================================")

    print("Number of labels:", len(LABELS))
    print("Labels:", LABELS)

    print(
        "Vocabulary size:",
        model.config.vocab_size
    )

    print(
        "Tokenizer vocabulary:",
        len(tokenizer)
    )

    print()
    print("========================================")
    print("CHECKING DATASET SAMPLES")
    print("========================================")

    # Check first 10 examples before training.
    for i in range(10):

        example = tokenized["train"][i]

        input_ids = example["input_ids"]
        labels = example["labels"]

        max_input_id = max(input_ids)
        min_input_id = min(input_ids)

        print(
            f"Sample {i:03d} | "
            f"seq_len={len(input_ids)} | "
            f"input_id_range=({min_input_id}, {max_input_id}) | "
            f"max_vocab={model.config.vocab_size - 1}"
        )

        if max_input_id >= model.config.vocab_size:
            print(
                "❌ INVALID INPUT ID!"
            )
            return

        if min_input_id < 0:
            print(
                "❌ NEGATIVE INPUT ID!"
            )
            return

        valid_labels = [
            x for x in labels
            if x != -100
        ]

        invalid_labels = [
            x for x in valid_labels
            if x < 0 or x >= len(LABELS)
        ]

        if invalid_labels:
            print(
                "❌ INVALID LABELS:",
                invalid_labels[:20]
            )
            return

    print()
    print("Dataset sanity check: OK")

    print()
    print("========================================")
    print("TESTING INDIVIDUAL SAMPLES")
    print("========================================")

    # Use a simple SGD optimizer first.
    #
    # This is intentional.
    # We are NOT trying to train the model here.
    #
    # We only want to determine whether
    # AdamW/optimizer updates are causing NaNs.

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-5
    )

    for i in range(10):

        print()
        print("----------------------------------------")
        print(f"Testing sample {i}")
        print("----------------------------------------")

        optimizer.zero_grad(set_to_none=True)

        example = tokenized["train"][i]

        # IMPORTANT:
        # Remove text/entities because the model does not need them.
        model_example = {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "token_type_ids": example["token_type_ids"],
            "labels": example["labels"],
        }

        batch = collator(
            [model_example]
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # -------------------------------------------------
        # FORWARD
        # -------------------------------------------------

        outputs = model(**batch)

        loss = outputs.loss

        print(
            "Loss before backward:",
            loss.item()
        )

        print(
            "Loss finite:",
            torch.isfinite(loss).item()
        )

        if not torch.isfinite(loss):

            print(
                "❌ LOSS IS ALREADY NaN/INF"
            )

            return

        if not check_tensor(
            "logits",
            outputs.logits
        ):
            return

        # -------------------------------------------------
        # BACKWARD
        # -------------------------------------------------

        loss.backward()

        print(
            "Backward:",
            "OK"
        )

        gradients_ok, max_grad, max_grad_name = (
            check_gradients(
                model,
                "after backward"
            )
        )

        print(
            "Max gradient:",
            max_grad
        )

        print(
            "Max gradient parameter:",
            max_grad_name
        )

        if not gradients_ok:
            return

        # -------------------------------------------------
        # PARAMETERS BEFORE OPTIMIZER
        # -------------------------------------------------

        if not check_model_parameters(
            model,
            "before optimizer.step()"
        ):
            return

        print(
            "Parameters before optimizer:",
            "FINITE"
        )

        # -------------------------------------------------
        # OPTIMIZER STEP
        # -------------------------------------------------

        optimizer.step()

        print(
            "Optimizer step:",
            "OK"
        )

        # -------------------------------------------------
        # PARAMETERS AFTER OPTIMIZER
        # -------------------------------------------------

        if not check_model_parameters(
            model,
            "after optimizer.step()"
        ):
            print()
            print(
                "❌ NaN was introduced by optimizer.step()"
            )
            return

        print(
            "Parameters after optimizer:",
            "FINITE"
        )

    print()
    print("========================================")
    print("✅ ALL 10 SAMPLES PASSED")
    print("========================================")

    print()
    print(
        "No NaN detected with SGD."
    )

    print(
        "This means the forward/backward pipeline "
        "is numerically stable."
    )


if __name__ == "__main__":
    main()

