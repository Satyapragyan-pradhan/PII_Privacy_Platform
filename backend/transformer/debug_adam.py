
import torch

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


def check_parameters(model, stage):

    for name, param in model.named_parameters():

        if not torch.isfinite(param).all():

            print()
            print("========================================")
            print("❌ NON-FINITE PARAMETER")
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


def check_gradients(model):

    max_grad = 0.0
    max_name = None

    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        if not torch.isfinite(param.grad).all():

            print()
            print("========================================")
            print("❌ NON-FINITE GRADIENT")
            print("Parameter:", name)
            print("========================================")

            print(
                "NaN:",
                torch.isnan(param.grad).sum().item()
            )

            print(
                "Inf:",
                torch.isinf(param.grad).sum().item()
            )

            return False

        current = param.grad.abs().max().item()

        if current > max_grad:
            max_grad = current
            max_name = name

    print(
        f"Max gradient: {max_grad} "
        f"({max_name})"
    )

    return True


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
    torch_dtype=torch.float32,
)

    model = model.float()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.train()

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    print()
    print("========================================")
    print("MODEL")
    print("========================================")

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("========================================")
    print("ADAMW TEST")
    print("========================================")

    # Use the same learning rate as train.py.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )

    for i in range(10):

        print()
        print("----------------------------------------")
        print(f"STEP {i}")
        print("----------------------------------------")

        optimizer.zero_grad(
            set_to_none=True
        )

        example = tokenized["train"][i]

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

        # -------------------------------
        # FORWARD
        # -------------------------------

        outputs = model(**batch)

        loss = outputs.loss

        print(
            "Loss:",
            loss.item()
        )

        print(
            "Loss finite:",
            torch.isfinite(loss).item()
        )

        if not torch.isfinite(loss):

            print(
                "❌ Loss became NaN/INF before backward."
            )

            return

        # -------------------------------
        # BACKWARD
        # -------------------------------

        loss.backward()

        print(
            "Backward: OK"
        )

        if not check_gradients(model):
            return

        # -------------------------------
        # BEFORE UPDATE
        # -------------------------------

        if not check_parameters(
            model,
            "before optimizer.step()"
        ):
            return

        # -------------------------------
        # ADAMW UPDATE
        # -------------------------------

        optimizer.step()

        print(
            "AdamW step: OK"
        )

        # -------------------------------
        # AFTER UPDATE
        # -------------------------------

        if not check_parameters(
            model,
            "after optimizer.step()"
        ):
            print()
            print(
                "❌ AdamW introduced NaN/INF."
            )

            return

        print(
            "Parameters after update: FINITE"
        )

    print()
    print("========================================")
    print("✅ ADAMW PASSED 10 STEPS")
    print("========================================")


if __name__ == "__main__":
    main()
