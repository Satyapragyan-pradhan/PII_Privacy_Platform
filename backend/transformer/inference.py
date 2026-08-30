import os
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "transformer",
    "model",
    "deberta-pii",
)


# ---------------------------------------------------------
# DeBERTa PII NER
# ---------------------------------------------------------

class DeBERTaPIIExtractor:

    def __init__(
        self,
        model_path: str = MODEL_PATH
    ):

        self.model_path = model_path

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"DeBERTa model not found at: "
                f"{self.model_path}"
            )

        print(
            f"Loading DeBERTa model from: "
            f"{self.model_path}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"DeBERTa loaded successfully "
            f"on {self.device}"
        )

        print(
            f"Number of labels: "
            f"{self.model.config.num_labels}"
        )


    # -----------------------------------------------------
    # Extract entities
    # -----------------------------------------------------

    def extract(
        self,
        text: str
    ) -> List[Dict[str, Any]]:

        if not text or not text.strip():
            return []

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = (
            encoded.pop("offset_mapping")[0]
            .tolist()
        )

        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )[0]

        confidences = torch.max(
            probabilities,
            dim=-1
        ).values[0]

        entities = []
        current_entity = None

        # -------------------------------------------------
        # BIO decoding
        # -------------------------------------------------

        for i, (
            prediction,
            confidence
        ) in enumerate(
            zip(
                predictions.tolist(),
                confidences.tolist()
            )
        ):

            start, end = offset_mapping[i]

            # Special tokens
            if start == end:
                continue

            label = self.model.config.id2label[
                prediction
            ]

            if label == "O":

                if current_entity:
                    entities.append(
                        current_entity
                    )

                current_entity = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split(
                "-",
                1
            )

            # -------------------------------------------------
            # Beginning of entity
            # -------------------------------------------------

            if prefix == "B":

                if current_entity:
                    entities.append(
                        current_entity
                    )

                current_entity = {
                    "type": entity_type,
                    "value": text[start:end],
                    "start": start,
                    "end": end,
                    "confidence": float(
                        confidence
                    ),
                    "source": "deberta",
                    "validated": False,
                }

            # -------------------------------------------------
            # Continuation
            # -------------------------------------------------

            elif prefix == "I":

                if (
                    current_entity
                    and current_entity["type"]
                    == entity_type
                ):

                    current_entity["end"] = end

                    current_entity[
                        "value"
                    ] = text[
                        current_entity["start"]:
                        end
                    ]

                    # Conservative confidence:
                    # keep the lowest token confidence.
                    current_entity[
                        "confidence"
                    ] = min(
                        current_entity[
                            "confidence"
                        ],
                        float(confidence)
                    )

                else:

                    # Invalid I- without matching B-
                    if current_entity:
                        entities.append(
                            current_entity
                        )

                    current_entity = {
                        "type": entity_type,
                        "value": text[start:end],
                        "start": start,
                        "end": end,
                        "confidence": float(
                            confidence
                        ),
                        "source": "deberta",
                        "validated": False,
                    }

        if current_entity:
            entities.append(
                current_entity
            )

        # -----------------------------------------------------
        # Final cleanup
        # -----------------------------------------------------

        cleaned = []

        for entity in entities:

            value = entity["value"].strip()

            if not value:
                continue

            entity["value"] = value

            entity["confidence"] = round(
                float(entity["confidence"]),
                4
            )

            cleaned.append(entity)

        return cleaned


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_extractor = None


def get_extractor():

    global _extractor

    if _extractor is None:
        _extractor = (
            DeBERTaPIIExtractor()
        )

    return _extractor


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def extract_pii(
    text: str
) -> List[Dict[str, Any]]:

    return get_extractor().extract(text)