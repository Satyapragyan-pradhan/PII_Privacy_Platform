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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "transformer",
    "model",
    "deberta-pii",
)


# ---------------------------------------------------------
# DeBERTa PII NER Model
# ---------------------------------------------------------

class DeBERTaPIIExtractor:
    """
    Local inference wrapper for the fine-tuned DeBERTa
    token-classification model.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"DeBERTa model not found at: {self.model_path}"
            )

        print(f"Loading DeBERTa model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model.eval()

        # Use GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        print(f"DeBERTa loaded successfully on {self.device}")
        print(f"Number of labels: {self.model.config.num_labels}")

    # -----------------------------------------------------
    # Extract entities from text
    # -----------------------------------------------------

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Run NER inference on a piece of text.

        Returns:
            [
                {
                    "entity": "NAME",
                    "text": "Rahul Sharma",
                    "start": 11,
                    "end": 23,
                    "confidence": 0.98
                }
            ]
        """

        if not text or not text.strip():
            return []

        # -------------------------------------------------
        # Tokenize
        # -------------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Character offsets corresponding to each token
        offset_mapping = inputs.pop(
            "offset_mapping"
        )[0].tolist()

        # Move tensors to GPU / CPU
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1,
        )[0]

        confidences = torch.max(
            probabilities,
            dim=-1,
        ).values[0]

        # -------------------------------------------------
        # Convert BIO predictions into entities
        # -------------------------------------------------

        raw_entities = []

        current_entity = None
        current_confidences = []

        for i, (prediction, confidence) in enumerate(
            zip(
                predictions.tolist(),
                confidences.tolist(),
            )
        ):

            token_start, token_end = offset_mapping[i]

            # Ignore special tokens
            if token_start == token_end:
                continue

            label = self.model.config.id2label[
                prediction
            ]

            # ---------------------------------------------
            # Outside any entity
            # ---------------------------------------------

            if label == "O":

                if current_entity:
                    raw_entities.append(
                        {
                            **current_entity,
                            "confidences": current_confidences,
                        }
                    )

                    current_entity = None
                    current_confidences = []

                continue

            # ---------------------------------------------
            # Parse BIO label
            # ---------------------------------------------

            if "-" not in label:
                continue

            prefix, entity_type = label.split(
                "-",
                1,
            )

            # ---------------------------------------------
            # Beginning of new entity
            # ---------------------------------------------

            if prefix == "B":

                if current_entity:
                    raw_entities.append(
                        {
                            **current_entity,
                            "confidences": current_confidences,
                        }
                    )

                current_entity = {
                    "entity": entity_type,
                    "start": token_start,
                    "end": token_end,
                }

                current_confidences = [confidence]

            # ---------------------------------------------
            # Continuation of entity
            # ---------------------------------------------

            elif prefix == "I":

                if (
                    current_entity
                    and current_entity["entity"]
                    == entity_type
                ):
                    current_entity["end"] = token_end

                    current_confidences.append(
                        confidence
                    )

                else:
                    # Unexpected I- tag.
                    # Treat it as a new entity safely.

                    if current_entity:
                        raw_entities.append(
                            {
                                **current_entity,
                                "confidences":
                                    current_confidences,
                            }
                        )

                    current_entity = {
                        "entity": entity_type,
                        "start": token_start,
                        "end": token_end,
                    }

                    current_confidences = [confidence]

        # -------------------------------------------------
        # Add final entity
        # -------------------------------------------------

        if current_entity:
            raw_entities.append(
                {
                    **current_entity,
                    "confidences": current_confidences,
                }
            )

        # -------------------------------------------------
        # Clean entity spans
        # -------------------------------------------------

        entities = []

        for entity in raw_entities:

            start = entity["start"]
            end = entity["end"]

            entity_text = text[start:end]

            # Remove whitespace from beginning/end
            stripped_text = entity_text.strip()

            if not stripped_text:
                continue

            # Adjust character offsets after stripping
            leading_spaces = len(
                entity_text
            ) - len(
                entity_text.lstrip()
            )

            trailing_spaces = len(
                entity_text
            ) - len(
                entity_text.rstrip()
            )

            start += leading_spaces
            end -= trailing_spaces

            confidences = entity["confidences"]

            if confidences:
                confidence = sum(
                    confidences
                ) / len(confidences)
            else:
                confidence = 0.0

            entities.append(
                {
                    "entity": entity["entity"],
                    "start": start,
                    "end": end,
                    "confidence": round(
                        float(confidence),
                        4,
                    ),
                    "text": text[start:end],
                }
            )

        # -------------------------------------------------
        # Merge adjacent same-type entities
        #
        # Example:
        #
        # Rahul  -> NAME
        # Sharma -> NAME
        #
        # becomes:
        #
        # Rahul Sharma -> NAME
        # -------------------------------------------------

        merged_entities = []

        for entity in entities:

            if not merged_entities:
                merged_entities.append(entity)
                continue

            previous = merged_entities[-1]

            gap = text[
                previous["end"]:
                entity["start"]
            ]

            same_entity_type = (
                previous["entity"]
                == entity["entity"]
            )

            whitespace_gap = (
                gap.strip() == ""
            )

            if (
                same_entity_type
                and whitespace_gap
            ):
                # Merge spans

                previous["end"] = entity["end"]

                previous["text"] = text[
                    previous["start"]:
                    previous["end"]
                ]

                # Weighted-ish average based on
                # character span length
                previous_length = max(
                    1,
                    previous["end"]
                    - previous["start"]
                    - (
                        entity["end"]
                        - entity["start"]
                    ),
                )

                current_length = max(
                    1,
                    entity["end"]
                    - entity["start"],
                )

                previous["confidence"] = round(
                    (
                        previous["confidence"]
                        * previous_length
                        + entity["confidence"]
                        * current_length
                    )
                    / (
                        previous_length
                        + current_length
                    ),
                    4,
                )

            else:
                merged_entities.append(entity)

        return merged_entities


# ---------------------------------------------------------
# Singleton model instance
# ---------------------------------------------------------

_extractor = None


def get_extractor() -> DeBERTaPIIExtractor:
    """
    Load the model once and reuse it.
    """

    global _extractor

    if _extractor is None:
        _extractor = DeBERTaPIIExtractor()

    return _extractor


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def extract_pii(
    text: str,
) -> List[Dict[str, Any]]:
    """
    Simple public interface.

    Example:
        entities = extract_pii(text)
    """

    extractor = get_extractor()

    return extractor.extract(text)