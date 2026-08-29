from typing import List, Dict, Any

from transformer.inference import extract_pii


# ---------------------------------------------------------
# DeBERTa contextual / NER extraction
# ---------------------------------------------------------

def contextual_extract(text: str) -> List[Dict[str, Any]]:
    """
    Extract PII entities using the fine-tuned DeBERTa model.

    DeBERTa is used as the ML-based contextual extractor.
    Regex remains responsible for deterministic entities.

    Returns entities in the common pipeline format:

    {
        "value": "...",
        "type": "...",
        "start": 0,
        "end": 10,
        "confidence": 0.98,
        "source": "deberta",
        "validated": False
    }
    """

    if not text or not text.strip():
        return []

    try:
        predictions = extract_pii(text)

        entities = []

        for prediction in predictions:

            entity_type = prediction.get(
                "entity",
                ""
            )

            value = prediction.get(
                "text",
                ""
            )

            if not entity_type or not value:
                continue

            entities.append({
                "value": value.strip(),
                "type": entity_type,
                "start": prediction.get("start"),
                "end": prediction.get("end"),
                "confidence": prediction.get(
                    "confidence",
                    0.0
                ),
                "source": "deberta",
                "validated": False
            })

        return entities

    except Exception as exc:

        print(
            f"DeBERTa contextual extraction failed: {exc}"
        )

        return []