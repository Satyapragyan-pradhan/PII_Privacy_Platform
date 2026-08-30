from transformer.inference import extract_pii


def extract_nlp_entities(text: str):

    try:
        entities = extract_pii(text)

        for entity in entities:
            entity["type"] = str(
                entity.get("type", "")
            ).upper()

            entity["source"] = "deberta"

            entity.setdefault(
                "validated",
                False
            )

        return entities

    except Exception as exc:

        print(
            f"DeBERTa extraction failed: {exc}"
        )

        return []