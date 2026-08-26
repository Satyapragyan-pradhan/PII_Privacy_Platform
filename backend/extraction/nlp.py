import spacy


try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def extract_nlp_entities(text: str):
    if nlp is None:
        return []

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            entities.append({
                "value": ent.text.strip(),
                "type": "Name",
                "confidence": 0.70,
                "source": "spacy_ner",
                "validated": False
            })

        elif ent.label_ in {
            "GPE",
            "LOC",
            "FAC"
        }:
            entities.append({
                "value": ent.text.strip(),
                "type": "Address",
                "confidence": 0.55,
                "source": "spacy_ner",
                "validated": False
            })

    return entities