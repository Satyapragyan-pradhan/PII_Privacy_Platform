import re


PATTERNS = {
    "Aadhaar": re.compile(
        r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
    ),

    "PAN": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        re.IGNORECASE
    ),

    "Phone": re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    ),

    "Email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "DOB": re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])"
        r"[-/.]"
        r"(?:0?[1-9]|1[0-2])"
        r"[-/.]"
        r"(?:19|20)\d{2}\b"
    ),

    "Voter ID": re.compile(
        r"\b[A-Z]{3}[0-9]{7}\b",
        re.IGNORECASE
    ),

    "Driving Licence": re.compile(
        r"\b[A-Z]{2}[-\s]?"
        r"\d{2}[-\s]?"
        r"\d{4,13}\b",
        re.IGNORECASE
    )
}


def normalize_value(entity_type: str, value: str):
    value = value.strip()

    if entity_type in {
        "Aadhaar",
        "Phone",
        "Driving Licence"
    }:
        value = re.sub(
            r"[\s-]",
            "",
            value
        )

    if entity_type in {
        "PAN",
        "Voter ID"
    }:
        value = value.upper()

    return value


def extract_regex_entities(text: str):
    entities = []

    for entity_type, pattern in PATTERNS.items():

        matches = pattern.findall(text)

        for match in matches:

            if isinstance(match, tuple):
                value = match[0]
            else:
                value = match

            value = normalize_value(
                entity_type,
                value
            )

            entities.append({
                "value": value,
                "type": entity_type,
                "confidence": 0.97,
                "source": "regex",
                "validated": True
            })

    return entities