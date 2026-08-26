import re
from typing import List, Dict, Any


PAN_PATTERN = re.compile(
    r"^[A-Z]{5}[0-9]{4}[A-Z]$"
)

AADHAAR_PATTERN = re.compile(
    r"^\d{4}\s?\d{4}\s?\d{4}$"
)

PHONE_PATTERN = re.compile(
    r"^[6-9]\d{9}$"
)

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)


def normalize_value(value: str) -> str:
    return " ".join(
        str(value).strip().split()
    )


def validate_entity(
    entity_type: str,
    value: str
) -> bool:

    value = normalize_value(value)

    if not value:
        return False

    if entity_type == "PAN":
        return bool(
            PAN_PATTERN.fullmatch(
                value.upper()
            )
        )

    if entity_type == "Aadhaar":
        return bool(
            AADHAAR_PATTERN.fullmatch(
                value
            )
        )

    if entity_type == "Phone":

        digits = re.sub(
            r"\D",
            "",
            value
        )

        return bool(
            PHONE_PATTERN.fullmatch(
                digits
            )
        )

    if entity_type == "Email":
        return bool(
            EMAIL_PATTERN.fullmatch(
                value
            )
        )

    # Contextual entities cannot be
    # completely validated using regex.
    if entity_type in {
        "Name",
        "Address",
        "Date of Birth",
        "Driving Licence",
        "Voter ID"
    }:
        return True

    return False


def _same_entity(
    entity1: Dict[str, Any],
    entity2: Dict[str, Any]
) -> bool:

    type1 = str(
        entity1.get("type", "")
    ).lower()

    type2 = str(
        entity2.get("type", "")
    ).lower()

    value1 = normalize_value(
        entity1.get("value", "")
    ).lower()

    value2 = normalize_value(
        entity2.get("value", "")
    ).lower()

    return (
        type1 == type2
        and value1 == value2
    )


def reconcile_entities(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reconcile candidates produced by regex,
    NLP and contextual/LLM extraction.

    Deterministic entities are preferred when
    multiple methods produce the same entity.
    """

    final_entities = []

    for candidate in entities:

        candidate = candidate.copy()

        candidate.setdefault(
            "source",
            "unknown"
        )

        candidate_type = candidate.get(
            "type",
            ""
        )

        candidate_value = candidate.get(
            "value",
            ""
        )

        # Validate format
        format_valid = validate_entity(
            candidate_type,
            candidate_value
        )

        candidate["format_valid"] = format_valid

        # Check whether another extraction
        # method produced the same entity.
        agreement = False

        for existing in final_entities:

            if _same_entity(
                candidate,
                existing
            ):

                agreement = True

                # Mark agreement on both entities
                existing["methods_agree"] = True

                # Prefer deterministic source
                if (
                    existing.get("source")
                    != "regex"
                    and candidate.get("source")
                    == "regex"
                ):
                    existing.update(
                        candidate
                    )

                break

        if agreement:
            continue

        candidate["methods_agree"] = False

        final_entities.append(
            candidate
        )

    return final_entities