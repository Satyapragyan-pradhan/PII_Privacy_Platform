import re


def validate_entity(entity):
    entity_type = entity.get("type")
    value = entity.get("value", "")

    if not value:
        return False

    if entity_type == "PAN":
        return bool(
            re.fullmatch(
                r"[A-Z]{5}[0-9]{4}[A-Z]",
                value.upper()
            )
        )

    if entity_type == "Aadhaar":
        digits = re.sub(
            r"\D",
            "",
            value
        )

        return len(digits) == 12

    if entity_type == "Phone":
        digits = re.sub(
            r"\D",
            "",
            value
        )

        if digits.startswith("91"):
            digits = digits[-10:]

        return (
            len(digits) == 10
            and digits[0] in "6789"
        )

    return True