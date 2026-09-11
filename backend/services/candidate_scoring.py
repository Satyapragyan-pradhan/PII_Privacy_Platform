from typing import Dict, Any


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:

    return max(
        minimum,
        min(value, maximum)
    )


def context_bonus(
    context_score: float
) -> float:

    context_score = max(
        -1.0,
        min(
            float(context_score),
            1.0
        )
    )

    return 0.30 * context_score


def candidate_quality(
    entity: Dict[str, Any]
) -> float:

    confidence = float(
        entity.get(
            "confidence",
            0.0
        )
    )

    context_score = float(
        entity.get(
            "context_score",
            0.0
        )
    )

    format_valid = bool(
        entity.get(
            "format_valid",
            False
        )
    )

    methods_agree = bool(
        entity.get(
            "methods_agree",
            False
        )
    )

    quality = confidence

    quality += context_bonus(
        context_score
    )

    if format_valid:
        quality += 0.10

    if methods_agree:
        quality += 0.10

    return quality


def source_priority(
    entity: Dict[str, Any]
) -> int:

    source = str(
        entity.get(
            "source",
            ""
        )
    ).lower()

    entity_type = str(
        entity.get(
            "type",
            ""
        )
    ).upper()

    structured = {
        "PAN",
        "AADHAAR",
        "DRIVING_LICENCE",
        "VOTER_ID",
        "PHONE",
        "EMAIL",
    }

    semantic = {
        "NAME",
        "ADDRESS",
        "DOB",
    }

    if entity_type in structured:

        if source == "regex":
            return 3

        if source == "deberta":
            return 2

        if "llm" in source:
            return 1

    if entity_type in semantic:

        if source == "deberta":
            return 3

        if source == "regex":
            return 2

        if "llm" in source:
            return 1

    return 0


def choose_better(
    a: Dict[str, Any],
    b: Dict[str, Any]
) -> Dict[str, Any]:

    qa = candidate_quality(a)
    qb = candidate_quality(b)

    if qb > qa:
        return b.copy()

    if qa > qb:
        return a.copy()

    pa = source_priority(a)
    pb = source_priority(b)

    if pb > pa:
        return b.copy()

    if pa > pb:
        return a.copy()

    ca = float(
        a.get(
            "confidence",
            0.0
        )
    )

    cb = float(
        b.get(
            "confidence",
            0.0
        )
    )

    if cb > ca:
        return b.copy()

    return a.copy()