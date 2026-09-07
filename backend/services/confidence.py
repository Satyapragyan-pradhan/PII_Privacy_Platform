def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:

    return max(
        minimum,
        min(value, maximum)
    )


def calculate_confidence(entity) -> float:
    """
    Calculate final confidence using:
    - model/source confidence
    - format validation
    - agreement between extraction methods
    """

    source = str(
        entity.get("source", "unknown")
    ).lower()

    format_valid = entity.get(
        "format_valid",
        False
    )

    methods_agree = entity.get(
        "methods_agree",
        False
    )

    raw_confidence = float(
        entity.get("confidence", 0.0)
    )

    # -----------------------------
    # Deterministic regex extraction
    # -----------------------------

    if source == "regex":

        score = 0.90

        if format_valid:
            score += 0.05

        if methods_agree:
            score += 0.05

    # -----------------------------
    # OCR + regex
    # -----------------------------

    elif source == "ocr+regex":

        score = 0.75

        if format_valid:
            score += 0.10

        if methods_agree:
            score += 0.10

    # -----------------------------
    # DeBERTa transformer
    # -----------------------------

    elif source == "deberta":

        score = raw_confidence

        if format_valid:
            score += 0.05

        if methods_agree:
            score += 0.10

    # -----------------------------
    # NLP
    # -----------------------------

    elif source == "nlp":

        score = raw_confidence or 0.65

        if methods_agree:
            score += 0.10

        if format_valid:
            score += 0.05

    # -----------------------------
    # LLM / contextual
    # -----------------------------

    elif "llm" in source or "context" in source:

        score = raw_confidence or 0.70

        if methods_agree:
            score += 0.10

        if format_valid:
            score += 0.05

    # -----------------------------
    # Unknown
    # -----------------------------

    else:

        score = 0.50

        if methods_agree:
            score += 0.10

        if format_valid:
            score += 0.10

    return round(
        clamp(score),
        2
    )


def confidence_label(
    score: float
) -> str:

    if score >= 0.90:
        return "high"

    if score >= 0.70:
        return "medium"

    return "low"