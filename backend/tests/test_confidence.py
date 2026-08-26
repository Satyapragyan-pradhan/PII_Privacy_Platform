from services.confidence import calculate_confidence


def test_confidence_multiple_sources():

    entity = {
        "confidence": 0.90,
        "source": "multiple",
        "validated": True
    }

    score = calculate_confidence(
        entity
    )

    assert score > 0.90
    assert score <= 0.99