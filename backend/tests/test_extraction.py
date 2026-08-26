from services.reconciliation import reconcile_entities


def test_reconciliation():

    entities = [
        {
            "type": "PAN",
            "value": "ABCDE1234F",
            "confidence": 0.97,
            "source": "regex",
            "validated": True
        },
        {
            "type": "PAN",
            "value": "ABCDE1234F",
            "confidence": 0.82,
            "source": "llm/context",
            "validated": False
        }
    ]

    result = reconcile_entities(
        entities
    )

    assert len(result) == 1
    assert result[0]["source"] == "multiple"
    assert result[0]["validated"] is True