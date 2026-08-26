from extraction.regex import extract_regex_entities


def test_pan_extraction():

    text = "PAN: ABCDE1234F"

    entities = extract_regex_entities(
        text
    )

    assert any(
        entity["type"] == "PAN"
        and entity["value"] == "ABCDE1234F"
        for entity in entities
    )


def test_phone_extraction():

    text = "Phone: 9876543210"

    entities = extract_regex_entities(
        text
    )

    assert any(
        entity["type"] == "Phone"
        and entity["value"] == "9876543210"
        for entity in entities
    )


def test_email_extraction():

    text = "Email: test@example.com"

    entities = extract_regex_entities(
        text
    )

    assert any(
        entity["type"] == "Email"
        and entity["value"] == "test@example.com"
        for entity in entities
    )