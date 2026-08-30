import json
import re
import requests

from core.config import settings


SYSTEM_PROMPT = """
You are a PII extraction system.

Extract only:
- Name
- Address
- Date of Birth

Return ONLY valid JSON:

{
    "entities": [
        {
            "type": "Name",
            "value": "..."
        }
    ]
}

Rules:
- Extract only information explicitly present in the document.
- Do not invent information.
- Do not infer information that is not present.
- Do not extract unrelated dates.
- Do not extract unrelated locations.
- Only extract an address when clearly associated with the person.
- Only extract a date as Date of Birth when clearly identified as DOB.
- If nothing is found, return {"entities": []}.
"""


def extract_json(text: str) -> dict:

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return {
            "entities": []
        }

    try:
        parsed = json.loads(
            match.group(0)
        )

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    return {
        "entities": []
    }


def find_entity_span(
    text: str,
    value: str,
    entity_type: str
):

    # Exact match
    match = re.search(
        re.escape(value),
        text,
        re.IGNORECASE
    )

    if match:
        return match.start(), match.end()

    # Normalize whitespace
    normalized_value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    if not normalized_value:
        return None, None

    # Flexible whitespace
    parts = normalized_value.split()

    pattern = r"\s+".join(
        re.escape(part)
        for part in parts
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.start(), match.end()

    return None, None


def contextual_extract(text: str):

    prompt = f"""
{SYSTEM_PROMPT}

DOCUMENT TEXT:

{text}
"""

    try:

        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        parsed = extract_json(
            result.get(
                "response",
                ""
            )
        )

        entities = []

        for entity in parsed.get(
            "entities",
            []
        ):

            if not isinstance(
                entity,
                dict
            ):
                continue

            entity_type = entity.get(
                "type"
            )

            value = entity.get(
                "value"
            )

            if entity_type not in {
                "Name",
                "Address",
                "Date of Birth"
            }:
                continue

            if not isinstance(
                value,
                str
            ):
                continue

            value = value.strip()

            if not value:
                continue

            start, end = find_entity_span(
                text,
                value,
                entity_type
            )

            entity_data = {
                "type": entity_type,
                "value": value,
                "confidence": 0.82,
                "source": "llm/context",
                "validated": False
            }

            if start is not None:
                entity_data["start"] = start
                entity_data["end"] = end

            entities.append(
                entity_data
            )

        return entities

    except Exception as exc:

        print(
            f"Contextual extraction failed: {exc}"
        )

        return []