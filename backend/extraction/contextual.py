import json
import re

import requests

from core.config import settings


SYSTEM_PROMPT = """
You are a PII extraction system.

Extract only the following entities:

- Name
- Address
- Date of Birth

Return ONLY valid JSON in this format:

{
  "entities": [
    {
      "type": "Name",
      "value": "..."
    }
  ]
}

Do not invent information.
Only extract information explicitly present in the text.
"""


def extract_json(text: str):
    """
    Attempts to extract JSON from an LLM response.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return {"entities": []}

    try:
        return json.loads(
            match.group(0)
        )
    except json.JSONDecodeError:
        return {"entities": []}


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
            result.get("response", "")
        )

        entities = []

        for entity in parsed.get(
            "entities",
            []
        ):

            entity_type = entity.get("type")
            value = entity.get("value")

            if entity_type not in {
                "Name",
                "Address",
                "Date of Birth"
            }:
                continue

            if not value:
                continue

            entities.append({
                "value": value.strip(),
                "type": entity_type,
                "confidence": 0.82,
                "source": "llm/context",
                "validated": False
            })

        return entities

    except Exception:
        return []