import json
import requests
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "pii-llama"


def extract_context_pii(text: str, candidates: list) -> list:
    prompt = f"""
You are a PII extraction and contextual reasoning system.

Extract ONLY PII belonging to the primary person described in the document.

PII TYPES:
NAME
ADDRESS
DOB
PAN
AADHAAR
PHONE
EMAIL
DRIVING_LICENCE
VOTER_ID

Rules:
1. Use the document context, not just patterns.
2. Ignore dates that are not the person's DOB.
3. Distinguish the person's address from addresses mentioned as examples,
   offices, publications, historical records, etc.
4. Correct obvious OCR errors such as "l" instead of "1" when context
   clearly indicates the intended PII.
5. If multiple addresses belong to the same person, return all of them.
6. Do not invent information.
7. Return ONLY valid JSON.
8. Use the exact value as it appears in the text whenever possible.
9. If an entity was detected by Regex or DeBERTa but context indicates it
   belongs to somebody else or is unrelated, reject it.

Existing model candidates:
{json.dumps(candidates, ensure_ascii=False)}

DOCUMENT:
{text}

Return:
[
  {{
    "type": "NAME|ADDRESS|DOB|PAN|AADHAAR|PHONE|EMAIL|DRIVING_LICENCE|VOTER_ID",
    "value": "...",
    "confidence": 0.0,
    "source": "llm"
  }}
]
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120
        )

        response.raise_for_status()

        raw = response.json().get("response", "").strip()

        if not raw:
            return []

        result = json.loads(raw)

        if isinstance(result, dict):
            result = result.get("entities", [])

        if not isinstance(result, list):
            return []

        cleaned = []

        for entity in result:
            if not isinstance(entity, dict):
                continue

            if not entity.get("type") or not entity.get("value"):
                continue

            cleaned.append({
                "type": str(entity["type"]).upper(),
                "value": str(entity["value"]).strip(),
                "confidence": float(entity.get("confidence", 0.85)),
                "source": "llm"
            })

        return cleaned

    except Exception as e:
        print(f"Context LLM error: {e}")
        return []