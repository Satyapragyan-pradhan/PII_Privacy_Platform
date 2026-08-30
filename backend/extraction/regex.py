import re

from typing import List, Dict, Any


# =========================================================
# Indian State / UT Codes
# =========================================================

INDIAN_DL_STATE_CODES = (
    "AN|AP|AR|AS|BR|CH|CG|DD|DL|DN|GA|GJ|HR|HP|"
    "JK|JH|KA|KL|LA|LD|MP|MH|MN|ML|MZ|NL|OD|OR|"
    "PY|PB|RJ|SK|TN|TS|TR|UP|UK|WB"
)


# =========================================================
# Deterministic PII Patterns
# =========================================================

PATTERNS = {

    # -----------------------------------------------------
    # Aadhaar
    # -----------------------------------------------------
    "Aadhaar": re.compile(
        r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
    ),

    # -----------------------------------------------------
    # PAN
    # -----------------------------------------------------
    "PAN": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        re.IGNORECASE
    ),

    # -----------------------------------------------------
    # Indian Phone Number
    #
    # Supports:
    #   9876543210
    #   +91 9876543210
    #   +91-9876543210
    #   91 9876543210
    #
    # Does NOT allow the number to be immediately preceded
    # by another alphanumeric character.
    # -----------------------------------------------------
    "Phone": re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:\+91[\s-]?)?"
        r"[6-9]\d{9}"
        r"(?!\d)"
    ),

    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------
    "Email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    # -----------------------------------------------------
    # Indian Voter ID
    # -----------------------------------------------------
    "Voter ID": re.compile(
        r"\b[A-Z]{3}[0-9]{7}\b",
        re.IGNORECASE
    ),

    # -----------------------------------------------------
    # Indian Driving Licence
    #
    # Examples:
    #   OD0120230012345
    #   OD-01-20230012345
    #   OD 01 20230012345
    #
    # IMPORTANT:
    #   - Must start with valid Indian state/UT code
    #   - Must contain RTO code
    #   - Must contain licence number
    #   - Cannot be immediately attached to another
    #     alphanumeric character
    # -----------------------------------------------------
    "Driving Licence": re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"(?:{INDIAN_DL_STATE_CODES})"
        rf"[-\s]?"
        rf"\d{{2}}"
        rf"[-\s]?"
        rf"\d{{4,13}}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE
    ),
}


# =========================================================
# Date Pattern
# =========================================================

DATE_PATTERN = re.compile(
    r"\b"
    r"(?:0?[1-9]|[12]\d|3[01])"
    r"[/-]"
    r"(?:0?[1-9]|1[0-2])"
    r"[/-]"
    r"(?:19|20)\d{2}"
    r"\b"
)


# =========================================================
# DOB Context Pattern
#
# We ONLY classify a date as DOB when the surrounding
# context explicitly indicates a birth date.
#
# Examples accepted:
#
#   date of birth is 15/08/2004
#   date of birth: 15/08/2004
#   DOB: 20/01/1999
#   born on 05/06/2000
#   birth date = 10/10/1998
#   birthdate was 10/10/1998
#
# Examples rejected:
#
#   published on 12/03/2021
#   records from 15-08-2010
#   meeting on 20/01/1999
# =========================================================

DOB_CONTEXT_PATTERN = re.compile(
    r"""
    (?:
        \bdate\s+of\s+birth\b
        |
        \bdob\b
        |
        \bborn\s+on\b
        |
        \bbirth\s+date\b
        |
        \bbirthdate\b
    )

    \s*
    (?:
        :
        |
        -
        |
        =
        |
        \bis\b
        |
        \bwas\b
    )?
    \s*

    (?P<date>
        (?:0?[1-9]|[12]\d|3[01])
        [/-]
        (?:0?[1-9]|1[0-2])
        [/-]
        (?:19|20)\d{2}
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


# =========================================================
# Normalization
# =========================================================

def normalize_regex_value(
    entity_type: str,
    value: str
) -> str:

    value = str(value).strip()

    if entity_type in {
        "Aadhaar",
        "Phone",
        "Driving Licence"
    }:
        value = re.sub(
            r"[\s-]",
            "",
            value
        )

    if entity_type in {
        "PAN",
        "Voter ID",
        "Driving Licence"
    }:
        value = value.upper()

    return value


# =========================================================
# Regex Extraction
# =========================================================

def extract_regex_entities(
    text: str
) -> List[Dict[str, Any]]:

    if not text or not text.strip():
        return []

    entities = []

    # -----------------------------------------------------
    # 1. Deterministic identifiers
    # -----------------------------------------------------

    for entity_type, pattern in PATTERNS.items():

        for match in pattern.finditer(text):

            value = normalize_regex_value(
                entity_type,
                match.group(0)
            )

            entities.append({
                "type": entity_type,
                "value": value,
                "confidence": 0.97,
                "source": "regex",
                "start": match.start(),
                "end": match.end(),
                "validated": True
            })

    # -----------------------------------------------------
    # 2. DOB
    #
    # IMPORTANT:
    # Do NOT run DATE_PATTERN independently.
    #
    # Only dates appearing in DOB_CONTEXT_PATTERN
    # become DOB entities.
    # -----------------------------------------------------

    for match in DOB_CONTEXT_PATTERN.finditer(text):

        date_match = match.group("date")

        # Calculate exact span of the date itself,
        # not the "date of birth is" context.
        date_start = match.start("date")
        date_end = match.end("date")

        entities.append({
            "type": "DOB",
            "value": date_match,
            "confidence": 0.97,
            "source": "regex",
            "start": date_start,
            "end": date_end,
            "validated": True
        })

    # -----------------------------------------------------
    # 3. Stable ordering
    # -----------------------------------------------------

    entities.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"]
        )
    )

    return entities