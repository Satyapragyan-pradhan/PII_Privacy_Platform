import re
from datetime import datetime
from typing import List, Dict, Any


# =========================================================
# Canonical Types
# =========================================================

TYPE_ALIASES = {
    "NAME": "NAME",
    "ADDRESS": "ADDRESS",
    "DOB": "DOB",
    "DATE OF BIRTH": "DOB",
    "PAN": "PAN",
    "AADHAAR": "AADHAAR",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "DRIVING LICENCE": "DRIVING_LICENCE",
    "DRIVING_LICENCE": "DRIVING_LICENCE",
    "VOTER ID": "VOTER_ID",
    "VOTER_ID": "VOTER_ID",
}


def canonical_type(entity_type: str) -> str:
    key = str(entity_type or "").strip().upper()
    return TYPE_ALIASES.get(
        key,
        key.replace(" ", "_")
    )


# =========================================================
# Validation Patterns
# =========================================================

PAN_PATTERN = re.compile(
    r"^[A-Z]{5}[0-9]{4}[A-Z]$"
)

AADHAAR_PATTERN = re.compile(
    r"^[2-9]\d{11}$"
)

PHONE_PATTERN = re.compile(
    r"^[6-9]\d{9}$"
)

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

DL_PATTERN = re.compile(
    r"^[A-Z]{2}\d{2}\d{4,13}$"
)

VOTER_PATTERN = re.compile(
    r"^[A-Z]{3}\d{7}$"
)

DOB_PATTERNS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
]


DL_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL",
    "DN", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL",
    "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD",
    "OR", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UP",
    "UK", "WB"
}


# =========================================================
# Basic Normalization
# =========================================================

def normalize(value: Any) -> str:
    return " ".join(
        str(value or "").strip().split()
    )


def compact(value: Any) -> str:
    return re.sub(
        r"[\s-]",
        "",
        normalize(value)
    )


# =========================================================
# Phone Normalization
# =========================================================

def normalize_phone(value: str) -> str:

    phone = re.sub(
        r"\D",
        "",
        normalize(value)
    )

    # +91XXXXXXXXXX -> XXXXXXXXXX
    if phone.startswith("91") and len(phone) == 12:
        phone = phone[2:]

    return phone


# =========================================================
# DOB Normalization
# =========================================================

def normalize_dob(value: str):

    value = normalize(value)

    for pattern in DOB_PATTERNS:

        try:

            date = datetime.strptime(
                value,
                pattern
            )

            # Canonical representation
            return date.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            continue

    return value


# =========================================================
# Validation
# =========================================================

def validate_entity(
    entity_type: str,
    value: str
) -> bool:

    t = canonical_type(entity_type)
    v = normalize(value)

    if not v:
        return False

    if t == "PAN":

        return bool(
            PAN_PATTERN.fullmatch(
                v.upper()
            )
        )

    if t == "AADHAAR":

        return bool(
            AADHAAR_PATTERN.fullmatch(
                compact(v)
            )
        )

    if t == "PHONE":

        return bool(
            PHONE_PATTERN.fullmatch(
                normalize_phone(v)
            )
        )

    if t == "EMAIL":

        return bool(
            EMAIL_PATTERN.fullmatch(v)
        )

    if t == "DRIVING_LICENCE":

        v = compact(v).upper()

        return bool(
            DL_PATTERN.fullmatch(v)
            and v[:2] in DL_STATE_CODES
        )

    if t == "VOTER_ID":

        return bool(
            VOTER_PATTERN.fullmatch(
                compact(v).upper()
            )
        )

    # Semantic entities are handled by
    # DeBERTa / contextual model.
    if t in {
        "NAME",
        "ADDRESS",
        "DOB"
    }:

        return True

    return False


# =========================================================
# Entity Key
# =========================================================

def entity_key(
    entity: Dict[str, Any]
):

    t = canonical_type(
        entity.get("type", "")
    )

    value = normalize(
        entity.get("value", "")
    )

    # Phone:
    # 9876543210
    # +91 9876543210
    # +919876543210
    # become identical.
    if t == "PHONE":

        value = normalize_phone(
            value
        )

    # DOB:
    # 15/08/2004
    # 15-08-2004
    # become identical.
    elif t == "DOB":

        value = normalize_dob(
            value
        )

    # Numeric identifiers
    elif t in {
        "AADHAAR",
        "DRIVING_LICENCE",
        "VOTER_ID"
    }:

        value = compact(value)

    else:

        value = value.lower()

    return t, value


# =========================================================
# Source Priority
# =========================================================

def source_priority(
    entity: Dict[str, Any]
) -> int:

    source = str(
        entity.get(
            "source",
            ""
        )
    ).lower()

    if source == "regex":
        return 3

    if source == "deberta":
        return 2

    if "llm" in source:
        return 1

    return 0


# =========================================================
# Choose Better Entity
# =========================================================

def choose_better(
    a: Dict[str, Any],
    b: Dict[str, Any]
):

    pa = source_priority(a)
    pb = source_priority(b)

    if pb > pa:
        return b.copy()

    if pa > pb:
        return a.copy()

    ca = float(
        a.get("confidence", 0)
    )

    cb = float(
        b.get("confidence", 0)
    )

    if cb > ca:
        return b.copy()

    return a.copy()


# =========================================================
# Span Overlap
# =========================================================

def spans_overlap(
    a: Dict[str, Any],
    b: Dict[str, Any]
) -> bool:

    s1 = a.get("start")
    e1 = a.get("end")

    s2 = b.get("start")
    e2 = b.get("end")

    if None in (
        s1,
        e1,
        s2,
        e2
    ):
        return False

    return (
        s1 < e2
        and s2 < e1
    )


# =========================================================
# Name Fragment Merge
# =========================================================

def merge_name_fragments(
    entities
):

    names = [
        e for e in entities
        if canonical_type(
            e.get("type")
        ) == "NAME"
        and e.get("start") is not None
        and e.get("end") is not None
    ]

    others = [
        e for e in entities
        if canonical_type(
            e.get("type")
        ) != "NAME"
    ]

    names.sort(
        key=lambda e: e["start"]
    )

    merged = []

    for current in names:

        current = current.copy()

        if not merged:

            merged.append(current)
            continue

        previous = merged[-1]

        # Example:
        # Rhu1 + Sharma
        if (
            current["start"]
            - previous["end"]
            <= 1
        ):

            previous["value"] = (
                normalize(
                    previous["value"]
                )
                + " "
                + normalize(
                    current["value"]
                )
            )

            previous["end"] = (
                current["end"]
            )

            previous["confidence"] = min(
                float(
                    previous.get(
                        "confidence",
                        0
                    )
                ),
                float(
                    current.get(
                        "confidence",
                        0
                    )
                )
            )

        else:

            merged.append(current)

    return others + merged


# =========================================================
# Address Fragment Merge
# =========================================================

def merge_address_fragments(
    entities
):

    addresses = [
        e for e in entities
        if canonical_type(
            e.get("type")
        ) == "ADDRESS"
        and e.get("start") is not None
        and e.get("end") is not None
    ]

    others = [
        e for e in entities
        if canonical_type(
            e.get("type")
        ) != "ADDRESS"
    ]

    addresses.sort(
        key=lambda e: e["start"]
    )

    merged = []

    for current in addresses:

        current = current.copy()

        if not merged:

            merged.append(current)
            continue

        previous = merged[-1]

        if (
            current["start"]
            - previous["end"]
            <= 2
        ):

            previous["value"] = (
                normalize(
                    previous["value"]
                )
                + " "
                + normalize(
                    current["value"]
                )
            )

            previous["end"] = (
                current["end"]
            )

            previous["confidence"] = min(
                float(
                    previous.get(
                        "confidence",
                        0
                    )
                ),
                float(
                    current.get(
                        "confidence",
                        0
                    )
                )
            )

        else:

            merged.append(current)

    return others + merged


# =========================================================
# Main Reconciliation
# =========================================================

def reconcile_entities(
    entities: List[Dict[str, Any]],
    text: str = ""
) -> List[Dict[str, Any]]:

    if not entities:
        return []

    # -----------------------------------------------------
    # 1. Normalize + validate
    # -----------------------------------------------------

    candidates = []

    for entity in entities:
     

     e = entity.copy()

     e["type"] = canonical_type(
        e.get("type", e.get("entity", ""))
    )

     e["value"] = normalize(
        e.get("value", e.get("text", ""))
    )

     if not e["type"] or not e["value"]:
        continue

     e["format_valid"] = validate_entity(
        e["type"],
        e["value"]
    )

    # -----------------------------------------------------
    # Reject invalid model predictions
    # -----------------------------------------------------

     if not e["format_valid"]:
        continue

     candidates.append(e)

    # -----------------------------------------------------
    # 2. Reconcile identical entities
    # -----------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = entity_key(
            candidate
        )

        if key not in unique:

            candidate[
                "methods_agree"
            ] = False

            unique[key] = candidate

            continue

        existing = unique[key]

        better = choose_better(
            existing,
            candidate
        )

        # Multiple extraction methods
        # found the same entity.
        better[
            "methods_agree"
        ] = True

        # If either method validated
        # the entity, preserve validation.
        better[
            "format_valid"
        ] = (
            existing.get(
                "format_valid",
                False
            )
            or candidate.get(
                "format_valid",
                False
            )
        )

        unique[key] = better

    reconciled = list(
        unique.values()
    )

    # -----------------------------------------------------
    # 3. Merge model fragments
    # -----------------------------------------------------

    reconciled = merge_name_fragments(
        reconciled
    )

    reconciled = merge_address_fragments(
        reconciled
    )

    # -----------------------------------------------------
    # 4. Resolve overlapping entities
    # -----------------------------------------------------

    reconciled.sort(
        key=lambda e: (
            e.get(
                "start",
                10**9
            ),
            -source_priority(e),
            -float(
                e.get(
                    "confidence",
                    0
                )
            )
        )
    )

    final = []

    for candidate in reconciled:

        conflict = False

        for existing in final:

            if not spans_overlap(
                candidate,
                existing
            ):
                continue

            # Same normalized entity
            if (
                entity_key(candidate)
                == entity_key(existing)
            ):

                conflict = True
                break

            # Same type + overlapping span
            # -> stronger source wins.
            if (
                canonical_type(
                    candidate["type"]
                )
                ==
                canonical_type(
                    existing["type"]
                )
            ):

                if (
                    source_priority(
                        existing
                    )
                    >=
                    source_priority(
                        candidate
                    )
                ):

                    conflict = True
                    break

        if not conflict:

            final.append(candidate)

    # -----------------------------------------------------
    # 5. Document order
    # -----------------------------------------------------

    final.sort(
        key=lambda e: e.get(
            "start",
            10**9
        )
    )

    return final