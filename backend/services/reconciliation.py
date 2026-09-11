import re

from datetime import datetime
from typing import List, Dict, Any

from services.candidate_scoring import (
    candidate_quality,
    choose_better,
)


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


def canonical_type(
    entity_type: str
) -> str:

    key = str(
        entity_type or ""
    ).strip().upper()

    return TYPE_ALIASES.get(
        key,
        key.replace(
            " ",
            "_"
        )
    )


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


def normalize(
    value: Any
) -> str:

    return " ".join(
        str(
            value or ""
        ).strip().split()
    )


def compact(
    value: Any
) -> str:

    return re.sub(
        r"[\s-]",
        "",
        normalize(value)
    )


def normalize_phone(
    value: str
) -> str:

    phone = re.sub(
        r"\D",
        "",
        normalize(value)
    )

    if (
        phone.startswith("91")
        and len(phone) == 12
    ):
        phone = phone[2:]

    return phone


def normalize_dob(
    value: str
):

    value = normalize(
        value
    )

    for pattern in DOB_PATTERNS:

        try:

            date = datetime.strptime(
                value,
                pattern
            )

            return date.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            continue

    return value


def validate_entity(
    entity_type: str,
    value: str
) -> bool:

    t = canonical_type(
        entity_type
    )

    v = normalize(
        value
    )

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
            EMAIL_PATTERN.fullmatch(
                v
            )
        )

    if t == "DRIVING_LICENCE":

        v = compact(
            v
        ).upper()

        return bool(
            DL_PATTERN.fullmatch(
                v
            )
            and v[:2] in DL_STATE_CODES
        )

    if t == "VOTER_ID":

        return bool(
            VOTER_PATTERN.fullmatch(
                compact(v).upper()
            )
        )

    if t in {
        "NAME",
        "ADDRESS",
        "DOB"
    }:

        return True

    return False


def entity_key(
    entity: Dict[str, Any]
):

    t = canonical_type(
        entity.get(
            "type",
            ""
        )
    )

    value = normalize(
        entity.get(
            "value",
            ""
        )
    )

    if t == "PHONE":

        value = normalize_phone(
            value
        )

    elif t == "DOB":

        value = normalize_dob(
            value
        )

    elif t in {
        "AADHAAR",
        "DRIVING_LICENCE",
        "VOTER_ID"
    }:

        value = compact(
            value
        )

    else:

        value = value.lower()

    return t, value


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


def merge_name_fragments(
    entities
):

    names = [
        e
        for e in entities
        if canonical_type(
            e.get("type")
        ) == "NAME"
        and e.get("start") is not None
        and e.get("end") is not None
    ]

    others = [
        e
        for e in entities
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

            merged.append(
                current
            )

            continue

        previous = merged[-1]

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

            previous["context_score"] = max(
                float(
                    previous.get(
                        "context_score",
                        0
                    )
                ),
                float(
                    current.get(
                        "context_score",
                        0
                    )
                )
            )

        else:

            merged.append(
                current
            )

    return others + merged


def merge_address_fragments(
    entities
):

    addresses = [
        e
        for e in entities
        if canonical_type(
            e.get("type")
        ) == "ADDRESS"
        and e.get("start") is not None
        and e.get("end") is not None
    ]

    others = [
        e
        for e in entities
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

            merged.append(
                current
            )

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

            previous["context_score"] = max(
                float(
                    previous.get(
                        "context_score",
                        0
                    )
                ),
                float(
                    current.get(
                        "context_score",
                        0
                    )
                )
            )

        else:

            merged.append(
                current
            )

    return others + merged


def reconcile_entities(
    entities: List[Dict[str, Any]],
    text: str = ""
) -> List[Dict[str, Any]]:

    if not entities:
        return []

    candidates = []

    for entity in entities:

        e = entity.copy()

        e["type"] = canonical_type(
            e.get(
                "type",
                e.get(
                    "entity",
                    ""
                )
            )
        )

        e["value"] = normalize(
            e.get(
                "value",
                e.get(
                    "text",
                    ""
                )
            )
        )

        if (
            not e["type"]
            or not e["value"]
        ):
            continue

        e["format_valid"] = (
            validate_entity(
                e["type"],
                e["value"]
            )
        )

        if not e["format_valid"]:
            continue

        candidates.append(
            e
        )

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

        better[
            "methods_agree"
        ] = True

        better[
            "format_valid"
        ] = (
            existing.get(
                "format_valid",
                False
            )
            or
            candidate.get(
                "format_valid",
                False
            )
        )

        unique[key] = better

    reconciled = list(
        unique.values()
    )

    reconciled = merge_name_fragments(
        reconciled
    )

    reconciled = merge_address_fragments(
        reconciled
    )

    reconciled.sort(
        key=lambda e: (
            e.get(
                "start",
                10**9
            ),
            -candidate_quality(e)
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

            if (
                entity_key(candidate)
                ==
                entity_key(existing)
            ):

                conflict = True
                break

            if (
                canonical_type(
                    candidate["type"]
                )
                ==
                canonical_type(
                    existing["type"]
                )
            ):

                candidate_quality_score = (
                    candidate_quality(
                        candidate
                    )
                )

                existing_quality_score = (
                    candidate_quality(
                        existing
                    )
                )

                if (
                    existing_quality_score
                    >=
                    candidate_quality_score
                ):

                    conflict = True
                    break

        if not conflict:

            final.append(
                candidate
            )

    final.sort(
        key=lambda e: e.get(
            "start",
            10**9
        )
    )

    return final