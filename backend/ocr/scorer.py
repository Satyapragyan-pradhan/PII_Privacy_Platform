import re


def normalize_text(text: str) -> str:
    """
    Normalize OCR output for scoring.
    """
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def score_ocr(text: str) -> int:
    """
    PII-aware OCR quality scorer.

    Higher score means the OCR output contains more useful
    document/PII signals.

    This is NOT a language-quality scorer.
    It is designed specifically for PII extraction.
    """

    if not text:
        return 0

    text = normalize_text(text)

    score = 0

    # ---------------------------------------------------------
    # 1. Aadhaar-like 12 digit number
    # ---------------------------------------------------------

    # Normal form:
    # 1234 5678 9012
    if re.search(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        text
    ):
        score += 50

    # OCR may introduce spaces/newlines
    digits = re.sub(r"\D", "", text)

    if len(digits) >= 12:
        # Look for any 12-digit sequence
        if re.search(r"\d{12}", digits):
            score += 35

    # ---------------------------------------------------------
    # 2. Date of birth
    # ---------------------------------------------------------

    dob_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2}\b",
    ]

    for pattern in dob_patterns:
        if re.search(pattern, text):
            score += 30
            break

    # DOB label
    if re.search(r"\bDOB\b", text, re.IGNORECASE):
        score += 15

    if re.search(
        r"date\s+of\s+birth",
        text,
        re.IGNORECASE
    ):
        score += 15

    # ---------------------------------------------------------
    # 3. Gender
    # ---------------------------------------------------------

    if re.search(
        r"\b(male|female)\b",
        text,
        re.IGNORECASE
    ):
        score += 10

    # ---------------------------------------------------------
    # 4. Name-like text
    # ---------------------------------------------------------

    # Multi-word alphabetic name
    name_matches = re.findall(
        r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b",
        text
    )

    if name_matches:
        score += min(len(name_matches) * 20, 40)

    # ---------------------------------------------------------
    # 5. Aadhaar-related keywords
    # ---------------------------------------------------------

    keywords = [
        "aadhaar",
        "aadhar",
        "uidai",
        "government",
        "india",
        "unique",
        "identification",
    ]

    for keyword in keywords:
        if keyword.lower() in text.lower():
            score += 5

    # ---------------------------------------------------------
    # 6. OCR sanity checks
    # ---------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        score += min(len(lines), 5)

    # Count useful alphanumeric characters
    alphanumeric = re.findall(
        r"[A-Za-z0-9]",
        text
    )

    if len(alphanumeric) >= 20:
        score += 5

    # ---------------------------------------------------------
    # 7. Penalize obvious OCR garbage
    # ---------------------------------------------------------

    garbage_chars = re.findall(
        r"[^A-Za-z0-9\s.,:/\-]",
        text
    )

    if len(garbage_chars) > 20:
        score -= 15

    if len(garbage_chars) > 40:
        score -= 20

    # Excessive short/noisy lines
    if len(lines) > 20:
        score -= 10

    return max(score, 0)