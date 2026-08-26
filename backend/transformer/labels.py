LABELS = [
    "O",

    "B-PERSON",
    "I-PERSON",

    "B-DATE_OF_BIRTH",
    "I-DATE_OF_BIRTH",

    "B-ADDRESS",
    "I-ADDRESS",

    "B-PHONE",
    "I-PHONE",

    "B-EMAIL",
    "I-EMAIL",

    "B-PAN",
    "I-PAN",

    "B-AADHAAR",
    "I-AADHAAR",
]


LABEL2ID = {
    label: idx
    for idx, label in enumerate(LABELS)
}


ID2LABEL = {
    idx: label
    for idx, label in enumerate(LABELS)
}