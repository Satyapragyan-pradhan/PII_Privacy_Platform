import json
import random
import re
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path("evaluation/training/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_DOCUMENTS = 10000


FIRST_NAMES = [
    "Rohan", "Rahul", "Amit", "Arjun", "Vikram",
    "Aditya", "Rajesh", "Suresh", "Karan", "Manish",
    "Pankaj", "Deepak", "Nisha", "Priya", "Ananya",
    "Ishita", "Riya", "Neha", "Kavita", "Sneha",
    "Apurva", "Chunni", "Meera", "Shreya", "Tanvi"
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Verma",
    "Gupta", "Das", "Sahani", "Bansal", "Mehta",
    "Reddy", "Nair", "Joshi", "Chopra", "Malhotra",
    "Fernandes", "Manchanda", "Pradhan", "Mishra", "Sethi"
]

CITIES = [
    ("Ahmedabad", "Gujarat", "380015"),
    ("Jaipur", "Rajasthan", "302017"),
    ("Kolkata", "West Bengal", "700073"),
    ("Bhubaneswar", "Odisha", "751001"),
    ("New Delhi", "Delhi", "110018"),
    ("Mumbai", "Maharashtra", "400001"),
    ("Pune", "Maharashtra", "411001"),
    ("Bengaluru", "Karnataka", "560001"),
    ("Hyderabad", "Telangana", "500001"),
    ("Chandigarh", "Chandigarh", "160017"),
    ("Panchkula", "Haryana", "134109"),
    ("Lucknow", "Uttar Pradesh", "226001")
]

STREETS = [
    "Green Park Lane",
    "Rose Garden Road",
    "College Square",
    "MG Road",
    "Station Road",
    "Lake View Road",
    "Park Street",
    "Civil Lines",
    "Sector 15",
    "Ashok Nagar",
    "Rajendra Nagar",
    "Gandhi Nagar"
]

EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "example.test",
    "mail.test"
]

ORG_NAMES = [
    "Example Technologies",
    "National Finance Services",
    "Bharat Solutions",
    "Eastern Industries",
    "Global Systems Pvt Ltd",
    "Odisha Digital Services",
    "Apex Consulting",
    "United Business Group"
]


def random_name():
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)


def random_phone():
    return str(random.randint(6, 9)) + "".join(
        str(random.randint(0, 9)) for _ in range(9)
    )


def random_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1970, 2004)

    return f"{day:02d}-{month:02d}-{year}"


def random_date_slash():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1970, 2004)

    return f"{day:02d}/{month:02d}/{year}"


def random_pan():
    letters1 = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    digits = "".join(str(random.randint(0, 9)) for _ in range(4))
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    return letters1 + digits + letter


def random_aadhaar():
    first = str(random.randint(2, 9))
    return first + "".join(
        str(random.randint(0, 9)) for _ in range(11)
    )


def random_voter():
    return "".join(
        random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for _ in range(3)
    ) + "".join(
        str(random.randint(0, 9))
        for _ in range(7)
    )


def random_dl():
    state = random.choice([
        "HR", "HP", "OD", "DL", "MH",
        "KA", "RJ", "UP", "WB"
    ])

    district = f"{random.randint(1, 99):02d}"
    year = random.randint(2014, 2024)
    serial = f"{random.randint(1, 99999999):08d}"

    return f"{state}{district}{year}{serial}"


def random_address():
    city, state, pin = random.choice(CITIES)
    house = random.randint(1, 9999)
    street = random.choice(STREETS)

    return (
        f"{house} {street}, "
        f"{city}, {state} - {pin}"
    )


def random_email(first, last):
    username = f"{first.lower()}.{last.lower()}"
    domain = random.choice(EMAIL_DOMAINS)

    return f"{username}@{domain}"


def add_entity(entities, text, entity_type, value):
    start = text.find(value)

    if start == -1:
        raise ValueError(
            f"Could not locate entity: {value}"
        )

    end = start + len(value)

    entities.append({
        "type": entity_type,
        "value": value,
        "start": start,
        "end": end
    })


def replace_at(text, start, end, replacement):
    return text[:start] + replacement + text[end:]


# ---------------------------------------------------------
# OCR corruption
# ---------------------------------------------------------

def ocr_character_noise(text):
    replacements = [
        ("O", "0"),
        ("0", "O"),
        ("I", "1"),
        ("1", "I"),
        ("S", "5"),
        ("5", "S")
    ]

    chars = list(text)

    for i in range(len(chars)):
        if random.random() < 0.008:
            original = chars[i]

            for a, b in replacements:
                if original == a:
                    chars[i] = b
                    break

    return "".join(chars)


def ocr_spacing_noise(text):
    text = re.sub(
        r"(\d)\s+([A-Z])",
        r"\1\2",
        text
    )

    text = re.sub(
        r"([A-Z])\s+(\d)",
        r"\1\2",
        text
    )

    return text


def ocr_punctuation_noise(text):
    if random.random() < 0.30:
        text = text.replace(",", "")

    if random.random() < 0.20:
        text = text.replace(":", "")

    return text


def ocr_line_noise(text):
    lines = text.splitlines()

    if len(lines) > 4 and random.random() < 0.20:
        i = random.randint(0, len(lines) - 2)

        if random.random() < 0.5:
            lines[i], lines[i + 1] = lines[i + 1], lines[i]

    return "\n".join(lines)


def apply_ocr_noise(text):
    noise_level = random.random()

    if noise_level < 0.40:
        return text

    text = ocr_character_noise(text)

    if noise_level >= 0.55:
        text = ocr_spacing_noise(text)

    if noise_level >= 0.65:
        text = ocr_punctuation_noise(text)

    if noise_level >= 0.75:
        text = ocr_line_noise(text)

    return text


# ---------------------------------------------------------
# Document generators
# ---------------------------------------------------------

def generate_aadhaar():
    first, last = random_name()

    name = f"{first} {last}"
    dob = random_date_slash()
    aadhaar = random_aadhaar()
    address = random_address()

    father_first, father_last = random_name()
    father = f"{father_first} {father_last}"

    text = f"""Government of India
AADHAAR
Name
{name}
DOB
{dob}
Gender
MALE
Aadhaar Number
{aadhaar}
Address
{address}
S/O {father}
Unique Identification Authority of India
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "AADHAAR", aadhaar)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_pan():
    first, last = random_name()

    name = f"{first} {last}"
    father_first, father_last = random_name()
    father = f"{father_first} {father_last}"

    pan = random_pan()
    dob = random_date_slash()

    text = f"""INCOME TAX DEPARTMENT
GOVT. OF INDIA
Permanent Account Number Card
PAN
{pan}
Name
{name}
Father's Name
{father}
Date of Birth
{dob}
"""

    entities = []

    add_entity(entities, text, "PAN", pan)
    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)

    return text, entities


def generate_driving_licence():
    first, last = random_name()

    name = f"{first} {last}"

    secondary_first, secondary_last = random_name()
    secondary = f"{secondary_first} {secondary_last}"

    dob = random_date()
    issue_date = random_date()
    validity_date = random_date()

    dl = random_dl()
    address = random_address()

    text = f"""UNION OF INDIA
Driving Licence
DLNUMBER {dl}
NAME
{name}
S/W/D
{secondary}
DOB
{dob}
Validity
{validity_date}
Address
{address}
Issue Date
{issue_date}
"""

    entities = []

    add_entity(entities, text, "DRIVING_LICENCE", dl)
    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_voter():
    first, last = random_name()

    name = f"{first} {last}"
    voter = random_voter()
    dob = random_date_slash()
    address = random_address()

    text = f"""ELECTION COMMISSION OF INDIA
VOTER ID
EPIC Number
{voter}
Name
{name}
Date of Birth
{dob}
Address
{address}
"""

    entities = []

    add_entity(entities, text, "VOTER_ID", voter)
    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_bank_kyc():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)
    dob = random_date_slash()
    address = random_address()
    pan = random_pan()
    aadhaar = random_aadhaar()

    secondary_first, secondary_last = random_name()
    secondary = f"{secondary_first} {secondary_last}"

    text = f"""CUSTOMER KYC FORM

Customer Name
{name}

Date of Birth
{dob}

Mobile Number
{phone}

Email Address
{email}

Residential Address
{address}

PAN
{pan}

Aadhaar Number
{aadhaar}

Nominee Name
{secondary}

Bank
National Finance Services
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)
    add_entity(entities, text, "PAN", pan)
    add_entity(entities, text, "AADHAAR", aadhaar)

    return text, entities


def generate_employee():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)
    dob = random_date_slash()
    address = random_address()

    emergency_first, emergency_last = random_name()
    emergency = f"{emergency_first} {emergency_last}"

    organization = random.choice(ORG_NAMES)

    text = f"""EMPLOYEE INFORMATION FORM

Employee Name: {name}
Date of Birth: {dob}
Mobile: {phone}
Email: {email}

Residential Address:
{address}

Emergency Contact:
{emergency}

Organization:
{organization}

HR Email:
hr@example.test
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_customer():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)
    address = random_address()

    text = f"""CUSTOMER REGISTRATION

Name
{name}

Phone
{phone}

Email
{email}

Address
{address}

Customer Service
help@example.test
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_hospital():
    first, last = random_name()

    name = f"{first} {last}"
    dob = random_date_slash()
    phone = random_phone()
    address = random_address()

    relative_first, relative_last = random_name()
    relative = f"{relative_first} {relative_last}"

    text = f"""HOSPITAL REGISTRATION

Patient Name
{name}

Date of Birth
{dob}

Mobile Number
{phone}

Residential Address
{address}

Emergency Contact
{relative}

Hospital
City General Hospital
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_insurance():
    first, last = random_name()

    name = f"{first} {last}"
    dob = random_date_slash()
    phone = random_phone()
    email = random_email(first, last)
    address = random_address()

    text = f"""INSURANCE PROPOSAL FORM

Policy Holder
{name}

DOB
{dob}

Contact Number
{phone}

Email
{email}

Permanent Address
{address}

Insurance Company
Apex Insurance Services
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_job_application():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)
    address = random_address()

    text = f"""JOB APPLICATION FORM

Applicant Name
{name}

Contact Number
{phone}

Email Address
{email}

Current Address
{address}

Recruitment Department
Example Technologies
recruitment@example.test
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_education():
    first, last = random_name()

    name = f"{first} {last}"
    dob = random_date_slash()
    phone = random_phone()
    address = random_address()

    guardian_first, guardian_last = random_name()
    guardian = f"{guardian_first} {guardian_last}"

    text = f"""SCHOOL / COLLEGE ADMISSION FORM

Student Name
{name}

Date of Birth
{dob}

Mobile
{phone}

Address
{address}

Parent / Guardian
{guardian}

Institution
Example University
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "DOB", dob)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_rental():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)
    address = random_address()

    owner_first, owner_last = random_name()
    owner = f"{owner_first} {owner_last}"

    text = f"""RENTAL TENANT FORM

Tenant Name
{name}

Phone
{phone}

Email
{email}

Current Address
{address}

Property Owner
{owner}

Rental Agreement
Residential property
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


def generate_travel():
    first, last = random_name()

    name = f"{first} {last}"
    phone = random_phone()
    email = random_email(first, last)

    address = random_address()

    text = f"""TRAVEL PASSENGER FORM

Passenger Name
{name}

Contact Phone
{phone}

Email
{email}

Residential Address
{address}

Transport Department
National Travel Services
"""

    entities = []

    add_entity(entities, text, "NAME", name)
    add_entity(entities, text, "PHONE", phone)
    add_entity(entities, text, "EMAIL", email)
    add_entity(entities, text, "ADDRESS", address)

    return text, entities


GENERATORS = [
    ("aadhaar", generate_aadhaar),
    ("pan", generate_pan),
    ("driving_licence", generate_driving_licence),
    ("voter_id", generate_voter),
    ("bank_kyc_form", generate_bank_kyc),
    ("employee_information", generate_employee),
    ("customer_registration", generate_customer),
    ("hospital_registration", generate_hospital),
    ("insurance_form", generate_insurance),
    ("job_application", generate_job_application),
    ("school_college_form", generate_education),
    ("rental_tenant_form", generate_rental),
    ("travel_passenger_form", generate_travel),
]


def generate_dataset():
    documents = []

    for i in range(TOTAL_DOCUMENTS):

        document_type, generator = random.choice(GENERATORS)

        clean_text, clean_entities = generator()

        noisy_text = apply_ocr_noise(
            clean_text
        )

        # -------------------------------------------------
        # Important:
        # OCR corruption may alter offsets.
        #
        # For the first version we only keep examples where
        # all entity strings survive the corruption exactly.
        # -------------------------------------------------

        valid_entities = []

        for entity in clean_entities:

            value = entity["value"]

            start = noisy_text.find(value)

            if start == -1:
                continue

            valid_entities.append({
                "type": entity["type"],
                "value": value,
                "start": start,
                "end": start + len(value)
            })

        documents.append({
            "document_id": f"train_{i + 1:06d}",
            "document_type": document_type,
            "text": noisy_text,
            "entities": valid_entities
        })

        if (i + 1) % 500 == 0:
            print(
                f"Generated {i + 1}/{TOTAL_DOCUMENTS}"
            )

    random.shuffle(documents)

    # -----------------------------------------------------
    # Split by document, not by OCR variant.
    # -----------------------------------------------------

    train_end = 8000
    validation_end = 9000

    train = documents[:train_end]
    validation = documents[train_end:validation_end]
    test = documents[validation_end:]

    with open(
        OUTPUT_DIR / "all_documents.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        OUTPUT_DIR / "train.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            train,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        OUTPUT_DIR / "validation.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            validation,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        OUTPUT_DIR / "test.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            test,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n======================================")
    print("DATASET GENERATION COMPLETE")
    print("======================================")
    print(f"Total      : {len(documents)}")
    print(f"Train      : {len(train)}")
    print(f"Validation : {len(validation)}")
    print(f"Test       : {len(test)}")
    print(f"Output     : {OUTPUT_DIR.resolve()}")
    print("======================================")


if __name__ == "__main__":
    generate_dataset()