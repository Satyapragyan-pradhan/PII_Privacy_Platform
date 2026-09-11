import re
from typing import Dict, Any, List, Optional


class ContextClassifier:

    PRIMARY_LABELS = {
        "NAME": [
            "full name",
            "applicant name",
            "customer name",
            "employee name",
            "candidate name",
            "patient name",
            "passenger name",
            "tenant name",
            "member name",
            "your name",
            "name",
        ],
        "DOB": [
            "date of birth",
            "date of b1rth",
            "dob",
            "birth date",
            "birthdate",
            "born on",
        ],
        "ADDRESS": [
            "residential address",
            "permanent address",
            "current address",
            "home address",
            "correspondence address",
            "address",
        ],
        "PHONE": [
            "mobile number",
            "mobile",
            "phone number",
            "phone",
            "contact number",
            "contact",
        ],
        "EMAIL": [
            "personal email",
            "email address",
            "e-mail",
            "email",
        ],
        "PAN": [
            "pan number",
            "pan no",
            "pan",
        ],
        "AADHAAR": [
            "aadhaar number",
            "aadhar number",
            "aadhaar",
            "aadhar",
        ],
        "VOTER_ID": [
            "voter id",
            "voter",
            "epic number",
            "epic no",
            "epic",
        ],
        "DRIVING_LICENCE": [
            "driving licence",
            "driving license",
            "licence number",
            "license number",
            "dl number",
        ],
    }

    NON_TARGET_LABELS = {
        "DOB": [
            "date of issue",
            "date issued",
            "issue date",
            "issued on",
            "date of expiry",
            "expiry date",
            "valid till",
            "validity",
            "renewal date",
            "application date",
            "registration date",
            "joining date",
            "admission date",
            "published on",
            "meeting date",
        ],
        "ADDRESS": [
            "office address",
            "company address",
            "business address",
            "work address",
            "employer address",
            "branch address",
            "hospital address",
            "school address",
            "college address",
            "postal office",
        ],
        "EMAIL": [
            "hr email",
            "hr e-mail",
            "company email",
            "company e-mail",
            "office email",
            "office e-mail",
            "support email",
            "support e-mail",
            "admin email",
            "admin e-mail",
            "organization email",
            "organisation email",
            "help email",
            "website email",
        ],
        "PHONE": [
            "emergency contact",
            "emergency phone",
            "emergency number",
            "office phone",
            "company phone",
            "helpline",
            "customer care",
        ],
    }

    SECONDARY_PERSON_LABELS = [
        "father's name",
        "father name",
        "fathers name",
        "mother's name",
        "mother name",
        "mothers name",
        "spouse's name",
        "spouse name",
        "spouses name",
        "husband's name",
        "husband name",
        "wife's name",
        "wife name",
        "guardian's name",
        "guardian name",
        "son of",
        "daughter of",
        "s/o",
        "d/o",
        "w/o",
        "c/o",
        "father",
        "mother",
        "spouse",
        "husband",
        "wife",
        "guardian",
        "emergency contact",
        "emergency name",
        "reference name",
        "nominee",
        "nominee name",
        "contact person",
        "alternate contact",
        "alternate name",
        "next of kin",
        "kin name",
        "dependent",
        "dependent name",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.model = None

        if model_path:
            self._try_load_model(model_path)

    def _try_load_model(self, model_path: str):
        try:
            import joblib

            self.model = joblib.load(model_path)

            print(
                f"[CONTEXT] Loaded context model: "
                f"{model_path}"
            )

        except Exception as exc:
            print(
                f"[CONTEXT] Learned model unavailable: "
                f"{exc}"
            )
            self.model = None

    @staticmethod
    def _clean(text: str) -> str:
        return (
            str(text or "")
            .lower()
            .replace("’", "'")
            .replace("`", "'")
        )

    @staticmethod
    def _line_context(
        text: str,
        start: int
    ) -> str:

        before = text[:start]

        line_start = before.rfind("\n")

        if line_start == -1:
            line_start = 0
        else:
            line_start += 1

        return text[line_start:start]

    @staticmethod
    def _nearby_context(
        text: str,
        start: int,
        window: int = 120
    ) -> str:

        return text[
            max(0, start - window):start
        ]

    @staticmethod
    def _label_position(
        context: str,
        labels: List[str]
    ) -> Optional[tuple]:

        best = None

        for label in labels:
            position = context.rfind(label)

            if position == -1:
                continue

            candidate = (
                position,
                len(label),
                label
            )

            if best is None or position > best[0]:
                best = candidate

        return best

    def _nearest_label(
        self,
        entity_type: str,
        text: str,
        start: int
    ) -> Dict[str, Any]:

        entity_type = entity_type.upper()

        line_context = self._clean(
            self._line_context(text, start)
        )

        nearby_context = self._clean(
            self._nearby_context(text, start)
        )

        primary = self._label_position(
            line_context,
            self.PRIMARY_LABELS.get(
                entity_type,
                []
            )
        )

        negative = self._label_position(
            line_context,
            self.NON_TARGET_LABELS.get(
                entity_type,
                []
            )
        )

        secondary = None

        if entity_type == "NAME":
            secondary = self._label_position(
                line_context,
                self.SECONDARY_PERSON_LABELS
            )

        if primary is None:
            primary = self._label_position(
                nearby_context,
                self.PRIMARY_LABELS.get(
                    entity_type,
                    []
                )
            )

        if negative is None:
            negative = self._label_position(
                nearby_context,
                self.NON_TARGET_LABELS.get(
                    entity_type,
                    []
                )
            )

        if secondary is None and entity_type == "NAME":
            secondary = self._label_position(
                nearby_context,
                self.SECONDARY_PERSON_LABELS
            )

        candidates = []

        if primary:
            candidates.append(
                ("primary", primary)
            )

        if negative:
            candidates.append(
                ("negative", negative)
            )

        if secondary:
            candidates.append(
                ("secondary", secondary)
            )

        if not candidates:
            return {
                "role": "unknown",
                "label": "",
                "score": 0.0,
            }

        role, selected = max(
            candidates,
            key=lambda item: item[1][0]
        )

        label = selected[2]

        if role == "secondary":
            return {
                "role": "secondary_person",
                "label": label,
                "score": -1.0,
            }

        if role == "negative":
            return {
                "role": "non_target",
                "label": label,
                "score": -1.0,
            }

        return {
            "role": "primary",
            "label": label,
            "score": 1.0,
        }

    def _learned_prediction(
        self,
        entity: Dict[str, Any],
        text: str
    ) -> Optional[Dict[str, Any]]:

        if self.model is None:
            return None

        start = entity.get("start")

        if start is None:
            return None

        entity_type = str(
            entity.get("type", "")
        ).upper()

        context = self._nearby_context(
            text,
            int(start)
        )

        try:
            prediction = self.model.predict(
                [context]
            )[0]

            probability = None

            if hasattr(
                self.model,
                "predict_proba"
            ):
                probabilities = (
                    self.model.predict_proba(
                        [context]
                    )[0]
                )

                probability = float(
                    max(probabilities)
                )

            role = str(
                prediction
            ).lower()

            if role in {
                "primary",
                "primary_person",
                "primary_dob",
                "primary_address",
                "primary_contact",
            }:
                return {
                    "role": "primary",
                    "label": "",
                    "score": probability or 1.0,
                    "model": "learned",
                }

            if role in {
                "secondary",
                "secondary_person",
                "related_person",
                "non_primary",
            }:
                return {
                    "role": "secondary_person",
                    "label": "",
                    "score": -(probability or 1.0),
                    "model": "learned",
                }

            if role in {
                "non_target",
                "contextual",
                "organization",
                "unrelated",
            }:
                return {
                    "role": "non_target",
                    "label": "",
                    "score": -(probability or 1.0),
                    "model": "learned",
                }

        except Exception as exc:
            print(
                f"[CONTEXT] Learned prediction failed: "
                f"{exc}"
            )

        return None

    def classify(
        self,
        entity: Dict[str, Any],
        text: str
    ) -> Dict[str, Any]:

        result = entity.copy()

        start = result.get("start")

        if start is None:
            result.update({
                "context_role": "unknown",
                "context_label": "",
                "context_score": 0.0,
                "context_model": "none",
            })
            return result

        learned = self._learned_prediction(
            result,
            text
        )

        if learned is not None:
            result.update({
                "context_role": learned["role"],
                "context_label": learned.get(
                    "label",
                    ""
                ),
                "context_score": round(
                    float(
                        learned.get(
                            "score",
                            0.0
                        )
                    ),
                    3
                ),
                "context_model": "learned",
            })
            return result

        fallback = self._nearest_label(
            str(
                result.get(
                    "type",
                    ""
                )
            ),
            text,
            int(start)
        )

        result.update({
            "context_role": fallback["role"],
            "context_label": fallback["label"],
            "context_score": round(
                float(fallback["score"]),
                3
            ),
            "context_model": "nearest_field",
        })

        return result


_classifier = ContextClassifier()


def get_context_classifier() -> ContextClassifier:
    return _classifier


def classify_entity_context(
    entity: Dict[str, Any],
    text: str
) -> Dict[str, Any]:

    return get_context_classifier().classify(
        entity,
        text
    )