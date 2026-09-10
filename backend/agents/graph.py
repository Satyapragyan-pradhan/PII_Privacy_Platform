from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from extraction.regex import extract_regex_entities
from extraction.nlp import extract_nlp_entities
from extraction.contextual import contextual_extract
from services.reconciliation import reconcile_entities


class GraphState(TypedDict):
    text: str

    regex_entities: List[Dict[str, Any]]
    nlp_entities: List[Dict[str, Any]]

    contextual_entities: List[Dict[str, Any]]

    preliminary_entities: List[Dict[str, Any]]
    final_entities: List[Dict[str, Any]]


def regex_node(state: GraphState):
    return {
        "regex_entities": extract_regex_entities(
            state["text"]
        )
    }


def nlp_node(state: GraphState):
    entities = extract_nlp_entities(
        state["text"]
    )

    print("\n========== DEBERTA ==========")

    for entity in entities:
        print(entity)

    print("=============================\n")

    return {
        "nlp_entities": entities
    }


def is_related_person_name(
    entity: Dict[str, Any],
    text: str
) -> bool:

    if str(
        entity.get("type", "")
    ).upper() != "NAME":
        return False

    start = entity.get("start")

    if start is None:
        return False

    context_start = max(
        0,
        int(start) - 100
    )

    preceding_text = text[
        context_start:int(start)
    ].lower()

    preceding_text = (
        preceding_text
        .replace("’", "'")
        .replace("`", "'")
    )

    related_labels = [
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
    ]

    for label in related_labels:

        if label in preceding_text:

            print(
                f"[ROLE] Related-person NAME detected: "
                f"{entity.get('value')} "
                f"(label: {label})"
            )

            return True

    return False


def filter_primary_person_entities(
    entities: List[Dict[str, Any]],
    text: str
) -> List[Dict[str, Any]]:

    filtered = []

    for entity in entities:

        if is_related_person_name(
            entity,
            text
        ):
            continue

        filtered.append(entity)

    return filtered


def preliminary_node(state: GraphState):

    entities = (
        state.get("regex_entities", [])
        +
        state.get("nlp_entities", [])
    )

    entities = filter_primary_person_entities(
        entities,
        state["text"]
    )

    preliminary = reconcile_entities(
        entities,
        state["text"]
    )

    print("\n========== PRELIMINARY ==========")

    for entity in preliminary:
        print(entity)

    print("=================================\n")

    return {
        "preliminary_entities": preliminary
    }


LOW_CONFIDENCE_THRESHOLD = 0.70


def detect_document_profile(preliminary):

    detected_types = {
        str(entity.get("type", "")).upper()
        for entity in preliminary
    }

    if "PAN" in detected_types:
        return {
            "document_type": "PAN",
            "expected_types": {
                "NAME",
                "DOB",
            }
        }

    if "AADHAAR" in detected_types:
        return {
            "document_type": "AADHAAR",
            "expected_types": {
                "NAME",
                "DOB",
                "ADDRESS",
            }
        }

    if "DRIVING_LICENCE" in detected_types:
        return {
            "document_type": "DRIVING_LICENCE",
            "expected_types": {
                "NAME",
                "DOB",
                "ADDRESS",
            }
        }

    if "VOTER_ID" in detected_types:
        return {
            "document_type": "VOTER_ID",
            "expected_types": {
                "NAME",
                "ADDRESS",
            }
        }

    return {
        "document_type": "GENERIC",
        "expected_types": set()
    }


def find_low_confidence_entities(preliminary):

    low_confidence = []

    for entity in preliminary:

        confidence = float(
            entity.get(
                "confidence",
                0.0
            )
        )

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence.append(entity)

    return low_confidence


def find_conflicts(preliminary):

    values_by_type = {}

    for entity in preliminary:

        entity_type = str(
            entity.get("type", "")
        ).upper()

        value = str(
            entity.get("value", "")
        ).strip().lower()

        if not entity_type or not value:
            continue

        values_by_type.setdefault(
            entity_type,
            set()
        ).add(value)

    conflicts = []

    for entity_type, values in values_by_type.items():

        if len(values) > 1:
            conflicts.append(entity_type)

    return conflicts


def context_node(state: GraphState):

    preliminary = state.get(
        "preliminary_entities",
        []
    )

    profile = detect_document_profile(
        preliminary
    )

    document_type = profile[
        "document_type"
    ]

    expected_types = profile[
        "expected_types"
    ]

    detected_types = {
        str(entity.get("type", "")).upper()
        for entity in preliminary
    }

    missing = (
        expected_types
        - detected_types
    )

    low_confidence = (
        find_low_confidence_entities(
            preliminary
        )
    )

    conflicts = find_conflicts(
        preliminary
    )

    print(
        "\n========== LLM FALLBACK CHECK =========="
    )

    print(
        "Document profile:",
        document_type
    )

    print(
        "Expected contextual types:",
        expected_types
    )

    print(
        "Detected types:",
        detected_types
    )

    print(
        "Missing relevant types:",
        missing
    )

    print(
        "Low-confidence entities:",
        low_confidence
    )

    print(
        "Conflicting types:",
        conflicts
    )

    llm_required = (
        bool(missing)
        or bool(low_confidence)
        or bool(conflicts)
    )

    if not llm_required:

        print(
            "LLM fallback skipped."
        )

        print(
            "Reason: extraction is sufficient "
            "and unambiguous."
        )

        print(
            "=======================================\n"
        )

        return {
            "contextual_entities": []
        }

    print(
        "LLM fallback activated."
    )

    print(
        "=======================================\n"
    )

    return {
        "contextual_entities":
            contextual_extract(
                state["text"]
            )
    }


def final_reconciliation_node(
    state: GraphState
):

    entities = (
        state.get("regex_entities", [])
        +
        state.get("nlp_entities", [])
        +
        state.get("contextual_entities", [])
    )

    entities = filter_primary_person_entities(
        entities,
        state["text"]
    )

    final = reconcile_entities(
        entities,
        state["text"]
    )

    return {
        "final_entities": final
    }


def build_graph():

    graph = StateGraph(
        GraphState
    )

    graph.add_node(
        "regex",
        regex_node
    )

    graph.add_node(
        "nlp",
        nlp_node
    )

    graph.add_node(
        "preliminary",
        preliminary_node
    )

    graph.add_node(
        "context",
        context_node
    )

    graph.add_node(
        "reconciliation",
        final_reconciliation_node
    )

    graph.set_entry_point(
        "regex"
    )

    graph.add_edge(
        "regex",
        "nlp"
    )

    graph.add_edge(
        "nlp",
        "preliminary"
    )

    graph.add_edge(
        "preliminary",
        "context"
    )

    graph.add_edge(
        "context",
        "reconciliation"
    )

    graph.add_edge(
        "reconciliation",
        END
    )

    return graph.compile()