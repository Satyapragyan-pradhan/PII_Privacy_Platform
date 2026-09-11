from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from extraction.regex import extract_regex_entities
from extraction.nlp import extract_nlp_entities
from extraction.contextual import contextual_extract

from models.context_classifier import (
    classify_entity_context
)

from services.reconciliation import (
    reconcile_entities
)


class GraphState(TypedDict):
    text: str

    regex_entities: List[Dict[str, Any]]
    nlp_entities: List[Dict[str, Any]]

    contextual_entities: List[Dict[str, Any]]

    preliminary_entities: List[Dict[str, Any]]
    final_entities: List[Dict[str, Any]]


def regex_node(
    state: GraphState
):

    entities = extract_regex_entities(
        state["text"]
    )

    return {
        "regex_entities": entities
    }


def nlp_node(
    state: GraphState
):

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


def classify_context(
    entities: List[Dict[str, Any]],
    text: str
) -> List[Dict[str, Any]]:

    return [
        classify_entity_context(
            entity,
            text
        )
        for entity in entities
    ]


def filter_non_primary(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    filtered = []

    for entity in entities:

        role = entity.get(
            "context_role",
            "unknown"
        )

        entity_type = str(
            entity.get(
                "type",
                ""
            )
        ).upper()

        if (
            role == "secondary_person"
            and entity_type == "NAME"
        ):

            print(
                f"[ROLE] Secondary NAME rejected: "
                f"{entity.get('value')} "
                f"(label: "
                f"{entity.get('context_label')})"
            )

            continue

        if (
            role == "non_target"
            and entity_type in {
                "EMAIL",
                "PHONE",
                "DOB",
                "ADDRESS"
            }
        ):

            print(
                f"[ROLE] Non-target "
                f"{entity_type} rejected: "
                f"{entity.get('value')} "
                f"(label: "
                f"{entity.get('context_label')})"
            )

            continue

        filtered.append(entity)

    return filtered


def preliminary_node(
    state: GraphState
):

    entities = (
        state.get(
            "regex_entities",
            []
        )
        +
        state.get(
            "nlp_entities",
            []
        )
    )

    entities = classify_context(
        entities,
        state["text"]
    )

    entities = filter_non_primary(
        entities
    )

    print(
        "\n========== CONTEXT =========="
    )

    for entity in entities:

        print(
            f"{entity.get('type')} | "
            f"{entity.get('value')} | "
            f"role={entity.get('context_role')} | "
            f"score={entity.get('context_score')} | "
            f"label={entity.get('context_label')} | "
            f"model={entity.get('context_model')}"
        )

    print(
        "=============================\n"
    )

    preliminary = reconcile_entities(
        entities,
        state["text"]
    )

    print(
        "\n========== PRELIMINARY =========="
    )

    for entity in preliminary:
        print(entity)

    print(
        "=================================\n"
    )

    return {
        "preliminary_entities": preliminary
    }


LOW_CONFIDENCE_THRESHOLD = 0.70


def find_low_confidence_entities(
    preliminary: List[Dict[str, Any]]
):

    low_confidence = []

    for entity in preliminary:

        confidence = float(
            entity.get(
                "confidence",
                0.0
            )
        )

        context_score = float(
            entity.get(
                "context_score",
                0.0
            )
        )

        role = entity.get(
            "context_role",
            "unknown"
        )

        # Strong non-target evidence should never
        # activate the LLM.
        if role == "non_target":
            continue

        if (
            confidence < LOW_CONFIDENCE_THRESHOLD
            and context_score <= 0
        ):
            low_confidence.append(
                entity
            )

    return low_confidence


def find_conflicts(
    preliminary: List[Dict[str, Any]]
):

    values_by_type = {}

    for entity in preliminary:

        entity_type = str(
            entity.get(
                "type",
                ""
            )
        ).upper()

        value = str(
            entity.get(
                "value",
                ""
            )
        ).strip().lower()

        if not entity_type or not value:
            continue

        values_by_type.setdefault(
            entity_type,
            []
        ).append(entity)

    conflicts = []

    for entity_type, entities in values_by_type.items():

        if len(entities) <= 1:
            continue

        values = {
            str(
                entity.get(
                    "value",
                    ""
                )
            ).strip().lower()
            for entity in entities
        }

        if len(values) <= 1:
            continue

        meaningful = [
            entity
            for entity in entities
            if entity.get(
                "context_role",
                "unknown"
            ) != "secondary_person"
            and entity.get(
                "context_role",
                "unknown"
            ) != "non_target"
        ]

        if len(meaningful) <= 1:
            continue

        if entity_type in {
            "PAN",
            "AADHAAR",
            "VOTER_ID",
            "DRIVING_LICENCE"
        }:

            conflicts.append(
                entity_type
            )

            continue

        strong_candidates = [
            entity
            for entity in meaningful
            if float(
                entity.get(
                    "context_score",
                    0.0
                )
            ) > 0
        ]

        if len(strong_candidates) > 1:
            conflicts.append(
                entity_type
            )

    return conflicts


def has_primary_name_ambiguity(
    preliminary: List[Dict[str, Any]]
):

    names = [
        entity
        for entity in preliminary
        if str(
            entity.get(
                "type",
                ""
            )
        ).upper() == "NAME"
        and entity.get(
            "context_role",
            "unknown"
        ) not in {
            "secondary_person",
            "non_target"
        }
    ]

    if len(names) <= 1:
        return False

    primary_candidates = [
        entity
        for entity in names
        if entity.get(
            "context_role"
        ) == "primary"
    ]

    if len(primary_candidates) == 1:
        return False

    return True


def has_unresolved_context(
    preliminary: List[Dict[str, Any]]
):

    for entity in preliminary:

        role = entity.get(
            "context_role",
            "unknown"
        )

        confidence = float(
            entity.get(
                "confidence",
                0.0
            )
        )

        context_score = float(
            entity.get(
                "context_score",
                0.0
            )
        )

        if role == "non_target":
            continue

        if (
            role == "unknown"
            and confidence < 0.80
            and context_score <= 0
        ):

            return True

    return False


def context_node(
    state: GraphState
):

    preliminary = state.get(
        "preliminary_entities",
        []
    )

    text = state.get(
        "text",
        ""
    )

    low_confidence = (
        find_low_confidence_entities(
            preliminary
        )
    )

    conflicts = find_conflicts(
        preliminary
    )

    primary_name_ambiguity = (
        has_primary_name_ambiguity(
            preliminary
        )
    )

    unresolved_context = (
        has_unresolved_context(
            preliminary
        )
    )

    print(
        "\n========== LLM FALLBACK CHECK =========="
    )

    print(
        "Relevant low-confidence:",
        low_confidence
    )

    print(
        "Conflicting types:",
        conflicts
    )

    print(
        "Primary-name ambiguity:",
        primary_name_ambiguity
    )

    print(
        "Unresolved context:",
        unresolved_context
    )

    llm_required = (
        bool(low_confidence)
        or bool(conflicts)
        or primary_name_ambiguity
        or unresolved_context
    )

    if not llm_required:

        print(
            "LLM fallback skipped."
        )

        print(
            "Reason: extraction is "
            "contextually sufficient."
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
                text
            )
    }


def final_reconciliation_node(
    state: GraphState
):

    entities = (
        state.get(
            "regex_entities",
            []
        )
        +
        state.get(
            "nlp_entities",
            []
        )
        +
        state.get(
            "contextual_entities",
            []
        )
    )

    entities = classify_context(
        entities,
        state["text"]
    )

    entities = filter_non_primary(
        entities
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