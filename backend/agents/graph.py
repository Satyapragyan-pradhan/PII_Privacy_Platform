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


# =========================================================
# REGEX
# =========================================================

def regex_node(state: GraphState):

    return {
        "regex_entities": extract_regex_entities(
            state["text"]
        )
    }


# =========================================================
# DEBERTA / NLP
# =========================================================

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


# =========================================================
# PRELIMINARY RECONCILIATION
# =========================================================

def preliminary_node(state: GraphState):

    entities = (
        state.get("regex_entities", [])
        +
        state.get("nlp_entities", [])
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


# =========================================================
# LLM FALLBACK
# =========================================================

def context_node(state: GraphState):

    preliminary = state.get(
        "preliminary_entities",
        []
    )

    # Only these entities require contextual understanding.
    contextual_types = {
        "NAME",
        "ADDRESS",
        "DOB"
    }

    detected = {
        str(entity.get("type", "")).upper()
        for entity in preliminary
    }

    missing = contextual_types - detected

    print(
        "\nMissing contextual types:",
        missing
    )

    # If DeBERTa/Regex already found all
    # contextual entities, DO NOT call LLM.
    if not missing:

        print(
            "LLM fallback skipped."
        )

        return {
            "contextual_entities": []
        }

    print(
        "LLM fallback activated."
    )

    return {
        "contextual_entities":
            contextual_extract(
                state["text"]
            )
    }


# =========================================================
# FINAL RECONCILIATION
# =========================================================

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

    final = reconcile_entities(
        entities,
        state["text"]
    )

    return {
        "final_entities": final
    }


# =========================================================
# BUILD GRAPH
# =========================================================

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