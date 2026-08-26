from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from extraction.regex import extract_regex_entities
from extraction.nlp import extract_nlp_entities
from extraction.contextual import contextual_extract

from services.reconciliation import reconcile_entities
from services.confidence import calculate_confidence


class GraphState(TypedDict):
    text: str
    regex_entities: List[Dict[str, Any]]
    nlp_entities: List[Dict[str, Any]]
    contextual_entities: List[Dict[str, Any]]
    final_entities: List[Dict[str, Any]]


def regex_node(state: GraphState):
    entities = extract_regex_entities(
        state["text"]
    )

    return {
        "regex_entities": entities
    }


def nlp_node(state: GraphState):
    entities = extract_nlp_entities(
        state["text"]
    )

    return {
        "nlp_entities": entities
    }


def context_node(state: GraphState):
    entities = contextual_extract(
        state["text"]
    )

    return {
        "contextual_entities": entities
    }


def reconciliation_node(state: GraphState):

    all_entities = (
        state.get("regex_entities", [])
        + state.get("nlp_entities", [])
        + state.get("contextual_entities", [])
    )

    reconciled = reconcile_entities(
        all_entities
    )

    final_entities = []

    for entity in reconciled:

        entity["confidence"] = (
            calculate_confidence(
                entity
            )
        )

        final_entities.append(entity)

    return {
        "final_entities": final_entities
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
        "context",
        context_node
    )

    graph.add_node(
        "reconciliation",
        reconciliation_node
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