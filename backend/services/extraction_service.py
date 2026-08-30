
from ingestion.loader import load_document
from ocr.engine import ocr_image, ocr_pdf_pages

from agents.graph import build_graph

from services.confidence import (
    calculate_confidence,
    confidence_label
)

from services.reconciliation import (
    validate_entity
)


GRAPH = build_graph()


def enrich_entity(entity):

    entity = entity.copy()

    entity_type = entity.get(
        "type",
        ""
    )

    value = entity.get(
        "value",
        ""
    )

    # -----------------------------------------------------
    # Final deterministic validation
    # -----------------------------------------------------

    format_valid = validate_entity(
        entity_type,
        value
    )

    entity["format_valid"] = (
        format_valid
    )

    # -----------------------------------------------------
    # Preserve reconciliation result
    # -----------------------------------------------------

    methods_agree = entity.get(
        "methods_agree",
        False
    )

    entity["methods_agree"] = (
        methods_agree
    )

    # -----------------------------------------------------
    # Calculate final confidence
    # -----------------------------------------------------

    confidence = calculate_confidence(
        entity
    )

    entity["confidence"] = (
        confidence
    )

    entity["validated"] = (
        format_valid
    )

    entity["confidence_level"] = (
        confidence_label(
            confidence
        )
    )

    return entity


def process_single_document(
    document
):

    filename = document[
        "filename"
    ]

    content = document[
        "content"
    ]

    # -----------------------------------------------------
    # Load document
    # -----------------------------------------------------

    loaded = load_document(
        filename,
        content
    )

    text = loaded.get(
        "text",
        ""
    )

    # -----------------------------------------------------
    # OCR fallback
    # -----------------------------------------------------

    if loaded.get(
        "needs_ocr"
    ):

        if loaded.get(
            "document"
        ) is not None:

            ocr_text = ocr_pdf_pages(
                loaded["document"]
            )

        else:

            ocr_text = ocr_image(
                content
            )

        if ocr_text:
            text = ocr_text

    # -----------------------------------------------------
    # No text
    # -----------------------------------------------------

    if not text.strip():

        return {
            "document": filename,
            "entities": [],
            "status": "no_text"
        }

    print(
        "\n========== OCR / INPUT TEXT =========="
    )

    print(text)

    print(
        "======================================\n"
    )

    # -----------------------------------------------------
    # Run extraction graph
    # -----------------------------------------------------

    result = GRAPH.invoke({

        "text": text,

        "regex_entities": [],

        "nlp_entities": [],

        "contextual_entities": [],
        "preliminary_entities": [],

        "final_entities": []

    })

    entities = result.get(
        "final_entities",
        []
    )

    # -----------------------------------------------------
    # Final enrichment
    # -----------------------------------------------------

    entities = [
        enrich_entity(entity)
        for entity in entities
    ]

    return {
        "document": filename,
        "entities": entities,
        "status": "success"
    }


def process_documents(
    documents
):

    results = []

    total_entities = 0

    high_confidence = 0

    medium_confidence = 0

    low_confidence = 0

    for document in documents:

        result = process_single_document(
            document
        )

        entities = result.get(
            "entities",
            []
        )

        total_entities += len(
            entities
        )

        for entity in entities:

            confidence = entity.get(
                "confidence",
                0
            )

            if confidence >= 0.90:

                high_confidence += 1

            elif confidence >= 0.70:

                medium_confidence += 1

            else:

                low_confidence += 1

        results.append(
            result
        )

    return {

        "status": "success",

        "documents_processed": len(
            documents
        ),

        "documents": results,

        "analytics": {

            "total_entities":
                total_entities,

            "high_confidence":
                high_confidence,

            "medium_confidence":
                medium_confidence,

            "low_confidence":
                low_confidence
        }
    }

