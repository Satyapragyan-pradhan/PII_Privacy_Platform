import time

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

    format_valid = validate_entity(
        entity_type,
        value
    )

    entity["format_valid"] = format_valid

    methods_agree = entity.get(
        "methods_agree",
        False
    )

    entity["methods_agree"] = methods_agree

    confidence = calculate_confidence(
        entity
    )

    entity["confidence"] = confidence

    entity["validated"] = format_valid

    entity["confidence_level"] = (
        confidence_label(
            confidence
        )
    )

    return entity


def calculate_analytics(entities):

    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0

    for entity in entities:

        confidence = float(
            entity.get(
                "confidence",
                0
            )
        )

        if confidence >= 0.90:
            high_confidence += 1

        elif confidence >= 0.70:
            medium_confidence += 1

        else:
            low_confidence += 1

    return {
        "total_entities": len(entities),
        "high_confidence": high_confidence,
        "medium_confidence": medium_confidence,
        "low_confidence": low_confidence
    }


def process_single_document(document):

    total_start = time.perf_counter()

    filename = document[
        "filename"
    ]

    content = document[
        "content"
    ]

    timings = {
        "loading_ms": 0,
        "ocr_ms": 0,
        "extraction_ms": 0,
        "enrichment_ms": 0,
        "total_ms": 0
    }

    # Load document

    start = time.perf_counter()

    loaded = load_document(
        filename,
        content
    )

    timings["loading_ms"] = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    text = loaded.get(
        "text",
        ""
    )

    # OCR

    if loaded.get("needs_ocr"):

        start = time.perf_counter()

        if loaded.get("document") is not None:

            ocr_text = ocr_pdf_pages(
                loaded["document"]
            )

        else:

            ocr_text = ocr_image(
                content
            )

        timings["ocr_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        if ocr_text:
            text = ocr_text

    # No text

    if not text.strip():

        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000,
            2
        )

        return {
            "document": filename,
            "entities": [],
            "status": "no_text",
            "analytics": calculate_analytics([]),
            "processing_time_ms": timings["total_ms"],
            "timings": timings
        }

    print(
        "\n========== OCR / INPUT TEXT =========="
    )

    print(text)

    print(
        "======================================\n"
    )

    # Extraction graph

    start = time.perf_counter()

    result = GRAPH.invoke({

        "text": text,

        "regex_entities": [],

        "nlp_entities": [],

        "contextual_entities": [],

        "preliminary_entities": [],

        "final_entities": []

    })

    timings["extraction_ms"] = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    entities = result.get(
        "final_entities",
        []
    )

    # Final enrichment

    start = time.perf_counter()

    entities = [
        enrich_entity(entity)
        for entity in entities
    ]

    timings["enrichment_ms"] = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    timings["total_ms"] = round(
        (time.perf_counter() - total_start) * 1000,
        2
    )

    analytics = calculate_analytics(
        entities
    )

    print(
        f"[TIMING] {filename}: "
        f"{timings['total_ms']} ms"
    )

    return {
        "document": filename,
        "entities": entities,
        "status": "success",
        "analytics": analytics,
        "processing_time_ms": timings["total_ms"],
        "timings": timings
    }


def process_documents(documents):

    results = []

    total_entities = 0
    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0

    batch_start = time.perf_counter()

    for document in documents:

        result = process_single_document(
            document
        )

        results.append(
            result
        )

        analytics = result.get(
            "analytics",
            {}
        )

        total_entities += analytics.get(
            "total_entities",
            0
        )

        high_confidence += analytics.get(
            "high_confidence",
            0
        )

        medium_confidence += analytics.get(
            "medium_confidence",
            0
        )

        low_confidence += analytics.get(
            "low_confidence",
            0
        )

    batch_time = round(
        (time.perf_counter() - batch_start) * 1000,
        2
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
        },

        "processing_time_ms":
            batch_time
    }