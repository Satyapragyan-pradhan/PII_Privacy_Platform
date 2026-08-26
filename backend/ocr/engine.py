import io

import pymupdf
import cv2
import numpy as np
import pytesseract

from PIL import Image

from core.config import settings

from ocr.variants import (
    create_variants,
    create_aadhaar_rois,
    create_roi_variants,
)

from ocr.scorer import score_ocr


if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = (
        settings.TESSERACT_CMD
    )


def run_tesseract(image, psm=6):
    """
    Run Tesseract OCR.
    """

    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm}"
    ).strip()


def ocr_image(content: bytes):
    """
    Main OCR pipeline.

    Strategy:

        Image
          |
          +--> Full-image variants
          |
          +--> Identity ROI
          |
          +--> Number ROI
          |
          +--> PII-aware scoring
          |
          +--> Merge useful OCR
    """

    # =========================================================
    # Load image
    # =========================================================

    image = Image.open(
        io.BytesIO(content)
    ).convert("RGB")

    image_array = np.array(
        image
    )

    opencv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # =========================================================
    # 1. Full image variants
    # =========================================================

    variants = create_variants(
        opencv_image
    )

    results = []

    for name, variant in variants.items():

        psm = 11 if name.endswith("psm11") else 6

        text = run_tesseract(
            variant,
            psm=psm
        )

        score = score_ocr(
            text
        )

        print(
            f"[OCR] {name} | score={score}"
        )

        results.append({
            "name": name,
            "text": text,
            "score": score,
        })

    # =========================================================
    # 2. Aadhaar-style ROIs
    # =========================================================

    rois = create_aadhaar_rois(
        opencv_image
    )

    roi_results = []

    for roi_name, roi in rois.items():

        roi_variants = create_roi_variants(
            roi
        )

        for variant_name, variant in roi_variants.items():

            text = run_tesseract(
                variant,
                psm=6
            )

            score = score_ocr(
                text
            )

            full_name = (
                f"{roi_name}_{variant_name}"
            )

            print(
                f"[OCR] {full_name} | score={score}"
            )

            roi_results.append({
                "name": full_name,
                "roi": roi_name,
                "text": text,
                "score": score,
            })

    # =========================================================
    # 3. Select best full-image OCR
    # =========================================================

    best_full = max(
        results,
        key=lambda x: x["score"],
        default=None
    )

    # =========================================================
    # 4. Select best result for each ROI
    # =========================================================

    best_identity = max(
        (
            r for r in roi_results
            if r["roi"] == "identity"
        ),
        key=lambda x: x["score"],
        default=None
    )

    best_number = max(
        (
            r for r in roi_results
            if r["roi"] == "number"
        ),
        key=lambda x: x["score"],
        default=None
    )

    # =========================================================
    # Debug output
    # =========================================================

    if best_full:
        print(
            f"[OCR] Best full image: "
            f"{best_full['name']} | "
            f"score={best_full['score']}"
        )

    if best_identity:
        print(
            f"[OCR] Best identity ROI: "
            f"{best_identity['name']} | "
            f"score={best_identity['score']}"
        )

    if best_number:
        print(
            f"[OCR] Best number ROI: "
            f"{best_number['name']} | "
            f"score={best_number['score']}"
        )

    # =========================================================
    # 5. Build combined OCR
    # =========================================================

    selected_parts = []

    # Identity is more valuable than garbage full-image OCR
    if (
        best_identity
        and best_identity["score"] > 0
    ):
        selected_parts.append(
            best_identity["text"]
        )

    # Number ROI
    if (
        best_number
        and best_number["score"] > 0
    ):
        selected_parts.append(
            best_number["text"]
        )

    # ---------------------------------------------------------
    # If ROIs produced nothing useful, fall back to full image
    # ---------------------------------------------------------

    if not selected_parts and best_full:
        selected_parts.append(
            best_full["text"]
        )

    final_text = "\n".join(
        part
        for part in selected_parts
        if part
    ).strip()

    print(
        "\n========== SELECTED OCR OUTPUT =========="
    )

    print(final_text)

    print(
        "=========================================="
    )

    return final_text


def ocr_pdf_pages(document):
    """
    OCR scanned PDF pages.
    """

    text_parts = []

    for page_number, page in enumerate(document):

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2)
        )

        image_bytes = pixmap.tobytes(
            "png"
        )

        text = ocr_image(
            image_bytes
        )

        if text:

            text_parts.append(
                f"Page {page_number + 1}\n{text}"
            )

    return "\n".join(
        text_parts
    )