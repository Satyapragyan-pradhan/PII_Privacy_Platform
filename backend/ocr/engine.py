import io

import pymupdf
import numpy as np
import pytesseract

from PIL import Image
from paddleocr import PaddleOCR

from core.config import settings


# =========================================================
# Tesseract configuration
# =========================================================

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = (
        settings.TESSERACT_CMD
    )


# =========================================================
# PaddleOCR singleton
# =========================================================

_paddle_ocr = None


def get_paddle_ocr():
    """
    Initialize PaddleOCR only once.

    The model loading cost is paid only on the first OCR
    request after the backend starts.
    """

    global _paddle_ocr

    if _paddle_ocr is None:

        print(
            "[OCR] Initializing PaddleOCR..."
        )

        _paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print(
            "[OCR] PaddleOCR initialized successfully."
        )

    return _paddle_ocr


# =========================================================
# Tesseract fallback
# =========================================================

def run_tesseract(image, psm=6):
    """
    Run Tesseract OCR as a fallback.
    """

    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm}"
    ).strip()


# =========================================================
# Extract PaddleOCR result
# =========================================================

def extract_paddle_result(result):
    """
    Extract recognized text and recognition scores
    from PaddleOCR output.

    Only rec_texts and rec_scores are passed into the
    downstream PII extraction pipeline.
    """

    texts = []
    scores = []

    for item in result:

        # PaddleOCR result objects support dictionary-style
        # access for fields such as rec_texts/rec_scores.
        try:
            rec_texts = item["rec_texts"]
            rec_scores = item["rec_scores"]

        except Exception:

            # Defensive fallback for object-style access.
            rec_texts = getattr(
                item,
                "rec_texts",
                []
            )

            rec_scores = getattr(
                item,
                "rec_scores",
                []
            )

        if rec_texts is None:
            rec_texts = []

        if rec_scores is None:
            rec_scores = []

        for text, score in zip(
            rec_texts,
            rec_scores
        ):

            text = str(text).strip()

            if not text:
                continue

            texts.append(text)
            scores.append(
                float(score)
            )

    return texts, scores


# =========================================================
# PaddleOCR
# =========================================================

def run_paddle_ocr(content: bytes):
    """
    Run PaddleOCR on an image.

    Returns:
        text
        average OCR confidence
    """

    image = Image.open(
        io.BytesIO(content)
    ).convert("RGB")

    image_array = np.array(
        image
    )

    ocr = get_paddle_ocr()

    print(
        "[OCR] Running PaddleOCR..."
    )

    result = ocr.predict(
        image_array
    )

    texts, scores = extract_paddle_result(
        result
    )

    if not texts:
        print(
            "[OCR] PaddleOCR returned no text."
        )

        return "", 0.0

    final_text = "\n".join(
        texts
    ).strip()

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    print(
        f"[OCR] PaddleOCR detected "
        f"{len(texts)} text regions."
    )

    print(
        f"[OCR] Average recognition score: "
        f"{average_score:.4f}"
    )

    print(
        "\n========== PADDLE OCR OUTPUT =========="
    )

    for text, score in zip(
        texts,
        scores
    ):
        print(
            f"[{score:.4f}] {text}"
        )

    print(
        "======================================="
    )

    return final_text, average_score


# =========================================================
# Main image OCR
# =========================================================

def ocr_image(content: bytes):
    """
    Main OCR pipeline.

    Strategy:

        Image
          |
          v
       PaddleOCR
          |
          +---- text found ----> downstream pipeline
          |
          +---- failed --------> Tesseract fallback
    """

    # -----------------------------------------------------
    # 1. Primary OCR: PaddleOCR
    # -----------------------------------------------------

    try:

        text, ocr_score = run_paddle_ocr(
            content
        )

        if text.strip():

            print(
                f"[OCR] PaddleOCR selected "
                f"(score={ocr_score:.4f})"
            )

            return text

        print(
            "[OCR] PaddleOCR produced no usable text."
        )

    except Exception as e:

        print(
            f"[OCR] PaddleOCR failed: {e}"
        )

    # -----------------------------------------------------
    # 2. Fallback: Tesseract
    # -----------------------------------------------------

    print(
        "[OCR] Falling back to Tesseract..."
    )

    try:

        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")

        text = run_tesseract(
            image,
            psm=6
        )

        if text.strip():

            print(
                "[OCR] Tesseract fallback succeeded."
            )

            return text

    except Exception as e:

        print(
            f"[OCR] Tesseract fallback failed: {e}"
        )

    # -----------------------------------------------------
    # 3. Nothing worked
    # -----------------------------------------------------

    print(
        "[OCR] All OCR engines failed."
    )

    return ""


# =========================================================
# Scanned PDF OCR
# =========================================================

def ocr_pdf_pages(document):
    """
    OCR scanned PDF pages.

    Each page is rendered to an image and passed through
    the same PaddleOCR-first pipeline.
    """

    text_parts = []

    for page_number, page in enumerate(
        document
    ):

        print(
            f"\n[OCR] Processing PDF page "
            f"{page_number + 1}"
        )

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
                f"Page {page_number + 1}\n"
                f"{text}"
            )

    return "\n".join(
        text_parts
    )