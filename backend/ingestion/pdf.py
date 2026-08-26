import io
import pymupdf


def extract_pdf_text(content: bytes):
    document = pymupdf.open(
        stream=io.BytesIO(content),
        filetype="pdf"
    )

    pages = []
    text_parts = []

    for page_number, page in enumerate(document):
        text = page.get_text("text").strip()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

        if text:
            text_parts.append(text)

    full_text = "\n".join(text_parts)

    return {
        "text": full_text,
        "needs_ocr": len(full_text.strip()) < 20,
        "pages": pages,
        "document": document
    }