from pathlib import Path

from ingestion.pdf import extract_pdf_text
from ingestion.docx import extract_docx_text
from ingestion.excel import extract_excel_text


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg"
}


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def load_document(filename: str, content: bytes):
    extension = get_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        return extract_pdf_text(content)

    if extension == ".docx":
        return extract_docx_text(content)

    if extension in {".xlsx", ".xls"}:
        return extract_excel_text(content)

    if extension in {".png", ".jpg", ".jpeg"}:
        return {
            "text": "",
            "needs_ocr": True,
            "pages": []
        }

    raise ValueError(
        f"Unsupported document type: {extension}"
    )