import time
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.extraction_service import process_documents

router = APIRouter()


@router.post("/extract")
async def extract_pii(
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    start_time = time.perf_counter()

    documents = []

    for file in files:

        if not file:
            continue

        content = await file.read()

        if not content:
            continue

        documents.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "content": content
        })

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="All uploaded files are empty"
        )

    result = process_documents(documents)

    result["processing_time_ms"] = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    return result


@router.get("/analytics")
def analytics():
    return {
        "status": "success",
        "message": "Analytics endpoint ready"
    }