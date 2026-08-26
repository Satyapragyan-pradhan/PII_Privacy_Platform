import time
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.extraction_service import process_documents

router = APIRouter()


@router.post("/extract")
async def extract_pii(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    start_time = time.perf_counter()

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    documents = [{
        "filename": file.filename,
        "content_type": file.content_type,
        "content": content
    }]

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