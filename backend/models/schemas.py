from typing import List, Optional

from pydantic import BaseModel, Field


class PIIEntity(BaseModel):
    value: str
    type: str
    confidence: float = Field(
        ge=0,
        le=1
    )
    source: str
    validated: bool = False


class DocumentResult(BaseModel):
    document: str
    entities: List[PIIEntity]
    status: str


class Analytics(BaseModel):
    total_entities: int
    high_confidence: int
    low_confidence: int


class ExtractionResponse(BaseModel):
    status: str
    documents_processed: int
    documents: List[DocumentResult]
    analytics: Analytics
    processing_time_ms: Optional[float] = None