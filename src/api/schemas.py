"""Pydantic schemas for API request/response models."""

from typing import Literal, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    mode: Literal["compressed", "full_rag", "baseline"] = "compressed"


class QueryResponse(BaseModel):
    answer: str
    mode: str
    tokens_used: int
    compression_ratio: Optional[float] = None
    time_ms: float


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    model: str
    index_size: int
