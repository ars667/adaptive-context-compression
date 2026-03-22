"""FastAPI route handlers."""

import time
import tempfile
import os
from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas import QueryRequest, QueryResponse, UploadResponse, HealthResponse
from src.pipeline import RAGPipeline

router = APIRouter(prefix="/api")

# Shared pipeline instance (initialized once on first request)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
def health():
    """Check service status."""
    pipeline = get_pipeline()
    index_size = pipeline.vector_store.index.ntotal if pipeline.vector_store.index else 0
    return HealthResponse(
        status="ok",
        model=pipeline.llm.model,
        index_size=index_size,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question using the RAG pipeline."""
    if request.mode not in ("compressed", "full_rag", "baseline"):
        raise HTTPException(status_code=400, detail="Invalid mode")

    pipeline = get_pipeline()
    start = time.perf_counter()
    result = pipeline.query(request.question, mode=request.mode)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result["answer"],
        mode=result["mode"],
        tokens_used=result["tokens_used"],
        compression_ratio=result.get("compression_ratio"),
        time_ms=round(elapsed_ms, 1),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pipeline = get_pipeline()

    # Save to temp file then index
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_indexed = pipeline.load_document(tmp_path)
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        status="ok",
        filename=file.filename,
        chunks_indexed=chunks_indexed,
    )
