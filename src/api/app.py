"""FastAPI application for the adaptive context compression web service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Веб-сервис адаптивного сжатия контекста",
    description=(
        "RAG-система для работы с учебной литературой. "
        "Поддерживает три режима: compressed, full_rag, baseline."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
