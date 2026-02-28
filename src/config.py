import os
# Fix PyTorch/FAISS/Tokenizers segfault on MacOS by restricting threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic_settings import BaseSettings as PydanticBaseSettings
from typing import Optional


class Settings(PydanticBaseSettings):
    GROQ_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_CHUNKS: int = 10
    COMPRESSION_RATIO: float = 0.4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_instance = None


def get_settings() -> Settings:
    global _instance
    if _instance is None:
        _instance = Settings()
    return _instance
