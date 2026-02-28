"""Main RAG pipeline with three modes: baseline, full_rag, compressed."""

from typing import Dict, Optional
from src.config import get_settings
from src.retrieval.vector_store import VectorStore
from src.compression.compressor import ContextCompressor
from src.llm.groq_client import GroqClient
from src.document_processing.loader import load_pdf


class RAGPipeline:
    """End-to-end RAG pipeline with adaptive context compression."""

    def __init__(self):
        settings = get_settings()
        self.top_k = int(settings.TOP_K_CHUNKS)

        # Initialize components
        self.vector_store = VectorStore()
        self.compressor = ContextCompressor()
        self.llm = GroqClient()

        self.current_document: Optional[str] = None

    def load_document(self, file_path: str) -> int:
        """
        Load and index a document.

        Args:
            file_path: Path to PDF/DOCX file

        Returns:
            Number of chunks indexed
        """
        # Load document (support PDF for now)
        if file_path.endswith(".pdf"):
            chunks = load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Add to vector store (this also saves the index)
        self.vector_store.add_documents(chunks)
        self.current_document = file_path

        return len(chunks)

    def query(self, question: str, mode: str = "compressed") -> Dict:
        """
        Process a query using specified mode.

        Args:
            question: User question
            mode: One of "baseline", "full_rag", "compressed"

        Returns:
            Dict with answer, mode, tokens_used, compression_ratio
        """
        result = {
            "answer": "",
            "mode": mode,
            "tokens_used": 0,
            "compression_ratio": None,
        }

        if mode == "baseline":
            # No context, direct LLM query
            response = self.llm.generate(context="", query=question)
            result["answer"] = response["answer"]
            result["tokens_used"] = response["tokens_used"]

        elif mode == "full_rag":
            # Retrieve top-k chunks, no compression
            chunks = self.vector_store.search(question, top_k=self.top_k)
            context = "\n\n".join([chunk["text"] for chunk in chunks])

            response = self.llm.generate(context=context, query=question)
            result["answer"] = response["answer"]
            result["tokens_used"] = response["tokens_used"]

        elif mode == "compressed":
            # Retrieve → compress → LLM
            chunks = self.vector_store.search(question, top_k=self.top_k)

            # Compress with protection
            compression_result = self.compressor.compress(chunks, query=question)
            context = compression_result["compressed_text"]

            response = self.llm.generate(context=context, query=question)
            result["answer"] = response["answer"]
            result["tokens_used"] = response["tokens_used"]
            result["compression_ratio"] = compression_result["compression_ratio"]

        else:
            result["answer"] = (
                f"Unknown mode: {mode}. Use: baseline, full_rag, compressed"
            )

        return result
