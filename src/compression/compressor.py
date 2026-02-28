"""Context compression using LLMLingua with protection for formulas/definitions."""

from typing import List, Dict
from llmlingua import PromptCompressor
from src.config import get_settings


class ContextCompressor:
    """Compresses retrieved chunks while protecting important content."""

    def __init__(self):
        settings = get_settings()
        self.compression_ratio = float(settings.COMPRESSION_RATIO)

        # Initialize LLMLingua-2 with multilingual BERT model
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )

    def compress(self, chunks: List[Dict], query: str) -> Dict:
        """
        Compress chunks while protecting formulas, definitions, and code.

        Args:
            chunks: List of chunk dicts from vector store
            query: User question for query-aware compression

        Returns:
            Dict with compressed_text, original_tokens, compressed_tokens, compression_ratio
        """
        if not chunks:
            return {
                "compressed_text": "",
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 1.0,
            }

        protected_parts = []
        compressible_parts = []

        # Separate protected and compressible content
        for chunk in chunks:
            text = chunk["text"]
            # Protect formulas, definitions, and code
            if (
                chunk.get("is_formula")
                or chunk.get("is_definition")
                or chunk.get("is_code")
            ):
                protected_parts.append(text)
            else:
                compressible_parts.append(text)

        # Count original tokens (rough estimate: 1 token ≈ 4 chars for multilingual)
        original_text = "\n\n".join(compressible_parts)
        original_tokens = len(original_text) // 4 if original_text else 0

        # Compress only the compressible parts
        if compressible_parts and self.compression_ratio < 1.0:
            try:
                # LLMLingua compress method
                compressed_result = self.compressor.compress_prompt(
                    compressible_parts,
                    question=query,
                    rate=self.compression_ratio,
                )
                compressed_text = compressed_result.get(
                    "compressed_prompt", original_text
                )
                compressed_tokens = len(compressed_text) // 4 if compressed_text else 0
            except Exception as e:
                print(f"⚠️ Compression failed: {e}, using original text")
                compressed_text = original_text
                compressed_tokens = original_tokens
        else:
            compressed_text = original_text
            compressed_tokens = original_tokens

        # Combine protected + compressed parts
        final_parts = (
            protected_parts + [compressed_text] if compressed_text else protected_parts
        )
        final_text = "\n\n".join(filter(None, final_parts))
        final_tokens = len(final_text) // 4 if final_text else 0

        # Calculate actual compression ratio
        actual_ratio = (
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )

        return {
            "compressed_text": final_text,
            "original_tokens": original_tokens
            + len(protected_parts) * 50,  # rough estimate
            "compressed_tokens": final_tokens + len(protected_parts) * 50,
            "compression_ratio": round(actual_ratio, 2),
        }
