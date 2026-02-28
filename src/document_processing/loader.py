import re
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict


def load_pdf(path: str) -> List[str]:
    """Load PDF file and extract text pages."""
    doc = fitz.open(path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append(text)
    doc.close()
    return pages


def load_docx(path: str) -> List[str]:
    """Load DOCX file and extract text pages."""
    doc = Document(path)
    pages = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            pages.append(paragraph.text)
    return pages


def split_into_chunks(pages: List[str], chunk_size: int, overlap: int) -> List[Dict]:
    """Split pages into chunks with metadata."""
    chunks = []
    chunk_id = 0

    for page_num, page_text in enumerate(pages):
        source = f"page_{page_num + 1}"

        # Simple splitting by sentences (can be improved)
        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunk = {
                        "text": current_chunk.strip(),
                        "chunk_id": chunk_id,
                        "source": source,
                        "is_formula": _is_formula(current_chunk.strip()),
                        "is_definition": _is_definition(current_chunk.strip()),
                        "is_code": _is_code(current_chunk.strip()),
                    }
                    chunks.append(chunk)
                    chunk_id += 1

                current_chunk = sentence + " "

                # Handle overlap
                if overlap > 0:
                    words = current_chunk.split()
                    if len(words) > overlap:
                        current_chunk = " ".join(words[-overlap:]) + " "
                    else:
                        current_chunk = ""

        # Add the last chunk
        if current_chunk.strip():
            chunk = {
                "text": current_chunk.strip(),
                "chunk_id": chunk_id,
                "source": source,
                "is_formula": _is_formula(current_chunk.strip()),
                "is_definition": _is_definition(current_chunk.strip()),
                "is_code": _is_code(current_chunk.strip()),
            }
            chunks.append(chunk)
            chunk_id += 1

    return chunks


def _is_formula(text: str) -> bool:
    """Check if text contains mathematical formulas."""
    pattern = r"\$.*\$|\$\$.*\$\$"
    return bool(re.search(pattern, text))


def _is_definition(text: str) -> bool:
    """Check if text contains definition or theorem keywords."""
    keywords = ["определение", "теорема", "definition", "theorem"]
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _is_code(text: str) -> bool:
    """Check if text contains code blocks."""
    return "```" in text
