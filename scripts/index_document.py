import argparse
import os
from src.document_processing.loader import load_pdf, load_docx, split_into_chunks
from src.retrieval.vector_store import VectorStore
from src.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Index a document")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the document file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return

    # Get settings
    settings = get_settings()

    # Determine file type and load
    file_extension = os.path.splitext(args.file)[1].lower()

    if file_extension == ".pdf":
        pages = load_pdf(args.file)
    elif file_extension == ".docx":
        pages = load_docx(args.file)
    else:
        print(
            f"Error: Unsupported file format '{file_extension}'. Only .pdf and .docx are supported."
        )
        return

    # Split into chunks
    chunks = split_into_chunks(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    if not chunks:
        print("❌ Ошибка: В документе не найдено текста для индексации.")
        print("   Убедитесь, что PDF содержит текстовый слой (не является картинкой).")
        return

    # Create vector store and add documents
    vector_store = VectorStore()
    vector_store.add_documents(chunks)

    print(f"✅ Успешно проиндексировано {len(chunks)} чанков")


if __name__ == "__main__":
    main()
