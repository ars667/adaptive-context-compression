import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from src.config import get_settings


class VectorStore:
    def __init__(self, index_path: str = "data/indexes/faiss_index.bin"):
        self.settings = get_settings()
        self.model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.index_path = index_path
        self.chunks_path = index_path.replace(".bin", "_chunks.pkl")

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add documents to the vector store and create FAISS index."""
        self.chunks = chunks

        # Extract texts and create embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # 🔧 Fix for empty or 1D arrays
        if len(embeddings) == 0:
            print("⚠️ Warning: No embeddings generated")
            return

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Add embeddings to index
        self.index.add(embeddings)

        # Save index and chunks
        self.save()
        print(f"✅ Индексировано {len(chunks)} чанков")

    def save(self) -> None:
        """Сохраняет FAISS индекс и чанки в файлы"""
        if self.index is None:
            return

        # Сохраняем FAISS индекс
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        # Сохраняем чанки через pickle
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"💾 Сохранено: {self.index_path}, {self.chunks_path}")

    @classmethod
    def load(
        cls, index_path: str = "data/indexes/faiss_index.bin", model_name: str = None
    ) -> "VectorStore":
        """Загружает FAISS индекс и чанки из файлов"""
        from src.config import get_settings

        settings = get_settings()
        model_name = model_name or settings.EMBEDDING_MODEL

        instance = cls(index_path=index_path)
        instance.model = SentenceTransformer(model_name)

        # Загружаем FAISS индекс
        if os.path.exists(index_path):
            instance.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"Index not found: {index_path}")

        # Загружаем чанки
        chunks_path = index_path.replace(".bin", "_chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                instance.chunks = pickle.load(f)
        else:
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        print(f"📂 Загружено: {len(instance.chunks)} чанков")
        return instance

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Search for top_k chunks similar to the query."""
        if self.index is None:
            raise ValueError("Index not created. Call add_documents() first.")

        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        elif query_embedding.ndim > 2:
            query_embedding = query_embedding.reshape(query_embedding.shape[0], -1)

        try:
            faiss.normalize_L2(query_embedding)
            query_embedding = query_embedding.astype(np.float32)
            D, I = self.index.search(query_embedding, top_k)
            distances, indices = D, I
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []

        # Get results with scores
        results = []
        if indices is not None and distances is not None:
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    distance = distances[0][i] if i < len(distances[0]) else 0
                    score = float(distance)  # Inner product score

                    # Bonus for special chunks
                    if chunk.get("is_formula", False) or chunk.get(
                        "is_definition", False
                    ):
                        score += 0.1

                    results.append(
                        {
                            "text": chunk["text"],
                            "chunk_id": chunk["chunk_id"],
                            "source": chunk["source"],
                            "score": score,
                            "is_formula": chunk.get("is_formula", False),
                            "is_definition": chunk.get("is_definition", False),
                            "is_code": chunk.get("is_code", False),
                        }
                    )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
