"""Streamlit UI for the Adaptive Context Compression RAG."""

import os
import streamlit as st
import time

# Set environment variables to prevent Segfault on MacOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline():
    """Load the RAG pipeline once and cache it."""
    return RAGPipeline()


def init_ui():
    st.set_page_config(
        page_title="Адаптивное сжатие контекста",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 RAG: Адаптивное сжатие контекста")
    st.markdown("Умный поиск по учебной литературе с использованием FAISS, LLMLingua и Llama-3.1")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Настройки")
        mode = st.radio(
            "Режим RAG:",
            options=["compressed", "full_rag", "baseline"],
            format_func=lambda x: {
                "compressed": "🗜️ Со сжатием контекста",
                "full_rag": "📚 Полный контекст (Full RAG)",
                "baseline": "🤖 Без контекста (Baseline)"
            }[x],
            index=0
        )
        
        st.divider()
        pipeline = get_pipeline()
        
        # File uploader
        uploaded_file = st.file_uploader("Загрузить новый учебник (PDF):", type=["pdf"])
        
        if uploaded_file is not None:
            # Check if this is a new file
            if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                with st.status(f"🚀 Обработка '{uploaded_file.name}'...") as status:
                    # Save to temp file
                    os.makedirs("data/raw", exist_ok=True)
                    file_path = os.path.join("data/raw", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write("🔍 Извлечение текста (и OCR если нужно)...")
                    num_chunks = pipeline.load_document(file_path)
                    
                    st.session_state.last_uploaded_file = uploaded_file.name
                    # Clear chat history for new document
                    st.session_state.messages = []
                    status.update(label=f"✅ Готово! Проиндексировано {num_chunks} чанков.", state="complete")
                    st.rerun()

        num_chunks = len(pipeline.vector_store.chunks) if pipeline.vector_store else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Загружено чанков", num_chunks)
        
        # Display current document name
        doc_name = "Не загружен"
        if pipeline.vector_store and pipeline.vector_store.chunks:
            # Try to get source from the first chunk
            doc_name = pipeline.vector_store.chunks[0].get("source", "Универсальный индекс")
            # If it's a temp file or generic name, try to make it prettier
            if doc_name.startswith("page_"):
                doc_name = "Текущий проиндексированный учебник"
        
        col2.metric("Источник", doc_name.split('/')[-1])
        
        if num_chunks == 0:
            st.warning("Внимание: Индекс пуст! Загрузите PDF выше.")

        return mode, pipeline


def main():
    mode, pipeline = init_ui()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "metrics" in message:
                metrics = message["metrics"]
                cols = st.columns(3)
                cols[0].metric("Токены LLM", metrics["tokens_used"])
                if metrics.get("compression_ratio") is not None:
                    cols[1].metric("Ratio сжатия", f"{metrics['compression_ratio']:.2f}")

    # Chat input
    if prompt := st.chat_input("Задайте вопрос по учебнику..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Генерация ответа (режим: {mode})..."):
                start_time = time.time()
                result = pipeline.query(question=prompt, mode=mode)
                end_time = time.time()
                
                answer = result["answer"]
                st.markdown(answer)
                
                # Display metrics
                st.write(f"⏱️ Время: {end_time - start_time:.2f} сек")
                
                cols = st.columns(3)
                cols[0].metric("Токены LLM", result["tokens_used"])
                if result.get("compression_ratio") is not None:
                    cols[1].metric("Ratio сжатия", f"{result['compression_ratio']:.2f}")

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "metrics": result
        })


if __name__ == "__main__":
    main()
