import os
import sys

# 🔧 КРИТИЧЕСКИЙ ФИКС ДЛЯ MACOS
# Предотвращает Segmentation Fault (exit code 139) при одновременной загрузке
# нескольких Transformer-моделей (sentence-transformers + llmlingua) и FAISS.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🌐 ОФФЛАЙН-РЕЖИМ ДЛЯ МОДЕЛЕЙ
# Если модели уже скачаны, эти флаги предотвращают попытки лезть в интернет
# и зависания из-за таймаутов Hugging Face.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
