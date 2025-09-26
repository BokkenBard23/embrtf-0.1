"""Конфигурационный файл для проекта CallCenter AI v3 (реплики + HyDE + Reranker)."""

import os
from pathlib import Path

# --- Основные пути ---
PROJECT_ROOT = Path(__file__).parent.resolve()
INPUT_ROOT = PROJECT_ROOT / "Input"
PROCESSED_ROOT = PROJECT_ROOT / "processed"
LOGS_ROOT = PROJECT_ROOT / "logs"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"      # Индексы FAISS
CACHE_ROOT = PROJECT_ROOT / "cache"                 # Кэш (например, search_cache)
EXPORTS_ROOT = PROJECT_ROOT / "exports"             # Экспорты из GUI
TEMP_ROOT = PROJECT_ROOT / "temp"                   # Временные файлы

# Создание всех папок
for path in [INPUT_ROOT, PROCESSED_ROOT, LOGS_ROOT, FAISS_INDEX_DIR, CACHE_ROOT, EXPORTS_ROOT, TEMP_ROOT]:
    os.makedirs(path, exist_ok=True)

# --- База данных ---
DATABASE_PATH = PROJECT_ROOT / "database.db"

# --- Модели ---
## Основная модель эмбеддингов (для реплик и диалогов)
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_DEVICE = "cuda"          # "cuda" или "cpu"
EMBEDDING_MODEL_PRECISION = "float16"    # "float16" для GPU, "float32" для CPU
EMBEDDING_MODEL_MAX_LENGTH = 4096
EMBEDDING_MODEL_DIMENSION = 1024

## Reranker (опционально)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
RERANKER_MAX_LENGTH = 512
RERANKER_ENABLED = True

## Модель для LLM (Ollama)
LLM_MODEL_NAME = "dimweb/ilyagusev-saiga_llama3_8b:kto_v5_Q4_K"
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")  # URL Ollama API

# --- Параметры обработки ---
PIPELINE_BATCH_SIZE = 32               # Размер батча при обработке файлов
PIPELINE_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PIPELINE_LOG_JSON_FORMAT = False         # True для JSON-логов (ELK, Grafana)

# --- Параметры индексации FAISS ---
FAISS_INDEX_TYPE = "IndexFlatIP"         # Тип индекса FAISS
FAISS_NLIST = 100                        # Для IVF индексов (если будешь менять тип)
FAISS_M = 32                             # Для HNSW (если понадобится)

# --- Параметры поиска и GUI ---
GUI_DEFAULT_TOP_K = 5                    # Сколько реплик показывать
GUI_DEFAULT_CHUNK_SIZE = 10              # Размер чанка для analysis_methods
GUI_DEFAULT_METHOD = "hierarchical"
ANALYSIS_METHODS = ["hierarchical", "rolling", "facts", "classification", "callback_classifier", "fast_phrase_classifier"]

# --- Настройки HyDE ---
HYDE_ENABLED = True
HYDE_PROMPT_TEMPLATE = """Ты — эксперт по анализу диалогов call-центра.
Напиши подробный, точный и естественный ответ на вопрос пользователя, как будто ты уже нашёл нужную информацию в базе диалогов.

Вопрос: {query}

Гипотетический ответ:"""

# --- Цветовая схема GUI (Темная тема) ---
GUI_THEME = {
    "dark_bg": '#2e2e2e',
    "dark_fg": '#ffffff',
    "dark_entry_bg": '#4d4d4d',
    "dark_button_bg": '#5a5a5a',
    "dark_frame_bg": '#3d3d3d',
    "success_color": '#4CAF50',   # Зеленый
    "warning_color": '#FF9800',   # Оранжевый
    "error_color": '#F44336',     # Красный
    "info_color": '#2196F3',      # Синий
    "highlight_color": '#9C27B0'  # Фиолетовый
}

# --- Системные настройки ---
MAX_WORKERS = 4                          # Для многопоточности в GUI
LLM_TIMEOUT = 60                         # Таймаут для LLM запросов (сек)
DEBUG_MODE = False                       # Режим отладки (больше логов)

# config.py (в конец файла)

# === Все 55 FREE моделей OpenRouter ===
OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "qwen/qwen3-coder:free",
    "z-ai/glm-4.5-air:free",
    "tngtech/deepseek-r1t-chimera:free",
    "moonshotai/kimi-k2:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "microsoft/mai-ds-r1:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "openai/gpt-oss-20b:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-4-maverick:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "qwen/qwen3-14b:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "mistralai/mistral-nemo:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "agentica-org/deepcoder-14b-preview:free",
    "moonshotai/kimi-dev-72b:free",
    "qwen/qwen3-30b-a3b:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-4-scout:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "qwen/qwq-32b:free",
    "qwen/qwen3-8b:free",
    "mistralai/devstral-small-2505:free",
    "qwen/qwen3-4b:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "shisa-ai/shisa-v2-llama3.3-70b:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "moonshotai/kimi-vl-a3b-thinking:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "tencent/hunyuan-a13b-instruct:free",
    "google/gemma-3-12b-it:free",
    "arliai/qwq-32b-arliai-rpr-v1:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "google/gemma-3n-e2b-it:free",
    "google/gemma-2-9b-it:free",
    "openai/gpt-oss-120b:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-4b-it:free",
    "rekaai/reka-flash-3:free"
]

# Твой ключ OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# config.py (добавить в конец)

# === База данных для облачной обработки ===
CLOUD_DATABASE_PATH = PROJECT_ROOT / "callback_cloud.db"

# config.py (добавить в конец)

# === Внутренний LLM API (Билайн) ===
LLM_INTERNAL_API_KEY = os.getenv("LLM_INTERNAL_API_KEY", "")
LLM_INTERNAL_API_URL = os.getenv("LLM_INTERNAL_API_URL", "")
LLM_INTERNAL_MODEL = os.getenv("LLM_INTERNAL_MODEL", "Qwen3-32B") # Или "RuadaptQwen" или "QwQ"