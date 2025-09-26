## Конфигурация (`config.py`)

### Пути и директории
- `PROJECT_ROOT`, `INPUT_ROOT`, `PROCESSED_ROOT`, `LOGS_ROOT`, `FAISS_INDEX_DIR`, `CACHE_ROOT`, `EXPORTS_ROOT`, `TEMP_ROOT` — создаются автоматически при импорте.
- `DATABASE_PATH` — SQLite база (по умолчанию `database.db`).

### Модели
- **Эмбеддинги**:
  - `EMBEDDING_MODEL_NAME` (пример: `Qwen/Qwen3-Embedding-0.6B`)
  - `EMBEDDING_MODEL_DEVICE`: `cuda` или `cpu`
  - `EMBEDDING_MODEL_PRECISION`: `float16`/`float32`
  - `EMBEDDING_MODEL_MAX_LENGTH`, `EMBEDDING_MODEL_DIMENSION`
- **Reranker (опционально)**:
  - `RERANKER_MODEL_NAME`, `RERANKER_MAX_LENGTH`, `RERANKER_ENABLED`
- **LLM для HyDE/чата**:
  - `LLM_MODEL_NAME`, `LLM_API_URL`, `LLM_TIMEOUT`

### Индексация FAISS
- `FAISS_INDEX_TYPE` (фактическая реализация — `IndexFlatIP`)
- `FAISS_NLIST`, `FAISS_M` — пригодятся при смене типа индекса.

### GUI и анализ
- `GUI_DEFAULT_TOP_K`, `GUI_DEFAULT_CHUNK_SIZE`, `GUI_DEFAULT_METHOD`, `ANALYSIS_METHODS`
- `GUI_THEME` — цвета интерфейса.
- `MAX_WORKERS`, `DEBUG_MODE`

### HyDE
- `HYDE_ENABLED` и `HYDE_PROMPT_TEMPLATE`. В текущем GUI HyDE отключён для поиска, но шаблон можно использовать в своих методах анализа.

### Внешние ключи и API
- `OPENROUTER_API_KEY` — ключ OpenRouter (если используете их API).
- `CLOUD_DATABASE_PATH` — отдельная БД под облачную обработку (если требуется).
- `LLM_INTERNAL_API_KEY`, `LLM_INTERNAL_API_URL`, `LLM_INTERNAL_MODEL` — внутренний LLM (корп. среда).

### Рекомендации по секретам
- Храните ключи в переменных окружения, не коммитьте в VCS.
- Смотрите секцию «Безопасность и ключи» в `README.md`.


