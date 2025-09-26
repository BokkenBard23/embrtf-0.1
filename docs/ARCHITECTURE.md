## Архитектура

### Компоненты
- `pipeline.py` — ETL: RTF → реплики → эмбеддинги → SQLite.
- `indexer.py` — построение и сохранение FAISS индексов + метаданные.
- `gui.py` — поиск, контекст, вызов методов анализа, экспорт.
- `analysis_methods.py` — стратегии генерации ответов на основе найденных реплик.
- `config.py` — централизованная конфигурация.
- `utils.py` — логирование и утилиты.

### Данные и БД
- Таблица `dialogs(id, text, metadata, source_theme, processed_at)`.
- Таблица `utterances(id, dialog_id, speaker, text, turn_order)`.
- Таблица `utterance_embeddings(utterance_id, vector BLOB)`.
- Таблица `faiss_indexes(theme, index_path, ids_path, built_at)`.

Индексы FAISS и JSON-списки ID хранятся на диске в `faiss_index/`.

### Потоки
1) Загрузка `.rtf` → `pipeline` сохраняет всё в БД.
2) `indexer` читает эмбеддинги → строит индекс(ы) на диск → пишет метаданные в БД.
3) `gui` загружает индексы и словарь `DATA_LOOKUPS` из БД → поиск → форматирование контекста → вызов анализа.

### Поиск
- Вектор запроса → поиск в FAISS (`IndexFlatIP`) по выбранному индексу.
- Сбор атрибутов по `utterance_id` из БД (`DATA_LOOKUPS`).
- Форматирование ближайшего контекста строк.
- (Опционально) rerank через CrossEncoder.


