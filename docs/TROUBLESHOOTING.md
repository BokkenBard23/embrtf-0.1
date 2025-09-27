## Troubleshooting / FAQ

### Нет найденных реплик в GUI
- Убедитесь, что выполнены `pipeline.py` и `indexer.py` без ошибок.
- Проверьте, что в БД есть записи в `utterances` и `utterance_embeddings`.
- Посмотрите логи: `logs/pipeline.log`, `logs/indexer.log`, `logs/gui.log`.

### Ошибка CUDA OOM при кодировании
- `pipeline` автоматически уменьшает batch. Можно также снизить `PIPELINE_BATCH_SIZE` в `config.py`.
- Очистите память и повторите.

### Индекс не загружается
- Проверьте существование файлов `faiss_index/faiss_index_<theme>.index` и `faiss_index/ids_<theme>.json`.
- Перестройте индексы: `python indexer.py`.

### LLM/HyDE не отвечает
- Проверьте `LLM_API_URL` и что сервис доступен локально.
- Увеличьте `LLM_TIMEOUT`.

### Где хранить ключи
- Используйте переменные окружения, не коммитьте ключи в репозиторий.


