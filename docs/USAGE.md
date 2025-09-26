## Примеры использования

### Быстрый конвейер
```bash
python pipeline.py   # 1) обработать новые .rtf
python indexer.py    # 2) переиндексировать
python gui.py        # 3) искать и анализировать в GUI
```

### Формат данных
- Каждая строка реплики в полном тексте: `Speaker: text [HH:MM:SS]` (квадратные скобки — опционально).
- `DATA_LOOKUPS` собирает для каждой реплики: `text, speaker, dialog_id, turn_order, full_dialog_text`.

### Методы анализа
- Выбираются из `analysis_methods.get_analysis_method(name)`.
- Передаётся: `question`, `found_with_scores`, `chunk_size`, `status_callback`.

### Советы
- Для больших коллекций увеличивайте batch в `pipeline` осторожно.
- Следите за `EMBEDDING_MODEL_MAX_LENGTH` (см. `analyze_dialogs.py`).
- GPU ускоряет `MODEL.encode`, но требует аккуратного batch и памяти.


