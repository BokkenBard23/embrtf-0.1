## Сборка проекта (Windows)

Инструкция по упаковке в исполняемые файлы с помощью PyInstaller. Цель — собрать отдельные приложения для `gui.py` и, при необходимости, консольных утилит (`pipeline.py`, `indexer.py`).

### Подготовка окружения
```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
pip install pyinstaller
```

### Важные замечания
- Приложение читает/записывает файлы рядом с рабочей директорией (папки `Input`, `processed`, `faiss_index`, `logs`, `exports`, `temp`). Убедитесь, что у процесса есть права записи.
- При первой сборке полезно запускать один-два раза из консоли для проверки путей и логов.
- Если используете GPU, убедитесь, что устанавливаете совместимые версии `torch`/CUDA перед сборкой.

### Сборка GUI (onefile)
Команда:
```bash
pyinstaller --noconfirm --clean ^
  --name CallCenterGUI ^
  --onefile ^
  --windowed ^
  --add-data "faiss_index;faiss_index" ^
  --add-data "Input;Input" ^
  --add-data "processed;processed" ^
  --add-data "exports;exports" ^
  --add-data "temp;temp" ^
  gui.py
```

Пояснения:
- `--windowed` скрывает консоль. Для отладки можно убрать этот флаг.
- `--add-data "src;dst"` копирует директории рядом с бинарём. На Windows разделитель `;`.
- Если директории пустые — создайте плейсхолдеры, иначе PyInstaller может их игнорировать.

Артефакт: `dist/CallCenterGUI.exe`.

### Сборка GUI (onedir)
```bash
pyinstaller --noconfirm --clean ^
  --name CallCenterGUI ^
  --onedir ^
  --windowed ^
  gui.py
```

Артефакт: папка `dist/CallCenterGUI/` с `CallCenterGUI.exe`. Проще добавлять ресурсы вручную.

### Сборка консольных утилит
Примеры для `pipeline.py` и `indexer.py`:
```bash
pyinstaller --noconfirm --clean --name pipeline_cli --onefile pipeline.py
pyinstaller --noconfirm --clean --name indexer_cli  --onefile indexer.py
```

### Ресурсы и динамические данные
- Пути в `config.py` рассчитываются от расположения файла `config.py` (корень проекта). При упаковке используйте структуру, где рядом с исполняемым лежат папки данных, либо адаптируйте `config.py` для режима `frozen` (пример ниже).

Пример адаптации путей для frozen-режима (опционально):
```python
from pathlib import Path
import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.resolve()
```

### Тестирование сборки
1) Запустите `dist/CallCenterGUI.exe`.
2) Проверьте логи в `logs/gui.log`.
3) Выполните тестовый поиск в теме `all`. Если индексы отсутствуют — соберите их командой `indexer_cli.exe` или вручную `python indexer.py`.

### Размер и оптимизация
- Уменьшайте зависимости в `requirements.txt`.
- Для onedir-режима можно включать UPX, если доступен (`--upx-dir`).

### Частые проблемы
- Отсутствуют индексы: соберите их после запуска `pipeline`.
- Нет прав записи: запустите из директории, где есть права, или от имени администратора.
- Ошибки CUDA при запуске: используйте CPU-сборку либо корректные версии CUDA/cuDNN.



