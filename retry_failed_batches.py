"""Скрипт для поиска и повторной обработки диалогов без эмбеддингов."""

import sqlite3
import config
import os
from pathlib import Path

def find_and_retry():
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    # Найти диалоги, у которых есть реплики, но нет эмбеддингов
    cursor.execute("""
        SELECT DISTINCT u.dialog_id
        FROM utterances u
        LEFT JOIN utterance_embeddings ue ON u.id = ue.utterance_id
        WHERE ue.utterance_id IS NULL
    """)
    dialog_ids = [row[0] for row in cursor.fetchall()]

    if not dialog_ids:
        print("✅ Все диалоги имеют эмбеддинги.")
        return

    print(f"⚠️ Найдено {len(dialog_ids)} диалогов без эмбеддингов. Перемещаем файлы обратно...")

    # Переместить файлы обратно в Input
    for dialog_id in dialog_ids:
        # Получаем путь к файлу из metadata
        cursor.execute("SELECT metadata FROM dialogs WHERE id = ?", (dialog_id,))
        row = cursor.fetchone()
        if not row:
            continue
        try:
            meta = json.loads(row[0])
            theme = meta.get("source_theme", "unknown")
            filename = f"{dialog_id}.rtf"  # или восстановить оригинальное имя
            proc_path = config.PROCESSED_ROOT / theme / filename
            input_path = config.INPUT_ROOT / theme / filename
            if proc_path.exists():
                proc_path.rename(input_path)
                print(f"  Перемещён: {filename}")
        except Exception as e:
            print(f"  ❌ Ошибка перемещения {dialog_id}: {e}")

    # Удалить записи из БД
    placeholders = ','.join('?' * len(dialog_ids))
    cursor.execute(f"DELETE FROM utterance_embeddings WHERE utterance_id IN (SELECT id FROM utterances WHERE dialog_id IN ({placeholders}))", dialog_ids)
    cursor.execute(f"DELETE FROM utterances WHERE dialog_id IN ({placeholders})", dialog_ids)
    cursor.execute(f"DELETE FROM dialogs WHERE id IN ({placeholders})", dialog_ids)
    conn.commit()
    conn.close()

    print("✅ Подготовлено к повторной обработке. Запусти pipeline.py снова.")

if __name__ == "__main__":
    find_and_retry()