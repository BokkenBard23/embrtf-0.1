"""Индексация эмбеддингов реплик (utterances) в FAISS. Поддержка тематических индексов.
   Оптимизирован: батчи, логи, проверка, экономия памяти."""

import os
import json
import pickle
import sqlite3
import logging
from pathlib import Path
from datetime import datetime  # Импорт для записи времени построения
import faiss
import numpy as np
from tqdm import tqdm

# === Импорт конфигурации ===
import config

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_ROOT / "indexer.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Indexer")

BATCH_SIZE = 1024  # Размер батча для добавления в FAISS

def get_themes_from_db(conn):
    """Получает список всех уникальных тем из БД."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source_theme FROM dialogs WHERE source_theme IS NOT NULL")
    themes = [row[0] for row in cursor.fetchall() if row[0]]
    return themes

def build_faiss_index_for_theme(theme_name, conn, index_path, ids_path):
    """Строит FAISS-индекс для заданной темы (или 'all') на основе реплик."""
    logger.info(f"🔍 Начало построения индекса для темы: '{theme_name}'...")

    # Запрос: подсчитываем количество реплик для темы
    if theme_name == "all":
        count_query = "SELECT COUNT(*) FROM utterance_embeddings ue JOIN utterances u ON ue.utterance_id = u.id"
        fetch_query = """
            SELECT ue.utterance_id, ue.vector
            FROM utterance_embeddings ue
            JOIN utterances u ON ue.utterance_id = u.id
        """
    else:
        count_query = """
            SELECT COUNT(*)
            FROM utterance_embeddings ue
            JOIN utterances u ON ue.utterance_id = u.id
            JOIN dialogs d ON u.dialog_id = d.id
            WHERE d.source_theme = ?
        """
        fetch_query = """
            SELECT ue.utterance_id, ue.vector
            FROM utterance_embeddings ue
            JOIN utterances u ON ue.utterance_id = u.id
            JOIN dialogs d ON u.dialog_id = d.id
            WHERE d.source_theme = ?
        """

    cursor = conn.cursor()
    cursor.execute(count_query, (theme_name,) if theme_name != "all" else ())
    total = cursor.fetchone()[0]

    if total == 0:
        logger.warning(f"⚠️ Для темы '{theme_name}' не найдено реплик для индексации.")
        return False

    logger.info(f"📊 Найдено {total} реплик для индексации.")

    # Создаём FAISS индекс
    dimension = config.EMBEDDING_MODEL_DIMENSION
    index = faiss.IndexFlatIP(dimension)  # Можно заменить на IndexIVFFlat при желании
    # Нормализация будет происходить перед добавлением векторов

    utterance_ids = []
    offset = 0

    pbar = tqdm(total=total, desc=f"Индексация '{theme_name}'", unit="реплика")

    # Цикл обработки батчей
    while offset < total:
        # Формируем параметры запроса
        if theme_name == "all":
            params = (BATCH_SIZE, offset)
        else:
            params = (theme_name, BATCH_SIZE, offset)

        cursor.execute(fetch_query + " LIMIT ? OFFSET ?", params)
        rows = cursor.fetchall()

        if not rows:
            break

        vectors = []
        batch_ids = []

        for row in rows:
            utterance_id, vector_blob = row
            try:
                vector = pickle.loads(vector_blob)
                if isinstance(vector, np.ndarray):
                    vector = vector.astype('float32')
                else:
                    vector = np.array(vector, dtype='float32')
                vectors.append(vector)
                batch_ids.append(utterance_id)
            except Exception as e:
                logger.warning(f"❌ Ошибка десериализации вектора {utterance_id}: {e}")
                continue

        if vectors:
            vectors = np.array(vectors)
            faiss.normalize_L2(vectors)
            index.add(vectors)
            utterance_ids.extend(batch_ids)

        pbar.update(len(batch_ids))
        offset += BATCH_SIZE

    pbar.close()

    # --- Сохранение индекса и ID ---
    try:
        # Убедимся, что директория существует
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # Сохраняем FAISS индекс (преобразуем Path в str)
        faiss.write_index(index, str(index_path))
        # Сохраняем список ID
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(utterance_ids, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Индекс для '{theme_name}' сохранён на диск: {index_path}, {len(utterance_ids)} реплик.")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения индекса/ID для '{theme_name}' на диск: {e}")
        return False

    # --- Сохранение метаданных индекса в БД ---
    try:
        meta_cursor = conn.cursor()
        meta_cursor.execute("""
            INSERT OR REPLACE INTO faiss_indexes (theme, index_path, ids_path, built_at)
            VALUES (?, ?, ?, ?)
        """, (theme_name, str(index_path), str(ids_path), datetime.now().isoformat()))
        conn.commit()
        logger.info(f"✅ Метаданные индекса для '{theme_name}' сохранены в БД.")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения метаданных индекса для '{theme_name}' в БД: {e}")
        conn.rollback() # Откатываем, если это было внутри транзакции
        # Не возвращаем False, так как индекс на диске уже создан

    return True

def main():
    """Основная функция для построения всех индексов."""
    logger.info("🚀 Начало построения FAISS-индексов для реплик...")
    
    # Подключение к БД
    conn = sqlite3.connect(config.DATABASE_PATH)
    
    # Получаем список тем
    themes = get_themes_from_db(conn)
    themes.append("all")  # Добавляем общую индексацию

    # Строим индекс для каждой темы
    for theme in themes:
        index_path = config.FAISS_INDEX_DIR / f"faiss_index_{theme}.index"
        ids_path = config.FAISS_INDEX_DIR / f"ids_{theme}.json"
        
        success = build_faiss_index_for_theme(theme, conn, index_path, ids_path)
        if not success:
            logger.error(f"❌ Не удалось построить индекс для темы '{theme}'.")

    conn.close()
    logger.info("🏁 Построение индексов завершено.")

if __name__ == "__main__":
    main()