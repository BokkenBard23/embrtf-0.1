# init_db.py
import sqlite3
import os

# Путь к базе данных
DB_PATH = "database.db"  # Убедись, что это тот же путь, что и у тебя

# Создание соединения
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Включаем поддержку foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

# Создание таблиц
tables_sql = [
    """
    CREATE TABLE IF NOT EXISTS dialogs (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        metadata TEXT NOT NULL,
        source_theme TEXT NOT NULL,
        processed_at TEXT NOT NULL
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS utterances (
        id TEXT PRIMARY KEY,
        dialog_id TEXT NOT NULL,
        speaker TEXT NOT NULL,
        text TEXT NOT NULL,
        turn_order INTEGER NOT NULL,
        FOREIGN KEY (dialog_id) REFERENCES dialogs(id)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS utterance_embeddings (
        utterance_id TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        FOREIGN KEY (utterance_id) REFERENCES utterances(id)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS embeddings (
        dialog_id TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        FOREIGN KEY (dialog_id) REFERENCES dialogs(id)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS faiss_indexes (
        theme TEXT PRIMARY KEY,
        index_path TEXT NOT NULL,
        ids_path TEXT NOT NULL,
        built_at TEXT NOT NULL
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS callback_phrases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phrase TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        category INTEGER NOT NULL,
        frequency INTEGER DEFAULT 1,
        verified BOOLEAN DEFAULT 0
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS qa_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        question TEXT NOT NULL,
        theme TEXT NOT NULL,
        method_used TEXT NOT NULL,
        parameters TEXT NOT NULL,
        answer TEXT NOT NULL,
        context_summary TEXT NOT NULL
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS raw_llm_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dialog_id TEXT NOT NULL,
        analysis_result TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (dialog_id) REFERENCES dialogs(id)
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL
    );
    """,

    """
    CREATE TABLE IF NOT EXISTS search_cache (
        query_hash TEXT PRIMARY KEY,
        results TEXT NOT NULL
    );
    """
]

# Выполняем создание всех таблиц
for sql in tables_sql:
    cursor.execute(sql)

# Сохраняем изменения
conn.commit()
conn.close()

print(f"✅ База данных '{DB_PATH}' инициализирована. Все таблицы созданы.")