import sqlite3
import os
import config


def init_db(db_path: str | None = None):
    """Инициализирует SQLite БД и создаёт все необходимые таблицы.

    Использует каскадное удаление для связанных сущностей.
    """
    path = str(db_path or config.DATABASE_PATH)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

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
            FOREIGN KEY (dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
        );
        """,

        """
        CREATE TABLE IF NOT EXISTS utterance_embeddings (
            utterance_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE
        );
        """,

        """
        CREATE TABLE IF NOT EXISTS embeddings (
            dialog_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY (dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
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
            FOREIGN KEY (dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
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

    for sql in tables_sql:
        cursor.execute(sql)

    conn.commit()
    conn.close()

    print(f"✅ База данных '{path}' инициализирована. Все таблицы созданы.")


if __name__ == "__main__":
    init_db()