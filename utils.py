# utils.py
"""Вспомогательные функции для проекта."""

import logging
import sqlite3
import config

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Настройка логгера с выводом в файл и поддержкой формата."""
    formatter = logging.Formatter(config.PIPELINE_LOG_FORMAT)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    
    # Чтобы логи не дублировались
    logger.propagate = False
    
    return logger


def get_db_connection(db_path: str | None = None, check_same_thread: bool = False):
    """Создает подключение к SQLite с включенными foreign_keys.

    Args:
        db_path: Путь к БД. По умолчанию берется из config.DATABASE_PATH.
        check_same_thread: Пробрасывается в sqlite3.connect для GUI-потоков.
    """
    path = str(db_path or config.DATABASE_PATH)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    return conn