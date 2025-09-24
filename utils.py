# utils.py
"""Вспомогательные функции для проекта."""

import logging
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