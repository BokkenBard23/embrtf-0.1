"""Модуль для управления данными в CallCenter AI v3."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import config
from utils import get_db_connection
import logging

# Проверяем наличие pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas не установлен. Некоторые функции будут недоступны.")

logger = logging.getLogger(__name__)

class DataManager:
    """Класс для управления данными системы."""
    
    def __init__(self):
        self.conn = get_db_connection()
        
    def get_dialogs_count(self) -> int:
        """Получить количество диалогов."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dialogs")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка получения количества диалогов: {e}")
            return 0
    
    def get_utterances_count(self) -> int:
        """Получить количество реплик."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM utterances")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка получения количества реплик: {e}")
            return 0
    
    def get_embeddings_count(self) -> int:
        """Получить количество эмбеддингов."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM utterance_embeddings")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка получения количества эмбеддингов: {e}")
            return 0
    
    def get_callback_phrases_count(self) -> int:
        """Получить количество фраз обратных звонков."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM callback_phrases")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка получения количества фраз: {e}")
            return 0
    
    def get_qa_pairs_count(self) -> int:
        """Получить количество пар вопрос-ответ."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM qa_pairs")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка получения количества QA-пар: {e}")
            return 0
    
    def get_dialogs_data(self, limit: int = 100, offset: int = 0):
        """Получить данные диалогов."""
        try:
            query = """
                SELECT id, text, metadata, source_theme, processed_at
                FROM dialogs
                ORDER BY processed_at DESC
                LIMIT ? OFFSET ?
            """
            if PANDAS_AVAILABLE:
                return pd.read_sql_query(query, self.conn, params=(limit, offset))
            else:
                # Возвращаем список словарей если pandas недоступен
                cursor = self.conn.cursor()
                cursor.execute(query, (limit, offset))
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка получения данных диалогов: {e}")
            return [] if not PANDAS_AVAILABLE else pd.DataFrame()
    
    def get_utterances_data(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Получить данные реплик."""
        try:
            query = """
                SELECT u.id, u.dialog_id, u.speaker, u.text, u.turn_order,
                       d.source_theme, d.processed_at
                FROM utterances u
                JOIN dialogs d ON u.dialog_id = d.id
                ORDER BY d.processed_at DESC, u.turn_order
                LIMIT ? OFFSET ?
            """
            return pd.read_sql_query(query, self.conn, params=(limit, offset))
        except Exception as e:
            logger.error(f"Ошибка получения данных реплик: {e}")
            return pd.DataFrame()
    
    def get_callback_phrases_data(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Получить данные фраз обратных звонков."""
        try:
            query = """
                SELECT id, phrase, source, category, frequency, verified, processed_at
                FROM callback_phrases
                ORDER BY frequency DESC, processed_at DESC
                LIMIT ? OFFSET ?
            """
            return pd.read_sql_query(query, self.conn, params=(limit, offset))
        except Exception as e:
            logger.error(f"Ошибка получения данных фраз: {e}")
            return pd.DataFrame()
    
    def get_qa_pairs_data(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Получить данные пар вопрос-ответ."""
        try:
            query = """
                SELECT id, timestamp, question, theme, method_used, 
                       parameters, answer, context_summary
                FROM qa_pairs
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            return pd.read_sql_query(query, self.conn, params=(limit, offset))
        except Exception as e:
            logger.error(f"Ошибка получения данных QA-пар: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить общую статистику системы."""
        try:
            stats = {
                'dialogs': self.get_dialogs_count(),
                'utterances': self.get_utterances_count(),
                'embeddings': self.get_embeddings_count(),
                'callback_phrases': self.get_callback_phrases_count(),
                'qa_pairs': self.get_qa_pairs_count()
            }
            
            # Дополнительная статистика
            cursor = self.conn.cursor()
            
            # Статистика по темам
            cursor.execute("""
                SELECT source_theme, COUNT(*) as count
                FROM dialogs
                GROUP BY source_theme
                ORDER BY count DESC
            """)
            stats['themes'] = dict(cursor.fetchall())
            
            # Статистика по категориям фраз
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM callback_phrases
                GROUP BY category
                ORDER BY count DESC
            """)
            stats['phrase_categories'] = dict(cursor.fetchall())
            
            # Статистика по методам анализа
            cursor.execute("""
                SELECT method_used, COUNT(*) as count
                FROM qa_pairs
                GROUP BY method_used
                ORDER BY count DESC
            """)
            stats['analysis_methods'] = dict(cursor.fetchall())
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def search_dialogs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск диалогов по тексту."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, text, metadata, source_theme, processed_at
                FROM dialogs
                WHERE text LIKE ?
                ORDER BY processed_at DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска диалогов: {e}")
            return []
    
    def search_utterances(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск реплик по тексту."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT u.id, u.dialog_id, u.speaker, u.text, u.turn_order,
                       d.source_theme, d.processed_at
                FROM utterances u
                JOIN dialogs d ON u.dialog_id = d.id
                WHERE u.text LIKE ?
                ORDER BY d.processed_at DESC, u.turn_order
                LIMIT ?
            """, (f"%{query}%", limit))
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска реплик: {e}")
            return []
    
    def search_callback_phrases(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск фраз обратных звонков."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, phrase, source, category, frequency, verified, processed_at
                FROM callback_phrases
                WHERE phrase LIKE ?
                ORDER BY frequency DESC, processed_at DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска фраз: {e}")
            return []
    
    def export_to_csv(self, data_type: str, filename: str, limit: int = 1000) -> bool:
        """Экспорт данных в CSV файл."""
        try:
            if data_type == 'dialogs':
                df = self.get_dialogs_data(limit=limit)
            elif data_type == 'utterances':
                df = self.get_utterances_data(limit=limit)
            elif data_type == 'callback_phrases':
                df = self.get_callback_phrases_data(limit=limit)
            elif data_type == 'qa_pairs':
                df = self.get_qa_pairs_data(limit=limit)
            else:
                logger.error(f"Неизвестный тип данных: {data_type}")
                return False
            
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Данные экспортированы в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}")
            return False
    
    def export_to_json(self, data_type: str, filename: str, limit: int = 1000) -> bool:
        """Экспорт данных в JSON файл."""
        try:
            if data_type == 'dialogs':
                df = self.get_dialogs_data(limit=limit)
            elif data_type == 'utterances':
                df = self.get_utterances_data(limit=limit)
            elif data_type == 'callback_phrases':
                df = self.get_callback_phrases_data(limit=limit)
            elif data_type == 'qa_pairs':
                df = self.get_qa_pairs_data(limit=limit)
            else:
                logger.error(f"Неизвестный тип данных: {data_type}")
                return False
            
            df.to_json(filename, orient='records', force_ascii=False, indent=2)
            logger.info(f"Данные экспортированы в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}")
            return False
    
    def get_theme_list(self) -> List[str]:
        """Получить список тем."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT source_theme FROM dialogs ORDER BY source_theme")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка получения списка тем: {e}")
            return []
    
    def get_category_list(self) -> List[str]:
        """Получить список категорий фраз."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM callback_phrases ORDER BY category")
            return [str(row[0]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка получения списка категорий: {e}")
            return []
    
    def close(self):
        """Закрытие соединения с БД."""
        if self.conn:
            self.conn.close()
