"""Оптимизированный процессор эмбеддингов для RTX 3060 12GB."""

import torch
import gc
import logging
from typing import List, Dict, Any, Optional, Generator
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
from pathlib import Path
import config
from utils import get_db_connection
import time

logger = logging.getLogger(__name__)

class OptimizedEmbeddingProcessor:
    """Оптимизированный процессор эмбеддингов с использованием GPU и батчевой обработки."""
    
    def __init__(self, model_name: str = None, batch_size: int = 32, max_length: int = 2048):
        """
        Инициализация процессора.
        
        Args:
            model_name: Название модели (по умолчанию из config)
            batch_size: Размер батча для обработки
            max_length: Максимальная длина последовательности
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.device = None
        self.conn = None
        
        # Определяем оптимальное устройство
        self._setup_device()
        
    def _setup_device(self):
        """Настройка устройства для обработки."""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Очищаем кэш CUDA
            torch.cuda.empty_cache()
            logger.info(f"✅ Используем GPU: {torch.cuda.get_device_name()}")
            logger.info(f"📊 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
            logger.info("⚠️ CUDA недоступна, используем CPU")
    
    def _load_model_lazy(self):
        """Ленивая загрузка модели только при необходимости."""
        if self.model is None:
            try:
                logger.info(f"🔄 Загрузка модели {self.model_name} на {self.device}...")
                
                # Загружаем модель с оптимизацией памяти
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=self.device,
                    cache_folder='./model_cache'  # Кэш для модели
                )
                
                # Настраиваем параметры для экономии памяти
                self.model.max_seq_length = self.max_length
                
                # Оптимизируем для GPU
                if self.device == "cuda":
                    # Используем mixed precision если доступно
                    if hasattr(torch.cuda, 'amp'):
                        self.model.half()  # Используем float16 для экономии памяти
                        logger.info("✅ Включен mixed precision (float16)")
                    
                    # Очищаем кэш после загрузки
                    torch.cuda.empty_cache()
                
                logger.info(f"✅ Модель загружена успешно на {self.device}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели: {e}")
                # Fallback на CPU
                if self.device == "cuda":
                    logger.warning("⚠️ Fallback на CPU...")
                    self.device = "cpu"
                    torch.cuda.empty_cache()
                    return self._load_model_lazy()
                else:
                    raise e
    
    def _unload_model(self):
        """Выгрузка модели из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("🗑️ Модель выгружена из памяти")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Получение информации об использовании памяти."""
        memory_info = {}
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            memory_info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return memory_info
    
    def _log_memory_usage(self, stage: str):
        """Логирование использования памяти."""
        memory = self._get_memory_usage()
        if self.device == "cuda":
            logger.info(f"💾 {stage} - GPU: {memory['gpu_allocated']:.2f}GB/{memory['gpu_total']:.2f}GB")
        else:
            logger.info(f"💾 {stage} - CPU режим")
    
    def process_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Обработка текстов батчами.
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Массив эмбеддингов
        """
        if not texts:
            return np.array([])
        
        # Загружаем модель если нужно
        self._load_model_lazy()
        
        try:
            self._log_memory_usage("Начало обработки батча")
            
            # Обрабатываем батчами
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                logger.info(f"🔄 Обработка батча {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
                # Обрабатываем батч
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=min(self.batch_size, len(batch_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Нормализуем для лучшего качества
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Очищаем промежуточные данные
                del batch_embeddings
                gc.collect()
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self._log_memory_usage(f"После батча {i//self.batch_size + 1}")
            
            # Объединяем все эмбеддинги
            result = np.vstack(all_embeddings)
            
            self._log_memory_usage("Завершение обработки")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки батча: {e}")
            raise e
    
    def process_dialogs_batch(self, dialog_ids: List[str], conn: sqlite3.Connection) -> Dict[str, np.ndarray]:
        """
        Обработка диалогов батчами.
        
        Args:
            dialog_ids: Список ID диалогов
            conn: Соединение с БД
            
        Returns:
            Словарь {dialog_id: embedding}
        """
        if not dialog_ids:
            return {}
        
        logger.info(f"🔄 Обработка {len(dialog_ids)} диалогов батчами по {self.batch_size}")
        
        cursor = conn.cursor()
        results = {}
        
        try:
            # Загружаем модель
            self._load_model_lazy()
            
            for i in range(0, len(dialog_ids), self.batch_size):
                batch_ids = dialog_ids[i:i + self.batch_size]
                
                logger.info(f"📊 Батч {i//self.batch_size + 1}/{(len(dialog_ids) + self.batch_size - 1)//self.batch_size}")
                
                # Получаем тексты диалогов
                placeholders = ','.join('?' * len(batch_ids))
                cursor.execute(f"""
                    SELECT id, text FROM dialogs 
                    WHERE id IN ({placeholders})
                """, batch_ids)
                
                batch_data = cursor.fetchall()
                
                if not batch_data:
                    continue
                
                # Извлекаем тексты
                batch_texts = [text for _, text in batch_data]
                batch_dialog_ids = [dialog_id for dialog_id, _ in batch_data]
                
                # Обрабатываем эмбеддинги
                batch_embeddings = self.process_texts_batch(batch_texts)
                
                # Сохраняем результаты
                for j, dialog_id in enumerate(batch_dialog_ids):
                    results[dialog_id] = batch_embeddings[j]
                
                # Очищаем память
                del batch_texts, batch_embeddings
                gc.collect()
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self._log_memory_usage(f"После обработки батча {i//self.batch_size + 1}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки диалогов: {e}")
            raise e
        finally:
            # Выгружаем модель после обработки
            self._unload_model()
    
    def process_utterances_batch(self, utterance_ids: List[str], conn: sqlite3.Connection) -> Dict[str, np.ndarray]:
        """
        Обработка реплик батчами.
        
        Args:
            utterance_ids: Список ID реплик
            conn: Соединение с БД
            
        Returns:
            Словарь {utterance_id: embedding}
        """
        if not utterance_ids:
            return {}
        
        logger.info(f"🔄 Обработка {len(utterance_ids)} реплик батчами по {self.batch_size}")
        
        cursor = conn.cursor()
        results = {}
        
        try:
            # Загружаем модель
            self._load_model_lazy()
            
            for i in range(0, len(utterance_ids), self.batch_size):
                batch_ids = utterance_ids[i:i + self.batch_size]
                
                logger.info(f"📊 Батч {i//self.batch_size + 1}/{(len(utterance_ids) + self.batch_size - 1)//self.batch_size}")
                
                # Получаем тексты реплик
                placeholders = ','.join('?' * len(batch_ids))
                cursor.execute(f"""
                    SELECT id, text FROM utterances 
                    WHERE id IN ({placeholders})
                """, batch_ids)
                
                batch_data = cursor.fetchall()
                
                if not batch_data:
                    continue
                
                # Извлекаем тексты
                batch_texts = [text for _, text in batch_data]
                batch_utterance_ids = [utterance_id for utterance_id, _ in batch_data]
                
                # Обрабатываем эмбеддинги
                batch_embeddings = self.process_texts_batch(batch_texts)
                
                # Сохраняем результаты
                for j, utterance_id in enumerate(batch_utterance_ids):
                    results[utterance_id] = batch_embeddings[j]
                
                # Очищаем память
                del batch_texts, batch_embeddings
                gc.collect()
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self._log_memory_usage(f"После обработки батча {i//self.batch_size + 1}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки реплик: {e}")
            raise e
        finally:
            # Выгружаем модель после обработки
            self._unload_model()
    
    def save_embeddings_to_db(self, embeddings: Dict[str, np.ndarray], 
                             table_name: str, conn: sqlite3.Connection):
        """
        Сохранение эмбеддингов в БД.
        
        Args:
            embeddings: Словарь {id: embedding}
            table_name: Название таблицы
            conn: Соединение с БД
        """
        if not embeddings:
            return
        
        logger.info(f"💾 Сохранение {len(embeddings)} эмбеддингов в таблицу {table_name}")
        
        cursor = conn.cursor()
        
        try:
            # Подготавливаем данные для вставки
            data_to_insert = []
            for item_id, embedding in embeddings.items():
                # Конвертируем в bytes для хранения в БД
                embedding_bytes = embedding.astype(np.float32).tobytes()
                data_to_insert.append((item_id, embedding_bytes))
            
            # Вставляем батчами
            batch_size = 100
            for i in range(0, len(data_to_insert), batch_size):
                batch_data = data_to_insert[i:i + batch_size]
                
                if table_name == "embeddings":
                    cursor.executemany("""
                        INSERT OR REPLACE INTO embeddings (dialog_id, vector)
                        VALUES (?, ?)
                    """, batch_data)
                elif table_name == "utterance_embeddings":
                    cursor.executemany("""
                        INSERT OR REPLACE INTO utterance_embeddings (utterance_id, vector)
                        VALUES (?, ?)
                    """, batch_data)
                
                conn.commit()
            
            logger.info(f"✅ Эмбеддинги сохранены в {table_name}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения эмбеддингов: {e}")
            raise e
    
    def get_optimal_batch_size(self) -> int:
        """Определение оптимального размера батча для текущей конфигурации."""
        if self.device == "cuda":
            # Для RTX 3060 12GB оптимальный размер батча
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 12:  # RTX 3060 12GB
                return 64
            elif gpu_memory >= 8:  # RTX 3070/3080 8GB
                return 32
            else:  # Меньше памяти
                return 16
        else:
            # Для CPU
            return 16
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        self._unload_model()

def main():
    """Тестирование процессора эмбеддингов."""
    logging.basicConfig(level=logging.INFO)
    
    # Создаем процессор
    processor = OptimizedEmbeddingProcessor(
        batch_size=32,  # Начнем с консервативного размера
        max_length=2048
    )
    
    # Тестируем на небольшом наборе данных
    test_texts = [
        "Привет, как дела?",
        "У меня проблема с интернетом",
        "Спасибо за помощь",
        "Когда будет готово?",
        "Хорошо, понял"
    ]
    
    logger.info("🧪 Тестирование процессора эмбеддингов...")
    
    try:
        # Тестируем обработку текстов
        embeddings = processor.process_texts_batch(test_texts)
        logger.info(f"✅ Обработано {len(embeddings)} текстов")
        logger.info(f"📊 Размер эмбеддинга: {embeddings.shape}")
        
        # Проверяем память
        memory = processor._get_memory_usage()
        if processor.device == "cuda":
            logger.info(f"💾 Использование GPU: {memory['gpu_allocated']:.2f}GB")
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")

if __name__ == "__main__":
    main()
