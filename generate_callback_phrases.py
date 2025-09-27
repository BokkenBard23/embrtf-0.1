"""Универсальный генератор фраз обратных звонков с поддержкой всех API."""

import sqlite3
import json
import requests
import time
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("callback_phrases.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Промпт для анализа диалогов
PROMPT_TEMPLATE = """Ты — строгий контролёр качества call-центра. Твоя задача — проанализировать диалог и определить, содержит ли он **ошибку оператора по правилу обратного звонка**.

Правила:
1. Ошибка есть, если:
   - Клиент говорит о намерении перезвонить/связаться/обратиться — **чётко и без неопределённости** (без слов "если что", "может быть", "как-нибудь", "не знаю", "с другого номера").
   - Оператор **НЕ** предложил обратный звонок.

2. Ошибка НЕТ, если:
   - Клиент дал **неопределённое обещание** (например, "если что перезвоню").
   - Оператор **предложил обратный звонок** (например, "мы вам перезвоним", "назначил звонок на 15:00").

Твоя задача:
- Проанализируй диалог.
- Если ошибка есть — выведи ТОЛЬКО одну строку в формате JSON: {"error": true, "client_phrase": "дословная фраза клиента", "operator_phrase": ""}
- Если ошибки нет — выведи ТОЛЬКО одну строку в формате JSON: {"error": false, "client_phrase": "дословная фраза клиента (если была)", "operator_phrase": "дословная фраза оператора (если была)"}

ВАЖНО:
- Фразы должны быть ДОСЛОВНЫМИ, как в диалоге.
- Не добавляй пояснений, не пиши "думаю", не добавляй markdown.
- Если фраз нет — верни пустую строку "".
- Не выдумывай фразы. Только то, что есть в тексте.

Диалог:
{dialog_text}"""

class CallbackPhraseGenerator:
    """Универсальный генератор фраз обратных звонков."""
    
    def __init__(self, api_type: str = "ollama", db_path: str = None):
        """
        Инициализация генератора.
        
        Args:
            api_type: Тип API ("ollama", "openrouter", "internal", "multi")
            db_path: Путь к базе данных
        """
        self.api_type = api_type
        self.db_path = db_path or config.DATABASE_PATH
        self.conn = None
        self.cursor = None
        
    def connect_db(self):
        """Подключение к базе данных."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Создаем таблицу если не существует
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS callback_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dialog_id TEXT,
                phrase TEXT NOT NULL,
                source TEXT NOT NULL,
                category INTEGER NOT NULL,
                frequency INTEGER DEFAULT 1,
                verified BOOLEAN DEFAULT 0,
                raw_response TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        
    def close_db(self):
        """Закрытие соединения с БД."""
        if self.conn:
            self.conn.close()
            
    def call_ollama_api(self, dialog_text: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Вызов Ollama API."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    config.LLM_API_URL,
                    json={
                        "model": config.LLM_MODEL_NAME,
                        "prompt": PROMPT_TEMPLATE.format(dialog_text=dialog_text),
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        logger.warning(f"Ollama error: {result['error']}")
                        continue
                        
                    raw_text = result.get("response", "").strip()
                    return self._parse_response(raw_text)
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        return None
        
    def call_openrouter_api(self, dialog_text: str, model: str = None) -> Optional[Dict[str, Any]]:
        """Вызов OpenRouter API."""
        if not config.OPENROUTER_API_KEY:
            logger.error("OpenRouter API key не настроен")
            return None
            
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        model = model or "deepseek/deepseek-chat-v3.1:free"
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": PROMPT_TEMPLATE.format(dialog_text=dialog_text)}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"]
                return self._parse_response(result_text)
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.call_openrouter_api(dialog_text, model)
            else:
                logger.error(f"OpenRouter error {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            
        return None
        
    def call_internal_api(self, dialog_text: str) -> Optional[Dict[str, Any]]:
        """Вызов внутреннего API."""
        if not config.LLM_INTERNAL_API_KEY or not config.LLM_INTERNAL_API_URL:
            logger.error("Internal API не настроен")
            return None
            
        headers = {
            "Authorization": f"Bearer {config.LLM_INTERNAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                config.LLM_INTERNAL_API_URL,
                headers=headers,
                json={
                    "model": config.LLM_INTERNAL_MODEL,
                    "messages": [
                        {"role": "user", "content": PROMPT_TEMPLATE.format(dialog_text=dialog_text)}
                    ],
                    "temperature": 0.1
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"]
                return self._parse_response(result_text)
            else:
                logger.error(f"Internal API error {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Internal API error: {e}")
            
        return None
        
    def _parse_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Парсинг ответа от API."""
        try:
            # Ищем JSON в ответе
            start = raw_text.find('{')
            if start == -1:
                return None
                
            for end in range(start + 1, len(raw_text) + 1):
                try:
                    candidate = raw_text[start:end]
                    result = json.loads(candidate)
                    if "error" in result and "client_phrase" in result and "operator_phrase" in result:
                        return result
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа: {e}")
            
        return None
        
    def process_dialog(self, dialog_id: str, dialog_text: str) -> bool:
        """Обработка одного диалога."""
        logger.info(f"Обработка диалога {dialog_id}...")
        
        # Выбираем API в зависимости от типа
        if self.api_type == "ollama":
            result = self.call_ollama_api(dialog_text)
        elif self.api_type == "openrouter":
            result = self.call_openrouter_api(dialog_text)
        elif self.api_type == "internal":
            result = self.call_internal_api(dialog_text)
        elif self.api_type == "multi":
            # Пробуем все модели OpenRouter
            for model in config.OPENROUTER_FREE_MODELS[:5]:  # Ограничиваем для тестирования
                result = self.call_openrouter_api(dialog_text, model)
                if result:
                    break
        else:
            logger.error(f"Неизвестный тип API: {self.api_type}")
            return False
            
        if not result:
            logger.warning(f"Не удалось обработать диалог {dialog_id}")
            return False
            
        # Сохраняем результат
        self._save_phrases(dialog_id, result)
        return True
        
    def _save_phrases(self, dialog_id: str, result: Dict[str, Any]):
        """Сохранение фраз в базу данных."""
        raw_response = json.dumps(result, ensure_ascii=False)
        
        # Сохраняем фразу клиента
        if result.get("client_phrase"):
            if result.get("error", False):
                source = "client_promise"
                category = 1
            else:
                source = "client_uncertain"
                category = 3
                
            self.cursor.execute("""
                INSERT OR IGNORE INTO callback_phrases (dialog_id, phrase, source, category, raw_response)
                VALUES (?, ?, ?, ?, ?)
            """, (dialog_id, result["client_phrase"], source, category, raw_response))
            
        # Сохраняем фразу оператора
        if result.get("operator_phrase"):
            self.cursor.execute("""
                INSERT OR IGNORE INTO callback_phrases (dialog_id, phrase, source, category, raw_response)
                VALUES (?, 'operator_offer', 2, ?)
            """, (dialog_id, result["operator_phrase"], raw_response))
            
        self.conn.commit()
        
    def process_all_dialogs(self, max_workers: int = 1):
        """Обработка всех диалогов."""
        self.connect_db()
        
        # Получаем все диалоги
        self.cursor.execute("SELECT id, text FROM dialogs")
        dialogs = self.cursor.fetchall()
        
        logger.info(f"Найдено {len(dialogs)} диалогов для обработки")
        
        if max_workers == 1:
            # Последовательная обработка
            for i, (dialog_id, dialog_text) in enumerate(dialogs, 1):
                logger.info(f"Обработка {i}/{len(dialogs)}")
                self.process_dialog(dialog_id, dialog_text)
        else:
            # Параллельная обработка
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_dialog, dialog_id, dialog_text)
                    for dialog_id, dialog_text in dialogs
                ]
                
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        future.result()
                        if i % 10 == 0:
                            logger.info(f"Обработано {i} диалогов")
                    except Exception as e:
                        logger.error(f"Ошибка обработки диалога: {e}")
                        
        self.close_db()
        logger.info("Обработка завершена")

def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Генератор фраз обратных звонков")
    parser.add_argument("--api", choices=["ollama", "openrouter", "internal", "multi"], 
                       default="ollama", help="Тип API для использования")
    parser.add_argument("--workers", type=int, default=1, help="Количество потоков")
    parser.add_argument("--db", help="Путь к базе данных")
    
    args = parser.parse_args()
    
    generator = CallbackPhraseGenerator(api_type=args.api, db_path=args.db)
    generator.process_all_dialogs(max_workers=args.workers)

if __name__ == "__main__":
    main()