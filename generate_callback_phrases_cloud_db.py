# generate_callback_phrases_cloud_db.py
"""Обработка диалогов через OpenRouter API (DeepSeek V3.1). Результаты сохраняются в отдельную БД."""

import sqlite3
import json
import requests
import time
import os
from pathlib import Path

# Конфигурация
OPENROUTER_API_KEY = "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"
MODEL = "deepseek/deepseek-chat-v3.1:free"  # DeepSeek V3.1 вместо OpenAI
REQUESTS_PER_MINUTE = 20  # Лимит OpenRouter
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # ~3 секунды

# Пути к базам данных
BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "db" / "call_center.db"  # Основная БД с диалогами
OPENROUTER_DB_PATH = BASE_DIR / "db" / "callback_phrases_openrouter.db"  # Новая БД для результатов

# Промпт (без использования .format())
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
"""

def ensure_db_directories():
    """Создает директории для БД, если они не существуют."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OPENROUTER_DB_PATH), exist_ok=True)

def find_dialogs_table(cursor):
    """Ищет таблицу с диалогами в базе данных."""
    # Получаем список всех таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Проверяем каждую таблицу на наличие полей id и text
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Проверяем названия столбцов
        has_id = False
        has_text = False
        for column in columns:
            col_name = column[1].lower()
            if "id" in col_name:
                has_id = True
            if "text" in col_name or "content" in col_name:
                has_text = True
        
        if has_id and has_text:
            print(f"✅ Найдена таблица с диалогами: {table_name}")
            return table_name
    
    print("⚠️ Таблица с диалогами не найдена. Проверьте структуру базы данных.")
    return None

def extract_phrases_with_openrouter(dialog_text):
    """Отправляет диалог в OpenRouter API с обработкой ошибок и повторами."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Формируем запрос БЕЗ использования .format() - напрямую добавляем текст
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE + dialog_text}
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    for attempt in range(3):  # 3 попытки
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"]
                # Обрабатываем возможные форматирования ответа
                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    # Если ответ не чистый JSON, пытаемся извлечь его из текста
                    if '{' in result_text and '}' in result_text:
                        json_start = result_text.find('{')
                        json_end = result_text.rfind('}') + 1
                        json_str = result_text[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        print(f"⚠️ Не удалось распарсить ответ: {result_text}")
                        result = {"error": False, "client_phrase": "", "operator_phrase": ""}
                return result
                
            elif response.status_code == 404:
                print(f"⚠️ Ошибка 404: API endpoint не найден. Проверьте URL.")
                return {"error": False, "client_phrase": "", "operator_phrase": ""}
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"⚠️ Превышен лимит. Ждем {retry_after} сек. (Заголовок Retry-After)")
                time.sleep(retry_after)
                continue
            else:
                print(f"⚠️ Ошибка HTTP {response.status_code}: {response.text}")
                time.sleep(DELAY_BETWEEN_REQUESTS * 2)
                continue
                
        except Exception as e:
            print(f"⚠️ Ошибка при вызове API: {e}")
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

    return {"error": False, "client_phrase": "", "operator_phrase": "", "error_msg": "Все попытки не удались"}

def main():
    """Основная функция обработки диалогов."""
    ensure_db_directories()
    
    # Подключение к ОСНОВНОЙ БД для чтения диалогов
    conn_main = sqlite3.connect(DATABASE_PATH)
    cursor_main = conn_main.cursor()
    
    # Ищем таблицу с диалогами
    dialogs_table = find_dialogs_table(cursor_main)
    if not dialogs_table:
        print("❌ Не удалось найти таблицу с диалогами. Выход.")
        return
    
    # Подключение к ОТДЕЛЬНОЙ БД для результатов OpenRouter
    conn_openrouter = sqlite3.connect(OPENROUTER_DB_PATH)
    cursor_openrouter = conn_openrouter.cursor()
    
    # Создаем таблицу в новой БД, если она не существует
    cursor_openrouter.execute("""
        CREATE TABLE IF NOT EXISTS callback_phrases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dialog_id TEXT NOT NULL,
            phrase TEXT NOT NULL,
            source TEXT NOT NULL,
            category INTEGER NOT NULL,
            frequency INTEGER DEFAULT 1,
            verified BOOLEAN DEFAULT 0,
            raw_response TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Получаем все диалоги из основной БД
    cursor_main.execute(f"SELECT id, text FROM {dialogs_table}")
    dialogs = cursor_main.fetchall()
    
    print(f"✅ Найдено {len(dialogs)} диалогов в основной БД.")
    
    # Обрабатываем диалоги с соблюдением лимитов
    processed_count = 0
    for i, (dialog_id, text) in enumerate(dialogs, 1):
        start_time = time.time()
        
        print(f"🔍 Обработка диалога {i}/{len(dialogs)} (ID: {dialog_id})...")
        result = extract_phrases_with_openrouter(text)
        
        # Сохраняем результат в БД OpenRouter
        if result.get("client_phrase") or result.get("operator_phrase"):
            raw_response_str = json.dumps(result, ensure_ascii=False)
            
            # Определяем категорию и источник
            if result.get("error", False):
                source = "client_promise"
                category = 1  # Ошибка
            else:
                source = "client_uncertain"
                category = 3  # Ненужный звонок
                
            # Сохраняем фразу клиента
            if result.get("client_phrase"):
                cursor_openrouter.execute("""
                    INSERT INTO callback_phrases (dialog_id, phrase, source, category, raw_response)
                    VALUES (?, ?, ?, ?, ?)
                """, (dialog_id, result["client_phrase"], source, category, raw_response_str))
            
            # Сохраняем фразу оператора (если есть)
            if result.get("operator_phrase"):
                cursor_openrouter.execute("""
                    INSERT INTO callback_phrases (dialog_id, phrase, source, category, raw_response)
                    VALUES (?, ?, 'operator_offer', 2, ?)
                """, (dialog_id, result["operator_phrase"], raw_response_str))
                
            conn_openrouter.commit()
            processed_count += 1
            print(f"✅ Диалог {dialog_id} обработан и сохранен.")
        else:
            print(f"ℹ️ Для диалога {dialog_id} не найдено подходящих фраз.")
        
        # Выдерживаем паузу между запросами
        processing_time = time.time() - start_time
        if processing_time < DELAY_BETWEEN_REQUESTS:
            sleep_time = DELAY_BETWEEN_REQUESTS - processing_time
            print(f"⏳ Ожидаем {sleep_time:.2f} сек. для соблюдения лимита...")
            time.sleep(sleep_time)
    
    # Закрываем соединения
    conn_main.close()
    conn_openrouter.close()
    print(f"✅ Обработка завершена. Обработано {processed_count} диалогов. Результаты сохранены в: {OPENROUTER_DB_PATH}")

if __name__ == "__main__":
    main()