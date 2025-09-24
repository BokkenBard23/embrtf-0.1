# generate_callback_phrases_multi_model.py
"""Обработка диалогов через 55 FREE моделей OpenRouter параллельно."""

import sqlite3
import json
import config
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# Промпт (тот же, что и раньше)
PROMPT_TEMPLATE = """Ты — строгий контролёр качества call-центра. Твоя задача — проанализировать диалог и определить, содержит ли он **ошибку оператора по правилу обратного звонка**.

Правила:
1. Ошибка есть, если:
   - Клиент говорит о намерении перезвонить/связаться/обратиться — **чётко и без неопределённости** (без слов “если что”, “может быть”, “как-нибудь”, “не знаю”, “с другого номера”).
   - Оператор **НЕ** предложил обратный звонок.

2. Ошибка НЕТ, если:
   - Клиент дал **неопределённое обещание** (например, “если что перезвоню”).
   - Оператор **предложил обратный звонок** (например, “мы вам перезвоним”, “назначил звонок на 15:00”).

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
{dialog_text}
"""

def extract_phrases_with_llm(dialog_text):
    """Отправляет диалог в случайную FREE модель OpenRouter с повторами при ошибке."""
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Пробуем каждую модель по очереди при ошибке
    for model in config.OPENROUTER_FREE_MODELS:
        for attempt in range(3):  # 3 попытки
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": PROMPT_TEMPLATE.format(dialog_text=dialog_text)}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                }

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result_text = response.json()["choices"][0]["message"]["content"]
                    result = json.loads(result_text)
                    return result
                elif response.status_code == 429:
                    print(f"⚠️ Лимит для модели {model}. Ждём 5 сек...")
                    time.sleep(5)
                    continue
                else:
                    print(f"⚠️ Ошибка {response.status_code} от {model}: {response.text}")
                    break  # Переход к следующей модели
            except Exception as e:
                print(f"⚠️ Ошибка при вызове {model} (попытка {attempt+1}): {e}")
                time.sleep(2)
        time.sleep(1)  # Пауза между моделями

    return {"error": False, "client_phrase": "", "operator_phrase": "", "error_msg": "Все модели не ответили"}

# Подключение к БД
conn = sqlite3.connect(config.DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS callback_phrases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phrase TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        category INTEGER NOT NULL,
        frequency INTEGER DEFAULT 1,
        verified BOOLEAN DEFAULT 0
    );
""")

cursor.execute("SELECT id, text FROM dialogs")
dialogs = cursor.fetchall()
print(f"✅ Найдено {len(dialogs)} диалогов. Начинаем обработку...")

def process_single_dialog(dialog_id, text):
    print(f"🔄 Обработка диалога ID: {dialog_id}...")
    result = extract_phrases_with_llm(text)

    if result.get("client_phrase"):
        if result["error"]:
            source = "client_promise"
            category = 1
        else:
            source = "client_uncertain"
            category = 3
        cursor.execute("""
            INSERT OR IGNORE INTO callback_phrases (phrase, source, category, frequency, verified)
            VALUES (?, ?, ?, 1, 0)
        """, (result["client_phrase"], source, category))
        cursor.execute("""
            UPDATE callback_phrases SET frequency = frequency + 1 WHERE phrase = ?
        """, (result["client_phrase"],))
    
    if result.get("operator_phrase"):
        cursor.execute("""
            INSERT OR IGNORE INTO callback_phrases (phrase, source, category, frequency, verified)
            VALUES (?, ?, 2, 1, 0)
        """, (result["operator_phrase"], "operator_offer"))
        cursor.execute("""
            UPDATE callback_phrases SET frequency = frequency + 1 WHERE phrase = ?
        """, (result["operator_phrase"],))

    return dialog_id

# === ЗАПУСК С 55 ПОТОКАМИ ===
with ThreadPoolExecutor(max_workers=55) as executor:
    futures = [executor.submit(process_single_dialog, dialog_id, text) for dialog_id, text in dialogs]
    
    for i, future in enumerate(as_completed(futures), 1):
        dialog_id = future.result()
        if i % 10 == 0:
            conn.commit()
            print(f"✅ Обработано {i} диалогов.")

conn.commit()
conn.close()

print("✅ Все диалоги обработаны. Фразы сохранены в таблицу 'callback_phrases'.")