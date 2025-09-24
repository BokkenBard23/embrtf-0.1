# generate_callback_phrases.py
import sqlite3
import json
import logging
import time
import requests
import re
from typing import Optional, Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_callback_errors.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Подключение к БД
DB_PATH = "database.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# МИНИМАЛИСТИЧНЫЙ ПРОМПТ — только суть, без воды
PROMPT_TEMPLATE = """Ты — JSON-генератор. Верни ТОЛЬКО валидный JSON без пояснений. Формат: {"error": true/false, "client_phrase": "...", "operator_phrase": "..."}

Правило:
- error = true: клиент ЧЁТКО сказал "перезвоню/свяжусь/наберу" БЕЗ "если/может/как-нибудь" → и оператор НЕ предложил звонок.
- error = false: клиент дал неопределённость ИЛИ оператор предложил звонок.

Фразы — дословно из диалога. Ничего лишнего.

Диалог:
{dialog_text}
"""


def escape_curly_brackets(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def extract_first_json(text: str) -> Optional[str]:
    start = text.find('{')
    if start == -1:
        return None
    for end in range(start + 1, len(text) + 1):
        try:
            candidate = text[start:end]
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def call_llm_with_retry(dialog_text: str, max_retries=2) -> Optional[Dict[str, Any]]:  # ← Уменьшил до 2 попыток
    for attempt in range(max_retries):
        try:
            safe_dialog_text = escape_curly_brackets(dialog_text)
            prompt = PROMPT_TEMPLATE.format(dialog_text=safe_dialog_text)

            # Возвращаемся к api/generate — меньше токенов → быстрее
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "qwen2.5:1.5b-instruct-q4_K_M",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 128,  # ← Ограничиваем длину ответа
                        "temperature": 0.1,  # ← Делаем ответ более детерминированным
                        "top_p": 0.9
                    }
                },
                timeout=30  # ← Уменьшаем таймаут
            )

            if response.status_code != 200:
                logging.warning(f"HTTP {response.status_code}: {response.text}")
                raise Exception(f"HTTP Error {response.status_code}")

            ollama_json = response.json()

            if "error" in ollama_json:
                error_msg = ollama_json["error"]
                logging.warning(f"Ollama error: {error_msg}")
                raise Exception(f"Ollama error: {error_msg}")

            raw_text = ollama_json.get("response", "").strip()
            logging.info(f"LLM raw response: {repr(raw_text)}")

            json_str = extract_first_json(raw_text)
            if not json_str:
                raise ValueError("No valid JSON found in response")

            result = json.loads(json_str)
            if "error" in result and "client_phrase" in result and "operator_phrase" in result:
                return result
            else:
                raise ValueError("Missing required keys in LLM JSON")

        except Exception as e:
            logging.warning(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # ← Короткая пауза
            else:
                logging.error(f"LLM не ответила валидно для диалога: {dialog_text[:100]}...")
                return None

    return None


# Получаем все диалоги
cursor.execute("SELECT id, text FROM dialogs")
dialogs = cursor.fetchall()

for i, (dialog_id, dialog_text) in enumerate(dialogs, 1):
    print(f"Обработка диалога {i}/{len(dialogs)} (ID: {dialog_id})...")

    result = call_llm_with_retry(dialog_text)
    if not result:
        continue

    error = result["error"]
    client_phrase = result["client_phrase"].strip() if result["client_phrase"] else ""
    operator_phrase = result["operator_phrase"].strip() if result["operator_phrase"] else ""

    if error and client_phrase:
        cursor.execute("""
            INSERT OR IGNORE INTO callback_phrases (phrase, source, category, verified)
            VALUES (?, ?, ?, 0)
        """, (client_phrase, 'client_promise', 1))

    if not error:
        if "если" in client_phrase or "может" in client_phrase or "как-нибудь" in client_phrase:
            cursor.execute("""
                INSERT OR IGNORE INTO callback_phrases (phrase, source, category, verified)
                VALUES (?, ?, ?, 0)
            """, (client_phrase, 'client_uncertain', 2))
        elif operator_phrase:
            cursor.execute("""
                INSERT OR IGNORE INTO callback_phrases (phrase, source, category, verified)
                VALUES (?, ?, ?, 0)
            """, (operator_phrase, 'operator_offer', 3))

    conn.commit()

conn.close()
print("✅ Все диалоги обработаны. Фразы сохранены в таблицу 'callback_phrases'.")
print("📄 Подробные ошибки и сырые ответы — в файле llm_callback_errors.log")