# generate_callback_phrases_internal_llm.py
import sqlite3
import json
import config
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
"""

def safe_request(text):
    """Отправляет запрос с текстом в теле, минуя форматирование строки"""
    headers = {
        "Authorization": f"Bearer {config.LLM_INTERNAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Формируем полное сообщение вручную
    full_prompt = PROMPT_TEMPLATE + text
    
    payload = {
        "model": config.LLM_INTERNAL_MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }

    print(f"\n[DEBUG] Отправляем запрос с текстом:\n{full_prompt[:500]}...\n")

    try:
        response = requests.post(
            config.LLM_INTERNAL_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"[DEBUG] Статус: {response.status_code}")
        print(f"[DEBUG] Заголовки: {response.headers}")
        
        if response.status_code == 200:
            result_data = response.json()
            print(f"[DEBUG] Полный ответ:\n{json.dumps(result_data, indent=2)}")
            
            result_text = result_data["choices"][0]["message"]["content"]
            print(f"[DEBUG] Raw response:\n{result_text}")
            
            # Парсинг без привязки к ключам
            try:
                result = json.loads(result_text)
                return result
            except:
                print(f"[ERROR] Не удалось распарсить JSON")
                return {"error": False, "client_phrase": "", "operator_phrase": ""}
        else:
            print(f"[ERROR] Ошибка API: {response.text}")
            return {"error": False, "client_phrase": "", "operator_phrase": ""}
            
    except Exception as e:
        print(f"[ERROR] Исключение: {str(e)}")
        return {"error": False, "client_phrase": "", "operator_phrase": ""}

# ... (остальной код без изменений)