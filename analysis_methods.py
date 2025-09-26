# analysis_methods.py
"""
Модуль с различными методами анализа диалогов.
Предполагается, что он будет использоваться в gui.py.
"""

import logging
import ollama
import time
import json

# === Импорт для fast_phrase_classifier ===
from classifier import classify_dialog_with_phrases, load_phrase_dict

# Настройка логирования
logger = logging.getLogger(__name__)

OLLAMA_MODEL_NAME = "dimweb/ilyagusev-saiga_llama3_8b:kto_v5_Q4_K"


# === Вспомогательная функция для вызова Ollama с повторными попытками ===
def call_ollama_with_retry(prompt, model_name=OLLAMA_MODEL_NAME, max_retries=3, delay=1):
    """
    Вызывает Ollama с повторными попытками при ошибках.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Вызов Ollama (попытка {attempt+1}/{max_retries})...")
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 4096,
                    "num_gpu": -1
                }
            )
            time.sleep(delay)
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ошибка при вызове Ollama (попытка {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * 2)
            else:
                raise e


# === Подход 1: Иерархический (многоступенчатый) анализ ===
def hierarchical_analysis(question, found_with_scores, chunk_size=10, status_callback=None):
    try:
        total_found = len(found_with_scores)
        if status_callback:
            status_callback(f"🧠 Иерархический анализ: обработка {total_found} диалогов...")

        summaries = []
        num_chunks = (total_found + chunk_size - 1) // chunk_size
        for i in range(0, len(found_with_scores), chunk_size):
            chunk_num = i // chunk_size + 1
            if status_callback:
                status_callback(f"🧠 Иерархический анализ: обработка группы {chunk_num}/{num_chunks}...")

            chunk = found_with_scores[i:i + chunk_size]
            chunk_context_parts = []
            for item, score in chunk:
                text_snippet = item['text'][:800] + ("..." if len(item['text']) > 800 else "")
                chunk_context_parts.append(f"[Схожесть: {score:.4f}] ID: {item['id']}\n{text_snippet}")
            
            chunk_context = "\n---\n".join(chunk_context_parts)
            chunk_prompt = (
                f"Вы — аналитик call-центра. Проанализируйте следующие фрагменты диалогов и кратко ответьте на вопрос пользователя. "
                f"Если информация отсутствует, ответьте 'Нет данных'. Отвечайте кратко и по существу.\n"
                f"Вопрос: {question}\n"
                f"Фрагменты диалогов:\n{chunk_context}\n"
                f"Краткий ответ:"
            )
            
            try:
                summary = call_ollama_with_retry(chunk_prompt)
                summaries.append(f"Группа {chunk_num} (диалоги {i+1}-{min(i+chunk_size, total_found)}):\n{summary}")
                logger.info(f"Обработана группа {chunk_num}/{num_chunks}")
            except Exception as e:
                error_msg = f"Ошибка при обработке группы {chunk_num}: {e}"
                logger.error(error_msg)
                summaries.append(f"Группа {chunk_num}: ОШИБКА - {str(e)}")

        if not summaries:
            return "Не удалось получить промежуточные результаты.", "Нет данных для контекста."

        if status_callback:
            status_callback("🧠 Иерархический анализ: финальная агрегация...")

        final_context = "\n\n".join(summaries)
        final_prompt = (
            f"Вы — аналитик call-центра. Ниже приведены краткие ответы по группам диалогов, относящихся к вопросу пользователя. "
            f"Проанализируйте эти ответы и дайте общий, развернутый и структурированный ответ на вопрос. "
            f"Если информация отсутствует, честно скажите об этом.\n"
            f"Вопрос пользователя: {question}\n"
            f"Ответы по группам:\n{final_context}\n"
            f"Общий ответ:"
        )
        
        try:
            final_answer = call_ollama_with_retry(final_prompt)
        except Exception as e:
            final_answer = f"Ошибка при финальной агрегации: {e}\n\nПромежуточные результаты:\n{final_context}"

        context_text = f"Всего найдено и проанализировано: {total_found} диалогов.\n\n"
        for item, score in found_with_scores:
            context_text += f"ID: {item['id']}\nСхожесть: {score:.4f}\nТекст: {item['text'][:300]}...\n---\n"

        return final_answer, context_text

    except Exception as e:
        error_msg = f"Ошибка в hierarchical_analysis: {e}"
        logger.error(error_msg)
        return f"Ошибка анализа: {error_msg}", f"Ошибка: {error_msg}"


# === Подход 2: Постепенное суммирование (Rolling Summary) ===
def rolling_summary_analysis(question, found_with_scores, chunk_size=5, status_callback=None):
    try:
        total_found = len(found_with_scores)
        if status_callback:
            status_callback(f"🧠 Постепенное суммирование: обработка {total_found} диалогов...")

        summary = "Начальный итог отсутствует."
        processed_count = 0

        for i in range(0, len(found_with_scores), chunk_size):
            if status_callback:
                status_callback(f"🧠 Постепенное суммирование: обработано {min(i+chunk_size, total_found)}/{total_found}...")

            chunk = found_with_scores[i:i + chunk_size]
            chunk_context_parts = []
            for item, score in chunk:
                text_snippet = item['text'][:500] + ("..." if len(item['text']) > 500 else "")
                chunk_context_parts.append(f"[Схожесть: {score:.4f}] ID: {item['id']}\n{text_snippet}")
            
            chunk_context = "\n---\n".join(chunk_context_parts)
            
            prompt = (
                f"Вы — аналитик call-центра. "
                f"Вопрос: {question}\n"
                f"Текущий промежуточный итог: {summary}\n"
                f"Новые диалоги для учета:\n{chunk_context}\n"
                f"Обнови промежуточный итог, учитывая новые диалоги. Ответь кратко."
            )
            
            try:
                summary = call_ollama_with_retry(prompt)
                processed_count += len(chunk)
                logger.info(f"Обновлен итог после {processed_count} диалогов")
            except Exception as e:
                logger.error(f"Ошибка при обновлении итога: {e}")
                summary += f"\n[Ошибка на шаге {processed_count//chunk_size + 1}: {e}]"

        final_prompt = (
            f"Вы — аналитик call-центра. На основе промежуточного итога, дай развернутый ответ на вопрос пользователя.\n"
            f"Вопрос: {question}\n"
            f"Промежуточный итог: {summary}\n"
            f"Ответ:"
        )
        
        if status_callback:
            status_callback("🧠 Постепенное суммирование: финальная формулировка ответа...")

        try:
            final_answer = call_ollama_with_retry(final_prompt)
        except Exception as e:
            final_answer = f"Ошибка при финальной формулировке: {e}\n\nПромежуточный итог:\n{summary}"

        context_text = f"Всего обработано: {total_found} диалогов методом постепенного суммирования.\n\n"
        for item, score in found_with_scores:
            context_text += f"ID: {item['id']}\nСхожесть: {score:.4f}\nТекст: {item['text'][:300]}...\n---\n"

        return final_answer, context_text

    except Exception as e:
        error_msg = f"Ошибка в rolling_summary_analysis: {e}"
        logger.error(error_msg)
        return f"Ошибка анализа: {error_msg}", f"Ошибка: {error_msg}"


# === Подход 3: Извлечение ключевых фактов (Information Extraction) ===
def fact_extraction_analysis(question, found_with_scores, chunk_size=10, status_callback=None):
    try:
        total_found = len(found_with_scores)
        if status_callback:
            status_callback(f"🧠 Извлечение фактов: обработка {total_found} диалогов...")

        all_facts = []
        num_chunks = (total_found + chunk_size - 1) // chunk_size
        for i in range(0, len(found_with_scores), chunk_size):
            chunk_num = i // chunk_size + 1
            if status_callback:
                status_callback(f"🧠 Извлечение фактов: обработка группы {chunk_num}/{num_chunks}...")

            chunk = found_with_scores[i:i + chunk_size]
            chunk_context_parts = []
            for item, score in chunk:
                text_snippet = item['text'][:1000] + ("..." if len(item['text']) > 1000 else "")
                chunk_context_parts.append(f"[Схожесть: {score:.4f}] ID: {item['id']}\n{text_snippet}")
            
            chunk_context = "\n---\n".join(chunk_context_parts)
            
            prompt = (
                f"Вы — аналитик call-центра. Ваша задача — извлечь ключевые факты из диалогов, "
                f"которые могут быть релевантны вопросу пользователя. "
                f"Формат ответа: список пунктов, каждый пункт - один факт. "
                f"Если в диалоге нет релевантной информации, напиши 'Нет данных'.\n"
                f"Вопрос пользователя: {question}\n"
                f"Диалоги:\n{chunk_context}\n"
                f"Извлеченные факты:"
            )
            
            try:
                facts = call_ollama_with_retry(prompt)
                all_facts.append(f"Группа {chunk_num}:\n{facts}")
                logger.info(f"Извлечены факты из группы {chunk_num}/{num_chunks}")
            except Exception as e:
                error_msg = f"Ошибка при извлечении фактов из группы {chunk_num}: {e}"
                logger.error(error_msg)
                all_facts.append(f"Группа {chunk_num}: ОШИБКА - {str(e)}")

        if not all_facts:
            return "Не удалось извлечь факты.", "Нет данных для контекста."

        if status_callback:
            status_callback("🧠 Извлечение фактов: агрегация и анализ...")

        facts_summary = "\n\n".join(all_facts)
        analysis_prompt = (
            f"Вы — аналитик call-центра. Ниже приведены извлеченные факты из диалогов по вопросу пользователя. "
            f"Проанализируйте эти факты и дайте структурированный ответ на вопрос. "
            f"Подсчитайте частоты, если это уместно. "
            f"Если информация отсутствует, честно скажите об этом.\n"
            f"Вопрос пользователя: {question}\n"
            f"Извлеченные факты:\n{facts_summary}\n"
            f"Анализ и ответ:"
        )
        
        try:
            final_answer = call_ollama_with_retry(analysis_prompt)
        except Exception as e:
            final_answer = f"Ошибка при анализе фактов: {e}\n\nИзвлеченные факты:\n{facts_summary}"

        context_text = f"Всего обработано: {total_found} диалогов. Извлечены факты.\n\n"
        context_text += f"Все извлеченные факты:\n{facts_summary}\n\n"
        context_text += "Полные тексты диалогов:\n"
        for item, score in found_with_scores:
            context_text += f"ID: {item['id']}\nСхожесть: {score:.4f}\nТекст: {item['text'][:300]}...\n---\n"

        return final_answer, context_text

    except Exception as e:
        error_msg = f"Ошибка в fact_extraction_analysis: {e}"
        logger.error(error_msg)
        return f"Ошибка анализа: {error_msg}", f"Ошибка: {error_msg}"


# === Подход 4: Классификация диалогов ===
def classification_analysis(question, found_with_scores, categories=None, status_callback=None):
    try:
        total_found = len(found_with_scores)
        if status_callback:
            status_callback(f"🧠 Классификация: обработка {total_found} диалогов...")

        if not categories:
            category_prompt = (
                f"Пользователь задал вопрос: '{question}'. "
                f"Определи 3-5 наиболее вероятных категории (типа звонка), "
                f"по которым можно классифицировать диалоги. "
                f"Ответь списком, по одному названию категории на строку."
            )
            try:
                categories_response = call_ollama_with_retry(category_prompt)
                categories = [cat.strip() for cat in categories_response.strip().split('\n') if cat.strip()]
                logger.info(f"Определены категории: {categories}")
            except Exception as e:
                logger.error(f"Ошибка при определении категорий: {e}")
                categories = ["Восстановление договора", "Техническая поддержка", "Финансовые вопросы", "Жалоба", "Консультация"]

        if status_callback:
            status_callback(f"🧠 Классификация: определены категории {categories}. Начинаем классификацию диалогов...")

        classified_dialogs = {cat: [] for cat in categories}
        classified_dialogs["Не классифицировано"] = []
        
        for i, (item, score) in enumerate(found_with_scores):
            if status_callback and i % 10 == 0:
                status_callback(f"🧠 Классификация: обработано {i+1}/{total_found} диалогов...")

            text_snippet = item['text'][:800] + ("..." if len(item['text']) > 800 else "")
            prompt = (
                f"Вы — аналитик call-центра. Классифицируйте следующий диалог по одной из указанных категорий. "
                f"Если ни одна категория не подходит, ответьте 'Не классифицировано'.\n"
                f"Категории: {', '.join(categories)}\n"
                f"Диалог:\n{text_snippet}\n"
                f"Категория:"
            )
            
            try:
                category_response = call_ollama_with_retry(prompt)
                determined_category = category_response.strip()
                if determined_category in classified_dialogs:
                    classified_dialogs[determined_category].append((item, score))
                else:
                    classified_dialogs["Не классифицировано"].append((item, score))
            except Exception as e:
                logger.error(f"Ошибка при классификации диалога {item['id']}: {e}")
                classified_dialogs["Не классифицировано"].append((item, score))

        if status_callback:
            status_callback("🧠 Классификация: анализ результатов...")

        analysis_lines = []
        context_lines = [f"Всего обработано: {total_found} диалогов.\n"]
        for category, dialogs in classified_dialogs.items():
            count = len(dialogs)
            analysis_lines.append(f"- {category}: {count} диалогов")
            if dialogs:
                context_lines.append(f"\nКатегория: {category} ({count} диалогов)")
                for item, score in dialogs[:3]:
                    context_lines.append(f"  ID: {item['id']} (Схожесть: {score:.4f})")
                if len(dialogs) > 3:
                    context_lines.append(f"  ... и ещё {len(dialogs) - 3} диалогов.")

        analysis_text = "\n".join(analysis_lines)
        context_text = "\n".join(context_lines)

        final_prompt = (
            f"Вы — аналитик call-центра. Ниже приведены результаты классификации диалогов. "
            f"Проанализируйте их и дайте краткий вывод о распределении типов звонков. "
            f"Если вопрос пользователя был о конкретной категории, сосредоточьтесь на ней.\n"
            f"Вопрос пользователя: {question}\n"
            f"Результаты классификации:\n{analysis_text}\n"
            f"Вывод:"
        )
        
        try:
            final_answer = call_ollama_with_retry(final_prompt)
        except Exception as e:
            final_answer = f"Ошибка при финальном анализе классификации: {e}\n\nРезультаты:\n{analysis_text}"

        return final_answer, context_text

    except Exception as e:
        error_msg = f"Ошибка в classification_analysis: {e}"
        logger.error(error_msg)
        return f"Ошибка анализа: {error_msg}", f"Ошибка: {error_msg}"


# === Подход 5: Классификация по обратным звонкам (Категории 1-4) — ЧИСТЫЙ LLM-АНАЛИЗ ===
def callback_classifier(question, found_with_scores, chunk_size=1, status_callback=None):
    try:
        total_found = len(found_with_scores)
        if status_callback:
            status_callback(f"📞 LLM-анализ обратных звонков: обработка {total_found} диалогов...")

        dialog_texts = {}
        for item, score in found_with_scores:
            dialog_id = item['dialog_id']
            if dialog_id not in dialog_texts:
                dialog_texts[dialog_id] = item['full_dialog_text']

        results = []
        context_lines = []

        for dialog_id, full_text in dialog_texts.items():
            if status_callback:
                status_callback(f"📞 Анализ диалога {dialog_id}...")

            prompt = f"""Ты — строгий контролёр качества call-центра. Проанализируй диалог и определи категорию по правилу обратного звонка.

Правила:
1. Если клиент говорит, что сам перезвонит (без слов "если что", "как-нибудь", "может быть", "возможно", "не знаю", "с другого номера") — оператор ОБЯЗАН предложить обратный звонок. Если не предложил — это ошибка (Категория 1).
2. Если оператор корректно предложил обратный звонок (с временем, с подтверждением, уместно) — Категория 2.
3. Если клиент сказал "если что", "как-нибудь потом", "может быть", "не знаю" — обратный звонок НЕ требуется. Если оператор его назначил — это ошибка (Категория 3).
4. Если обратный звонок не требовался и не назначен — всё правильно (Категория 4).

Твоя задача:
- Проанализируй диалог.
- Определи категорию (1, 2, 3, 4).
- Выведи ТОЛЬКО одну строку в формате JSON: {{"category": N, "client_phrases": ["фраза1", "фраза2"], "operator_phrases": ["фраза1", "фраза2"]}}
- Фразы должны быть ДОСЛОВНЫМИ, как в диалоге.
- Не добавляй пояснений, не пиши "думаю", не добавляй markdown.
- Если фраз нет — верни пустой список [].
- Не выдумывай фразы. Только то, что есть в тексте.

Пример правильного ответа:
{{"category": 1, "client_phrases": ["я сам перезвоню вечером"], "operator_phrases": []}}

Диалог:
{full_text}
"""

            try:
                llm_response = call_ollama_with_retry(prompt)
                result = json.loads(llm_response)
                if "category" not in result or "client_phrases" not in result or "operator_phrases" not in result:
                    raise ValueError("Неверный формат ответа от LLM")
                result["dialog_id"] = dialog_id
                results.append(result)

                cat_names = {
                    1: "❌ Пропущен обязательный звонок",
                    2: "✅ Корректно предложенный звонок",
                    3: "⚠️ Ненужный звонок назначен",
                    4: "✔️ Звонок не требовался и не назначен"
                }
                context_lines.append(f"ID: {dialog_id} | {cat_names[result['category']]}")
                if result['client_phrases']:
                    context_lines.append(f"  Клиент: {', '.join(result['client_phrases'])}")
                if result['operator_phrases']:
                    context_lines.append(f"  Оператор: {', '.join(result['operator_phrases'])}")
                context_lines.append("---")

            except Exception as e:
                logger.error(f"Ошибка анализа диалога {dialog_id}: {e}")
                results.append({
                    "dialog_id": dialog_id,
                    "category": 0,
                    "client_phrases": [],
                    "operator_phrases": [],
                    "error": str(e)
                })
                context_lines.append(f"ID: {dialog_id} | ❌ ОШИБКА АНАЛИЗА: {e}")
                context_lines.append("---")

        final_answer = json.dumps(results, ensure_ascii=False, indent=2)
        context_text = "📞 Результаты LLM-классификации по обратным звонкам:\n\n" + "\n".join(context_lines)

        return final_answer, context_text

    except Exception as e:
        error_msg = f"Ошибка в callback_classifier: {e}"
        logger.error(error_msg)
        return f"Ошибка анализа: {error_msg}", f"Ошибка: {error_msg}"


# === БЫСТРЫЙ КЛАССИФИКАТОР НА ОСНОВЕ СЛОВАРЯ ===
def fast_phrase_classifier(question, found_with_scores, chunk_size=1, status_callback=None):
    phrase_dict = None
    try:
        phrase_dict = load_phrase_dict()
    except Exception:
        phrase_dict = None

    results = []
    for item, score in found_with_scores:
        # Извлекаем только реплики клиента из диалога
        client_lines = []
        dialog_text = item['full_dialog_text']
        lines = dialog_text.splitlines()
        for line in lines:
            if line.startswith("Клиент:") or line.startswith("клиент:") or line.startswith("КЛИЕНТ:"):
                client_lines.append(line)
        
        # Проверяем каждую реплику клиента
        category = 4  # По умолчанию — всё нормально
        for client_line in client_lines:
            # Убираем "Клиент:" и пробелы
            clean_client_line = client_line.replace("Клиент:", "").replace("клиент:", "").strip()
            if clean_client_line:
                # Ищем в словаре
                if phrase_dict and any(phrase in clean_client_line for phrase in phrase_dict.get("category_1_phrases", {}).get("client", [])):
                    # Проверяем исключения
                    if any(excl in clean_client_line for excl in phrase_dict.get("category_3_phrases", {}).get("client", [])):
                        category = 3
                    elif any(excl in clean_client_line for excl in phrase_dict.get("category_2_phrases", {}).get("operator", [])):
                        category = 2
                    else:
                        category = 1
                    break
        
        results.append({
            "dialog_id": item['dialog_id'],
            "category": category,
            "client_phrases": [],
            "operator_phrases": []
        })
    final_answer = json.dumps(results, ensure_ascii=False, indent=2)
    context_text = "Классификация завершена по словарю."
    return final_answer, context_text


# === Функция для выбора метода по названию ===
def get_analysis_method(method_name):
    methods = {
        "hierarchical": hierarchical_analysis,
        "rolling": rolling_summary_analysis,
        "facts": fact_extraction_analysis,
        "classification": classification_analysis,
        "callback_classifier": callback_classifier,
        "fast_phrase_classifier": fast_phrase_classifier,
    }
    return methods.get(method_name)