# classifier.py — классификатор диалогов по словарю фраз

import json
import os

# Путь к словарю фраз
PHRASE_DICT_PATH = "data/aggregated_phrases.json"

def load_phrase_dict():
    """Загружает словарь фраз из JSON-файла."""
    if not os.path.exists(PHRASE_DICT_PATH):
        raise FileNotFoundError(f"Словарь фраз не найден: {PHRASE_DICT_PATH}")
    
    with open(PHRASE_DICT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_dialog_with_phrases(dialog_text, phrase_dict=None):
    """
    Классифицирует диалог в одну из 4 категорий на основе словаря фраз.
    
    Args:
        dialog_text (str): Текст диалога.
        phrase_dict (dict): Словарь фраз (если None — загружает из файла).
    
    Returns:
        int: Категория (1, 2, 3, 4).
    """
    if phrase_dict is None:
        phrase_dict = load_phrase_dict()
    
    # Ищем фразы клиента из категории 1
    for phrase in phrase_dict["category_1_phrases"]["client"]:
        if phrase in dialog_text:
            # Проверяем исключения: фразы из категории 2 (оператор) и 3 (клиент)
            for excl_phrase in phrase_dict["category_3_phrases"]["client"]:
                if excl_phrase in dialog_text:
                    return 3  # Клиент дал неопределённость → звонок не нужен, но назначен
            
            for excl_phrase in phrase_dict["category_2_phrases"]["operator"]:
                if excl_phrase in dialog_text:
                    return 2  # Оператор корректно предложил звонок
            
            return 1  # Ошибка: клиент обещал перезвонить → оператор не предложил
    
    return 4  # Звонок не требовался и не назначен