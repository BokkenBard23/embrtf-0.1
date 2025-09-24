import json
from collections import defaultdict

# Загрузи результаты (предположим, что они в файле 'callback_results.json')
with open('callback_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Агрегация
aggregated = {
    "category_1_phrases": {"client": [], "operator": []},
    "category_2_phrases": {"client": [], "operator": []},
    "category_3_phrases": {"client": [], "operator": []},
    "category_4_phrases": {"client": [], "operator": []}
}

for item in results:
    cat = item["category"]
    key = f"category_{cat}_phrases"
    aggregated[key]["client"].extend(item["client_phrases"])
    aggregated[key]["operator"].extend(item["operator_phrases"])

# Убираем дубликаты и сортируем
for cat_key in aggregated:
    for speaker in ["client", "operator"]:
        # Убираем дубли, сохраняя порядок
        seen = set()
        unique_phrases = []
        for phrase in aggregated[cat_key][speaker]:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        aggregated[cat_key][speaker] = sorted(unique_phrases, key=str.lower)

# Сохраняем
with open('aggregated_phrases.json', 'w', encoding='utf-8') as f:
    json.dump(aggregated, f, ensure_ascii=False, indent=2)

print("✅ Агрегация завершена. Результат: aggregated_phrases.json")