import sqlite3
import json

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Загружаем все фразы
cursor.execute("SELECT phrase, source, category FROM callback_phrases ORDER BY category, phrase")
rows = cursor.fetchall()

# Группируем по категориям
aggregated = {
    "category_1_phrases": {"client": [], "operator": []},
    "category_2_phrases": {"client": [], "operator": []},
    "category_3_phrases": {"client": [], "operator": []},
    "category_4_phrases": {"client": [], "operator": []}
}

for phrase, source, category in rows:
    key = f"category_{category}_phrases"
    if source == "client_promise":
        aggregated[key]["client"].append(phrase)
    elif source == "operator_offer":
        aggregated[key]["operator"].append(phrase)
    elif source == "client_uncertain":
        aggregated[key]["client"].append(phrase)

# Сохраняем в JSON
with open("data/aggregated_phrases.json", "w", encoding="utf-8") as f:
    json.dump(aggregated, f, ensure_ascii=False, indent=2)

print("✅ Словарь экспортирован: data/aggregated_phrases.json")
conn.close()