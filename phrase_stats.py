import json

with open('aggregated_phrases.json', 'r', encoding='utf-8') as f:
    agg = json.load(f)

with open('callback_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Считаем количество диалогов по категориям
cat_counts = {1:0, 2:0, 3:0, 4:0}
for item in results:
    cat_counts[item["category"]] += 1

print("📊 СТАТИСТИКА ПО КАТЕГОРИЯМ:")
for cat in [1,2,3,4]:
    print(f"Категория {cat}: {cat_counts[cat]} диалогов")

print("\n🔤 УНИКАЛЬНЫЕ ФРАЗЫ:")
for cat_key in agg:
    client_count = len(agg[cat_key]["client"])
    op_count = len(agg[cat_key]["operator"])
    print(f"{cat_key}: клиент — {client_count}, оператор — {op_count}")

# Топ-10 самых частых фраз в категории 1 (клиент)
from collections import Counter
cat1_client_phrases = [item["client_phrases"] for item in results if item["category"] == 1]
all_cat1_client = [phrase for sublist in cat1_client_phrases for phrase in sublist]
phrase_freq = Counter(all_cat1_client).most_common(10)

print("\n🔝 ТОП-10 ФРАЗ КЛИЕНТА В КАТЕГОРИИ 1 (ошибка оператора):")
for phrase, count in phrase_freq:
    print(f"{count:3d}x | {phrase}")