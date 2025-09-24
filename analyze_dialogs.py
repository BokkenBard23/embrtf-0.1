# analyze_dialogs.py
import json
import config
from pathlib import Path
from striprtf.striprtf import rtf_to_text
import ollama # Для будущего использования или подсчета токенов

# --- Настройки для анализа ---
# Определим пороги длины в токенах
LENGTH_THRESHOLDS = [512, 1024, 2048, 4096]

def count_tokens_approx(text):
    """Приблизительный подсчет токенов."""
    # Очень грубая оценка: 1 токен ~= 4 символа
    return len(text) / 4

def analyze():
    """Анализирует диалоги и выводит статистику по их длине."""
    print("🔍 Начинаем анализ диалогов по тематическим папкам...")
    
    total_chars = 0
    total_tokens_approx = 0
    dialog_count = 0
    token_counts = []
    
    # Словарь для подсчета по темам
    theme_stats = {}

    # --- НОВОЕ: Счетчики для порогов ---
    threshold_counts = {thresh: 0 for thresh in LENGTH_THRESHOLDS}
    above_max_threshold = 0 # Счетчик для диалогов длиннее самого большого порога

    for theme_folder in config.INPUT_ROOT.iterdir():
        if theme_folder.is_dir():
            theme_name = theme_folder.name
            print(f"  📂 Анализ папки темы: {theme_name}")
            
            theme_dialog_count = 0
            theme_total_tokens = 0
            theme_token_counts = []

            for file in theme_folder.glob("*.rtf"):
                try:
                    raw_text = rtf_to_text(file.read_text(encoding="utf-8", errors="ignore"))
                    chars = len(raw_text)
                    tokens = count_tokens_approx(raw_text)
                    
                    total_chars += chars
                    total_tokens_approx += tokens
                    token_counts.append(tokens)
                    dialog_count += 1
                    
                    theme_dialog_count += 1
                    theme_total_tokens += tokens
                    theme_token_counts.append(tokens)
                    
                    # --- НОВОЕ: Проверка порогов ---
                    for thresh in LENGTH_THRESHOLDS:
                        if tokens <= thresh:
                            threshold_counts[thresh] += 1
                    if tokens > LENGTH_THRESHOLDS[-1]: # Длиннее самого большого порога
                         above_max_threshold += 1

                except Exception as e:
                    print(f"    ❌ Ошибка обработки файла {file}: {e}")

            # Сохраняем статистику по теме
            if theme_dialog_count > 0:
                theme_stats[theme_name] = {
                    "count": theme_dialog_count,
                    "avg_tokens": theme_total_tokens / theme_dialog_count,
                    "max_tokens": max(theme_token_counts),
                    "min_tokens": min(theme_token_counts),
                }

    # --- Вывод результатов ---
    if dialog_count > 0:
        avg_chars = total_chars / dialog_count
        avg_tokens = total_tokens_approx / dialog_count
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        
        print(f"\n--- Общая статистика по всем диалогам ---")
        print(f"Всего обработано файлов: {dialog_count}")
        print(f"Среднее количество символов: {avg_chars:.2f}")
        print(f"Среднее количество токенов (приблизительно): {avg_tokens:.2f}")
        print(f"Максимальное количество токенов (приблизительно): {max_tokens:.2f}")
        print(f"Минимальное количество токенов (приблизительно): {min_tokens:.2f}")
        
        if max_tokens > LENGTH_THRESHOLDS[0]:
            print(f"\n--- Распределение по длинам ---")
            print(f"(Используются пороги: {LENGTH_THRESHOLDS})")
            for thresh in LENGTH_THRESHOLDS:
                percentage = (threshold_counts[thresh] / dialog_count) * 100
                print(f"  ⏱️  Диалогов с <= {thresh} токенов: {threshold_counts[thresh]} ({percentage:.2f}%)")
            if above_max_threshold > 0:
                percentage_above = (above_max_threshold / dialog_count) * 100
                print(f"  ⏱️  Диалогов с > {LENGTH_THRESHOLDS[-1]} токенов: {above_max_threshold} ({percentage_above:.2f}%)")
            
            # Рекомендация
            print(f"\n--- Рекомендации ---")
            # Найдем порог, который покрывает >= 95% диалогов
            target_coverage = 95.0
            recommended_length = None
            for thresh in sorted(LENGTH_THRESHOLDS, reverse=True):
                percentage = (threshold_counts[thresh] / dialog_count) * 100
                if percentage >= target_coverage:
                    recommended_length = thresh
            
            if recommended_length:
                print(f"  ✅ Рекомендуется MODEL.max_seq_length = {recommended_length} (покрывает ~{target_coverage}% диалогов).")
            else:
                # Если даже самый большой порог не покрывает 95%
                if above_max_threshold > 0:
                    percentage_above = (above_max_threshold / dialog_count) * 100
                    print(f"  ⚠️  Нет порога, покрывающего {target_coverage}%. {percentage_above:.2f}% диалогов длиннее {LENGTH_THRESHOLDS[-1]} токенов.")
                    print(f"  💡 Рассмотрите MODEL.max_seq_length = {LENGTH_THRESHOLDS[-1]} и стратегии для очень длинных диалогов.")
                else:
                     # Все диалоги укладываются в максимальный порог
                     print(f"  ✅ MODEL.max_seq_length = {LENGTH_THRESHOLDS[-1]} покроет 100% диалогов.")
                     
        else:
            print(f"\n✅ Все диалоги короче {LENGTH_THRESHOLDS[0]} токенов. MODEL.max_seq_length = 512 будет достаточным.")
            
        # --- Статистика по темам ---
        if theme_stats:
            print(f"\n--- Статистика по темам ---")
            for theme, stats in theme_stats.items():
                print(f"  📁 {theme}:")
                print(f"    - Файлов: {stats['count']}")
                print(f"    - Средняя длина (токены): {stats['avg_tokens']:.2f}")
                print(f"    - Мин/Макс длина: {stats['min_tokens']:.2f} / {stats['max_tokens']:.2f}")

    else:
        print("Не найдено .rtf файлов для анализа.")

if __name__ == "__main__":
    analyze()