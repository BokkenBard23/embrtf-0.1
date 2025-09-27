"""
Упрощенная версия Hierarchical Dictionary Builder
Без зависимости от txtai для тестирования
"""

import json
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

class HierDictSimple:
    def __init__(self, texts):
        """
        Упрощенная инициализация иерархического словаря.
        
        Args:
            texts: List[str] – исходные диалоги для тестов
        """
        self.texts = texts
        logger.info(f"Инициализирован HierDictSimple с {len(texts)} текстами")

    def _keywords(self, texts, top_k=20):
        """
        Упрощенное извлечение ключевых слов на основе частоты.
        
        Args:
            texts: Список текстов для анализа
            top_k: Количество ключевых слов для извлечения
            
        Returns:
            Список ключевых слов
        """
        try:
            # Простой подсчет частоты слов
            word_count = defaultdict(int)
            for text in texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 2:  # Игнорируем короткие слова
                        word_count[word] += 1
            
            # Сортируем по частоте
            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, count in sorted_words[:top_k]]
            
            logger.info(f"Извлечено {len(keywords)} ключевых слов")
            return keywords
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых слов: {e}")
            return []

    def build(self, out_file="simple_dict.json"):
        """
        Построение упрощенного словаря.
        
        Args:
            out_file: Путь для сохранения словаря
            
        Returns:
            Словарь с parent, child и exclude словами
        """
        try:
            logger.info("Начинаем построение упрощенного словаря...")
            
            # Извлекаем ключевые слова
            parent = self._keywords(self.texts, 15)
            child = self._keywords(self.texts, 15)
            
            # Exclude = слова с низкой частотой
            word_count = defaultdict(int)
            for text in self.texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 2:
                        word_count[word] += 1
            
            # Находим слова с частотой 1 (редкие)
            exclude = [word for word, count in word_count.items() if count == 1][:10]

            dictionary = {
                "parent": {"must": parent, "exclude": exclude},
                "child": {"must": child, "exclude": []},
            }
            
            # Сохраняем словарь
            with open(out_file, "w", encoding="utf8") as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Словарь сохранён → {out_file}")
            logger.info(f"Parent слов: {len(parent)}")
            logger.info(f"Child слов: {len(child)}")
            logger.info(f"Exclude слов: {len(exclude)}")
            
            return dictionary
            
        except Exception as e:
            logger.error(f"Ошибка построения словаря: {e}")
            return {}

    def test(self, k=5):
        """
        Упрощенный тест качества.
        
        Args:
            k: Количество результатов для анализа
            
        Returns:
            Значение качества (простая метрика)
        """
        try:
            logger.info(f"Запуск упрощенного теста...")
            
            # Простая метрика: доля текстов с ключевыми словами
            parent_words = set(self._keywords(self.texts, 10))
            relevant = sum(1 for text in self.texts if any(word in text.lower() for word in parent_words))
            quality = relevant / len(self.texts) if self.texts else 0
            
            logger.info(f"Качество: {quality:.2f}")
            logger.info(f"Релевантных текстов: {relevant} из {len(self.texts)}")
            
            return quality
            
        except Exception as e:
            logger.error(f"Ошибка тестирования: {e}")
            return 0.0

    def get_statistics(self):
        """
        Получение статистики по словарю.
        
        Returns:
            Словарь со статистикой
        """
        try:
            parent = self._keywords(self.texts, 15)
            child = self._keywords(self.texts, 15)
            
            return {
                "total_texts": len(self.texts),
                "parent_words": len(parent),
                "child_words": len(child),
                "parent_words_list": parent,
                "child_words_list": child
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def export_to_smart_logger(self, output_file="simple_smart_logger_dict.json"):
        """
        Экспорт словаря в формате для Smart Logger.
        
        Args:
            output_file: Путь для сохранения файла
            
        Returns:
            Путь к сохранённому файлу
        """
        try:
            dictionary = self.build(output_file)
            
            # Дополнительная информация для Smart Logger
            smart_logger_dict = {
                "metadata": {
                    "created_by": "CallCenter AI v3 (Simple)",
                    "version": "1.0",
                    "total_texts": len(self.texts),
                    "description": "Упрощенный иерархический словарь для классификации диалогов"
                },
                "dictionary": dictionary
            }
            
            with open(output_file, "w", encoding="utf8") as f:
                json.dump(smart_logger_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Словарь экспортирован для Smart Logger → {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в Smart Logger: {e}")
            return None


# ---------- Быстрый self-test ----------
if __name__ == "__main__":
    """Пример использования на тестовых данных."""
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Тестовые данные
    test_texts = [
        "Здравствуйте, бонус не начислили, хотя обещали подарок.",
        "Оператор: Проверю информацию и вернусь.",
        "Клиент: Спасибо, жду компенсацию.",
        "Проблема с интернетом, не работает уже два дня.",
        "Техническая поддержка: Перезагрузите роутер.",
        "Клиент: Спасибо, помогло!",
        "Хочу отменить подписку на услуги.",
        "Оператор: Оформлю отмену, деньги вернутся в течение 5 дней.",
        "Клиент: Отлично, спасибо за помощь.",
        "Не могу войти в личный кабинет, забыл пароль.",
        "Оператор: Отправлю ссылку для восстановления пароля.",
        "Клиент: Получил, спасибо!",
        "Проблема с мобильным приложением, постоянно вылетает.",
        "Техническая поддержка: Обновите приложение до последней версии.",
        "Клиент: Обновил, теперь работает нормально."
    ]
    
    try:
        print("🧪 Тестирование HierDictSimple...")
        
        # Создаем упрощенный иерархический словарь
        hd = HierDictSimple(test_texts)
        
        # Строим словарь
        dictionary = hd.build("test_simple_dict.json")
        
        if dictionary:
            print("✅ Словарь создан успешно")
            print(f"📚 Parent слов: {len(dictionary.get('parent', {}).get('must', []))}")
            print(f"📚 Child слов: {len(dictionary.get('child', {}).get('must', []))}")
            print(f"📚 Exclude слов: {len(dictionary.get('parent', {}).get('exclude', []))}")
            
            # Показываем примеры слов
            parent_words = dictionary.get('parent', {}).get('must', [])
            child_words = dictionary.get('child', {}).get('must', [])
            exclude_words = dictionary.get('parent', {}).get('exclude', [])
            
            print(f"\n📝 Примеры Parent слов: {parent_words[:5]}")
            print(f"📝 Примеры Child слов: {child_words[:5]}")
            print(f"📝 Примеры Exclude слов: {exclude_words[:5]}")
        
        # Тестируем качество
        print("\n🧪 Тестирование качества...")
        quality = hd.test()
        print(f"✅ Качество: {quality:.2f}")
        
        # Получаем статистику
        stats = hd.get_statistics()
        print(f"\n📊 Статистика:")
        print(f"  📚 Всего текстов: {stats.get('total_texts', 0)}")
        print(f"  📚 Parent слов: {stats.get('parent_words', 0)}")
        print(f"  📚 Child слов: {stats.get('child_words', 0)}")
        
        # Экспортируем для Smart Logger
        print("\n💾 Экспорт для Smart Logger...")
        output_file = hd.export_to_smart_logger("test_simple_smart_logger_dict.json")
        if output_file:
            print(f"✅ Экспорт завершен: {output_file}")
        
        print("\n🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
