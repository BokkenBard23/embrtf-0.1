#!/usr/bin/env python3
"""Тестирование модуля иерархических словарей."""

import sys
import os
import logging
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Тестирование модуля иерархических словарей."""
    try:
        print("🧪 Тестирование модуля иерархических словарей")
        print("=" * 60)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("hier_dict_test.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Проверяем зависимости
        print("📚 Проверка зависимостей...")
        
        try:
            import pandas as pd
            print("✅ pandas доступен")
        except ImportError:
            print("❌ pandas не установлен")
            return 1
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            print("✅ scikit-learn доступен")
        except ImportError as e:
            print(f"❌ scikit-learn не установлен: {e}")
            return 1
        
        try:
            from txtai import Embeddings
            print("✅ txtai доступен")
        except ImportError as e:
            print(f"❌ txtai не установлен: {e}")
            return 1
        
        # Импортируем модуль
        from hier_dict import HierDict
        print("✅ hier_dict модуль загружен")
        
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
        
        print(f"\n📊 Тестовых текстов: {len(test_texts)}")
        
        # Создаем эмбеддинги
        print("\n🔄 Создание эмбеддингов...")
        
        # Эмбеддинги для диалогов
        dial_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
        dial_emb.index(test_texts)
        print("✅ Эмбеддинги диалогов созданы")
        
        # Эмбеддинги для предложений
        sentences = []
        for text in test_texts:
            sentences.extend([s.strip() for s in text.split('.') if s.strip()])
        
        sent_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
        sent_emb.index(sentences)
        print(f"✅ Эмбеддинги предложений созданы ({len(sentences)} предложений)")
        
        # Эмбеддинги для фраз
        phrases = []
        for text in test_texts:
            phrases.extend([p.strip() for p in text.split() if len(p.strip()) > 2])
        
        phrase_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
        phrase_emb.index(phrases)
        print(f"✅ Эмбеддинги фраз созданы ({len(phrases)} фраз)")
        
        # Создаем иерархический словарь
        print("\n🔄 Создание иерархического словаря...")
        hd = HierDict(dial_emb, sent_emb, phrase_emb, test_texts)
        
        # Строим словарь
        dictionary = hd.build("test_hierarchical_dict.json")
        
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
        precision = hd.test(k=3)
        print(f"✅ Precision@3: {precision:.2f}")
        
        # Получаем статистику
        stats = hd.get_statistics()
        print(f"\n📊 Статистика:")
        print(f"  📚 Всего текстов: {stats.get('total_texts', 0)}")
        print(f"  📚 Parent слов: {stats.get('parent_words', 0)}")
        print(f"  📚 Child слов: {stats.get('child_words', 0)}")
        
        # Экспортируем для Smart Logger
        print("\n💾 Экспорт для Smart Logger...")
        output_file = hd.export_to_smart_logger("test_smart_logger_dict.json")
        if output_file:
            print(f"✅ Экспорт завершен: {output_file}")
        
        print("\n🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
