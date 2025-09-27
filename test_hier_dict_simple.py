#!/usr/bin/env python3
"""Упрощенное тестирование модуля иерархических словарей."""

import sys
import os
import logging
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Упрощенное тестирование модуля."""
    try:
        print("🧪 Упрощенное тестирование модуля иерархических словарей")
        print("=" * 60)
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        
        # Проверяем зависимости по одной
        print("📚 Проверка зависимостей...")
        
        try:
            import pandas as pd
            print("✅ pandas доступен")
        except ImportError as e:
            print(f"❌ pandas не установлен: {e}")
            return 1
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            print("✅ scikit-learn доступен")
        except ImportError as e:
            print(f"❌ scikit-learn не установлен: {e}")
            return 1
        
        print("🔄 Попытка импорта txtai...")
        try:
            from txtai import Embeddings
            print("✅ txtai доступен")
        except ImportError as e:
            print(f"❌ txtai не установлен: {e}")
            print("💡 Установите: pip install txtai")
            return 1
        except Exception as e:
            print(f"❌ Ошибка импорта txtai: {e}")
            return 1
        
        print("🔄 Импорт hier_dict...")
        try:
            from hier_dict import HierDict
            print("✅ hier_dict модуль загружен")
        except ImportError as e:
            print(f"❌ hier_dict не найден: {e}")
            return 1
        except Exception as e:
            print(f"❌ Ошибка импорта hier_dict: {e}")
            return 1
        
        print("\n🎉 Все зависимости загружены успешно!")
        print("✅ Модуль готов к использованию")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
