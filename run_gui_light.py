#!/usr/bin/env python3
"""Скрипт запуска облегченного GUI CallCenter AI v3."""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Главная функция запуска облегченного GUI."""
    try:
        print("🚀 Запуск CallCenter AI v3 - Облегченный GUI")
        print("=" * 50)
        print("⚠️  Эта версия работает без тяжелых зависимостей")
        print("📋 Доступные функции: просмотр данных, поиск, статистика")
        print("=" * 50)
        
        # Проверяем наличие необходимых модулей
        try:
            import gui_light
            print("✅ Модуль GUI загружен")
        except ImportError as e:
            print(f"❌ Ошибка импорта GUI: {e}")
            return 1
        
        try:
            import config
            print("✅ Конфигурация загружена")
        except ImportError as e:
            print(f"❌ Ошибка импорта config: {e}")
            return 1
        
        try:
            from utils import get_db_connection
            print("✅ Утилиты загружены")
        except ImportError as e:
            print(f"❌ Ошибка импорта utils: {e}")
            return 1
        
        # Запускаем GUI
        print("🎨 Запуск облегченного интерфейса...")
        gui_light.main()
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
