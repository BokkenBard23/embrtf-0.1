#!/usr/bin/env python3
"""Скрипт запуска улучшенного GUI CallCenter AI v3."""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Главная функция запуска GUI."""
    try:
        print("🚀 Запуск CallCenter AI v3 - Улучшенный GUI")
        print("=" * 50)
        
        # Проверяем наличие необходимых модулей
        try:
            import gui_ru
            print("✅ Модуль GUI загружен")
        except ImportError as e:
            print(f"❌ Ошибка импорта GUI: {e}")
            return 1
        
        try:
            import data_manager
            print("✅ Модуль управления данными загружен")
        except ImportError as e:
            print(f"❌ Ошибка импорта data_manager: {e}")
            return 1
        
        try:
            import config
            print("✅ Конфигурация загружена")
        except ImportError as e:
            print(f"❌ Ошибка импорта config: {e}")
            return 1
        
        # Запускаем GUI
        print("🎨 Запуск интерфейса...")
        gui_ru.main()
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
