"""Единый интерактивный тест всех систем проекта CallCenter AI v3."""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SystemTester:
    """Интерактивный тестер всех систем проекта."""
    
    def __init__(self):
        self.results = {}
        self.test_data = {}
        
    def print_header(self, title: str):
        """Печать заголовка теста."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
        
    def print_result(self, test_name: str, success: bool, message: str = ""):
        """Печать результата теста."""
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{status} {test_name}")
        if message:
            print(f"    {message}")
        self.results[test_name] = success
        
    def test_basic_imports(self) -> bool:
        """Тест базовых импортов."""
        self.print_header("БАЗОВЫЕ ИМПОРТЫ")
        
        try:
            import sqlite3
            self.print_result("sqlite3", True)
        except ImportError as e:
            self.print_result("sqlite3", False, str(e))
            return False
            
        try:
            import tkinter
            self.print_result("tkinter", True)
        except ImportError as e:
            self.print_result("tkinter", False, str(e))
            
        try:
            import numpy
            self.print_result("numpy", True)
        except ImportError as e:
            self.print_result("numpy", False, str(e))
            
        try:
            import faiss
            self.print_result("faiss", True)
        except ImportError as e:
            self.print_result("faiss", False, str(e))
            
        try:
            import requests
            self.print_result("requests", True)
        except ImportError as e:
            self.print_result("requests", False, str(e))
            
        return True
        
    def test_config(self) -> bool:
        """Тест конфигурации."""
        self.print_header("КОНФИГУРАЦИЯ")
        
        try:
            import config
            self.print_result("config.py", True)
            
            # Проверяем основные параметры
            required_attrs = [
                'DATABASE_PATH', 'EMBEDDING_MODEL_NAME', 'EMBEDDING_MODEL_DEVICE',
                'LLM_MODEL_NAME', 'LLM_API_URL', 'ANALYSIS_METHODS'
            ]
            
            for attr in required_attrs:
                if hasattr(config, attr):
                    self.print_result(f"config.{attr}", True)
                else:
                    self.print_result(f"config.{attr}", False, "Отсутствует")
                    
            return True
            
        except Exception as e:
            self.print_result("config.py", False, str(e))
            return False
            
    def test_database(self) -> bool:
        """Тест базы данных."""
        self.print_header("БАЗА ДАННЫХ")
        
        try:
            import config
            from utils import get_db_connection
            
            # Подключение к БД
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Проверяем таблицы
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['dialogs', 'utterances', 'utterance_embeddings', 'faiss_indexes']
            for table in required_tables:
                if table in tables:
                    self.print_result(f"Таблица {table}", True)
                else:
                    self.print_result(f"Таблица {table}", False, "Отсутствует")
                    
            # Проверяем данные
            cursor.execute("SELECT COUNT(*) FROM utterances")
            utterance_count = cursor.fetchone()[0]
            self.print_result(f"Реплик в БД", True, f"{utterance_count} записей")
            
            cursor.execute("SELECT COUNT(*) FROM dialogs")
            dialog_count = cursor.fetchone()[0]
            self.print_result(f"Диалогов в БД", True, f"{dialog_count} записей")
            
            conn.close()
            return True
            
        except Exception as e:
            self.print_result("База данных", False, str(e))
            return False
            
    def test_faiss_indexes(self) -> bool:
        """Тест FAISS индексов."""
        self.print_header("FAISS ИНДЕКСЫ")
        
        try:
            import faiss
            import json
            from pathlib import Path
            
            # Проверяем файлы индексов
            faiss_dir = Path("faiss_index")
            if not faiss_dir.exists():
                self.print_result("Директория faiss_index", False, "Не существует")
                return False
                
            self.print_result("Директория faiss_index", True)
            
            # Проверяем индексы
            index_files = list(faiss_dir.glob("*.index"))
            if not index_files:
                self.print_result("Файлы индексов", False, "Не найдены")
                return False
                
            self.print_result("Файлы индексов", True, f"{len(index_files)} файлов")
            
            # Проверяем ID файлы
            id_files = list(faiss_dir.glob("ids_*.json"))
            self.print_result("ID файлы", True, f"{len(id_files)} файлов")
            
            # Загружаем один индекс для проверки
            try:
                index = faiss.read_index(str(index_files[0]))
                self.print_result("Загрузка индекса", True, f"Размер: {index.ntotal}")
            except Exception as e:
                self.print_result("Загрузка индекса", False, str(e))
                
            return True
            
        except Exception as e:
            self.print_result("FAISS индексы", False, str(e))
            return False
            
    def test_analysis_methods(self) -> bool:
        """Тест методов анализа."""
        self.print_header("МЕТОДЫ АНАЛИЗА")
        
        try:
            import analysis_methods
            
            methods = ["hierarchical", "rolling", "facts", "classification", "callback_classifier", "fast_phrase_classifier"]
            
            for method in methods:
                func = analysis_methods.get_analysis_method(method)
                if func:
                    self.print_result(f"Метод {method}", True)
                else:
                    self.print_result(f"Метод {method}", False, "Не найден")
                    
            return True
            
        except Exception as e:
            self.print_result("Методы анализа", False, str(e))
            return False
            
    def test_gui_components(self) -> bool:
        """Тест компонентов GUI."""
        self.print_header("КОМПОНЕНТЫ GUI")
        
        try:
            # Тестируем импорт GUI модулей
            import gui
            self.print_result("gui.py импорт", True)
            
            # Тестируем функции GUI
            from gui import load_faiss_indexes, load_data_lookups
            
            # Загружаем индексы
            try:
                load_faiss_indexes()
                self.print_result("Загрузка индексов", True)
            except Exception as e:
                self.print_result("Загрузка индексов", False, str(e))
                
            # Загружаем данные
            try:
                load_data_lookups()
                self.print_result("Загрузка данных", True)
            except Exception as e:
                self.print_result("Загрузка данных", False, str(e))
                
            return True
            
        except Exception as e:
            self.print_result("GUI компоненты", False, str(e))
            return False
            
    def test_model_loading(self) -> bool:
        """Тест загрузки модели (опционально)."""
        self.print_header("ЗАГРУЗКА МОДЕЛИ")
        
        try:
            import torch
            import config
            
            # Проверяем CUDA
            cuda_available = torch.cuda.is_available()
            self.print_result("CUDA доступна", cuda_available)
            
            # Проверяем конфигурацию
            device = config.EMBEDDING_MODEL_DEVICE
            self.print_result(f"Устройство в конфиге", True, device)
            
            # Пробуем загрузить модель (только если пользователь согласится)
            response = input("Загрузить модель для тестирования? (y/n): ").lower().strip()
            if response == 'y':
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
                    self.print_result("Загрузка модели", True, "Успешно")
                except Exception as e:
                    self.print_result("Загрузка модели", False, str(e))
            else:
                self.print_result("Загрузка модели", True, "Пропущено")
                
            return True
            
        except Exception as e:
            self.print_result("Загрузка модели", False, str(e))
            return False
            
    def test_api_connectivity(self) -> bool:
        """Тест подключения к API."""
        self.print_header("API ПОДКЛЮЧЕНИЯ")
        
        try:
            import requests
            import config
            
            # Тест Ollama
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code == 200:
                    self.print_result("Ollama API", True, "Доступен")
                else:
                    self.print_result("Ollama API", False, f"Код {response.status_code}")
            except Exception as e:
                self.print_result("Ollama API", False, "Недоступен")
                
            # Тест OpenRouter (если настроен)
            if config.OPENROUTER_API_KEY:
                try:
                    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
                    if response.status_code == 200:
                        self.print_result("OpenRouter API", True, "Доступен")
                    else:
                        self.print_result("OpenRouter API", False, f"Код {response.status_code}")
                except Exception as e:
                    self.print_result("OpenRouter API", False, "Недоступен")
            else:
                self.print_result("OpenRouter API", True, "Не настроен")
                
            return True
            
        except Exception as e:
            self.print_result("API подключения", False, str(e))
            return False
            
    def run_quick_test(self):
        """Быстрый тест основных функций."""
        self.print_header("БЫСТРЫЙ ТЕСТ СИСТЕМЫ")
        
        tests = [
            ("Базовые импорты", self.test_basic_imports),
            ("Конфигурация", self.test_config),
            ("База данных", self.test_database),
            ("FAISS индексы", self.test_faiss_indexes),
            ("Методы анализа", self.test_analysis_methods),
            ("GUI компоненты", self.test_gui_components),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.print_result(test_name, False, f"Ошибка: {e}")
                
    def run_full_test(self):
        """Полный тест всех систем."""
        self.print_header("ПОЛНЫЙ ТЕСТ СИСТЕМЫ")
        
        tests = [
            ("Базовые импорты", self.test_basic_imports),
            ("Конфигурация", self.test_config),
            ("База данных", self.test_database),
            ("FAISS индексы", self.test_faiss_indexes),
            ("Методы анализа", self.test_analysis_methods),
            ("GUI компоненты", self.test_gui_components),
            ("Загрузка модели", self.test_model_loading),
            ("API подключения", self.test_api_connectivity),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.print_result(test_name, False, f"Ошибка: {e}")
                
    def print_summary(self):
        """Печать итогового отчета."""
        self.print_header("ИТОГОВЫЙ ОТЧЕТ")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for success in self.results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Всего тестов: {total_tests}")
        print(f"Пройдено: {passed_tests}")
        print(f"Провалено: {failed_tests}")
        print(f"Успешность: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ ПРОВАЛЕННЫЕ ТЕСТЫ:")
            for test_name, success in self.results.items():
                if not success:
                    print(f"  - {test_name}")
                    
        print(f"\n{'='*60}")
        if failed_tests == 0:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к работе.")
        else:
            print("⚠️ Есть проблемы. Проверьте проваленные тесты.")
        print(f"{'='*60}")

def main():
    """Главная функция."""
    print("🧪 CallCenter AI v3 - Интерактивный тестер систем")
    print("=" * 60)
    
    tester = SystemTester()
    
    while True:
        print("\nВыберите тип тестирования:")
        print("1. Быстрый тест (основные функции)")
        print("2. Полный тест (все системы)")
        print("3. Выход")
        
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == "1":
            tester.run_quick_test()
            tester.print_summary()
        elif choice == "2":
            tester.run_full_test()
            tester.print_summary()
        elif choice == "3":
            print("До свидания! 👋")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")
            
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()
