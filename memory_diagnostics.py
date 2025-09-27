#!/usr/bin/env python3
"""Диагностика использования памяти для CallCenter AI v3."""

import sys
import os
import gc
import psutil
import logging
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def get_system_memory_info():
    """Получение информации о системной памяти."""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024**3,
        'available': memory.available / 1024**3,
        'used': memory.used / 1024**3,
        'percent': memory.percent
    }

def get_gpu_memory_info():
    """Получение информации о памяти GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            }
    except ImportError:
        pass
    return None

def test_model_loading():
    """Тестирование загрузки модели с диагностикой памяти."""
    print("🧪 Тестирование загрузки модели с диагностикой памяти")
    print("=" * 60)
    
    # Системная память до загрузки
    print("\n📊 Системная память ДО загрузки:")
    sys_mem = get_system_memory_info()
    print(f"  💾 Всего: {sys_mem['total']:.1f} GB")
    print(f"  ✅ Доступно: {sys_mem['available']:.1f} GB")
    print(f"  📈 Использовано: {sys_mem['used']:.1f} GB ({sys_mem['percent']:.1f}%)")
    
    # GPU память до загрузки
    gpu_mem_before = get_gpu_memory_info()
    if gpu_mem_before:
        print(f"\n🎮 GPU память ДО загрузки:")
        print(f"  💾 Всего: {gpu_mem_before['total']:.1f} GB")
        print(f"  ✅ Свободно: {gpu_mem_before['free']:.1f} GB")
        print(f"  📈 Использовано: {gpu_mem_before['allocated']:.1f} GB")
    
    try:
        print(f"\n🔄 Загрузка модели...")
        
        # Импортируем и загружаем модель
        from sentence_transformers import SentenceTransformer
        import config
        
        # Загружаем модель
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cuda")
        model.max_seq_length = 2048
        
        print(f"✅ Модель загружена успешно")
        
        # Системная память после загрузки
        print(f"\n📊 Системная память ПОСЛЕ загрузки:")
        sys_mem_after = get_system_memory_info()
        print(f"  💾 Всего: {sys_mem_after['total']:.1f} GB")
        print(f"  ✅ Доступно: {sys_mem_after['available']:.1f} GB")
        print(f"  📈 Использовано: {sys_mem_after['used']:.1f} GB ({sys_mem_after['percent']:.1f}%)")
        print(f"  📊 Изменение: {sys_mem_after['used'] - sys_mem['used']:.1f} GB")
        
        # GPU память после загрузки
        gpu_mem_after = get_gpu_memory_info()
        if gpu_mem_after:
            print(f"\n🎮 GPU память ПОСЛЕ загрузки:")
            print(f"  💾 Всего: {gpu_mem_after['total']:.1f} GB")
            print(f"  ✅ Свободно: {gpu_mem_after['free']:.1f} GB")
            print(f"  📈 Использовано: {gpu_mem_after['allocated']:.1f} GB")
            print(f"  📊 Изменение: {gpu_mem_after['allocated'] - gpu_mem_before['allocated']:.1f} GB")
        
        # Тестируем обработку текста
        print(f"\n🧪 Тестирование обработки текста...")
        test_text = "Это тестовый текст для проверки работы модели эмбеддингов."
        
        embedding = model.encode([test_text])
        print(f"✅ Текст обработан, размер эмбеддинга: {embedding.shape}")
        
        # Очищаем память
        del model, embedding
        gc.collect()
        
        # Проверяем память после очистки
        print(f"\n🗑️ Память ПОСЛЕ очистки:")
        sys_mem_clean = get_system_memory_info()
        print(f"  📈 Использовано: {sys_mem_clean['used']:.1f} GB ({sys_mem_clean['percent']:.1f}%)")
        
        gpu_mem_clean = get_gpu_memory_info()
        if gpu_mem_clean:
            print(f"  🎮 GPU использовано: {gpu_mem_clean['allocated']:.1f} GB")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False
    
    return True

def main():
    """Главная функция диагностики."""
    print("🔍 Диагностика памяти CallCenter AI v3")
    print("=" * 60)
    
    try:
        # Проверяем доступность библиотек
        print("📚 Проверка библиотек...")
        
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            print(f"🎮 CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🎮 CUDA версия: {torch.version.cuda}")
                print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        except ImportError:
            print("❌ PyTorch не установлен")
        
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ SentenceTransformers доступен")
        except ImportError:
            print("❌ SentenceTransformers не установлен")
        
        try:
            import psutil
            print("✅ psutil доступен")
        except ImportError:
            print("❌ psutil не установлен")
        
        # Запускаем тестирование
        if test_model_loading():
            print("\n🎉 Диагностика завершена успешно!")
        else:
            print("\n❌ Диагностика завершена с ошибками")
            return 1
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
