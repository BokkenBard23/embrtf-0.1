#!/usr/bin/env python3
"""Тестирование оптимизированного процессора эмбеддингов."""

import sys
import os
import logging
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Тестирование процессора эмбеддингов."""
    try:
        print("🧪 Тестирование оптимизированного процессора эмбеддингов")
        print("=" * 60)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("processor_test.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Импортируем процессор
        from optimized_embedding_processor import OptimizedEmbeddingProcessor
        
        # Создаем процессор с оптимальными параметрами для RTX 3060
        processor = OptimizedEmbeddingProcessor(
            batch_size=32,  # Оптимальный размер для RTX 3060
            max_length=2048  # Полный контекст
        )
        
        print(f"🎯 Устройство: {processor.device}")
        print(f"📊 Размер батча: {processor.batch_size}")
        print(f"📏 Максимальная длина: {processor.max_length}")
        
        # Тестируем на различных размерах данных
        test_cases = [
            ("Маленький тест", 5),
            ("Средний тест", 50),
            ("Большой тест", 200)
        ]
        
        for test_name, num_texts in test_cases:
            print(f"\n🔬 {test_name} ({num_texts} текстов)")
            print("-" * 40)
            
            # Генерируем тестовые тексты
            test_texts = [
                f"Тестовый текст номер {i+1} для проверки работы процессора эмбеддингов. "
                f"Это довольно длинный текст, который должен проверить работу с контекстом {processor.max_length} символов. "
                f"Содержит различные слова и фразы для создания качественных эмбеддингов."
                for i in range(num_texts)
            ]
            
            try:
                # Обрабатываем тексты
                print(f"🔄 Обработка {len(test_texts)} текстов...")
                embeddings = processor.process_texts_batch(test_texts)
                
                print(f"✅ Успешно обработано: {len(embeddings)} текстов")
                print(f"📊 Размер эмбеддинга: {embeddings.shape}")
                print(f"💾 Тип данных: {embeddings.dtype}")
                
                # Проверяем память
                memory = processor._get_memory_usage()
                if processor.device == "cuda":
                    print(f"💾 GPU память: {memory['gpu_allocated']:.2f}GB/{memory['gpu_total']:.2f}GB")
                    print(f"📈 Использование: {memory['gpu_allocated']/memory['gpu_total']*100:.1f}%")
                
                # Очищаем память
                del embeddings
                processor._unload_model()
                
            except Exception as e:
                print(f"❌ Ошибка в {test_name}: {e}")
                continue
        
        print("\n🎉 Тестирование завершено!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
