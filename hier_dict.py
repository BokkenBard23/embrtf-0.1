"""
Hierarchical Dictionary Builder for txtai 3-level embeddings
parent → child → exclude  +  auto-test (Precision@k)

Usage:
    from hier_dict import HierDict
    hd = HierDict(dial_emb, sent_emb, phrase_emb, texts)
    dictionary = hd.build()
    hd.test(k=5)           # выводит Precision@k
"""
import json
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class HierDict:
    def __init__(self, dial_embeddings, sent_embeddings, phrase_embeddings, texts):
        """
        Инициализация иерархического словаря.
        
        Args:
            dial_embeddings: txtai.Embeddings (диалоги)
            sent_embeddings: txtai.Embeddings (предложения)
            phrase_embeddings: txtai.Embeddings (фразы)
            texts: List[str] – исходные диалоги для тестов
        """
        self.dial_emb = dial_embeddings
        self.sent_emb = sent_embeddings
        self.phrase_emb = phrase_embeddings
        self.texts = texts
        
        logger.info(f"Инициализирован HierDict с {len(texts)} текстами")

    # ---------- 1. Авто-извлечение ключевых слов ----------
    def _keywords(self, emb, texts, top_k=20):
        """
        Возвращает top-k слов с наибольшей частотой и векторной длиной.
        
        Args:
            emb: txtai.Embeddings объект
            texts: Список текстов для анализа
            top_k: Количество ключевых слов для извлечения
            
        Returns:
            Список ключевых слов
        """
        try:
            # Просто считаем TF + векторную «важность»
            words = [w.lower() for t in texts for w in t.split()]
            # Вектор каждого слова
            unique = list(set(words))
            if not unique:
                return []
            
            w_vec = emb.embed(unique)
            # Сумма косинусов = «центральность»
            centr = cosine_similarity(w_vec, w_vec).sum(axis=1)
            df = pd.DataFrame({"word": unique, "score": centr})
            
            keywords = df.sort_values("score", ascending=False).head(top_k)["word"].tolist()
            logger.info(f"Извлечено {len(keywords)} ключевых слов")
            return keywords
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых слов: {e}")
            return []

    # ---------- 2. Построение 3-уровня ----------
    def build(self, out_file="smart_logger_dict.json"):
        """
        Построение иерархического словаря.
        
        Args:
            out_file: Путь для сохранения словаря
            
        Returns:
            Словарь с parent, child и exclude словами
        """
        try:
            logger.info("Начинаем построение иерархического словаря...")
            
            # Извлекаем ключевые слова для каждого уровня
            parent = self._keywords(self.dial_emb, self.texts, 15)
            child = self._keywords(self.sent_emb, self.texts, 15)
            
            # Exclude = антонимы через embeddings (самые далёкие)
            all_w = list(set(" ".join(self.texts).lower().split()))
            if all_w:
                w_vec = self.phrase_emb.embed(all_w)
                parent_vec = self.phrase_emb.embed(parent)
                dist = cosine_similarity(parent_vec, w_vec).mean(axis=0)
                exclude = [all_w[i] for i in dist.argsort()[-10:]]  # 10 самых «дальних»
            else:
                exclude = []

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

    # ---------- 3. Авто-тест Precision@k ----------
    def test(self, k=5):
        """
        Считаем Precision@k: считаем первые k результатов поиска по parent-фразам.
        
        Args:
            k: Количество результатов для анализа
            
        Returns:
            Значение Precision@k
        """
        try:
            logger.info(f"Запуск теста Precision@{k}...")
            
            # Условный запрос
            query_text = " ".join(self.texts[:50])
            parent_phrases = self.dial_emb.embed([query_text])
            results = self.dial_emb.search(query_text, limit=k)
            
            # Считаем «релевантными» те, где есть хотя бы 1 слово из parent
            parent_words = set(self._keywords(self.dial_emb, self.texts[:50], 10))
            relevant = sum(1 for r in results if any(w in r["text"] for w in parent_words))
            precision = relevant / k if k > 0 else 0
            
            logger.info(f"Precision@{k} = {precision:.2f}")
            logger.info(f"Релевантных результатов: {relevant} из {k}")
            
            return precision
            
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
            parent = self._keywords(self.dial_emb, self.texts, 15)
            child = self._keywords(self.sent_emb, self.texts, 15)
            
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

    def export_to_smart_logger(self, output_file="smart_logger_dict.json"):
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
                    "created_by": "CallCenter AI v3",
                    "version": "1.0",
                    "total_texts": len(self.texts),
                    "description": "Иерархический словарь для классификации диалогов"
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


# ---------- 4. Быстрый self-test ----------
if __name__ == "__main__":
    """Пример использования на тестовых данных."""
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Тестовые данные
    texts = [
        "Здравствуйте, бонус не начислили, хотя обещали подарок.",
        "Оператор: Проверю информацию и вернусь.",
        "Клиент: Спасибо, жду компенсацию.",
        "Проблема с интернетом, не работает уже два дня.",
        "Техническая поддержка: Перезагрузите роутер.",
        "Клиент: Спасибо, помогло!",
        "Хочу отменить подписку на услуги.",
        "Оператор: Оформлю отмену, деньги вернутся в течение 5 дней.",
        "Клиент: Отлично, спасибо за помощь."
    ]

    try:
        from txtai import Embeddings

        print("🧪 Тестирование HierDict...")
        
        # Создаем эмбеддинги
        dial_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
        sent_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
        phr_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})

        # Индексируем данные
        dial_emb.index(texts)
        sent_emb.index([s for t in texts for s in t.split(".") if s.strip()])
        phr_emb.index([p for t in texts for p in t.split() if p.strip()])

        # Создаем иерархический словарь
        hd = HierDict(dial_emb, sent_emb, phr_emb, texts)
        
        # Строим словарь
        dictionary = hd.build("test_dict.json")
        
        # Тестируем
        precision = hd.test(k=3)
        
        # Получаем статистику
        stats = hd.get_statistics()
        print(f"📊 Статистика: {stats}")
        
        # Экспортируем для Smart Logger
        hd.export_to_smart_logger("smart_logger_test.json")
        
        print("✅ Тестирование завершено успешно!")
        
    except ImportError:
        print("❌ txtai не установлен. Установите: pip install txtai")
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
