"""Облегченная версия GUI для CallCenter AI v3 без тяжелых зависимостей."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sqlite3
import json
import threading
import time
from pathlib import Path
import config
from utils import get_db_connection
import logging
from datetime import datetime
from tkinter import simpledialog
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gui_light.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightCallCenterGUI:
    """Облегченная версия GUI без тяжелых зависимостей."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CallCenter AI v3 - Облегченный GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Переменные для прогресс-баров
        self.progress_vars = {}
        self.status_vars = {}
        
        # Инициализация
        self.init_chat_db()
        self.create_widgets()
        self.load_initial_data()
        
    def init_chat_db(self):
        """Инициализация базы данных для чата."""
        try:
            self.conn = get_db_connection()
            logger.info("✅ База данных для чата инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")
            messagebox.showerror("Ошибка", f"Не удалось подключиться к базе данных: {e}")
    
    def create_widgets(self):
        """Создание всех виджетов интерфейса."""
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка 1: Управление данными
        self.create_data_management_tab()
        
        # Вкладка 2: Поиск
        self.create_search_tab()
        
        # Вкладка 3: Статистика
        self.create_statistics_tab()
        
        # Вкладка 4: Логи
        self.create_logs_tab()
        
    def create_data_management_tab(self):
        """Создание вкладки управления данными."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Управление данными")
        
        # Заголовок
        title_label = tk.Label(self.data_frame, text="Управление данными CallCenter AI", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для группировки
        self.create_data_controls()
        self.create_data_tables()
        
    def create_data_controls(self):
        """Создание элементов управления данными."""
        # Фрейм для кнопок управления
        controls_frame = ttk.LabelFrame(self.data_frame, text="Операции с данными", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопки управления
        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        # Просмотр диалогов
        self.view_dialogs_btn = tk.Button(buttons_frame, text="💬 Просмотр диалогов", 
                                         command=self.view_dialogs, 
                                         bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                         width=20, height=2)
        self.view_dialogs_btn.pack(side='left', padx=5, pady=5)
        
        # Просмотр реплик
        self.view_utterances_btn = tk.Button(buttons_frame, text="🗣️ Просмотр реплик", 
                                            command=self.view_utterances, 
                                            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                            width=20, height=2)
        self.view_utterances_btn.pack(side='left', padx=5, pady=5)
        
        # Просмотр фраз
        self.view_phrases_btn = tk.Button(buttons_frame, text="📝 Просмотр фраз", 
                                         command=self.view_phrases, 
                                         bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                         width=20, height=2)
        self.view_phrases_btn.pack(side='left', padx=5, pady=5)
        
        # Экспорт данных
        self.export_btn = tk.Button(buttons_frame, text="📤 Экспорт данных", 
                                   command=self.export_data, 
                                   bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
                                   width=20, height=2)
        self.export_btn.pack(side='left', padx=5, pady=5)
        
    def create_data_tables(self):
        """Создание таблиц для отображения данных."""
        tables_frame = ttk.LabelFrame(self.data_frame, text="Таблицы данных", padding=10)
        tables_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Создаем Treeview для данных
        columns = ('ID', 'Текст', 'Источник', 'Дата')
        self.data_tree = ttk.Treeview(tables_frame, columns=columns, show='headings', height=15)
        
        # Настройка колонок
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=200)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(tables_frame, orient='vertical', command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(tables_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Размещение
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tables_frame.grid_rowconfigure(0, weight=1)
        tables_frame.grid_columnconfigure(0, weight=1)
        
    def create_search_tab(self):
        """Создание вкладки поиска."""
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="🔍 Поиск")
        
        # Заголовок
        title_label = tk.Label(self.search_frame, text="Поиск по данным", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для поиска
        self.create_search_controls()
        self.create_search_results()
        
    def create_search_controls(self):
        """Создание элементов управления поиском."""
        # Фрейм для поиска
        search_frame = ttk.LabelFrame(self.search_frame, text="Параметры поиска", padding=10)
        search_frame.pack(fill='x', padx=10, pady=5)
        
        # Поле поиска
        search_input_frame = tk.Frame(search_frame)
        search_input_frame.pack(fill='x', pady=5)
        
        tk.Label(search_input_frame, text="Поисковый запрос:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.search_entry = tk.Entry(search_input_frame, font=('Arial', 12), width=60)
        self.search_entry.pack(fill='x', pady=2)
        self.search_entry.bind('<Return>', lambda e: self.perform_search())
        
        # Кнопки поиска
        search_buttons_frame = tk.Frame(search_frame)
        search_buttons_frame.pack(fill='x', pady=10)
        
        self.search_btn = tk.Button(search_buttons_frame, text="🔍 Найти", 
                                   command=self.perform_search, 
                                   bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                   width=15, height=2)
        self.search_btn.pack(side='left', padx=5)
        
        self.clear_results_btn = tk.Button(search_buttons_frame, text="🗑️ Очистить", 
                                          command=self.clear_results, 
                                          bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                          width=15, height=2)
        self.clear_results_btn.pack(side='left', padx=5)
        
    def create_search_results(self):
        """Создание области результатов поиска."""
        results_frame = ttk.LabelFrame(self.search_frame, text="Результаты поиска", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Создаем Treeview для результатов
        columns = ('ID', 'Текст', 'Тип', 'Дата')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        # Настройка колонок
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=200)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(results_frame, orient='horizontal', command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Размещение
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
    def create_statistics_tab(self):
        """Создание вкладки статистики."""
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="📈 Статистика")
        
        # Заголовок
        title_label = tk.Label(self.stats_frame, text="Статистика системы", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для статистики
        self.create_stats_controls()
        self.create_stats_display()
        
    def create_stats_controls(self):
        """Создание элементов управления статистикой."""
        controls_frame = ttk.LabelFrame(self.stats_frame, text="Обновление статистики", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопка обновления
        self.refresh_stats_btn = tk.Button(controls_frame, text="🔄 Обновить статистику", 
                                          command=self.refresh_statistics, 
                                          bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                          width=20, height=2)
        self.refresh_stats_btn.pack(pady=5)
        
    def create_stats_display(self):
        """Создание области отображения статистики."""
        stats_display_frame = ttk.LabelFrame(self.stats_frame, text="Статистика данных", padding=10)
        stats_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Область статистики
        self.stats_display = scrolledtext.ScrolledText(stats_display_frame, height=20, width=80, 
                                                      font=('Arial', 10))
        self.stats_display.pack(fill='both', expand=True)
        
    def create_logs_tab(self):
        """Создание вкладки логов."""
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📋 Логи")
        
        # Заголовок
        title_label = tk.Label(self.logs_frame, text="Логи системы", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для логов
        self.create_logs_controls()
        self.create_logs_display()
        
    def create_logs_controls(self):
        """Создание элементов управления логами."""
        controls_frame = ttk.LabelFrame(self.logs_frame, text="Управление логами", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопки управления
        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        self.clear_logs_btn = tk.Button(buttons_frame, text="🗑️ Очистить логи", 
                                       command=self.clear_logs, 
                                       bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                       width=15, height=2)
        self.clear_logs_btn.pack(side='left', padx=5, pady=5)
        
        self.save_logs_btn = tk.Button(buttons_frame, text="💾 Сохранить логи", 
                                      command=self.save_logs, 
                                      bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                      width=15, height=2)
        self.save_logs_btn.pack(side='left', padx=5, pady=5)
        
    def create_logs_display(self):
        """Создание области отображения логов."""
        logs_display_frame = ttk.LabelFrame(self.logs_frame, text="Логи выполнения", padding=10)
        logs_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Область логов
        self.logs_display = scrolledtext.ScrolledText(logs_display_frame, height=20, width=80, 
                                                     font=('Arial', 10))
        self.logs_display.pack(fill='both', expand=True)
        
    def load_initial_data(self):
        """Загрузка начальных данных."""
        try:
            self.log_message("🔄 Загрузка начальных данных...")
            self.refresh_statistics()
            self.log_message("✅ Начальные данные загружены успешно")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки данных: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить начальные данные: {e}")
    
    def log_message(self, message):
        """Добавление сообщения в лог."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_display.see(tk.END)
        self.root.update_idletasks()
        
    def clear_logs(self):
        """Очистка логов."""
        self.logs_display.delete(1.0, tk.END)
        
    def save_logs(self):
        """Сохранение логов в файл."""
        filename = filedialog.asksaveasfilename(defaultextension=".txt", 
                                               filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.logs_display.get(1.0, tk.END))
            messagebox.showinfo("Успех", "Логи сохранены")
    
    # Методы управления данными
    def view_dialogs(self):
        """Просмотр диалогов."""
        self.log_message("💬 Загрузка диалогов...")
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, text, source_theme, processed_at FROM dialogs ORDER BY processed_at DESC LIMIT 100")
            results = cursor.fetchall()
            
            # Очищаем таблицу
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Заполняем данными
            for row in results:
                text_preview = row[1][:100] + '...' if len(row[1]) > 100 else row[1]
                self.data_tree.insert('', 'end', values=(
                    row[0], text_preview, row[2], row[3]
                ))
            
            self.log_message(f"✅ Загружено {len(results)} диалогов")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки диалогов: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить диалоги: {e}")
    
    def view_utterances(self):
        """Просмотр реплик."""
        self.log_message("🗣️ Загрузка реплик...")
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT u.id, u.text, u.speaker, d.processed_at
                FROM utterances u
                JOIN dialogs d ON u.dialog_id = d.id
                ORDER BY d.processed_at DESC, u.turn_order
                LIMIT 100
            """)
            results = cursor.fetchall()
            
            # Очищаем таблицу
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Заполняем данными
            for row in results:
                text_preview = row[1][:100] + '...' if len(row[1]) > 100 else row[1]
                self.data_tree.insert('', 'end', values=(
                    row[0], text_preview, row[2], row[3]
                ))
            
            self.log_message(f"✅ Загружено {len(results)} реплик")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки реплик: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить реплики: {e}")
    
    def view_phrases(self):
        """Просмотр фраз обратных звонков."""
        self.log_message("📝 Загрузка фраз...")
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, phrase, source, processed_at FROM callback_phrases ORDER BY processed_at DESC LIMIT 100")
            results = cursor.fetchall()
            
            # Очищаем таблицу
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Заполняем данными
            for row in results:
                phrase_preview = row[1][:100] + '...' if len(row[1]) > 100 else row[1]
                self.data_tree.insert('', 'end', values=(
                    row[0], phrase_preview, row[2], row[3]
                ))
            
            self.log_message(f"✅ Загружено {len(results)} фраз")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки фраз: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить фразы: {e}")
    
    def export_data(self):
        """Экспорт данных."""
        data_type = simpledialog.askstring("Экспорт данных", 
                                          "Выберите тип данных (dialogs/utterances/callback_phrases):")
        if not data_type:
            return
        
        filename = filedialog.asksaveasfilename(defaultextension=".json", 
                                               filetypes=[("JSON files", "*.json")])
        if not filename:
            return
        
        try:
            self.log_message(f"📤 Экспорт {data_type} в {filename}")
            
            cursor = self.conn.cursor()
            if data_type == "dialogs":
                cursor.execute("SELECT * FROM dialogs")
            elif data_type == "utterances":
                cursor.execute("SELECT * FROM utterances")
            elif data_type == "callback_phrases":
                cursor.execute("SELECT * FROM callback_phrases")
            else:
                messagebox.showerror("Ошибка", "Неверный тип данных")
                return
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Конвертируем в список словарей
            data = [dict(zip(columns, row)) for row in results]
            
            # Сохраняем в JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.log_message("✅ Данные экспортированы успешно")
            messagebox.showinfo("Успех", "Данные экспортированы успешно")
                
        except Exception as e:
            self.log_message(f"❌ Ошибка экспорта: {e}")
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    # Методы поиска
    def perform_search(self):
        """Выполнение поиска."""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Предупреждение", "Введите поисковый запрос")
            return
        
        self.log_message(f"🔍 Поиск: {query}")
        try:
            # Очищаем предыдущие результаты
            self.clear_results()
            
            # Выполняем поиск по репликам
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT u.id, u.text, u.speaker, d.source_theme, d.processed_at
                FROM utterances u
                JOIN dialogs d ON u.dialog_id = d.id
                WHERE u.text LIKE ?
                ORDER BY d.processed_at DESC
                LIMIT 50
            """, (f"%{query}%",))
            
            results = cursor.fetchall()
            
            # Отображаем результаты
            for result in results:
                text_preview = result[1][:100] + '...' if len(result[1]) > 100 else result[1]
                self.results_tree.insert('', 'end', values=(
                    result[0], text_preview, result[2], result[4]
                ))
            
            self.log_message(f"✅ Найдено {len(results)} результатов")
            
        except Exception as e:
            self.log_message(f"❌ Ошибка поиска: {e}")
            messagebox.showerror("Ошибка", f"Ошибка поиска: {e}")
    
    def clear_results(self):
        """Очистка результатов."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.log_message("🗑️ Результаты очищены")
    
    # Методы статистики
    def refresh_statistics(self):
        """Обновление статистики."""
        self.log_message("🔄 Обновление статистики...")
        try:
            stats = self.get_statistics()
            self.display_statistics(stats)
            self.log_message("✅ Статистика обновлена")
        except Exception as e:
            self.log_message(f"❌ Ошибка обновления статистики: {e}")
            messagebox.showerror("Ошибка", f"Не удалось обновить статистику: {e}")
    
    def get_statistics(self):
        """Получение статистики."""
        stats = {}
        cursor = self.conn.cursor()
        
        # Основная статистика
        cursor.execute("SELECT COUNT(*) FROM dialogs")
        stats['dialogs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM utterances")
        stats['utterances'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM utterance_embeddings")
        stats['embeddings'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM callback_phrases")
        stats['callback_phrases'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        stats['qa_pairs'] = cursor.fetchone()[0]
        
        # Статистика по темам
        cursor.execute("SELECT source_theme, COUNT(*) FROM dialogs GROUP BY source_theme")
        stats['themes'] = dict(cursor.fetchall())
        
        return stats
    
    def display_statistics(self, stats):
        """Отображение статистики."""
        self.stats_display.delete(1.0, tk.END)
        
        # Основная статистика
        self.stats_display.insert(tk.END, "=== ОСНОВНАЯ СТАТИСТИКА ===\n\n")
        self.stats_display.insert(tk.END, f"📊 Диалогов: {stats.get('dialogs', 0):,}\n")
        self.stats_display.insert(tk.END, f"🗣️ Реплик: {stats.get('utterances', 0):,}\n")
        self.stats_display.insert(tk.END, f"🧠 Эмбеддингов: {stats.get('embeddings', 0):,}\n")
        self.stats_display.insert(tk.END, f"📝 Фраз обратных звонков: {stats.get('callback_phrases', 0):,}\n")
        self.stats_display.insert(tk.END, f"❓ Пар вопрос-ответ: {stats.get('qa_pairs', 0):,}\n\n")
        
        # Статистика по темам
        if 'themes' in stats and stats['themes']:
            self.stats_display.insert(tk.END, "=== СТАТИСТИКА ПО ТЕМАМ ===\n\n")
            for theme, count in stats['themes'].items():
                self.stats_display.insert(tk.END, f"📁 {theme}: {count:,} диалогов\n")
            self.stats_display.insert(tk.END, "\n")
        
        # Общая информация
        self.stats_display.insert(tk.END, "=== ИНФОРМАЦИЯ О СИСТЕМЕ ===\n\n")
        self.stats_display.insert(tk.END, f"🕒 Время обновления: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.stats_display.insert(tk.END, f"💾 База данных: {config.DATABASE_PATH}\n")
        self.stats_display.insert(tk.END, f"🧠 Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}\n")
        self.stats_display.insert(tk.END, f"🤖 LLM модель: {config.LLM_MODEL_NAME}\n")

def main():
    """Главная функция."""
    root = tk.Tk()
    app = LightCallCenterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
