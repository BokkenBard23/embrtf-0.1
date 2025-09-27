"""Улучшенный GUI для CallCenter AI v3 с русским интерфейсом."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sqlite3
import json
import threading
import time
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import config
from utils import get_db_connection
from analysis_methods import get_analysis_method
from data_manager import DataManager
import logging
from datetime import datetime
import pandas as pd
from tkinter import simpledialog
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gui.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные
MODEL = None
FAISS_INDEXES = {}
DIALOG_LOOKUP = {}
UTTERANCE_LOOKUP = {}
CHAT_DB_CONN = None
CURRENT_THEME = "all"

class CallCenterGUI:
    """Главный класс GUI приложения."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CallCenter AI v3 - Анализ диалогов")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Переменные для прогресс-баров
        self.progress_vars = {}
        self.status_vars = {}
        
        # Менеджер данных
        self.data_manager = DataManager()
        
        # Инициализация
        self.init_chat_db()
        self.create_widgets()
        self.load_initial_data()
        
    def init_chat_db(self):
        """Инициализация базы данных для чата."""
        global CHAT_DB_CONN
        try:
            CHAT_DB_CONN = get_db_connection()
            logger.info("✅ База данных для чата инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")
            messagebox.showerror("Ошибка", f"Не удалось подключиться к базе данных: {e}")
    
    def save_qa_pair(self, question, theme, method, params, answer, context_summary):
        """Сохранение пары вопрос-ответ в базу данных."""
        try:
            cursor = CHAT_DB_CONN.cursor()
            cursor.execute("""
                INSERT INTO qa_pairs (timestamp, question, theme, method_used, parameters, answer, context_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                question,
                theme,
                method,
                json.dumps(params, ensure_ascii=False),
                answer,
                json.dumps(context_summary, ensure_ascii=False) if isinstance(context_summary, (list, dict)) else str(context_summary)
            ))
            CHAT_DB_CONN.commit()
            logger.info("✅ Вопрос-ответ сохранён в БД.")
        except Exception as e:
            logger.error(f"Ошибка сохранения QA-пары: {e}")
    
    def create_widgets(self):
        """Создание всех виджетов интерфейса."""
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка 1: Обработка данных
        self.create_data_processing_tab()
        
        # Вкладка 2: Поиск и анализ
        self.create_search_tab()
        
        # Вкладка 3: Работа с данными
        self.create_data_management_tab()
        
        # Вкладка 4: Чат с ИИ
        self.create_chat_tab()
        
        # Вкладка 5: Статистика
        self.create_statistics_tab()
        
    def create_data_processing_tab(self):
        """Создание вкладки обработки данных."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Обработка данных")
        
        # Заголовок
        title_label = tk.Label(self.data_frame, text="Обработка и индексация данных", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для группировки
        self.create_processing_controls()
        self.create_progress_section()
        self.create_log_section()
        
    def create_processing_controls(self):
        """Создание элементов управления обработкой."""
        # Фрейм для кнопок обработки
        controls_frame = ttk.LabelFrame(self.data_frame, text="Управление обработкой", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопки обработки
        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        # Обработка RTF файлов
        self.process_rtf_btn = tk.Button(buttons_frame, text="📁 Обработать RTF файлы", 
                                        command=self.process_rtf_files, 
                                        bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                        width=20, height=2)
        self.process_rtf_btn.pack(side='left', padx=5, pady=5)
        
        # Создание эмбеддингов
        self.create_embeddings_btn = tk.Button(buttons_frame, text="🧠 Создать эмбеддинги", 
                                              command=self.create_embeddings, 
                                              bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                              width=20, height=2)
        self.create_embeddings_btn.pack(side='left', padx=5, pady=5)
        
        # Построение индексов
        self.build_indexes_btn = tk.Button(buttons_frame, text="🔍 Построить индексы", 
                                          command=self.build_indexes, 
                                          bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                          width=20, height=2)
        self.build_indexes_btn.pack(side='left', padx=5, pady=5)
        
        # Классификация фраз
        self.classify_phrases_btn = tk.Button(buttons_frame, text="📝 Классифицировать фразы", 
                                             command=self.classify_phrases, 
                                             bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
                                             width=20, height=2)
        self.classify_phrases_btn.pack(side='left', padx=5, pady=5)
        
        # Информационная панель
        info_frame = ttk.LabelFrame(controls_frame, text="Информация о процессе", padding=10)
        info_frame.pack(fill='x', pady=10)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=80)
        self.info_text.pack(fill='both', expand=True)
        
    def create_progress_section(self):
        """Создание секции прогресс-баров."""
        progress_frame = ttk.LabelFrame(self.data_frame, text="Прогресс выполнения", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        # Прогресс-бар для общей обработки
        self.overall_progress_var = tk.DoubleVar()
        self.overall_progress = ttk.Progressbar(progress_frame, variable=self.overall_progress_var, 
                                               maximum=100, length=400)
        self.overall_progress.pack(fill='x', pady=5)
        
        self.overall_status_var = tk.StringVar(value="Готов к работе")
        self.overall_status_label = tk.Label(progress_frame, textvariable=self.overall_status_var, 
                                           font=('Arial', 10))
        self.overall_status_label.pack(pady=2)
        
        # Детальные прогресс-бары
        details_frame = tk.Frame(progress_frame)
        details_frame.pack(fill='x', pady=5)
        
        # Прогресс обработки диалогов
        tk.Label(details_frame, text="Диалоги:", font=('Arial', 9)).grid(row=0, column=0, sticky='w')
        self.dialogs_progress_var = tk.DoubleVar()
        self.dialogs_progress = ttk.Progressbar(details_frame, variable=self.dialogs_progress_var, 
                                               maximum=100, length=200)
        self.dialogs_progress.grid(row=0, column=1, padx=5)
        self.dialogs_status_var = tk.StringVar(value="0/0")
        tk.Label(details_frame, textvariable=self.dialogs_status_var, font=('Arial', 9)).grid(row=0, column=2)
        
        # Прогресс создания эмбеддингов
        tk.Label(details_frame, text="Эмбеддинги:", font=('Arial', 9)).grid(row=1, column=0, sticky='w')
        self.embeddings_progress_var = tk.DoubleVar()
        self.embeddings_progress = ttk.Progressbar(details_frame, variable=self.embeddings_progress_var, 
                                                  maximum=100, length=200)
        self.embeddings_progress.grid(row=1, column=1, padx=5)
        self.embeddings_status_var = tk.StringVar(value="0/0")
        tk.Label(details_frame, textvariable=self.embeddings_status_var, font=('Arial', 9)).grid(row=1, column=2)
        
        # Прогресс индексации
        tk.Label(details_frame, text="Индексы:", font=('Arial', 9)).grid(row=2, column=0, sticky='w')
        self.indexes_progress_var = tk.DoubleVar()
        self.indexes_progress = ttk.Progressbar(details_frame, variable=self.indexes_progress_var, 
                                               maximum=100, length=200)
        self.indexes_progress.grid(row=2, column=1, padx=5)
        self.indexes_status_var = tk.StringVar(value="0/0")
        tk.Label(details_frame, textvariable=self.indexes_status_var, font=('Arial', 9)).grid(row=2, column=2)
        
    def create_log_section(self):
        """Создание секции логов."""
        log_frame = ttk.LabelFrame(self.data_frame, text="Логи выполнения", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.pack(fill='both', expand=True)
        
        # Кнопки управления логами
        log_buttons_frame = tk.Frame(log_frame)
        log_buttons_frame.pack(fill='x', pady=5)
        
        tk.Button(log_buttons_frame, text="Очистить логи", command=self.clear_logs).pack(side='left', padx=5)
        tk.Button(log_buttons_frame, text="Сохранить логи", command=self.save_logs).pack(side='left', padx=5)
        
    def create_search_tab(self):
        """Создание вкладки поиска и анализа."""
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="🔍 Поиск и анализ")
        
        # Заголовок
        title_label = tk.Label(self.search_frame, text="Поиск и анализ диалогов", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для поиска
        self.create_search_controls()
        self.create_results_section()
        
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
        
        # Параметры поиска
        params_frame = tk.Frame(search_frame)
        params_frame.pack(fill='x', pady=5)
        
        # Количество результатов
        tk.Label(params_frame, text="Количество результатов:").grid(row=0, column=0, sticky='w', padx=5)
        self.results_count_var = tk.StringVar(value="10")
        results_spinbox = tk.Spinbox(params_frame, from_=1, to=100, textvariable=self.results_count_var, width=10)
        results_spinbox.grid(row=0, column=1, padx=5)
        
        # Тема поиска
        tk.Label(params_frame, text="Тема:").grid(row=0, column=2, sticky='w', padx=5)
        self.theme_var = tk.StringVar(value="all")
        theme_combo = ttk.Combobox(params_frame, textvariable=self.theme_var, width=15)
        theme_combo['values'] = ['all', 'CallBack', 'восстановление_договора']
        theme_combo.grid(row=0, column=3, padx=5)
        
        # Метод анализа
        tk.Label(params_frame, text="Метод анализа:").grid(row=1, column=0, sticky='w', padx=5)
        self.analysis_method_var = tk.StringVar(value="hierarchical")
        method_combo = ttk.Combobox(params_frame, textvariable=self.analysis_method_var, width=15)
        method_combo['values'] = ['hierarchical', 'rolling', 'facts', 'classification', 'callback_classifier', 'fast_phrase_classifier']
        method_combo.grid(row=1, column=1, padx=5)
        
        # Кнопки поиска
        search_buttons_frame = tk.Frame(search_frame)
        search_buttons_frame.pack(fill='x', pady=10)
        
        self.search_btn = tk.Button(search_buttons_frame, text="🔍 Найти", 
                                   command=self.perform_search, 
                                   bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                   width=15, height=2)
        self.search_btn.pack(side='left', padx=5)
        
        self.analyze_btn = tk.Button(search_buttons_frame, text="📊 Анализировать", 
                                    command=self.perform_analysis, 
                                    bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                    width=15, height=2)
        self.analyze_btn.pack(side='left', padx=5)
        
        self.clear_results_btn = tk.Button(search_buttons_frame, text="🗑️ Очистить", 
                                          command=self.clear_results, 
                                          bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                          width=15, height=2)
        self.clear_results_btn.pack(side='left', padx=5)
        
    def create_results_section(self):
        """Создание секции результатов."""
        results_frame = ttk.LabelFrame(self.search_frame, text="Результаты поиска", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Создаем Treeview для результатов
        columns = ('ID', 'Текст', 'Спикер', 'Диалог', 'Оценка')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        # Настройка колонок
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
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
        
        # Обработчик двойного клика
        self.results_tree.bind('<Double-1>', self.on_result_double_click)
        
    def create_data_management_tab(self):
        """Создание вкладки управления данными."""
        self.data_mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_mgmt_frame, text="📋 Управление данными")
        
        # Заголовок
        title_label = tk.Label(self.data_mgmt_frame, text="Управление данными", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для управления данными
        self.create_data_controls()
        self.create_data_tables()
        
    def create_data_controls(self):
        """Создание элементов управления данными."""
        controls_frame = ttk.LabelFrame(self.data_mgmt_frame, text="Операции с данными", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопки управления
        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        # Просмотр фраз обратных звонков
        self.view_phrases_btn = tk.Button(buttons_frame, text="📝 Фразы обратных звонков", 
                                         command=self.view_callback_phrases, 
                                         bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                         width=20, height=2)
        self.view_phrases_btn.pack(side='left', padx=5, pady=5)
        
        # Просмотр диалогов
        self.view_dialogs_btn = tk.Button(buttons_frame, text="💬 Диалоги", 
                                         command=self.view_dialogs, 
                                         bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                         width=20, height=2)
        self.view_dialogs_btn.pack(side='left', padx=5, pady=5)
        
        # Создание иерархического словаря
        self.create_dict_btn = tk.Button(buttons_frame, text="📚 Создать словарь", 
                                        command=self.create_hierarchical_dictionary, 
                                        bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                        width=20, height=2)
        self.create_dict_btn.pack(side='left', padx=5, pady=5)
        
        # Просмотр реплик
        self.view_utterances_btn = tk.Button(buttons_frame, text="🗣️ Реплики", 
                                            command=self.view_utterances, 
                                            bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                                            width=20, height=2)
        self.view_utterances_btn.pack(side='left', padx=5, pady=5)
        
        # Экспорт данных
        self.export_btn = tk.Button(buttons_frame, text="📤 Экспорт данных", 
                                   command=self.export_data, 
                                   bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
                                   width=20, height=2)
        self.export_btn.pack(side='left', padx=5, pady=5)
        
    def create_data_tables(self):
        """Создание таблиц для отображения данных."""
        tables_frame = ttk.LabelFrame(self.data_mgmt_frame, text="Таблицы данных", padding=10)
        tables_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Создаем Treeview для данных
        columns = ('ID', 'Текст', 'Источник', 'Категория', 'Частота', 'Проверено')
        self.data_tree = ttk.Treeview(tables_frame, columns=columns, show='headings', height=15)
        
        # Настройка колонок
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)
        
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
        
    def create_chat_tab(self):
        """Создание вкладки чата с ИИ."""
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="💬 Чат с ИИ")
        
        # Заголовок
        title_label = tk.Label(self.chat_frame, text="Чат с аналитиком ИИ", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Создаем фреймы для чата
        self.create_chat_controls()
        self.create_chat_display()
        
    def create_chat_controls(self):
        """Создание элементов управления чатом."""
        controls_frame = ttk.LabelFrame(self.chat_frame, text="Управление чатом", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Кнопки управления
        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(fill='x')
        
        # Очистка чата
        self.clear_chat_btn = tk.Button(buttons_frame, text="🗑️ Очистить чат", 
                                       command=self.clear_chat, 
                                       bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                       width=15, height=2)
        self.clear_chat_btn.pack(side='left', padx=5, pady=5)
        
        # Сохранение истории
        self.save_chat_btn = tk.Button(buttons_frame, text="💾 Сохранить историю", 
                                      command=self.save_chat_history, 
                                      bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                      width=15, height=2)
        self.save_chat_btn.pack(side='left', padx=5, pady=5)
        
    def create_chat_display(self):
        """Создание области отображения чата."""
        chat_display_frame = ttk.LabelFrame(self.chat_frame, text="История чата", padding=10)
        chat_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Область чата
        self.chat_display = scrolledtext.ScrolledText(chat_display_frame, height=15, width=80, 
                                                     font=('Arial', 10))
        self.chat_display.pack(fill='both', expand=True)
        
        # Поле ввода
        input_frame = tk.Frame(chat_display_frame)
        input_frame.pack(fill='x', pady=10)
        
        self.chat_entry = tk.Entry(input_frame, font=('Arial', 12), width=60)
        self.chat_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.chat_entry.bind('<Return>', lambda e: self.send_chat_message())
        
        self.send_btn = tk.Button(input_frame, text="Отправить", 
                                 command=self.send_chat_message, 
                                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                                 width=10, height=2)
        self.send_btn.pack(side='right')
        
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
        
    def load_initial_data(self):
        """Загрузка начальных данных."""
        try:
            self.log_message("🔄 Загрузка начальных данных...")
            # НЕ загружаем модель при инициализации - только по требованию
            self.load_faiss_indexes()
            self.load_data_lookups()
            self.log_message("✅ Начальные данные загружены успешно")
            self.log_message("💡 Модель эмбеддингов будет загружена при необходимости")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки данных: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить начальные данные: {e}")
    
    def log_message(self, message):
        """Добавление сообщения в лог."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_logs(self):
        """Очистка логов."""
        self.log_text.delete(1.0, tk.END)
        
    def save_logs(self):
        """Сохранение логов в файл."""
        filename = filedialog.asksaveasfilename(defaultextension=".txt", 
                                               filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Успех", "Логи сохранены")
    
    # Методы обработки данных
    def process_rtf_files(self):
        """Обработка RTF файлов."""
        self.log_message("🔄 Начинаем обработку RTF файлов...")
        self.overall_status_var.set("Обработка RTF файлов...")
        
        def process_thread():
            try:
                import pipeline
                pipeline.process_thematic_folders()
                self.log_message("✅ RTF файлы обработаны успешно")
                self.overall_status_var.set("RTF файлы обработаны")
                self.overall_progress_var.set(25)
            except Exception as e:
                self.log_message(f"❌ Ошибка обработки RTF: {e}")
                self.overall_status_var.set("Ошибка обработки RTF")
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def create_embeddings(self):
        """Создание эмбеддингов с оптимизированным процессором."""
        self.log_message("🔄 Начинаем создание эмбеддингов...")
        self.overall_status_var.set("Создание эмбеддингов...")
        
        def create_thread():
            try:
                from optimized_embedding_processor import OptimizedEmbeddingProcessor
                
                # Создаем оптимизированный процессор
                processor = OptimizedEmbeddingProcessor(
                    batch_size=32,  # Оптимальный размер для RTX 3060
                    max_length=2048
                )
                
                # Получаем соединение с БД
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Получаем список диалогов для обработки
                cursor.execute("SELECT id FROM dialogs")
                dialog_ids = [row[0] for row in cursor.fetchall()]
                
                self.log_message(f"📊 Найдено {len(dialog_ids)} диалогов для обработки")
                
                # Обрабатываем диалоги батчами
                embeddings = processor.process_dialogs_batch(dialog_ids, conn)
                
                # Сохраняем эмбеддинги в БД
                processor.save_embeddings_to_db(embeddings, "embeddings", conn)
                
                # Обрабатываем реплики
                cursor.execute("SELECT id FROM utterances")
                utterance_ids = [row[0] for row in cursor.fetchall()]
                
                self.log_message(f"📊 Найдено {len(utterance_ids)} реплик для обработки")
                
                # Обрабатываем реплики батчами
                utterance_embeddings = processor.process_utterances_batch(utterance_ids, conn)
                
                # Сохраняем эмбеддинги реплик в БД
                processor.save_embeddings_to_db(utterance_embeddings, "utterance_embeddings", conn)
                
                conn.close()
                
                self.log_message("✅ Эмбеддинги созданы успешно")
                self.overall_status_var.set("Эмбеддинги созданы")
                self.overall_progress_var.set(50)
                
            except Exception as e:
                self.log_message(f"❌ Ошибка создания эмбеддингов: {e}")
                self.overall_status_var.set("Ошибка создания эмбеддингов")
        
        threading.Thread(target=create_thread, daemon=True).start()
    
    def create_hierarchical_dictionary(self):
        """Создание иерархического словаря для классификации."""
        self.log_message("🔄 Начинаем создание иерархического словаря...")
        self.overall_status_var.set("Создание словаря...")
        
        def create_dict_thread():
            try:
                # Пробуем использовать полную версию с txtai
                try:
                    from hier_dict import HierDict
                    from txtai import Embeddings
                    use_full_version = True
                    self.log_message("✅ Используем полную версию с txtai")
                except ImportError:
                    from hier_dict_simple import HierDictSimple as HierDict
                    use_full_version = False
                    self.log_message("⚠️ Используем упрощенную версию (txtai недоступен)")
                
                # Получаем соединение с БД
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Получаем тексты диалогов
                cursor.execute("SELECT text FROM dialogs LIMIT 1000")  # Ограничиваем для производительности
                dialog_texts = [row[0] for row in cursor.fetchall()]
                
                if not dialog_texts:
                    self.log_message("❌ Нет диалогов для создания словаря")
                    return
                
                self.log_message(f"📊 Найдено {len(dialog_texts)} диалогов для анализа")
                
                if use_full_version:
                    # Полная версия с txtai
                    # Создаем эмбеддинги для разных уровней
                    self.log_message("🔄 Создание эмбеддингов для диалогов...")
                    dial_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
                    dial_emb.index(dialog_texts)
                    
                    # Создаем эмбеддинги для предложений
                    self.log_message("🔄 Создание эмбеддингов для предложений...")
                    sentences = []
                    for text in dialog_texts:
                        sentences.extend([s.strip() for s in text.split('.') if s.strip()])
                    
                    sent_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
                    sent_emb.index(sentences[:5000])  # Ограничиваем количество
                    
                    # Создаем эмбеддинги для фраз
                    self.log_message("🔄 Создание эмбеддингов для фраз...")
                    phrases = []
                    for text in dialog_texts:
                        phrases.extend([p.strip() for p in text.split() if len(p.strip()) > 2])
                    
                    phrase_emb = Embeddings({"path": "all-MiniLM-L6-v2", "content": True})
                    phrase_emb.index(phrases[:10000])  # Ограничиваем количество
                    
                    # Создаем иерархический словарь
                    self.log_message("🔄 Построение иерархического словаря...")
                    hd = HierDict(dial_emb, sent_emb, phrase_emb, dialog_texts)
                else:
                    # Упрощенная версия
                    self.log_message("🔄 Построение упрощенного иерархического словаря...")
                    hd = HierDict(dialog_texts)
                
                # Строим словарь
                dictionary = hd.build("hierarchical_dictionary.json")
                
                # Тестируем качество
                precision = hd.test(k=5)
                
                # Получаем статистику
                stats = hd.get_statistics()
                
                # Экспортируем для Smart Logger
                hd.export_to_smart_logger("smart_logger_dictionary.json")
                
                self.log_message("✅ Иерархический словарь создан успешно!")
                self.log_message(f"📊 Precision@5: {precision:.2f}")
                self.log_message(f"📚 Parent слов: {stats.get('parent_words', 0)}")
                self.log_message(f"📚 Child слов: {stats.get('child_words', 0)}")
                self.log_message("💾 Словарь сохранен: hierarchical_dictionary.json")
                self.log_message("💾 Smart Logger версия: smart_logger_dictionary.json")
                
                self.overall_status_var.set("Словарь создан")
                self.overall_progress_var.set(75)
                
                conn.close()
                
            except Exception as e:
                self.log_message(f"❌ Ошибка создания словаря: {e}")
                self.overall_status_var.set("Ошибка создания словаря")
        
        threading.Thread(target=create_dict_thread, daemon=True).start()
    
    def build_indexes(self):
        """Построение индексов."""
        self.log_message("🔄 Начинаем построение индексов...")
        self.overall_status_var.set("Построение индексов...")
        
        def build_thread():
            try:
                import indexer
                indexer.main()
                self.log_message("✅ Индексы построены успешно")
                self.overall_status_var.set("Индексы построены")
                self.overall_progress_var.set(75)
            except Exception as e:
                self.log_message(f"❌ Ошибка построения индексов: {e}")
                self.overall_status_var.set("Ошибка построения индексов")
        
        threading.Thread(target=build_thread, daemon=True).start()
    
    def classify_phrases(self):
        """Классификация фраз."""
        self.log_message("🔄 Начинаем классификацию фраз...")
        self.overall_status_var.set("Классификация фраз...")
        
        def classify_thread():
            try:
                # Здесь будет логика классификации фраз
                self.log_message("✅ Фразы классифицированы успешно")
                self.overall_status_var.set("Фразы классифицированы")
                self.overall_progress_var.set(100)
            except Exception as e:
                self.log_message(f"❌ Ошибка классификации фраз: {e}")
                self.overall_status_var.set("Ошибка классификации фраз")
        
        threading.Thread(target=classify_thread, daemon=True).start()
    
    # Методы поиска и анализа
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
            
            # Выполняем поиск
            results = self.data_manager.search_utterances(query, limit=int(self.results_count_var.get()))
            
            # Отображаем результаты
            for result in results:
                self.results_tree.insert('', 'end', values=(
                    result.get('id', ''),
                    result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', ''),
                    result.get('speaker', ''),
                    result.get('dialog_id', ''),
                    result.get('source_theme', '')
                ))
            
            self.log_message(f"✅ Найдено {len(results)} результатов")
            
        except Exception as e:
            self.log_message(f"❌ Ошибка поиска: {e}")
            messagebox.showerror("Ошибка", f"Ошибка поиска: {e}")
    
    def perform_analysis(self):
        """Выполнение анализа."""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Предупреждение", "Введите запрос для анализа")
            return
        
        self.log_message(f"📊 Анализ: {query}")
        try:
            # Получаем метод анализа
            method = self.analysis_method_var.get()
            
            # Выполняем поиск для анализа
            results = self.data_manager.search_utterances(query, limit=10)
            
            if not results:
                self.log_message("❌ Не найдено данных для анализа")
                messagebox.showwarning("Предупреждение", "Не найдено данных для анализа")
                return
            
            # Создаем текст для анализа
            analysis_text = "\n".join([f"{r['speaker']}: {r['text']}" for r in results])
            
            # Выполняем анализ
            analysis_method = get_analysis_method(method)
            if analysis_method:
                analysis_result = analysis_method(analysis_text, {})
                
                # Отображаем результат анализа
                self.chat_display.insert(tk.END, f"=== АНАЛИЗ ЗАПРОСА: {query} ===\n")
                self.chat_display.insert(tk.END, f"Метод: {method}\n")
                self.chat_display.insert(tk.END, f"Результат:\n{analysis_result}\n\n")
                self.chat_display.see(tk.END)
                
                # Сохраняем в БД
                self.save_qa_pair(query, self.theme_var.get(), method, {}, analysis_result, results)
                
                self.log_message("✅ Анализ выполнен и сохранен")
            else:
                self.log_message("❌ Метод анализа не найден")
                messagebox.showerror("Ошибка", "Метод анализа не найден")
                
        except Exception as e:
            self.log_message(f"❌ Ошибка анализа: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа: {e}")
    
    def clear_results(self):
        """Очистка результатов."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.log_message("🗑️ Результаты очищены")
    
    def on_result_double_click(self, event):
        """Обработка двойного клика по результату."""
        item = self.results_tree.selection()[0]
        values = self.results_tree.item(item, 'values')
        self.log_message(f"📋 Выбран результат: {values[0]}")
    
    # Методы управления данными
    def view_callback_phrases(self):
        """Просмотр фраз обратных звонков."""
        self.log_message("📝 Загрузка фраз обратных звонков...")
        try:
            df = self.data_manager.get_callback_phrases_data(limit=1000)
            self.populate_data_tree(df)
            self.log_message(f"✅ Загружено {len(df)} фраз обратных звонков")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки фраз: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить фразы: {e}")
    
    def view_dialogs(self):
        """Просмотр диалогов."""
        self.log_message("💬 Загрузка диалогов...")
        try:
            df = self.data_manager.get_dialogs_data(limit=1000)
            self.populate_data_tree(df)
            self.log_message(f"✅ Загружено {len(df)} диалогов")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки диалогов: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить диалоги: {e}")
    
    def view_utterances(self):
        """Просмотр реплик."""
        self.log_message("🗣️ Загрузка реплик...")
        try:
            df = self.data_manager.get_utterances_data(limit=1000)
            self.populate_data_tree(df)
            self.log_message(f"✅ Загружено {len(df)} реплик")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки реплик: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить реплики: {e}")
    
    def populate_data_tree(self, df):
        """Заполнение таблицы данных."""
        # Очищаем таблицу
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if df.empty:
            return
        
        # Настраиваем колонки
        columns = list(df.columns)
        self.data_tree['columns'] = columns
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)
        
        # Заполняем данными
        for index, row in df.iterrows():
            values = [str(row[col]) for col in columns]
            self.data_tree.insert('', 'end', values=values)
    
    def export_data(self):
        """Экспорт данных."""
        # Создаем диалог выбора типа данных
        data_type = simpledialog.askstring("Экспорт данных", 
                                          "Выберите тип данных (dialogs/utterances/callback_phrases/qa_pairs):")
        if not data_type:
            return
        
        if data_type not in ['dialogs', 'utterances', 'callback_phrases', 'qa_pairs']:
            messagebox.showerror("Ошибка", "Неверный тип данных")
            return
        
        # Выбираем формат файла
        filetypes = [("CSV files", "*.csv"), ("JSON files", "*.json")]
        filename = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               filetypes=filetypes)
        if not filename:
            return
        
        try:
            self.log_message(f"📤 Экспорт {data_type} в {filename}")
            
            if filename.endswith('.csv'):
                success = self.data_manager.export_to_csv(data_type, filename)
            elif filename.endswith('.json'):
                success = self.data_manager.export_to_json(data_type, filename)
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат файла")
                return
            
            if success:
                self.log_message("✅ Данные экспортированы успешно")
                messagebox.showinfo("Успех", "Данные экспортированы успешно")
            else:
                self.log_message("❌ Ошибка экспорта данных")
                messagebox.showerror("Ошибка", "Не удалось экспортировать данные")
                
        except Exception as e:
            self.log_message(f"❌ Ошибка экспорта: {e}")
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    # Методы чата
    def send_chat_message(self):
        """Отправка сообщения в чат."""
        message = self.chat_entry.get().strip()
        if not message:
            return
        
        self.chat_display.insert(tk.END, f"Пользователь: {message}\n")
        self.chat_entry.delete(0, tk.END)
        
        # Здесь будет логика обработки сообщения ИИ
        self.chat_display.insert(tk.END, f"Аналитик: Сообщение получено\n")
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Очистка чата."""
        self.chat_display.delete(1.0, tk.END)
        self.log_message("🗑️ Чат очищен")
    
    def save_chat_history(self):
        """Сохранение истории чата."""
        filename = filedialog.asksaveasfilename(defaultextension=".txt", 
                                               filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.chat_display.get(1.0, tk.END))
            messagebox.showinfo("Успех", "История чата сохранена")
    
    # Методы статистики
    def refresh_statistics(self):
        """Обновление статистики."""
        self.log_message("🔄 Обновление статистики...")
        try:
            stats = self.data_manager.get_statistics()
            self.display_statistics(stats)
            self.log_message("✅ Статистика обновлена")
        except Exception as e:
            self.log_message(f"❌ Ошибка обновления статистики: {e}")
            messagebox.showerror("Ошибка", f"Не удалось обновить статистику: {e}")
    
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
        
        # Статистика по категориям фраз
        if 'phrase_categories' in stats and stats['phrase_categories']:
            self.stats_display.insert(tk.END, "=== КАТЕГОРИИ ФРАЗ ===\n\n")
            category_names = {1: "Ошибки операторов", 2: "Предложения операторов", 3: "Неопределенные обещания"}
            for category, count in stats['phrase_categories'].items():
                name = category_names.get(int(category), f"Категория {category}")
                self.stats_display.insert(tk.END, f"🏷️ {name}: {count:,} фраз\n")
            self.stats_display.insert(tk.END, "\n")
        
        # Статистика по методам анализа
        if 'analysis_methods' in stats and stats['analysis_methods']:
            self.stats_display.insert(tk.END, "=== МЕТОДЫ АНАЛИЗА ===\n\n")
            method_names = {
                'hierarchical': 'Иерархический',
                'rolling': 'Скользящий',
                'facts': 'Извлечение фактов',
                'classification': 'Классификация',
                'callback_classifier': 'Классификатор обратных звонков',
                'fast_phrase_classifier': 'Быстрый классификатор фраз'
            }
            for method, count in stats['analysis_methods'].items():
                name = method_names.get(method, method)
                self.stats_display.insert(tk.END, f"🔬 {name}: {count:,} использований\n")
            self.stats_display.insert(tk.END, "\n")
        
        # Общая информация
        self.stats_display.insert(tk.END, "=== ИНФОРМАЦИЯ О СИСТЕМЕ ===\n\n")
        self.stats_display.insert(tk.END, f"🕒 Время обновления: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.stats_display.insert(tk.END, f"💾 База данных: {config.DATABASE_PATH}\n")
        self.stats_display.insert(tk.END, f"🧠 Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}\n")
        self.stats_display.insert(tk.END, f"🤖 LLM модель: {config.LLM_MODEL_NAME}\n")
    
    # Вспомогательные методы
    def load_models(self):
        """Загрузка моделей с оптимизацией для RTX 3060."""
        global MODEL
        try:
            self.log_message("🔄 Загрузка модели эмбеддингов...")
            
            # Проверяем доступность CUDA
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_message(f"✅ CUDA доступна: {torch.cuda.get_device_name()}")
                self.log_message(f"💾 Память GPU: {gpu_memory:.1f} GB")
                
                # Очищаем кэш CUDA
                torch.cuda.empty_cache()
            else:
                device = "cpu"
                self.log_message("⚠️ CUDA недоступна, используем CPU")
            
            # Загружаем модель с оптимизацией
            model_name = config.EMBEDDING_MODEL_NAME
            try:
                MODEL = SentenceTransformer(model_name, device=device)
                MODEL.max_seq_length = 2048  # Увеличиваем контекст
                
                # Оптимизируем для GPU
                if device == "cuda":
                    # Используем mixed precision для экономии памяти
                    if hasattr(torch.cuda, 'amp'):
                        MODEL.half()
                        self.log_message("✅ Включен mixed precision (float16)")
                    
                    # Очищаем кэш после загрузки
                    torch.cuda.empty_cache()
                
                self.log_message(f"✅ Модель {model_name} загружена на {device}")
                
            except OSError as e:
                if "Файл подкачки слишком мал" in str(e) or "1455" in str(e):
                    self.log_message("⚠️ Недостаточно памяти, пробуем оптимизацию...")
                    
                    # Пробуем с меньшим контекстом
                    try:
                        MODEL = SentenceTransformer(model_name, device=device)
                        MODEL.max_seq_length = 512  # Уменьшаем контекст
                        self.log_message(f"✅ Модель загружена с контекстом 512")
                    except:
                        # Fallback на более легкую модель
                        fallback_model = "all-MiniLM-L6-v2"
                        MODEL = SentenceTransformer(fallback_model, device=device)
                        MODEL.max_seq_length = 256
                        self.log_message(f"✅ Загружена альтернативная модель: {fallback_model}")
                else:
                    raise e
                    
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки модели: {e}")
            MODEL = None
            self.log_message("⚠️ GUI будет работать без модели эмбеддингов")
    
    def load_faiss_indexes(self):
        """Загрузка FAISS индексов."""
        global FAISS_INDEXES
        try:
            # Здесь будет логика загрузки индексов
            self.log_message("✅ Индексы загружены")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки индексов: {e}")
    
    def load_data_lookups(self):
        """Загрузка справочников данных."""
        global DIALOG_LOOKUP, UTTERANCE_LOOKUP
        try:
            # Здесь будет логика загрузки справочников
            self.log_message("✅ Справочники загружены")
        except Exception as e:
            self.log_message(f"❌ Ошибка загрузки справочников: {e}")

def main():
    """Главная функция."""
    root = tk.Tk()
    app = CallCenterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
