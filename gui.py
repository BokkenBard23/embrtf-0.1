"""GUI для анализа диалогов call-центра — обновлённая версия с поддержкой поиска по репликам, HyDE, Reranker, сохранением QA и чатом."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import numpy as np
import faiss
import sqlite3
import logging
import threading
import requests
import pickle
from datetime import datetime
from pathlib import Path

# === Импорт конфигурации и утилит ===
import config
from utils import setup_logger, get_db_connection

# === Импорт рабочих модулей ===
import pipeline
import indexer
import init_db

# === Логирование ===
logger = setup_logger('GUI', config.LOGS_ROOT / "gui.log")

# === Импорт моделей ===
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
    reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
except Exception as e:
    logger.warning(f"Reranker не доступен: {e}")
    RERANKER_AVAILABLE = False
    reranker = None

# === Глобальные переменные ===
MODEL = None
INDEXES = {}        # {theme: faiss_index}
IDS = {}            # {theme: [utterance_id1, ...]}
DATA_LOOKUPS = {}   # {utterance_id: {text, speaker, dialog_id, turn_order, full_dialog_text}}
CHAT_DB_CONN = None
CURRENT_THEME = "all"

# === Инициализация БД для чата и QA ===
def init_chat_db():
    global CHAT_DB_CONN
    try:
        CHAT_DB_CONN = get_db_connection(check_same_thread=False)
        logger.info("✅ База данных для чата и QA подключена.")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД чата: {e}")
        messagebox.showerror("Ошибка БД", f"Не удалось подключиться к базе данных чата: {e}")

# === Загрузка моделей и данных ===
def load_models():
    global MODEL
    logger.info(f"Загрузка embedding-модели: {config.EMBEDDING_MODEL_NAME}")
    
    # Определяем устройство с fallback на CPU
    device = "cpu"  # По умолчанию CPU для стабильности
    if config.EMBEDDING_MODEL_DEVICE == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("✅ Используем CUDA")
            else:
                logger.warning("⚠️ CUDA недоступна, используем CPU")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка проверки CUDA: {e}, используем CPU")
    
    try:
        MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        MODEL.max_seq_length = config.EMBEDDING_MODEL_MAX_LENGTH
        logger.info(f"✅ Модель загружена на устройстве: {device}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        # Fallback на CPU
        try:
            MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
            MODEL.max_seq_length = config.EMBEDDING_MODEL_MAX_LENGTH
            logger.info("✅ Модель загружена на CPU (fallback)")
        except Exception as e2:
            logger.error(f"❌ Критическая ошибка загрузки модели: {e2}")
            raise e2

def load_faiss_indexes():
    global INDEXES, IDS
    logger.info("Загрузка FAISS-индексов...")
    INDEXES = {}
    IDS = {}
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT theme, index_path, ids_path FROM faiss_indexes")
    rows = cursor.fetchall()
    conn.close()
    
    for theme, index_path, ids_path in rows:
        if Path(index_path).exists() and Path(ids_path).exists():
            try:
                INDEXES[theme] = faiss.read_index(index_path)
                with open(ids_path, 'r', encoding='utf-8') as f:
                    IDS[theme] = json.load(f)
                logger.info(f"✅ Загружен индекс: {theme} ({len(IDS[theme])} реплик)")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки индекса {theme}: {e}")

def load_data_lookups():
    global DATA_LOOKUPS
    logger.info("Загрузка данных реплик и диалогов...")
    DATA_LOOKUPS = {}
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, dialog_id, speaker, text, turn_order FROM utterances")
    utterances = cursor.fetchall()
    
    cursor.execute("SELECT id, text FROM dialogs")
    dialog_texts = {row[0]: row[1] for row in cursor.fetchall()}
    
    for utterance_id, dialog_id, speaker, text, turn_order in utterances:
        DATA_LOOKUPS[utterance_id] = {
            "text": text,
            "speaker": speaker,
            "dialog_id": dialog_id,
            "turn_order": turn_order,
            "full_dialog_text": dialog_texts.get(dialog_id, "")
        }
    
    conn.close()
    logger.info(f"✅ Загружено {len(DATA_LOOKUPS)} реплик.")


# === Утилиты UI ===
def set_ui_busy(is_busy: bool):
    ask_btn.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    btn_send.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    export_answer_btn.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    export_context_btn.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    btn_run_pipeline.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    btn_run_indexer.config(state=tk.DISABLED if is_busy else tk.NORMAL)
    btn_reload_indexes.config(state=tk.DISABLED if is_busy else tk.NORMAL)


# === Фоновые операции: pipeline / indexer / reload ===
def run_pipeline_background():
    def worker():
        try:
            set_ui_busy(True)
            status_label.config(text="🚀 Обработка новых диалогов (pipeline)...")
            root.update_idletasks()
            init_db.init_db()
            pipeline.process_thematic_folders()
            status_label.config(text="✅ Обработка завершена. Запустите индексацию.")
        except Exception as e:
            logger.error(f"Ошибка pipeline: {e}")
            messagebox.showerror("Pipeline", f"Ошибка обработки: {e}")
        finally:
            set_ui_busy(False)
    threading.Thread(target=worker, daemon=True).start()


def run_indexer_background():
    def worker():
        try:
            set_ui_busy(True)
            status_label.config(text="🔧 Построение FAISS индексов...")
            root.update_idletasks()
            init_db.init_db()
            indexer.main()
            status_label.config(text="✅ Индексация завершена. Перезагружаю индексы...")
            load_faiss_indexes()
            load_data_lookups()
            themes_for_combo = ["all"] + [t for t in INDEXES.keys() if t != "all"]
            theme_menu['values'] = themes_for_combo
            if themes_for_combo:
                theme_var.set(themes_for_combo[0])
            status_label.config(text="✅ Индексы обновлены.")
        except Exception as e:
            logger.error(f"Ошибка индексации: {e}")
            messagebox.showerror("Индексация", f"Ошибка индексации: {e}")
        finally:
            set_ui_busy(False)
    threading.Thread(target=worker, daemon=True).start()


def reload_indexes_and_data():
    try:
        set_ui_busy(True)
        status_label.config(text="🔄 Перезагрузка индексов и данных...")
        root.update_idletasks()
        load_faiss_indexes()
        load_data_lookups()
        themes_for_combo = ["all"] + [t for t in INDEXES.keys() if t != "all"]
        theme_menu['values'] = themes_for_combo
        if themes_for_combo:
            theme_var.set(themes_for_combo[0])
        status_label.config(text="✅ Перезагрузка завершена.")
    except Exception as e:
        logger.error(f"Ошибка перезагрузки индексов/данных: {e}")
        messagebox.showerror("Перезагрузка", f"Ошибка перезагрузки: {e}")
    finally:
        set_ui_busy(False)

# === HyDE ===
def generate_hypothetical_answer(query):
    try:
        hyde_prompt = f"""Ты — эксперт по анализу диалогов call-центра.
Напиши подробный, точный и естественный ответ на вопрос пользователя, как будто ты уже нашёл нужную информацию в базе диалогов.

Вопрос: {query}

Гипотетический ответ:"""
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": config.LLM_MODEL_NAME,
                "prompt": hyde_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ошибка HyDE: {e}")
    return query

# === Reranker ===
def rerank_results(query, candidates):
    if not RERANKER_AVAILABLE or not candidates:
        return candidates
    try:
        pairs = [(query, cand["text"]) for cand in candidates]
        scores = reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info("✅ Результаты переранжированы.")
    except Exception as e:
        logger.warning(f"Ошибка reranker: {e}")
    return candidates

# === Поиск по репликам ===
def find_similar_utterances(query, theme="all", top_k=5):
    if theme not in INDEXES:
        logger.error(f"Индекс для темы '{theme}' не загружен.")
        return []
    
    # ✅ Просто кодируем запрос — без HyDE
    query_vector = MODEL.encode([query], convert_to_tensor=False)[0].astype('float32')
    
    index = INDEXES[theme]
    ids_list = IDS[theme]
    distances, indices = index.search(np.array([query_vector]), top_k)
    
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(ids_list): continue
        utterance_id = ids_list[idx]
        if utterance_id not in DATA_LOOKUPS: continue
        item = DATA_LOOKUPS[utterance_id].copy()
        item["id"] = utterance_id
        item["faiss_score"] = float(distances[0][i])
        candidates.append(item)
    
    # ✅ Без reranker
    return candidates

# === Форматирование контекста с соседними репликами ===
def format_context_for_llm(results):
    context_parts = []
    for item in results:
        dialog_id = item["dialog_id"]
        turn_order = item["turn_order"]
        full_text = item["full_dialog_text"]
        
        surrounding_lines = []
        dialog_lines = full_text.splitlines()
        target_line = f"{item['speaker']}: {item['text']}"
        
        for i, line in enumerate(dialog_lines):
            if line == target_line or (line.startswith(item['speaker'] + ":") and item['text'] in line):
                start = max(0, i - 2)
                end = min(len(dialog_lines), i + 3)
                surrounding_lines = dialog_lines[start:end]
                break
        
        context_parts.append(
            f"[Схожесть: {item.get('rerank_score', item['faiss_score']):.3f}] ID: {item['id']}\n" +
            "\n".join(surrounding_lines) +
            "\n---"
        )
    return "\n\n".join(context_parts)

# === Сохранение QA-пары ===
def save_qa_pair(question, theme, method, params, answer, context_summary):
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

# === Анализ через analysis_methods.py ===
def run_ask():
    question = entry.get().strip()
    if not question:
        messagebox.showwarning("Предупреждение", "Введите вопрос!")
        return

    theme = theme_var.get()
    top_k = int(top_k_var.get())
    chunk_size = int(chunk_size_var.get())
    selected_method_name = method_var.get()

    ask_btn.config(state=tk.DISABLED, text="Ищу...")
    status_label.config(text=f"🔍 Поиск релевантных реплик (тема: {theme})...")
    root.update_idletasks()

    def update_status(text):
        status_label.config(text=text)
        root.update_idletasks()

    def worker():
        try:
            results = find_similar_utterances(question, theme=theme, top_k=top_k)
            if not results:
                answer_text = "Извините, не удалось найти релевантные фрагменты."
                context_text = "Нет найденных реплик."
            else:
                import analysis_methods
                analysis_func = analysis_methods.get_analysis_method(selected_method_name)
                if not analysis_func:
                    raise ValueError(f"Неизвестный метод: {selected_method_name}")

                context_text = format_context_for_llm(results)
                answer_text = analysis_func(
                    question=question,
                    found_with_scores=[(r, r.get("rerank_score", r["faiss_score"])) for r in results],
                    chunk_size=chunk_size,
                    status_callback=update_status
                )
                
                save_qa_pair(
                    question=question,
                    theme=theme,
                    method=selected_method_name,
                    params={"top_k": top_k, "chunk_size": chunk_size},
                    answer=answer_text,
                    context_summary=[r["id"] for r in results]
                )

            text_answer.delete(1.0, tk.END)
            text_answer.insert(tk.END, answer_text)
            text_context.delete(1.0, tk.END)
            text_context.insert(tk.END, context_text)
            status_label.config(text=f"✅ Готов. Тема: {theme}. Найдено {len(results)} реплик.")
        except Exception as e:
            error_msg = f"Ошибка: {e}"
            logger.error(error_msg)
            text_answer.delete(1.0, tk.END)
            text_answer.insert(tk.END, error_msg)
            status_label.config(text=f"❌ {error_msg}")
        finally:
            ask_btn.config(state=tk.NORMAL, text="Спросить")

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

# === Экспорт ===
def export_text(widget, default_name):
    content = widget.get(1.0, tk.END)
    if not content.strip():
        messagebox.showwarning("Экспорт", "Нет данных для экспорта.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        messagebox.showinfo("Экспорт", f"Данные экспортированы:\n{file_path}")

def export_answer():
    export_text(text_answer, f"Ответ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def export_context():
    export_text(text_context, f"Контекст_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# === Чат с уточнением ===
def on_send_click():
    user_message = entry_chat.get().strip()
    if not user_message: return
    chat_history.insert(tk.END, f"Вы: {user_message}\n")
    chat_history.see(tk.END)
    entry_chat.delete(0, tk.END)
    threading.Thread(target=process_user_message, args=(user_message,), daemon=True).start()

def process_user_message(user_message):
    try:
        if user_message.lower().startswith(("найти", "поиск", "найди", "search")):
            prompt = f"Задай 1-2 уточняющих вопроса по запросу: '{user_message}'"
        else:
            prompt = f"Ответь как аналитик call-центра: {user_message}"
        
        # Проверяем доступность Ollama
        try:
            test_response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if test_response.status_code != 200:
                chat_history.insert(tk.END, f"Аналитик: Ollama недоступен (код {test_response.status_code})\n")
                return
        except requests.exceptions.RequestException:
            chat_history.insert(tk.END, f"Аналитик: Ollama не запущен. Запустите: ollama serve\n")
            return
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": config.LLM_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            answer = response.json().get("response", "Ошибка генерации").strip()
            chat_history.insert(tk.END, f"Аналитик: {answer}\n")
        else:
            chat_history.insert(tk.END, f"Аналитик: Ошибка LLM ({response.status_code})\n")
    except Exception as e:
        chat_history.insert(tk.END, f"Аналитик: Ошибка: {e}\n")
    chat_history.see(tk.END)

# === GUI ===
def create_gui():
    global root, status_label, ask_btn, btn_send, entry, text_answer, text_context
    global top_k_var, chunk_size_var, method_var, theme_var, entry_chat, chat_history, theme_menu
    global export_answer_btn, export_context_btn

    root = tk.Tk()
    root.title("CallCenter AI Анализатор v3 (реплики + HyDE + Reranker)")
    root.geometry("1350x900")

    # Темная тема
    theme_colors = config.GUI_THEME
    dark_bg = theme_colors["dark_bg"]
    dark_fg = theme_colors["dark_fg"]
    dark_entry_bg = theme_colors["dark_entry_bg"]
    dark_button_bg = theme_colors["dark_button_bg"]
    dark_frame_bg = theme_colors["dark_frame_bg"]
    success_color = theme_colors["success_color"]

    root.configure(bg=dark_bg)
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background=dark_frame_bg)
    style.configure("TLabel", background=dark_bg, foreground=dark_fg)
    style.configure("TButton", background=dark_button_bg, foreground=dark_fg)
    style.configure("TCombobox", fieldbackground=dark_entry_bg, background=dark_button_bg, foreground=dark_fg)
    style.configure("TNotebook", background=dark_bg)
    style.configure("TNotebook.Tab", background=dark_button_bg, foreground=dark_fg)
    style.map("TNotebook.Tab", background=[('selected', dark_bg)], foreground=[('selected', dark_fg)])

    # Верхняя панель
    frame_top = tk.Frame(root, bg=dark_frame_bg)
    frame_top.pack(pady=10, padx=10, fill=tk.X)

    tk.Label(frame_top, text="Ваш вопрос:", bg=dark_frame_bg, fg=dark_fg, font=('Arial', 10, 'bold')).pack(anchor=tk.W)
    entry = tk.Entry(frame_top, width=70, font=('Arial', 10), bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    settings_frame = tk.Frame(frame_top, bg=dark_frame_bg)
    settings_frame.pack(side=tk.LEFT, padx=(10, 5))

    tk.Label(settings_frame, text="Тема:", bg=dark_frame_bg, fg=dark_fg).pack(side=tk.LEFT)
    theme_var = tk.StringVar(value="Загрузка...")
    theme_menu = ttk.Combobox(settings_frame, textvariable=theme_var, values=["Загрузка..."], state="readonly", width=15)
    theme_menu.pack(side=tk.LEFT, padx=2)

    tk.Label(settings_frame, text="Метод:", bg=dark_frame_bg, fg=dark_fg).pack(side=tk.LEFT)
    method_var = tk.StringVar(value=config.GUI_DEFAULT_METHOD)
    method_menu = ttk.Combobox(settings_frame, textvariable=method_var, values=config.ANALYSIS_METHODS, state="readonly", width=12)
    method_menu.pack(side=tk.LEFT, padx=2)

    tk.Label(settings_frame, text="Найти:", bg=dark_frame_bg, fg=dark_fg).pack(side=tk.LEFT)
    top_k_var = tk.StringVar(value=str(config.GUI_DEFAULT_TOP_K))
    top_k_spinbox = tk.Spinbox(settings_frame, from_=1, to=10000, textvariable=top_k_var, width=6, bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    top_k_spinbox.pack(side=tk.LEFT, padx=2)

    tk.Label(settings_frame, text="реплик, по", bg=dark_frame_bg, fg=dark_fg).pack(side=tk.LEFT)
    chunk_size_var = tk.StringVar(value=str(config.GUI_DEFAULT_CHUNK_SIZE))
    chunk_size_spinbox = tk.Spinbox(settings_frame, from_=1, to=1000, textvariable=chunk_size_var, width=4, bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    chunk_size_spinbox.pack(side=tk.LEFT, padx=2)

    tk.Label(settings_frame, text="шт.", bg=dark_frame_bg, fg=dark_fg).pack(side=tk.LEFT)

    ask_btn = tk.Button(frame_top, text="Спросить", command=run_ask, state=tk.DISABLED, bg=success_color, fg='white')
    ask_btn.pack(side=tk.RIGHT)

    export_frame = tk.Frame(frame_top, bg=dark_frame_bg)
    export_frame.pack(side=tk.RIGHT, padx=(0, 10))
    export_answer_btn = tk.Button(export_frame, text="Экспорт Ответа", command=export_answer, state=tk.DISABLED, bg=dark_button_bg, fg=dark_fg)
    export_answer_btn.pack(side=tk.LEFT, padx=2)
    export_context_btn = tk.Button(export_frame, text="Экспорт Контекста", command=export_context, state=tk.DISABLED, bg=dark_button_bg, fg=dark_fg)
    export_context_btn.pack(side=tk.LEFT, padx=2)

    status_label = tk.Label(root, text="⏳ Загрузка...", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=dark_entry_bg, fg=dark_fg)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Панель данных/управления
    controls_frame = tk.Frame(root, bg=dark_frame_bg)
    controls_frame.pack(pady=(0, 5), padx=10, fill=tk.X)
    global btn_run_pipeline, btn_run_indexer, btn_reload_indexes
    btn_run_pipeline = tk.Button(controls_frame, text="Обработать новые (.rtf)", command=run_pipeline_background, bg=dark_button_bg, fg=dark_fg)
    btn_run_pipeline.pack(side=tk.LEFT, padx=2)
    btn_run_indexer = tk.Button(controls_frame, text="Переиндексировать (FAISS)", command=run_indexer_background, bg=dark_button_bg, fg=dark_fg)
    btn_run_indexer.pack(side=tk.LEFT, padx=2)
    btn_reload_indexes = tk.Button(controls_frame, text="Обновить индексы/данные", command=reload_indexes_and_data, bg=dark_button_bg, fg=dark_fg)
    btn_reload_indexes.pack(side=tk.LEFT, padx=2)

    # Вкладки
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

    tab_answer = ttk.Frame(notebook)
    notebook.add(tab_answer, text='Ответ')
    text_answer = scrolledtext.ScrolledText(tab_answer, wrap=tk.WORD, font=('Arial', 10), bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    text_answer.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    tab_context = ttk.Frame(notebook)
    notebook.add(tab_context, text='Найденные реплики')
    text_context = scrolledtext.ScrolledText(tab_context, wrap=tk.WORD, font=('Arial', 10), bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    text_context.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    tab_chat = ttk.Frame(notebook)
    notebook.add(tab_chat, text='Чат с Аналитиком')
    chat_frame = tk.Frame(tab_chat, bg=dark_frame_bg)
    chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=('Arial', 10), bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    input_frame = tk.Frame(chat_frame, bg=dark_frame_bg)
    input_frame.pack(fill=tk.X, padx=5, pady=5)
    entry_chat = tk.Entry(input_frame, font=('Arial', 10), bg=dark_entry_bg, fg=dark_fg, insertbackground=dark_fg)
    entry_chat.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    btn_send = tk.Button(input_frame, text="Отправить", command=on_send_click, state=tk.DISABLED, bg=dark_button_bg, fg=dark_fg)
    btn_send.pack(side=tk.RIGHT)

    # Загрузка
    def delayed_init():
        init_chat_db()
        init_db.init_db()
        load_models()
        load_faiss_indexes()
        load_data_lookups()
        themes_for_combo = ["all"] + [t for t in INDEXES.keys() if t != "all"]
        theme_menu['values'] = themes_for_combo
        if themes_for_combo:
            theme_var.set(themes_for_combo[0])
        ask_btn.config(state=tk.NORMAL)
        btn_send.config(state=tk.NORMAL)
        export_answer_btn.config(state=tk.NORMAL)
        export_context_btn.config(state=tk.NORMAL)
        status_label.config(text="✅ Готов к работе.")
        if not INDEXES:
            if messagebox.askyesno("Индексы не найдены", "Не найдены индексы FAISS. Построить сейчас?"):
                run_indexer_background()

    root.after(100, delayed_init)
    root.mainloop()

if __name__ == "__main__":
    create_gui()