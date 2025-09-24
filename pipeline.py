"""Обработка RTF-файлов, извлечение текста/метаданных, кодирование (sentence-transformers GPU batch) и сохранение в SQLite.
   ОБНОВЛЕНО: теперь сохраняет реплики (utterances) и их эмбеддинги отдельно.
   + Динамический батч, TF32, логирование, автоочистка, PYTORCH_CUDA_ALLOC_CONF.
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import gc

# === Установка PYTORCH_CUDA_ALLOC_CONF ДО импорта torch ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === sentence-transformers для максимальной скорости ===
from sentence_transformers import SentenceTransformer
import torch

# === Включить TF32 для ускорения ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === Для обработки RTF ===
from striprtf.striprtf import rtf_to_text

# === Импорт конфигурации ===
import config

# === Настройка логирования ===
from utils import setup_logger
logger = setup_logger('Pipeline', config.LOGS_ROOT / "pipeline.log")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('🚨 %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.propagate = False

def log_to_file_only(msg):
    logger.info(msg)

def log(msg):
    logger.info(msg)

# === Автоматическая инициализация БД ===
def ensure_db_initialized():
    log_to_file_only("Проверка и инициализация структуры БД...")
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Проверка dialogs
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dialogs';")
        if cursor.fetchone() is None:
            log_to_file_only("Таблицы в БД не найдены. Запуск инициализации...")
            import init_db
            init_db.init_db()
            print("✅ Структура БД инициализирована.")
        
        # Создание таблицы utterances, если не существует
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utterances (
                id TEXT PRIMARY KEY,
                dialog_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                turn_order INTEGER NOT NULL,
                FOREIGN KEY (dialog_id) REFERENCES dialogs(id) ON DELETE CASCADE
            );
        """)
        
        # Создание таблицы utterance_embeddings, если не существует
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utterance_embeddings (
                utterance_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON DELETE CASCADE
            );
        """)
        
        conn.commit()
        conn.close()
        log_to_file_only("✅ Структура БД и таблиц utterances/utterance_embeddings в порядке.")
    except Exception as e:
        log_to_file_only(f"❌ Критическая ошибка при инициализации БД: {e}")
        print(f"❌ Критическая ошибка при инициализации БД: {e}")
        raise e

# === Загрузка модели sentence-transformers ===
def load_embedding_model():
    log_to_file_only(f"Загрузка модели {config.EMBEDDING_MODEL_NAME} на {config.EMBEDDING_MODEL_DEVICE}...")
    print(f"⏳ Загрузка модели {config.EMBEDDING_MODEL_NAME} на {config.EMBEDDING_MODEL_DEVICE}...")
    
    device = torch.device(config.EMBEDDING_MODEL_DEVICE if torch.cuda.is_available() or config.EMBEDDING_MODEL_DEVICE != "cuda" else "cpu")
    
    model_kwargs = {}
    if config.EMBEDDING_MODEL_PRECISION == "float16" and device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=str(device), model_kwargs=model_kwargs)
    model.max_seq_length = config.EMBEDDING_MODEL_MAX_LENGTH
    
    log_to_file_only(f"✅ Модель загружена. Используется устройство: {device}, точность: {config.EMBEDDING_MODEL_PRECISION}")
    print(f"✅ Модель загружена. Используется устройство: {device}, точность: {config.EMBEDDING_MODEL_PRECISION}")
    return model, device

def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def extract_metadata_from_filename(filename_stem: str) -> dict:
    import re
    pattern = r'^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_([A-Za-z0-9_]+)_([0-9]+)$'
    match = re.match(pattern, filename_stem)
    if match:
        year, month, day, hour, minute, second, operator_login, client_number = match.groups()
        if re.match(r'^\d{10,11}$', client_number):
            return {
                "date_from_filename": f"{year}-{month}-{day}",
                "time_from_filename": f"{hour}:{minute}:{second}",
                "operator_login_from_filename": operator_login,
                "client_number_from_filename": client_number
            }
    return {}

def clean_dialog_text_no_filter(text, file_metadata_from_name, file_id):
    import re
    from datetime import datetime
    lines = [line.rstrip() for line in text.splitlines() if line.rstrip()]
    if not lines:
        return [], file_metadata_from_name
    dialog_id = file_id
    dialog_datetime = None
    participants_line = None
    participants = {}
    dialog_type = "unknown"
    for i, line in enumerate(lines):
        id_datetime_match = re.match(r'^(\d+)\s*\((\d{2}\.\d{2}\.\d{4}\s\d{1,2}:\d{2}:\d{2})\)', line.strip())
        if id_datetime_match:
            dialog_id = id_datetime_match.group(1)
            try:
                dt_obj = datetime.strptime(id_datetime_match.group(2), '%d.%m.%Y %H:%M:%S')
                dialog_datetime = dt_obj.isoformat()
                dialog_type = "voice"
            except ValueError:
                pass
            if i + 1 < len(lines):
                participants_line = lines[i + 1].strip()
                p_match = re.search(r'([A-Za-z0-9@._\-]+)\s*(->|<-)\s*([0-9a-fA-F\-]+)', participants_line)
                if p_match:
                    op_raw = p_match.group(1)
                    arrow = p_match.group(2)
                    client_raw = p_match.group(3)
                    op_login = op_raw.split('@')[0] if '@' in op_raw else op_raw
                    participants["operator_login"] = op_login
                    participants["operator_raw"] = op_raw
                    participants["arrow"] = arrow
                    participants["client_id"] = client_raw
                    if client_raw.isdigit() and len(client_raw) >= 10:
                        participants["client_number"] = client_raw
            break
    if lines and "Rtf export" in lines[0] and "call(s)" in lines[0]:
        dialog_type = "chat"
    dialog_lines = []
    i = 0
    if participants_line:
        try:
            start_idx = lines.index(participants_line) + 1
            i = start_idx
        except ValueError:
            i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().lower() in ["rtf export, 1 call(s)", "порог чувствительности: 5,00 с"]:
            i += 1
            continue
        if re.match(r'^\d+\s*\(\d{2}\.\d{2}\.\d{4}\s\d{1,2}:\d{2}:\d{2}\)', line.strip()) and i > 2:
             break
        time_match = re.search(r'\t(\d+:\d+:\d+)$', line)
        line_text = line
        line_time = None
        if time_match:
            line_time = time_match.group(1)
            line_text = line[:time_match.start()].rstrip("\t")
        line_text = line_text.strip()
        if not line_text:
            i += 1
            continue
        parts = line_text.split('\t', 1)
        potential_speaker = parts[0].strip()
        replica_text = parts[1].strip() if len(parts) > 1 else line_text
        if potential_speaker and len(potential_speaker) < 100 and replica_text:
            speaker = potential_speaker.split('@')[0] if '@' in potential_speaker else potential_speaker
            time_str = f" [{line_time}]" if line_time else ""
            dialog_lines.append(f"{speaker}: {replica_text}{time_str}")
        elif dialog_lines and replica_text:
            dialog_lines[-1] += f" {replica_text}"
        else:
             dialog_lines.append(line_text)
        i += 1

    final_metadata = file_metadata_from_name.copy()
    final_metadata.update({
        "dialog_id": dialog_id,
        "dialog_datetime": dialog_datetime,
        "participants_raw": participants_line,
        "dialog_type": dialog_type,
        "participants": participants
    })
    return dialog_lines, final_metadata  # Возвращаем список реплик

# === Основная логика обработки ===
def process_thematic_folders():
    ensure_db_initialized()
    MODEL, device = load_embedding_model()
    
    log_to_file_only("🔍 Поиск тематических папок в Input...")
    theme_folders = [f for f in config.INPUT_ROOT.iterdir() if f.is_dir()]

    if not theme_folders:
        msg = "⚠️ В папке 'Input' не найдено тематических подпапок."
        log_to_file_only(msg)
        print(msg)
        return

    msg = f"📁 Найдены тематические папки: {[f.name for f in theme_folders]}"
    log_to_file_only(msg)
    print(msg)

    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    total_new_dialogs = 0
    
    for theme_folder in theme_folders:
        theme_name = theme_folder.name
        msg = f"📂 Начало обработки темы: {theme_name}"
        log_to_file_only(msg)
        print(msg)
        
        theme_proc_path = config.PROCESSED_ROOT / theme_name
        theme_proc_path.mkdir(parents=True, exist_ok=True)
        
        already_processed_files = {hash_file(p): p.stem for p in theme_proc_path.glob("*.rtf")}
        files_to_process = [p for p in theme_folder.glob("*.rtf") if hash_file(p) not in already_processed_files]
        
        if not files_to_process:
            msg = f"✅ Все файлы в '{theme_name}' уже обработаны."
            log_to_file_only(msg)
            print(msg)
            continue

        msg = f"📁 Найдено {len(files_to_process)} новых файлов в '{theme_name}' для обработки."
        log_to_file_only(msg)

        BATCH_SIZE = config.PIPELINE_BATCH_SIZE
        for i in tqdm(range(0, len(files_to_process), BATCH_SIZE), desc=f"Обработка '{theme_name}' (батчами по {BATCH_SIZE})", unit="батч"):
            batch_files = files_to_process[i:i + BATCH_SIZE]
            batch_dialogs_to_save = []
            batch_utterances_to_save = []
            batch_texts_to_encode = []
            batch_ids_for_embeddings = []
            
            # Этап 1: Обработка файлов
            for file in batch_files:
                try:
                    log_to_file_only(f"  📄 Обработка файла: {file.name}")
                    file_metadata = extract_metadata_from_filename(file.stem)
                    file_metadata["source_theme"] = theme_name
                    
                    raw_text = rtf_to_text(file.read_text(encoding="utf-8", errors="ignore"))
                    file_hash = hash_file(file)[:16]
                    dialog_lines, dialog_metadata = clean_dialog_text_no_filter(raw_text, file_metadata, file_hash)
                    
                    final_metadata = file_metadata.copy()
                    final_metadata.update(dialog_metadata)
                    dialog_id = final_metadata.get("dialog_id", file_hash)
                    
                    cursor.execute("SELECT id FROM dialogs WHERE id = ?", (dialog_id,))
                    if cursor.fetchone() is None:
                        # Сохраняем диалог
                        batch_dialogs_to_save.append({
                            "id": dialog_id,
                            "text": "\n".join(dialog_lines),
                            "metadata": json.dumps(final_metadata, ensure_ascii=False),
                            "source_theme": theme_name,
                            "processed_at": datetime.now().isoformat()
                        })
                        total_new_dialogs += 1
                        
                        # Сохраняем каждую реплику
                        for idx, line in enumerate(dialog_lines):
                            if ": " in line:
                                speaker_part, text_part = line.split(": ", 1)
                                if " [" in text_part:
                                    text_part = text_part.split(" [")[0]
                            else:
                                speaker_part = "Unknown"
                                text_part = line
                            
                            utterance_id = f"{dialog_id}_u{idx+1:03d}"
                            batch_utterances_to_save.append({
                                "id": utterance_id,
                                "dialog_id": dialog_id,
                                "speaker": speaker_part,
                                "text": text_part,
                                "turn_order": idx + 1
                            })
                            batch_texts_to_encode.append(text_part)
                            batch_ids_for_embeddings.append(utterance_id)
                    
                    # Перемещаем файл только после успешной подготовки данных
                    file.rename(theme_proc_path / file.name)
                        
                except Exception as e:
                    error_msg = f"  ❌ Ошибка при обработке файла {file.name}: {e}"
                    log_to_file_only(error_msg)
                    print(error_msg)
                    if file.exists():
                        file.rename(theme_proc_path / file.name)

            # Этап 2: Сохранение диалогов
            if batch_dialogs_to_save:
                try:
                    cursor.executemany("""
                        INSERT INTO dialogs (id, text, metadata, source_theme, processed_at)
                        VALUES (:id, :text, :metadata, :source_theme, :processed_at)
                    """, batch_dialogs_to_save)
                    conn.commit()
                except Exception as e:
                    log_to_file_only(f"  ❌ Ошибка при сохранении диалогов: {e}")
                    conn.rollback()

            # Этап 3: Сохранение реплик
            if batch_utterances_to_save:
                try:
                    cursor.executemany("""
                        INSERT INTO utterances (id, dialog_id, speaker, text, turn_order)
                        VALUES (:id, :dialog_id, :speaker, :text, :turn_order)
                    """, batch_utterances_to_save)
                    conn.commit()
                except Exception as e:
                    log_to_file_only(f"  ❌ Ошибка при сохранении реплик: {e}")
                    conn.rollback()

            # === ЛОГИРОВАНИЕ ДЛИН РЕПЛИК ===
            if batch_texts_to_encode:
                lengths = [len(text) for text in batch_texts_to_encode]
                log_to_file_only(f"  📏 Длины реплик: max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

            # Этап 4: Кодирование и сохранение эмбеддингов реплик
            if batch_texts_to_encode:
                current_batch_size = min(BATCH_SIZE, len(batch_texts_to_encode))
                while current_batch_size > 0:
                    try:
                        log_to_file_only(f"  🧠 Кодирование {len(batch_texts_to_encode)} реплик на {device} с батчем {current_batch_size}...")
                        with torch.no_grad():
                            batch_embeddings = MODEL.encode(
                                batch_texts_to_encode,
                                convert_to_tensor=False,
                                show_progress_bar=False,
                                device=device,
                                batch_size=current_batch_size
                            )
                        break  # Успешно
                    except torch.cuda.OutOfMemoryError as e:
                        log_to_file_only(f"  ⚠️ CUDA OOM при батче {current_batch_size}. Пробуем уменьшить...")
                        current_batch_size = max(1, current_batch_size // 2)
                        torch.cuda.empty_cache()
                        gc.collect()
                else:
                    log_to_file_only("  ❌ Не удалось закодировать реплики даже с батчем 1.")
                    conn.rollback()
                    continue

                # Сохранение эмбеддингов
                try:
                    import pickle
                    embeddings_to_save = []
                    for uid, emb in zip(batch_ids_for_embeddings, batch_embeddings):
                        emb_blob = pickle.dumps(emb)
                        embeddings_to_save.append((uid, emb_blob))
                    
                    cursor.executemany("""
                        INSERT OR REPLACE INTO utterance_embeddings (utterance_id, vector)
                        VALUES (?, ?)
                    """, embeddings_to_save)
                    conn.commit()
                    
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    log_to_file_only(f"  ❌ Ошибка при кодировании/сохранении эмбеддингов реплик: {e}")
                    conn.rollback()

        msg = f"✅ Завершена обработка темы: {theme_name}."
        log_to_file_only(msg)
        print(msg)

    conn.close()
    final_msg = f"🏁 Завершена обработка всех тематических папок. Всего новых диалогов: {total_new_dialogs}"
    log_to_file_only(final_msg)
    print(final_msg)

if __name__ == "__main__":
    process_thematic_folders()