@echo off
chcp 65001 >nul
setlocal

title 🖥️ Запуск GUI для анализа реплик | CallCenter AI v3

:: === Путь к проекту ===
set "PROJECT_PATH=%~dp0"
if "%PROJECT_PATH:~-1%"=="\" set "PROJECT_PATH=%PROJECT_PATH:~0,-1%"

:: === Лог-файл ===
set "LOG_FILE=%PROJECT_PATH%\logs\run_gui.log"
echo [%date% %time%] Запуск GUI >> "%LOG_FILE%"

:: === Проверка папки проекта ===
if not exist "%PROJECT_PATH%" (
    echo ❌ ОШИБКА: Папка проекта не найдена: %PROJECT_PATH% >> "%LOG_FILE%"
    echo.
    echo ❌ ОШИБКА: Папка проекта не найдена: %PROJECT_PATH%
    pause
    exit /b 1
)

cd /d "%PROJECT_PATH%"
echo 📁 Текущий путь: %cd% >> "%LOG_FILE%"

:: === Проверка venv ===
if not exist ".\venv\Scripts\Activate.bat" (
    echo ❌ Виртуальное окружение не найдено. >> "%LOG_FILE%"
    echo.
    echo ❌ ОШИБКА: Виртуальное окружение не найдено.
    echo Убедитесь, что папка 'venv' существует и содержит 'Scripts\Activate.bat'.
    pause
    exit /b 1
)

call .\venv\Scripts\Activate.bat
if "%VIRTUAL_ENV%"=="" (
    echo ❌ Не удалось активировать venv. >> "%LOG_FILE%"
    echo.
    echo ❌ ОШИБКА: Не удалось активировать виртуальное окружение.
    pause
    exit /b 1
)
echo ✅ venv активировано >> "%LOG_FILE%"

:: === Проверка Ollama ===
where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama не найдена. Убедитесь, что она запущена. >> "%LOG_FILE%"
    echo.
    echo ⚠️  Ollama не найдена или не запущена.
    echo Скачайте и запустите: https://ollama.com/
    pause
)

:: === Проверка БД и таблиц ===
if not exist "database.db" (
    echo ⚠️  database.db не найдена. >> "%LOG_FILE%"
    set "MISSING=1"
)

:: Проверка таблицы utterances (через SQLite)
sqlite3 "database.db" "SELECT COUNT(*) FROM utterances LIMIT 1;" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Таблица utterances не существует или пуста. >> "%LOG_FILE%"
    set "MISSING=1"
)

:: Проверка FAISS-индексов реплик
powershell -Command "Get-ChildItem -Path 'faiss_index' -Filter *index.index -ErrorAction SilentlyContinue | Select-Object -First 1" | findstr /C:".index" >nul
if errorlevel 1 (
    echo ⚠️  FAISS-индексы реплик не найдены. >> "%LOG_FILE%"
    set "MISSING=1"
)

:: Если что-то отсутствует — предупреждение
if defined MISSING (
    echo.
    echo ⚠️  ⚠️  ⚠️  ВНИМАНИЕ: Не хватает данных для полноценной работы GUI.
    echo     - Убедитесь, что вы запускали pipeline.bat для обработки диалогов.
    echo     - Убедитесь, что indexer.py создал индексы для реплик.
    echo     - Проверьте logs\pipeline.log и logs\indexer.log.
    echo.
    pause
)

:: === Запуск GUI ===
echo.
echo ========================================
echo 🚀 Запуск GUI для анализа диалогов и реплик...
echo ========================================
echo 📌 Доступен метод: "callback_classifier" — для поиска ошибок по обратным звонкам.
echo 📌 Используются индексы реплик — поиск стал точнее.
echo.

python gui.py
if %errorlevel% neq 0 (
    echo ❌ Ошибка при запуске GUI >> "%LOG_FILE%"
    echo.
    echo ❌ ОШИБКА при запуске GUI!
    echo Подробнее: logs\gui.log
    pause
    exit /b %errorlevel%
)

echo ✅ GUI закрыто >> "%LOG_FILE%"
echo.
pause