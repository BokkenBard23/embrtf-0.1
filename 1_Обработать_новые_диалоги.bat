@echo off
chcp 65001 >nul
setlocal

set "PROJECT_PATH=E:\CallCenterAI_v3"
set "LOG_FILE=%PROJECT_PATH%\logs\install.log"

mkdir "%PROJECT_PATH%\logs" 2>nul
echo [%date% %time%] === ЗАПУСК === > "%LOG_FILE%"

cd /d "%PROJECT_PATH%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ❌ Не удалось перейти в %PROJECT_PATH%
    pause
    exit /b 1
)

python --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ❌ Python не найден.
    pause
    exit /b 1
)

if exist "venv" (
    rmdir /s /q "venv" >> "%LOG_FILE%" 2>&1
)

python -m venv venv >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ❌ Не удалось создать venv.
    pause
    exit /b 1
)

call "venv\Scripts\Activate.bat" >> "%LOG_FILE%" 2>&1
if "%VIRTUAL_ENV%"=="" (
    echo ❌ Не удалось активировать venv.
    pause
    exit /b 1
)

pip install --upgrade pip >nul 2>&1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >> "%LOG_FILE%" 2>&1
pip install sentence-transformers faiss-cpu ollama striprtf tqdm numpy cross-encoder sqlite3 >> "%LOG_FILE%" 2>&1

for %%D in (Input processed logs faiss_index cache exports temp) do (
    if not exist "%%D" mkdir "%%D" >> "%LOG_FILE%" 2>&1
)

python pipeline.py
if errorlevel 1 (
    echo ❌ Ошибка в pipeline.py
    pause
    exit /b 1
)

python indexer.py
if errorlevel 1 (
    echo ❌ Ошибка в indexer.py
    pause
    exit /b 1
)

echo ✅ ВСЁ ГОТОВО!
pause