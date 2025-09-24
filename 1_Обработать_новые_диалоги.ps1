# Set UTF-8 output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Project path
$PROJECT_PATH = "E:\CallCenterAI_v3"
$LOG_FILE = "$PROJECT_PATH\logs\install.log"

# Create logs folder
if (!(Test-Path "$PROJECT_PATH\logs")) {
    New-Item -ItemType Directory -Path "$PROJECT_PATH\logs" | Out-Null
}

# Log start
"[$(Get-Date)] === START ===" | Out-File $LOG_FILE -Encoding UTF8

# Change directory
Set-Location $PROJECT_PATH
"[$(Get-Date)] Changed to $PROJECT_PATH" | Out-File $LOG_FILE -Encoding UTF8 -Append

# Check Python
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Python not found." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit 1
} else {
    "[$(Get-Date)] System Python: $pythonCheck" | Out-File $LOG_FILE -Encoding UTF8 -Append
}

# Remove old venv
if (Test-Path "venv") {
    Remove-Item -Recurse -Force "venv"
    "[$(Get-Date)] Removed old venv" | Out-File $LOG_FILE -Encoding UTF8 -Append
}

# Create new venv
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create venv." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to create venv." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit 1
}

# Explicitly define Python and Pip from venv
$PYTHON = "$PROJECT_PATH\venv\Scripts\python.exe"
$PIP = "$PROJECT_PATH\venv\Scripts\pip.exe"

# Verify venv Python
$pythonInVenv = & $PYTHON --version 2>&1
"[$(Get-Date)] Venv Python: $pythonInVenv" | Out-File $LOG_FILE -Encoding UTF8 -Append

# Upgrade pip
& $PYTHON -m pip install --upgrade pip 2>&1 | Out-File $LOG_FILE -Encoding UTF8 -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upgrade pip." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to upgrade pip." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit 1
}

# Install dependencies
"[$(Get-Date)] Installing dependencies..." | Out-File $LOG_FILE -Encoding UTF8 -Append
# --- NOTE: 'cross-encoder' removed as it's not a standalone package ---
& $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --disable-pip-version-check 2>&1 | Out-File $LOG_FILE -Encoding UTF8 -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to install PyTorch." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit 1
}

& $PIP install sentence-transformers faiss-cpu ollama striprtf tqdm numpy requests --disable-pip-version-check 2>&1 | Out-File $LOG_FILE -Encoding UTF8 -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install other dependencies." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to install other dependencies." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit 1
}

# Create folders
foreach ($dir in @("Input", "processed", "logs", "faiss_index", "cache", "exports", "temp")) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

# Run pipeline.py and capture output/errors
"[$(Get-Date)] Running pipeline.py..." | Out-File $LOG_FILE -Encoding UTF8 -Append
& $PYTHON pipeline.py 2>&1 | Out-File $LOG_FILE -Encoding UTF8 -Append
$pipelineExitCode = $LASTEXITCODE
if ($pipelineExitCode -ne 0) {
    Write-Host "ERROR: pipeline.py failed with exit code $pipelineExitCode. Check logs/install.log for details." -ForegroundColor Red
    "[$(Get-Date)] ERROR: pipeline.py failed with exit code $pipelineExitCode." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit $pipelineExitCode
}

# Run indexer.py and capture output/errors
"[$(Get-Date)] Running indexer.py..." | Out-File $LOG_FILE -Encoding UTF8 -Append
& $PYTHON indexer.py 2>&1 | Out-File $LOG_FILE -Encoding UTF8 -Append
$indexerExitCode = $LASTEXITCODE
if ($indexerExitCode -ne 0) {
    Write-Host "ERROR: indexer.py failed with exit code $indexerExitCode. Check logs/install.log for details." -ForegroundColor Red
    "[$(Get-Date)] ERROR: indexer.py failed with exit code $indexerExitCode." | Out-File $LOG_FILE -Encoding UTF8 -Append
    pause
    exit $indexerExitCode
}

Write-Host "=== ALL DONE ===" -ForegroundColor Green
"[$(Get-Date)] === ALL DONE ===" | Out-File $LOG_FILE -Encoding UTF8 -Append
pause