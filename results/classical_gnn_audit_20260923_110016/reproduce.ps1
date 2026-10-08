# ==============================================================================
# REPRODUCIBILITY SCRIPT: AUDIT RUN 20260923_110016
# ==============================================================================
# Host requirements: Python 3.11.9, Intel/AMD x86_64, Windows OS
# Run command: powershell -ExecutionPolicy Bypass -File .\reproduce.ps1

Write-Host "Starting reproduction of Classical ML & Periodic Crystal GNN Audit ..." -ForegroundColor Cyan

$VENV_PYTHON = "d:\Desktop\Material_science_qml\venv\Scripts\python.exe"
$SCRIPT_PATH = "d:\Desktop\Material_science_qml\src\08_adversarial_reproducibility_audit.py"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Error "Virtual environment Python executable not found at $VENV_PYTHON"
    exit 1
}

Write-Host "Executing frozen audit engine: $SCRIPT_PATH" -ForegroundColor Yellow
& $VENV_PYTHON $SCRIPT_PATH "$PSScriptRoot"

Write-Host "Audit reproduction finished successfully." -ForegroundColor Green
