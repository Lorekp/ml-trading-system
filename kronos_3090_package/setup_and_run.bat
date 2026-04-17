@echo off
echo ============================================
echo  Kronos Fine-Tuning Setup (Windows)
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

REM Fix: use a path WITHOUT spaces for the venv
set "VENV_DIR=C:\kronos_venv"

echo [1/4] Creating virtual environment at %VENV_DIR%...
echo   (Using C:\ to avoid OneDrive/spaces path issues)
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create venv. Trying local fallback...
    python -m venv venv
    set "VENV_DIR=%~dp0venv"
)

echo [2/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [3/4] Installing dependencies (this takes a few minutes)...
echo   Installing PyTorch with CUDA support...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 (
    echo WARNING: CUDA install failed, trying default...
    python -m pip install torch torchvision --quiet
)

echo   Installing other packages...
python -m pip install numpy pandas einops huggingface_hub tqdm safetensors matplotlib pyyaml --quiet

echo.
echo [4/4] Starting training...
echo ============================================
echo  Training will take approximately 15-20 hours.
echo  You can close this window - it will keep running.
echo  DO NOT turn off your PC until it finishes.
echo  Look for "Training completed" in the output.
echo ============================================
echo.

REM Run from the package root directory so relative paths work
cd /d "%~dp0"
cd kronos\finetune_csv
python train_sequential.py --config "%~dp0config_3090.yaml" --skip-tokenizer

echo.
echo ============================================
if errorlevel 1 (
    echo  Training FAILED. Please send me a screenshot of the error above.
) else (
    echo  Training COMPLETE!
    echo  Please send the entire "output" folder back to me.
    echo  It should be at: %~dp0output\
)
echo ============================================
pause
