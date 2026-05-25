@echo off

title TDA Imputation Evaluator

echo ==========================================
echo   TDA-Based Imputation Quality Evaluator
echo ==========================================
echo.

cd /d %~dp0

echo [1/4] Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not added to PATH.
    pause
    exit /b
)

echo.
echo [2/4] Creating virtual environment if needed...

if not exist venv (
    python -m venv venv
)

echo.
echo [3/4] Activating virtual environment...

call venv\Scripts\activate

echo.
echo [4/4] Installing / checking dependencies...

pip install -r requirements.txt

echo.
echo Launching app...

start cmd /c "timeout /t 5 >nul && start http://127.0.0.1:8000"

shiny run app.py

pause