@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo EmotionTunes - Setup and Run
echo ========================================

if not exist .venv (
    echo Creating Python virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python 3.11 was not found.
        echo Install Python 3.11 and run this file again.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run music.py

pause
