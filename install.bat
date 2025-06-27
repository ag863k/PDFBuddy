@echo off
echo Installing PDFBuddy requirements...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Installing Python packages...
pip install --upgrade pip
pip install -r requirements.txt

echo Testing installation...
python test_setup.py

echo.
echo Installation complete!
echo Next steps:
echo 1. Install Ollama from https://ollama.ai
echo 2. Run 'run.bat' to start the application
pause
