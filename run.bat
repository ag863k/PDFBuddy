@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting PDFBuddy Production Setup...

where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama not found. Please install from https://ollama.ai
    pause
    exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Starting Ollama...
    start /b ollama serve
    timeout /t 5 /nobreak >nul
)

ollama list | findstr "mistral" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 Downloading Mistral model...
    ollama pull mistral
)

echo 🌐 Starting PDFBuddy on http://localhost:5000
python app.py
