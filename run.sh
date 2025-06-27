#!/bin/bash
set -e

echo "🚀 Starting PDFBuddy Production Setup..."

if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install from https://ollama.ai"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Starting Ollama..."
    ollama serve &
    sleep 5
fi

if ! ollama list | grep -q "mistral"; then
    echo "📥 Downloading Mistral model..."
    ollama pull mistral
fi

echo "🌐 Starting PDFBuddy on http://localhost:5000"
python app.py
