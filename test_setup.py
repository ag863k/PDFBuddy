#!/usr/bin/env python3

import sys
import importlib
import subprocess

required_packages = [
    'flask',
    'PyMuPDF', 
    'faiss',
    'sentence_transformers',
    'numpy',
    'requests',
    'werkzeug'
]

def test_imports():
    print("Testing package imports...")
    failed = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.lower().replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed.append(package)
    
    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\nAll packages imported successfully!")
    return True

def test_ollama():
    print("\nTesting Ollama connection...")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
        else:
            print("✗ Ollama not responding")
            return False
    except:
        print("✗ Ollama not available")
        print("Install Ollama from https://ollama.ai and run: ollama serve")
        return False

if __name__ == '__main__':
    print("PDFBuddy System Check")
    print("=" * 30)
    
    imports_ok = test_imports()
    ollama_ok = test_ollama()
    
    print("\n" + "=" * 30)
    if imports_ok and ollama_ok:
        print("✓ System ready! Run: python app.py")
    elif imports_ok:
        print("⚠ Python packages OK, but Ollama needed")
    else:
        print("✗ Install missing packages first")
