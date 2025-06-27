# PDFBuddy - Enterprise PDF Chatbot

Production-ready PDF chatbot with local AI models. No API keys required.

## Features

- **100% Free** - No paid APIs or services
- **Local AI** - Ollama with Mistral/Phi-3 models
- **Enterprise Architecture** - Thread-safe, connection pooling
- **Smart PDF Analysis** - Advanced text extraction
- **Vector Search** - FAISS semantic search
- **Modern UI** - Responsive web interface
- **Performance Monitoring** - Health checks and metrics

## Quick Start

### Windows

1. **Quick Install:**
```cmd
git clone https://github.com/yourusername/PDFBuddy.git
cd PDFBuddy
install.bat
```

2. **Run:**
```cmd
run.bat
```

### Manual Setup

1. **Install Python packages:**
```bash
pip install -r requirements.txt
```

2. **Install Ollama:**
Download from https://ollama.ai/download

3. **Start:**
```bash
ollama pull mistral
python app.py
```

4. **Open browser:**
http://localhost:5000

## Usage

1. Upload PDF (max 16MB)
2. Ask questions about your document
3. View sources and conversation history

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask + Gunicorn |
| AI Model | Ollama + Mistral |
| PDF Processing | PyMuPDF |
| Vector DB | FAISS |
| Embeddings | sentence-transformers |
| Database | SQLite WAL mode |

## Docker

```bash
docker-compose up -d
```

## Configuration

Copy `.env.example` to `.env` for custom settings.

## API Endpoints

- `POST /upload` - Upload PDF
- `POST /chat` - Ask questions
- `GET /history` - Conversation history
- `GET /health` - Service status

## Performance

| Usage | CPU | RAM | Storage |
|-------|-----|-----|---------|
| Development | 2 cores | 4GB | 5GB |
| Small Team | 4 cores | 8GB | 20GB |
| Enterprise | 8+ cores | 16GB+ | 50GB+ |

## License

MIT License - see LICENSE file.
