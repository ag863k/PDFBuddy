# PDFBuddy - AI-Powered PDF Chatbot

A production-ready PDF chatbot built with a completely free and open-source stack. Chat with your PDF documents using local AI models - no API keys required.

## 🚀 Features

- **100% Free & Open Source** - No paid APIs or services required
- **Local AI Processing** - Uses Ollama with Mistral/Phi-3 models
- **Smart PDF Analysis** - Advanced text extraction and chunking
- **Vector Search** - Semantic search using FAISS and sentence transformers
- **Modern UI** - Clean, responsive web interface
- **Conversation History** - Track and review previous conversations
- **Production Ready** - Proper error handling, logging, and security

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |
| **Backend** | Flask (Python) |
| **AI Model** | Ollama + Mistral 7B / Phi-3 |
| **PDF Processing** | PyMuPDF |
| **Vector Database** | FAISS |
| **Embeddings** | sentence-transformers |
| **Database** | SQLite |

## 🎯 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - [Install Ollama](https://ollama.ai/download)

### Installation

1. **Clone and setup:**
```bash
git clone https://github.com/yourusername/PDFBuddy.git
cd PDFBuddy
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

3. **Quick start:**
```bash
# Linux/Mac
chmod +x run.sh
./run.sh

# Windows
run.bat
```

Or manually:
```bash
ollama pull mistral
python app.py
```

4. **Open your browser:**
Navigate to `http://localhost:5000`

## 📖 Usage

1. **Upload PDF** - Drag and drop or select a PDF file (max 16MB)
2. **Wait for Processing** - The system extracts and indexes the content
3. **Start Chatting** - Ask questions about your document
4. **View Sources** - See relevant excerpts that informed each answer
5. **Check History** - Review previous conversations

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 🌟 Environment Configuration

Copy `.env.example` to `.env` and configure:

```env
SECRET_KEY=your-super-secret-key-here
FLASK_ENV=production
OLLAMA_MODEL=mistral
PORT=5000
```

## 🔧 Production Deployment

### Free Hosting Options

1. **Oracle Cloud Always Free** - 1-4 ARM cores, 6-24GB RAM
2. **AWS EC2 Free Tier** - t2.micro instance, 750 hours/month
3. **Self-hosted** - Your own server or laptop

### Security Considerations

- Always set a strong `SECRET_KEY` in production
- Use HTTPS in production environments
- Implement rate limiting for public deployments
- Consider file upload restrictions

## 📊 Performance

### Hardware Requirements

| Usage Level | CPU | RAM | Storage |
|-------------|-----|-----|---------|
| **Light** (1-5 users) | 2 cores | 4GB | 10GB |
| **Medium** (5-20 users) | 4 cores | 8GB | 20GB |
| **Heavy** (20+ users) | 8+ cores | 16GB+ | 50GB+ |

### Model Options

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **phi3** | 2.3GB | Fast | Good | Low resources |
| **mistral** | 4.1GB | Medium | Better | Balanced |
| **llama3** | 4.7GB | Slow | Best | High quality |

## 🆘 Troubleshooting

### Common Issues

**Ollama not responding:**
```bash
curl http://localhost:11434/api/tags
ollama serve
```

**PDF processing errors:**
- Ensure PDF is not password protected
- Check file size (max 16MB)
- Verify PDF contains extractable text

**Performance issues:**
- Use lighter models (phi3 instead of mistral)
- Reduce chunk size in configuration
- Consider upgrading hardware for large documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [sentence-transformers](https://www.sbert.net/) - Text embeddings
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF processing
- [Flask](https://flask.palletsprojects.com/) - Web framework

---

**Built with ❤️ for the open-source community**
