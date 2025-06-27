import os
import sqlite3
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from flask import Flask, request, jsonify, render_template, session, g
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import uuid
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    UPLOAD_FOLDER = Path(os.environ.get('UPLOAD_FOLDER', 'uploads'))
    ALLOWED_EXTENSIONS = {'pdf'}
    OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 500))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 50))
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'chatbot.db')
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))
    MAX_CHUNKS_SEARCH = int(os.environ.get('MAX_CHUNKS_SEARCH', 5))
    SESSION_LIFETIME = timedelta(hours=int(os.environ.get('SESSION_LIFETIME_HOURS', 24)))

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
app.permanent_session_lifetime = Config.SESSION_LIFETIME

Config.UPLOAD_FOLDER.mkdir(exist_ok=True)

class ValidationError(Exception):
    pass

class ProcessingError(Exception):
    pass

def require_document(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'document_id' not in session:
            return jsonify({'error': 'No document uploaded. Please upload a PDF first.'}), 400
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({'error': str(e)}), 400
        except ProcessingError as e:
            logger.error(f"Processing error: {e}")
            return jsonify({'error': str(e)}), 500
        except RequestEntityTooLarge:
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred.'}), 500
    return decorated_function

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH, timeout=20.0)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=1000')
        conn.execute('PRAGMA temp_store=memory')
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

class PDFChatbot:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self._model_lock = threading.Lock()
        self._init_embedding_model()
        self.init_database()
        self._initialized = True
        
    def _init_embedding_model(self):
        try:
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ProcessingError("Failed to initialize embedding model")
        
    def init_database(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    UNIQUE(file_hash)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processing_time REAL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON conversations(document_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
            conn.commit()
        
    def _validate_file(self, filename: str, file_size: int) -> None:
        if not filename:
            raise ValidationError("No filename provided")
            
        if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS):
            raise ValidationError("Only PDF files are allowed")
            
        if file_size > Config.MAX_CONTENT_LENGTH:
            raise ValidationError(f"File too large. Maximum size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _check_duplicate_document(self, file_hash: str) -> Optional[str]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM documents WHERE file_hash = ?', (file_hash,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not Path(pdf_path).exists():
            raise ProcessingError("PDF file not found")
            
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            doc = PyMuPDF.open(pdf_path)
            text = ""
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                    
            doc.close()
            
            if not text.strip():
                raise ProcessingError("No extractable text found in PDF")
                
            logger.info(f"Successfully extracted {len(text)} characters from {page_count} pages")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            if 'doc' in locals():
                doc.close()
            raise ProcessingError(f"Failed to extract text from PDF: {str(e)}")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        if not text.strip():
            raise ProcessingError("No text to split into chunks")
            
        words = text.split()
        if not words:
            raise ProcessingError("No words found in text")
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= Config.CHUNK_SIZE:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_words = (current_chunk[-Config.CHUNK_OVERLAP:] 
                               if len(current_chunk) > Config.CHUNK_OVERLAP 
                               else current_chunk)
                current_chunk = overlap_words
                current_length = sum(len(word) + 1 for word in overlap_words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        if not chunks:
            raise ProcessingError("Failed to create text chunks")
            
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        if not chunks:
            raise ProcessingError("No chunks to create embeddings for")
            
        try:
            with self._model_lock:
                logger.info(f"Creating embeddings for {len(chunks)} chunks")
                embeddings = self.embedding_model.encode(
                    chunks, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=32
                )
                embeddings_array = np.array(embeddings).astype('float32')
                logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
                return embeddings_array
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise ProcessingError(f"Failed to create embeddings: {str(e)}")
    
    def build_vector_index(self, embeddings: np.ndarray) -> None:
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            logger.info(f"Built vector index with {embeddings.shape[0]} vectors")
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise ProcessingError(f"Failed to build vector index: {str(e)}")
    
    def process_document(self, file_path: str, filename: str) -> str:
        start_time = time.time()
        file_size = Path(file_path).stat().st_size
        
        try:
            self._validate_file(filename, file_size)
            
            file_hash = self._calculate_file_hash(file_path)
            existing_doc_id = self._check_duplicate_document(file_hash)
            
            if existing_doc_id:
                logger.info(f"Document already exists with ID: {existing_doc_id}")
                session['document_id'] = existing_doc_id
                return existing_doc_id
            
            document_id = str(uuid.uuid4())
            
            text = self.extract_text_from_pdf(file_path)
            chunks = self.split_text_into_chunks(text)
            
            embeddings = self.create_embeddings(chunks)
            self.build_vector_index(embeddings)
            self.chunks = chunks
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents 
                    (id, filename, file_hash, upload_date, chunk_count, file_size) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    document_id, 
                    filename, 
                    file_hash,
                    datetime.now().isoformat(), 
                    len(chunks),
                    file_size
                ))
                conn.commit()
            
            session['document_id'] = document_id
            session.permanent = True
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed successfully: {filename}, "
                       f"{len(chunks)} chunks, {processing_time:.2f}s")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise
    
    def find_relevant_chunks(self, query: str, k: Optional[int] = None) -> List[str]:
        if not query.strip():
            raise ValidationError("Query cannot be empty")
            
        if self.index is None or not self.chunks:
            logger.warning("No document indexed for search")
            return []
        
        k = k or Config.MAX_CHUNKS_SEARCH
        
        try:
            with self._model_lock:
                query_embedding = self.embedding_model.encode(
                    [query], 
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
            
            query_embedding = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
            
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks) and distances[0][i] < 1.5:  # similarity threshold
                    relevant_chunks.append(self.chunks[idx])
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            return []
    
    def _validate_ollama_connection(self) -> bool:
        try:
            response = requests.get(
                Config.OLLAMA_URL.replace('/api/generate', '/api/tags'),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def query_ollama(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")
            
        if not self._validate_ollama_connection():
            raise ProcessingError("Ollama service is not available")
            
        try:
            payload = {
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                Config.OLLAMA_URL, 
                json=payload, 
                timeout=Config.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                if not answer:
                    return "I could not generate a response to your question."
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise ProcessingError(f"AI model returned error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            raise ProcessingError("Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise ProcessingError("AI model is not available. Please ensure Ollama is running.")
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        start_time = time.time()
        
        if not question.strip():
            raise ValidationError("Please ask a question about the document.")
        
        if len(question) > 1000:
            raise ValidationError("Question is too long. Please keep it under 1000 characters.")
        
        try:
            relevant_chunks = self.find_relevant_chunks(question)
            
            if not relevant_chunks:
                return ("I couldn't find relevant information in the document to answer your question. "
                       "Please try rephrasing your question or ask about different aspects of the document."), []
            
            context = "\n\n".join(relevant_chunks[:3])  # Limit context size
            
            prompt = f"""Based on the following context from a PDF document, answer the question precisely and concisely using only the provided information. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Answer only based on the provided context
- Be precise and factual
- If information is incomplete, acknowledge it
- Keep the answer concise but complete

Answer:"""
            
            answer = self.query_ollama(prompt)
            
            processing_time = time.time() - start_time
            
            if 'document_id' in session:
                self.save_conversation(session['document_id'], question, answer, processing_time)
            
            logger.info(f"Question answered in {processing_time:.2f}s")
            return answer, relevant_chunks
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def save_conversation(self, document_id: str, question: str, answer: str, processing_time: float = 0) -> None:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (id, document_id, question, answer, timestamp, processing_time) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()), 
                    document_id, 
                    question, 
                    answer, 
                    datetime.now().isoformat(),
                    processing_time
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT filename, chunk_count, file_size, upload_date 
                    FROM documents WHERE id = ?
                ''', (document_id,))
                doc_info = cursor.fetchone()
                
                cursor.execute('''
                    SELECT COUNT(*), AVG(processing_time) 
                    FROM conversations WHERE document_id = ?
                ''', (document_id,))
                conv_stats = cursor.fetchone()
                
                if doc_info:
                    return {
                        'filename': doc_info[0],
                        'chunk_count': doc_info[1],
                        'file_size': doc_info[2],
                        'upload_date': doc_info[3],
                        'conversation_count': conv_stats[0] or 0,
                        'avg_processing_time': conv_stats[1] or 0
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}

chatbot = PDFChatbot()

@app.before_request
def setup_logging():
    if not hasattr(g, 'logging_setup') and not app.debug:
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        g.logging_setup = True

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found.'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error.'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@handle_errors
def upload_file():
    if 'file' not in request.files:
        raise ValidationError('No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        raise ValidationError('No file selected')
    
    filename = secure_filename(file.filename)
    if not filename:
        raise ValidationError('Invalid filename')
    
    file_path = Config.UPLOAD_FOLDER / filename
    
    try:
        file.save(str(file_path))
        
        document_id = chatbot.process_document(str(file_path), filename)
        
        return jsonify({
            'success': True,
            'message': 'PDF uploaded and processed successfully',
            'document_id': document_id,
            'filename': filename,
            'stats': chatbot.get_document_stats(document_id)
        })
        
    finally:
        if file_path.exists():
            file_path.unlink()

@app.route('/chat', methods=['POST'])
@handle_errors
@require_document
def chat():
    data = request.get_json()
    if not data:
        raise ValidationError('No JSON data provided')
    
    question = data.get('question', '').strip()
    if not question:
        raise ValidationError('Question is required')
    
    answer, sources = chatbot.answer_question(question)
    
    return jsonify({
        'answer': answer,
        'sources': sources[:3],  # Limit sources in response
        'timestamp': datetime.now().isoformat(),
        'source_count': len(sources)
    })

@app.route('/history')
@handle_errors
@require_document
def get_history():
    limit = min(int(request.args.get('limit', 20)), 100)  # Max 100 items
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT question, answer, timestamp, processing_time 
            FROM conversations 
            WHERE document_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session['document_id'], limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'question': row[0],
                'answer': row[1],
                'timestamp': row[2],
                'processing_time': row[3]
            })
        
        return jsonify({
            'conversations': conversations,
            'total': len(conversations)
        })

@app.route('/document/stats')
@handle_errors
@require_document
def get_document_stats():
    stats = chatbot.get_document_stats(session['document_id'])
    return jsonify(stats)

@app.route('/reset', methods=['POST'])
@handle_errors
def reset_session():
    session.clear()
    chatbot.index = None
    chatbot.chunks = []
    return jsonify({'success': True, 'message': 'Session reset successfully'})

@app.route('/health')
def health_check():
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'services': {
            'database': 'connected',
            'embedding_model': 'loaded' if chatbot.embedding_model else 'error',
            'ollama': 'connected' if chatbot._validate_ollama_connection() else 'disconnected'
        }
    }
    
    status_code = 200
    if not chatbot.embedding_model or not chatbot._validate_ollama_connection():
        health_data['status'] = 'degraded'
        status_code = 503
    
    return jsonify(health_data), status_code

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        
        logger.info(f"Starting PDFBuddy on port {port}")
        logger.info(f"Debug mode: {debug_mode}")
        logger.info(f"Database: {Config.DATABASE_PATH}")
        logger.info(f"Ollama URL: {Config.OLLAMA_URL}")
        logger.info(f"Model: {Config.OLLAMA_MODEL}")
        
        app.run(
            debug=debug_mode, 
            host='0.0.0.0', 
            port=port,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
