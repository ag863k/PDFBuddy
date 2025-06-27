import os
import sqlite3
import secrets
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import uuid
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 50))
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'chatbot.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PDFChatbot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        conn.commit()
        conn.close()
        
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = PyMuPDF.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= CHUNK_SIZE:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_words = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(word) + 1 for word in overlap_words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        return np.array(embeddings).astype('float32')
    
    def build_vector_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def process_document(self, file_path: str, filename: str) -> str:
        try:
            document_id = str(uuid.uuid4())
            
            text = self.extract_text_from_pdf(file_path)
            chunks = self.split_text_into_chunks(text)
            
            if not chunks:
                raise ValueError("No text could be extracted from the PDF")
            
            embeddings = self.create_embeddings(chunks)
            self.build_vector_index(embeddings)
            self.chunks = chunks
            
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO documents (id, filename, upload_date, chunk_count) VALUES (?, ?, ?, ?)',
                (document_id, filename, datetime.now().isoformat(), len(chunks))
            )
            conn.commit()
            conn.close()
            
            session['document_id'] = document_id
            
            logger.info(f"Document processed successfully: {filename}, {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def find_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        if self.index is None or not self.chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks
    
    def query_ollama(self, prompt: str) -> str:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "Sorry, there was an error connecting to the AI model."
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return "Sorry, the AI model is not available. Please ensure Ollama is running."
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        if not question.strip():
            return "Please ask a question about the document.", []
        
        relevant_chunks = self.find_relevant_chunks(question)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""Based on the following context from a PDF document, answer the question precisely using only the provided information.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.query_ollama(prompt)
        
        if 'document_id' in session:
            self.save_conversation(session['document_id'], question, answer)
        
        return answer, relevant_chunks
    
    def save_conversation(self, document_id: str, question: str, answer: str):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (id, document_id, question, answer, timestamp) VALUES (?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), document_id, question, answer, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

chatbot = PDFChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not chatbot.allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        document_id = chatbot.process_document(file_path, filename)
        
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': 'PDF uploaded and processed successfully',
            'document_id': document_id,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if 'document_id' not in session:
            return jsonify({'error': 'No document uploaded. Please upload a PDF first.'}), 400
        
        answer, sources = chatbot.answer_question(question)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500

@app.route('/history')
def get_history():
    try:
        if 'document_id' not in session:
            return jsonify([])
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT question, answer, timestamp FROM conversations WHERE document_id = ? ORDER BY timestamp DESC LIMIT 20',
            (session['document_id'],)
        )
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'question': row[0],
                'answer': row[1],
                'timestamp': row[2]
            })
        
        conn.close()
        return jsonify(conversations)
        
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify([])

@app.route('/reset', methods=['POST'])
def reset_session():
    session.clear()
    chatbot.index = None
    chatbot.chunks = []
    return jsonify({'success': True, 'message': 'Session reset successfully'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
