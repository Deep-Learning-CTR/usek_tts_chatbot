import os
import json
import base64
import tempfile
import numpy as np
import eventlet
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from pathlib import Path
from dotenv import load_dotenv
import whisper
from gtts import gTTS
from pydub import AudioSegment
import io

# Import existing pipeline
from integrated_pipeline import CrawlerRAGPipeline
from cerebras.cloud.sdk import Cerebras

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables
load_dotenv()
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not cerebras_api_key:
    print("WARNING: CEREBRAS_API_KEY not found in .env")

# Initialize Cerebras client
client = Cerebras(api_key=cerebras_api_key) if cerebras_api_key else None

# Global variables for models and pipeline
whisper_model = None
rag_pipeline = None
LLM_MODEL = "llama3.1-8b"

def init_resources():
    """Initialize heavy resources (Whisper, RAG Pipeline)"""
    global whisper_model, rag_pipeline
    
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("small.en")
    
    print("Loading RAG Pipeline...")
    root = Path(__file__).parent.parent
    vectordb_path = root / "indexes" / "crawler_vectordb"
    
    rag_pipeline = CrawlerRAGPipeline(
        embedding_model_name="toshk0/nomic-embed-text-v2-moe:Q6_K",
        vector_db_type="FAISS",
        vector_db_path=str(vectordb_path),
        chunk_size=800,
        chunk_overlap=200,
        ollama_base_url="http://localhost:11434",
        max_workers=12
    )
    print("Resources loaded successfully!")

def query_llm(prompt, context):
    """Query the LLM with context"""
    if not client:
        return "Error: Cerebras API key not configured."

    full_prompt = f"""
You are an AI assistant. You have access to crawled USEK university data.

Your job:
- Use the provided context **as the main knowledge source**
- If the context is incomplete or partially relevant, combine it with general knowledge
- Provide a conversational, concise answer suitable for voice chat
- Keep answers relatively short (2-3 sentences) unless asked for details

CONTEXT:
{context}

USER QUESTION:
{prompt}

FULL ANSWER:
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    return render_template('voice_call.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Receive audio chunk from client, transcribe, and respond"""
    # In a real streaming setup, we would buffer and process streams.
    # For this implementation, we'll accept a full "utterance" blob from the client
    # which simplifies the VAD logic (done on client side).
    
    try:
        audio_data = base64.b64decode(data['audio'])
        
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        # 1. Transcribe
        emit('status', {'state': 'transcribing'})
        result = whisper_model.transcribe(tmp_path, fp16=False)
        text = result["text"].strip()
        os.unlink(tmp_path)
        
        if not text:
            emit('status', {'state': 'listening'})
            return

        print(f"User said: {text}")
        emit('transcript', {'role': 'user', 'text': text})
        
        # 2. RAG Search
        emit('status', {'state': 'thinking'})
        
        context = ""
        if rag_pipeline:
            results = rag_pipeline.query(text, top_k=3)
            docs = results["documents"][0]
            context = "\n".join(docs)
        
        # 3. LLM Response
        answer = query_llm(text, context)
        print(f"AI response: {answer}")
        emit('transcript', {'role': 'assistant', 'text': answer})
        
        # 4. TTS
        emit('status', {'state': 'speaking'})
        tts = gTTS(answer, lang="en")
        
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Convert to base64 to send to client
        audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        
        emit('audio_response', {'audio': audio_b64})
        emit('status', {'state': 'listening'})
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        emit('error', {'message': str(e)})
        emit('status', {'state': 'listening'})

if __name__ == '__main__':
    init_resources()
    print("Starting Flask SocketIO server on http://localhost:5000")
    socketio.run(app, debug=True, port=5000)
