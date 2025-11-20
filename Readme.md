# USEK RAG Chatbot with Voice Interface 🎤

A sophisticated AI chatbot for USEK university that combines **RAG (Retrieval-Augmented Generation)** with advanced voice capabilities. The project features two distinct interfaces: a standard Streamlit chat application and a real-time voice calling interface similar to ChatGPT's voice mode.

## 🌟 Features

*   **RAG Architecture**: Retrieves accurate information from a custom vector database of USEK documents and web crawls.
*   **LLM Integration**: Powered by Llama 3.1 (via Cerebras) for intelligent, context-aware responses.
*   **Two Interfaces**:
    1.  **Streamlit Chat**: Text-based chat with optional voice input/output and source citation.
    2.  **Voice Call Interface**: Real-time, bidirectional voice conversation with a modern, animated UI.
*   **Speech Capabilities**:
    *   **STT (Speech-to-Text)**: Uses OpenAI Whisper for accurate transcription.
    *   **TTS (Text-to-Speech)**: Uses gTTS for natural-sounding audio responses.
*   **Voice Activity Detection (VAD)**: Automatically detects when you speak and when you stop.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd usek_tts_chatbot
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Setup**:
    Create a `.env` file in the root directory with your API keys:
    ```env
    CEREBRAS_API_KEY=your_api_key_here
    ```

## 🚀 Usage

### Option 1: Voice Call Interface (New! 🎙️)
A real-time, hands-free voice conversation experience.

1.  Run the Flask server:
    ```bash
    python src/voice_call_interface.py
    ```
2.  Open your browser to: **http://localhost:5000**
3.  Click the **Call** button and start talking!

### Option 2: Streamlit Chat Interface
The classic chat interface with text history and source citations.

1.  Run the Streamlit app:
    ```bash
    streamlit run src/streamlit_chat.py
    ```
2.  The app will open automatically in your browser.

## 📂 Project Structure

*   `src/`
    *   `voice_call_interface.py`: Flask backend for the voice call mode.
    *   `streamlit_chat.py`: Main Streamlit application.
    *   `integrated_pipeline.py`: Core RAG logic (retrieval + generation).
    *   `vector_db.py`: Vector database management (FAISS/Chroma).
    *   `templates/` & `static/`: Frontend assets for the voice interface.
*   `indexes/`: Stores the FAISS vector database.
*   `crawl_data/`: Raw data from the web crawler.

## 📝 Documentation

For a detailed explanation of how the voice interface was built, see [VOICE_INTERFACE_GUIDE.md](VOICE_INTERFACE_GUIDE.md).