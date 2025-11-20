import os
import streamlit as st
import requests
from pathlib import Path
import tempfile
from datetime import datetime

import whisper
from gtts import gTTS
from pydub import AudioSegment
from audiorecorder import audiorecorder

from integrated_pipeline import CrawlerRAGPipeline
from vector_db import FAISSStore

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

LLM_MODEL = "llama3.1-8b"

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Voice-Enabled RAG Chatbot",
    page_icon="🎤",
    layout="wide"
)

# SESSION INIT
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "llm_models" not in st.session_state:
    st.session_state.llm_models = []

if "last_processed_prompt" not in st.session_state:
    st.session_state.last_processed_prompt = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "pending_transcription" not in st.session_state:
    st.session_state.pending_transcription = None

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# -----------------------------------------
# INIT PIPELINE
# -----------------------------------------
def init_pipeline():
    try:
        root = Path(__file__).parent.parent
        vectordb_path = root / "indexes" / "crawler_vectordb"

        pipeline = CrawlerRAGPipeline(
            embedding_model_name="toshk0/nomic-embed-text-v2-moe:Q6_K",
            vector_db_type="FAISS",
            vector_db_path=str(vectordb_path),
            chunk_size=800,
            chunk_overlap=200,
            ollama_base_url="http://localhost:11434",
            max_workers=12
        )

        cnt = pipeline.vector_db.count()
        st.success(f"Loaded FAISS index with **{cnt:,} vectors**")

        return pipeline

    except Exception as e:
        st.error(f"Pipeline init error: {e}")
        return None



# -----------------------------------------
# LLM QUERY
# -----------------------------------------
load_dotenv()
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")

if not cerebras_api_key:
    raise ValueError("Missing CEREBRAS_API_KEY in .env")

# Initialize Cerebras client once
client = Cerebras(api_key=cerebras_api_key)

def query_llm(model: str, prompt: str, context: str):
    full_prompt = f"""
You are an AI assistant. You have access to crawled USEK university data.

Your job:
- Use the provided context **as the main knowledge source**
- If the context is incomplete or partially relevant, combine it with general knowledge
- Provide a full, detailed answer

CONTEXT:
{context}

USER QUESTION:
{prompt}

FULL ANSWER:
"""

    try:
        response = client.chat.completions.create(
            model=model,  # "llama-4-scout-17b-16e-instruct"
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=2048,
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[CEREBRAS ERROR]: {e}"




# -----------------------------------------
# STT + TTS
# -----------------------------------------
whisper_model = whisper.load_model("small.en")


def convert_audio_to_text(wav_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        result = whisper_model.transcribe(tmp.name, fp16=False)
        return result["text"].strip()


def speak_text(text):
    tts = gTTS(text, lang="en")
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_mp3.name)

    audio = AudioSegment.from_mp3(temp_mp3.name)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_file.name, format="wav")

    return wav_file.name



# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider("Top-K Chunks Used", 3, 10, 5)

    st.markdown("---")

    if st.session_state.pipeline is None:
        if st.button("Initialize Pipeline"):
            st.session_state.pipeline = init_pipeline()
            st.rerun()
    else:
        st.success("Pipeline Loaded ✔")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_processed_prompt = None
        st.rerun()



# -----------------------------------------
# MAIN UI
# -----------------------------------------
st.title("🎤 Voice-Enabled RAG Chatbot")
st.caption("Ask via text OR voice. Full history + improved answers.")



# -----------------------------------------
# DISPLAY CHAT HISTORY
# -----------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Speak button
        if msg["role"] == "assistant":
            if st.button("🔊 Speak", key=f"speak_hist_{idx}"):
                audio_path = speak_text(msg["content"])
                st.audio(open(audio_path, "rb").read(), format="audio/wav")

        # Sources for assistant messages (NEW FEATURE)
        if msg["role"] == "assistant" and "sources" in msg:
            urls = list({s["source"] for s in msg["sources"] if s["source"]})

            if urls:
                st.markdown("### 🔗 Sources Used")
                for u in urls:
                    st.markdown(f"- [{u}]({u})")



# -----------------------------------------
# INPUT SECTION (BELOW CHAT)
# -----------------------------------------
st.markdown("---")

# Single row: [Input field] [Mic button]
col_input, col_voice = st.columns([9.5, 0.5])

with col_voice:
    audio = audiorecorder("🎤", "⏹️")

# Process audio transcription
if len(audio) > 0:
    # Check if this is new audio by comparing with last processed
    audio_bytes = audio.export(format="wav").read()

    # Create a simple hash to detect new recordings
    import hashlib
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None

    # Only transcribe if it's a new recording
    if st.session_state.last_audio_hash != audio_hash:
        with st.spinner("🎤 Transcribing your voice..."):
            transcribed = convert_audio_to_text(audio_bytes)
            st.session_state.pending_transcription = transcribed
            st.session_state.last_audio_hash = audio_hash
        st.rerun()

# Chat input (with transcription if available)
prompt = None
with col_input:
    # If we have a pending transcription, show it in a form
    # If we have a pending transcription, show it in a cleaner UI
    if st.session_state.pending_transcription:
        col_text, col_btn = st.columns([8, 1])

        def submit_transcription():
            st.session_state.submit_voice = True

        with col_text:
            user_input = st.text_input(
                "Message",
                value=st.session_state.pending_transcription,
                label_visibility="collapsed",
                key="transcribed_input",
                on_change=submit_transcription
            )

        with col_btn:
            send_clicked = st.button("➤", on_click=submit_transcription, use_container_width=True)

        if st.session_state.get("submit_voice", False):
            prompt = user_input
            st.session_state.pending_transcription = None
            st.session_state.last_audio_hash = None
            st.session_state.submit_voice = False
    else:
        # Normal chat input for typing
        prompt = st.chat_input("💬 Type your question here...")
        if prompt:
            st.session_state.last_audio_hash = None



# -----------------------------------------
# PROCESS NEW QUESTION
# -----------------------------------------
if (
    prompt
    and st.session_state.pipeline
    and prompt != st.session_state.last_processed_prompt
):

    st.session_state.last_processed_prompt = prompt

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Search
    with st.spinner("🔍 Searching documents..."):
        results = st.session_state.pipeline.query(prompt, top_k=top_k)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        context_parts = []
        sources_collected = []

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            context_parts.append(f"[{i}] {doc}\n")
            sources_collected.append({
                "score": float(dist),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "content": doc
            })

        full_context = "\n".join(context_parts)
        st.session_state.last_sources = sources_collected

    # LLM Response
    with st.spinner("🧠 Generating answer..."):
        answer = query_llm(LLM_MODEL, prompt, full_context)

    # Save bot answer WITH sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_collected
    })

    # Display bot answer
    with st.chat_message("assistant"):
        st.markdown(answer)

        # Speak button (live)
        if st.button("🔊 Speak", key=f"speak_new_{datetime.now().timestamp()}"):
            audio_path = speak_text(answer)
            st.audio(open(audio_path, "rb").read(), format="audio/wav")

        # Display sources (NEW)
        urls = list({s["source"] for s in sources_collected if s["source"]})
        if urls:
            st.markdown("### 🔗 Sources Used")
            for u in urls:
                st.markdown(f"- [{u}]({u})")

    st.rerun()



# -----------------------------------------
# SHOW RETRIEVED CHUNKS (EXPANDER)
# -----------------------------------------
if st.session_state.last_sources:
    with st.expander("📚 Retrieved Chunks (Top 5)", expanded=False):
        for i, src in enumerate(st.session_state.last_sources[:5], start=1):
            st.markdown(f"### Chunk {i}")
            st.markdown(f"**Score:** {src['score']:.4f}")
            st.markdown(f"**Title:** {src['title']}")
            st.markdown(f"**Source URL:** {src['source']}")
            st.text_area(
                label=f"Chunk_{i}_{datetime.now().timestamp()}",
                value=src["content"],
                height=140,
                disabled=True
            )
            st.markdown("---")
