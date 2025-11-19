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
# VOICE RECORDING
# -----------------------------------------
st.subheader("🎙 Voice Input")

audio = audiorecorder("🎤 Start Recording", "⏹ Stop")
spoken_text = None

if len(audio) > 0:
    wav_bytes = audio.export(format="wav").read()

    with st.spinner("Transcribing..."):
        spoken_text = convert_audio_to_text(wav_bytes)

    st.success(f"🗣 You said: **{spoken_text}**")



# -----------------------------------------
# TEXT INPUT ALWAYS AVAILABLE
# -----------------------------------------
typed_text = st.chat_input("Type here...")

prompt = spoken_text if spoken_text else typed_text



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
            # Create a mapping of URLs to titles for better display
            url_to_info = {}
            for s in msg["sources"]:
                url = s.get("source", "")
                if url and url not in url_to_info:
                    title = s.get("title", "").strip()
                    url_to_info[url] = title if title else url.split("/")[-1] or "Source"

            if url_to_info:
                st.markdown("### 🔗 Sources Used")
                for url, display_text in url_to_info.items():
                    st.markdown(f"- [{display_text}]({url})")



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
        # Create a mapping of URLs to titles for better display
        url_to_info = {}
        for s in sources_collected:
            url = s.get("source", "")
            if url and url not in url_to_info:
                title = s.get("title", "").strip()
                url_to_info[url] = title if title else url.split("/")[-1] or "Source"
        
        if url_to_info:
            st.markdown("### 🔗 Sources Used")
            for url, display_text in url_to_info.items():
                st.markdown(f"- [{display_text}]({url})")

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
