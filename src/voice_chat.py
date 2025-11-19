import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import edge_tts
import asyncio
from playsound import playsound
from rag_search import answer_query  # we will create this next

############################
#   1. LOAD WHISPER MODEL  #
############################
print("Loading Whisper STT model... (this may take 30 sec)")
model = whisper.load_model("small")

############################
#   2. RECORD MICROPHONE   #
############################
def record_audio(seconds=5, fs=44100):
    print("🎙 Speak now...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    print("🎤 Recording stopped")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, fs)
    return tmp.name

############################
#   3. SPEECH → TEXT       #
############################
def speech_to_text(audio_path):
    print("🧠 Transcribing...")
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    print(f"🗣 You said: {text}")
    return text

############################
#   4. TEXT → SPEECH       #
############################
async def text_to_speech(text):
    print("🔊 Speaking...")
    voice = "en-US-JennyNeural"
    tts = edge_tts.Communicate(text, voice)
    
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    await tts.save(out.name)
    playsound(out.name)

############################
#   5. CHAT LOOP           #
############################
def voice_chat_loop():
    print("\n🎧 Voice Assistant Ready!")
    print("Say something... (or 'stop' to exit)\n")

    while True:
        audio_path = record_audio()

        query = speech_to_text(audio_path)

        if query.lower() in ["stop", "exit", "quit"]:
            print("👋 Goodbye!")
            break

        # RAG ANSWER HERE
        response = answer_query(query)

        print(f"🤖 BOT: {response}")

        asyncio.run(text_to_speech(response))

if __name__ == "__main__":
    voice_chat_loop()
