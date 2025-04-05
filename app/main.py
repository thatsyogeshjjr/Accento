import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
import tempfile
from jiwer import wer

st.set_page_config(page_title="Pronunciation Scorer", layout="centered")

# Load pretrained Wav2Vec2 model & processor
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

st.title("🎙️ Real-Time Pronunciation Scorer")
st.markdown("Upload your audio and compare it with correct pronunciation.")

# Target Text
target_text = st.text_input("Enter the sentence you'd like to pronounce:", "The quick brown fox jumps over the lazy dog")

# Generate TTS (target audio) using gTTS
if st.button("Generate TTS for Target Text"):
    tts = gTTS(target_text)
    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(tts_path)
    st.audio(tts_path, format="audio/mp3")
    st.success("TTS Audio Generated Successfully!")

# Upload user audio
uploaded_file = st.file_uploader("Upload your .wav audio", type=["wav"])

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()
    return transcription

# Pronunciation Scoring
if uploaded_file:
    # Save user audio to a temporary file
    user_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio_data, sr = librosa.load(uploaded_file, sr=16000)
    sf.write(user_audio_path, audio_data, sr)

    user_transcription = transcribe(user_audio_path)
    st.markdown("#### 🗣️ Your Transcription:")
    st.write(user_transcription)

    # Compute Word Error Rate (WER)
    error = wer(target_text.lower(), user_transcription)
    score = max(0, int((1 - error) * 100))  # out of 100

    st.markdown("#### 🧠 Pronunciation Score:")
    st.metric(label="Score", value=f"{score}/100")

    if score >= 80:
        st.success("Great pronunciation!")
    elif score >= 50:
        st.warning("Good, but you can improve.")
    else:
        st.error("Needs improvement. Try again!")

