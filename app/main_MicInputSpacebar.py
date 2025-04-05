import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import numpy as np
import tempfile
from jiwer import wer

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()


st.title("🎙️ Real-Time Pronunciation Scorer")
st.markdown("Record your pronunciation and compare it with the correct pronunciation.")
target_text = st.text_input("Enter the sentence you'd like to pronounce:", "The quick brown fox jumps over the lazy dog")


if st.button("Generate TTS for Target Text"):
    tts = gTTS(target_text)
    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(tts_path)
    st.audio(tts_path, format="audio/mp3")
    st.success("TTS Audio Generated Successfully!")
st.markdown("## 🎤 Record Your Pronunciation")
audio_input = st.audio_input("Press 'Start recording' to record your pronunciation.")



def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    # Load the audio file
    waveform, sample_rate = torchaudio.load(tmpfile_path, normalize=True)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if not already
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Ensure the waveform is 2D: [1, num_samples]
    waveform = waveform.squeeze(0)

    # Process the waveform
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()

    return transcription


if audio_input:
    user_transcription = transcribe(audio_input.read())
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

add_keyboard_shortcuts({
    ' ': 'Start Recording',
})