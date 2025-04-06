import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import numpy as np
import tempfile
from jiwer import wer
import random

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

# Sample list of 50 common British English phrases
phrases = [
      "Good morning!",
    "How are you?",
    "Have a great day!",
    "Thank you very much.",
    "Excuse me, where is the nearest restroom?",
    "I'm sorry, I didn't mean to offend you.",
    "Can you help me with this task?",
    "I don't understand.",
    "Could you please repeat that?",
    "What time is it?",
    "How much does this cost?",
    "Can I have the bill, please?",
    "Do you have any recommendations?",
    "I'm looking forward to the weekend.",
    "Let's go for a walk.",
    "I'm feeling tired today.",
    "It's a beautiful day!",
    "Could you please pass the salt?",
    "I need a vacation.",
    "Congratulations on your success!",
    "I'm sorry to hear that.",
    "Where can I find a taxi?",
    "This food tastes delicious.",
    "I'm running late.",
    "I can't find my keys.",
    "Can you give me a hand?",
    "That's a great idea!",
    "I'm so excited!",
    "Have a safe trip!",
    "What are your plans for today?",
    "Let's meet at 5 PM.",
    "Are you free this weekend?",
    "I'll let you know.",
    "Sorry, I'm busy.",
    "Maybe next time.",
    "That works for me.",
    "Where should we go?",
    "I had a busy day.",
    "Let's take a break.",
    "I need to clean my room.",
    "I forgot my keys!",
    "My phone battery is low.",
    "It's raining outside.",
    "The weather is nice today.",
    "It's getting colder.",
    "I love sunny days!",
    "It's very windy today.",
    "I hope it doesn't rain.",
    "It's freezing!",
    "I didn't quite catch that. Could you repeat it?",
    "Please forgive me.",
]

# Initialize session state
if 'remaining_phrases' not in st.session_state:
    st.session_state.remaining_phrases = phrases.copy()
    st.session_state.current_phrase = random.choice(st.session_state.remaining_phrases)

# Layout
st.title("British English Phrase Pronunciation Practice")

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

# Function to update the current phrase
def update_phrase():
    if st.session_state.remaining_phrases:
        st.session_state.current_phrase = random.choice(st.session_state.remaining_phrases)
    else:
        st.session_state.current_phrase = None

# Section for current phrase
current_phrase = st.session_state.current_phrase

if current_phrase:
    st.header(current_phrase)

    # Button to play TTS
    if st.button("Listen to Pronunciation"):
        tts = gTTS(current_phrase)
        tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(tts_path)
        st.audio(tts_path, format="audio/mp3")
        st.success("TTS Audio Generated Successfully!")

    # Recording section
    st.subheader("Record Your Pronunciation")
    audio_input = st.audio_input("Record your pronunciation")

    # if audio_input:
    #     # Placeholder for transcription and scoring logic
    #     # user_transcription = transcribe(audio_input.read())
    #     # score = evaluate_pronunciation(current_phrase, user_transcription)
    #     score = 85  # Placeholder score for demonstration

    


    if audio_input:
        user_transcription = transcribe(audio_input.read())
        st.markdown("#### 🗣️ Your Transcription:")
        st.write(user_transcription)

        # Compute Word Error Rate (WER)
        error = wer(current_phrase.lower(), user_transcription)
        score = max(0, int((1 - error) * 100))  # out of 100


        if score >= 80:
            st.success("Great pronunciation!")
            st.session_state.remaining_phrases.remove(current_phrase)
            update_phrase()
        elif score >= 50:
            st.warning("Good, but you can improve.")
        else:
            st.error("It'k ok. We all learn")

            st.markdown("#### 🧠 Pronunciation Score:")
            st.metric(label="Score", value=f"{score}/100")


else:
    st.success("Congratulations! You've mastered all 50 phrases.")
