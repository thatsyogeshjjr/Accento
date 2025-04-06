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

# Sample vocabulary words and their meanings
vocabulary = {
    "abate": "to become less intense or widespread.",
    "benevolent": "well-meaning and kindly.",
    "candid": "truthful and straightforward; frank.",
    "debilitate": "to make someone weak.",
    "ebullient": "cheerful and full of energy.",
    "facetious": "treating serious issues with deliberately inappropriate humor.",
    "garrulous": "excessively talkative, especially on trivial matters.",
    "haphazard": "lacking any obvious principle of organization.",
    "juxtaposition": "the fact of two things being seen or placed close together.",
    "keen": "having or showing eagerness or enthusiasm.",
    "loquacious": "tending to talk a great deal; talkative.",
    "meticulous": "showing great attention to detail; very careful and precise.",
    "nefarious": "wicked or criminal.",
    "obfuscate": "to deliberately make something difficult to understand.",
    "pugnacious": "eager or quick to argue, quarrel, or fight.",
    "quintessential": "representing the most perfect or typical example of a quality or class.",
    "rambunctious": "uncontrollably exuberant; boisterous.",
    "sagacious": "having or showing keen mental discernment and good judgment.",
    "tenacious": "holding firm to a purpose or opinion.",
    "ubiquitous": "present, appearing, or found everywhere.",
    "vicarious": "experienced in the imagination through the feelings or actions of another person.",
    "wary": "feeling or showing caution about possible dangers or problems.",
    "zealous": "having or showing zeal; fervent.",
    "ambiguous": "open to more than one interpretation; not having one obvious meaning.",
    "benevolence": "the quality of being well-meaning.",
    "cacophony": "a harsh, discordant mixture of sounds.",
    "deference": "humble submission and respect.",
    "empathy": "the ability to understand and share the feelings of another.",
    "frivolous": "not having any serious purpose or value.",
    "gregarious": "fond of company; sociable.",
    "hubris": "excessive pride or self-confidence.",
}

# Initialize session state for selected word
if 'selected_word' not in st.session_state:
    st.session_state.selected_word = list(vocabulary.keys())[0]

# Layout
st.title("Vocabulary Builder")

# Function to update the selected word
def update_word(word):
    st.session_state.selected_word = word

# Section for highlighted word
selected_word = st.session_state.selected_word
st.header(selected_word)
st.markdown(f"*{selected_word}*")
st.write("Meaning:", vocabulary[selected_word])

# Button to play TTS
if st.button("Listen to Pronunciation"):
    tts = gTTS(selected_word)
    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(tts_path)
    st.audio(tts_path, format="audio/mp3")
    st.success("TTS Audio Generated Successfully!")
    

# Recording section
st.subheader("Record Pronunciation")
audio_input = st.audio_input("Record your pronunciation")

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
    error = wer(selected_word.lower(), user_transcription)
    score = max(0, int((1 - error) * 100))  # out of 100

    st.markdown("#### 🧠 Pronunciation Score:")
    st.metric(label="Score", value=f"{score}/100")

    if score >= 80:
        st.success("Great pronunciation!")
    elif score >= 50:
        st.warning("Good, but you can improve.")
    else:
        st.error("Needs improvement. Try again!")
# List of words
st.markdown("""
    <style>
        .vocab-button-container {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: flex-start;
        }
        .vocab-button {
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
            transition: background-color 0.3s;
        }
        .vocab-button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

st.subheader("List of Words")
st.markdown('<div class="vocab-button-container">', unsafe_allow_html=True)
for word in vocabulary.keys():
    if st.button(word, key=word, on_click=update_word, args=(word,)):
        pass
st.markdown('</div>', unsafe_allow_html=True)    


