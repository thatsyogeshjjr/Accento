import streamlit as st
import pyttsx3  # For Text-to-Speech

# Initialize the TTS engine
engine = pyttsx3.init()

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
    engine.say(selected_word)
    engine.runAndWait()

# Recording section
st.subheader("Record Pronunciation")
audio_value = st.audio_input("Record your pronunciation")

if audio_value:
    st.audio(audio_value)
    # Here, you can add functionality to process the recorded audio if needed

# List of words
st.subheader("List of Words")
for word in vocabulary.keys():
    if st.button(word, key=word, on_click=update_word, args=(word,)):
        pass


