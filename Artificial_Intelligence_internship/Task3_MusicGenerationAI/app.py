import os
import time
import wave
import numpy as np
import streamlit as st
from music21 import converter, note, chord
from generate_music import generate_music

st.set_page_config(
    page_title="Music Generation with AI",
    page_icon="🎵",
    layout="wide"
)

# -----------------------------
# File Paths
# -----------------------------
midi_file = "generated_music/generated_music.mid"
wav_file = "generated_music/generated_music.wav"

# -----------------------------
# MIDI to WAV Converter
# -----------------------------
def midi_to_wav(midi_path, wav_path, sample_rate=44100):
    midi_stream = converter.parse(midi_path)

    audio = np.array([], dtype=np.float32)

    for element in midi_stream.flat.notes:
        duration = float(element.duration.quarterLength)
        duration_seconds = max(duration * 0.5, 0.25)

        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)

        if isinstance(element, note.Note):
            frequency = element.pitch.frequency
            sound = 0.4 * np.sin(2 * np.pi * frequency * t)

        elif isinstance(element, chord.Chord):
            sound = np.zeros_like(t)
            for pitch in element.pitches:
                frequency = pitch.frequency
                sound += 0.25 * np.sin(2 * np.pi * frequency * t)

        else:
            continue

        fade_length = min(500, len(sound))
        sound[:fade_length] *= np.linspace(0, 1, fade_length)
        sound[-fade_length:] *= np.linspace(1, 0, fade_length)

        audio = np.concatenate((audio, sound))

    if len(audio) == 0:
        raise ValueError("No playable notes found in MIDI file.")

    audio = audio / np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)

    with wave.open(wav_path, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}

section[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(135deg, #0F172A, #1E3A8A);
    padding: 48px 32px;
    border-radius: 26px;
    text-align: center;
    color: white;
    box-shadow: 0px 12px 32px rgba(15, 23, 42, 0.28);
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 900;
    margin-bottom: 12px;
}

.hero p {
    font-size: 19px;
    color: #CBD5E1;
}

.info-card {
    background: #FFFFFF;
    padding: 26px;
    border-radius: 22px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid #E2E8F0;
    min-height: 150px;
    margin-bottom: 24px;
}

.info-card h3 {
    color: #0F172A;
    font-size: 24px;
    margin-bottom: 10px;
}

.info-card p {
    color: #475569;
    font-size: 16px;
    line-height: 1.6;
}

.generate-panel {
    background: #FFFFFF;
    padding: 46px 36px;
    border-radius: 30px;
    text-align: center;
    box-shadow: 0px 16px 42px rgba(15, 23, 42, 0.15);
    border: 2px solid #CBD5E1;
    margin-top: 12px;
    margin-bottom: 24px;
}

.generate-panel h2 {
    color: #0F172A;
    font-size: 40px;
    font-weight: 900;
    margin-bottom: 10px;
}

.generate-panel p {
    color: #64748B;
    font-size: 18px;
}

.stButton > button {
    background: linear-gradient(135deg, #14B8A6, #2563EB);
    color: white;
    border: none;
    border-radius: 18px;
    height: 82px;
    font-size: 28px;
    font-weight: 900;
    box-shadow: 0px 14px 30px rgba(37, 99, 235, 0.35);
    transition: 0.25s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0F766E, #1D4ED8);
    color: white;
    transform: translateY(-2px) scale(1.01);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0F172A, #334155);
    color: white;
    border: none;
    border-radius: 14px;
    height: 58px;
    font-size: 18px;
    font-weight: 800;
}

.result-box {
    background: #ECFDF5;
    padding: 24px;
    border-radius: 20px;
    border-left: 7px solid #10B981;
    margin-top: 24px;
    margin-bottom: 18px;
    color: #065F46;
    font-size: 18px;
    font-weight: 600;
}

.audio-card {
    background: #FFFFFF;
    padding: 26px;
    border-radius: 22px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid #E2E8F0;
    margin-top: 18px;
}

.audio-card h3 {
    color: #0F172A;
    font-size: 25px;
    margin-bottom: 8px;
}

.audio-card p {
    color: #64748B;
    font-size: 15px;
}

.warning-box {
    background: #FEF3C7;
    padding: 20px;
    border-radius: 18px;
    border-left: 7px solid #F59E0B;
    color: #92400E;
    font-size: 17px;
    margin-top: 18px;
}

.footer {
    text-align: center;
    margin-top: 38px;
    padding: 18px;
    color: #64748B;
    font-size: 14px;
}

@media screen and (max-width: 768px) {
    .hero h1 {
        font-size: 34px;
    }

    .hero p {
        font-size: 15px;
    }

    .generate-panel h2 {
        font-size: 28px;
    }

    .stButton > button {
        height: 68px;
        font-size: 21px;
    }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎵 AI Music Generator")
st.sidebar.write("**CodeAlpha AI Internship**")
st.sidebar.write("**Task 3:** Music Generation with AI")

st.sidebar.markdown("---")
st.sidebar.write("### Built With")
st.sidebar.write("Python")
st.sidebar.write("Streamlit")
st.sidebar.write("music21")
st.sidebar.write("TensorFlow")
st.sidebar.write("LSTM Model")

# -----------------------------
# File Checks
# -----------------------------
model_exists = os.path.exists("music_model.keras")
notes_exists = os.path.exists("notes.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎵 Music Generation with AI</h1>
    <p>Generate, play, and download AI-created music using a trained LSTM model.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Info Cards
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📌 Project Overview</h3>
        <p>This app creates new music from patterns learned from a MIDI dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ AI Process</h3>
        <p>The LSTM model learns note sequences and generates a new music pattern.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Generate Music
# -----------------------------
st.markdown("""
<div class="generate-panel">
    <h2>🎼 Generate AI Music</h2>
    <p>Click below to create a new track and play it directly in the app.</p>
</div>
""", unsafe_allow_html=True)

generate_btn = st.button("🚀 GENERATE MUSIC NOW", use_container_width=True)

if generate_btn:
    if not model_exists or not notes_exists:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Model files are missing. Please run <b>python train_model.py</b> first.
        </div>
        """, unsafe_allow_html=True)

    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.write("🎵 Loading trained model...")
            progress_bar.progress(25)
            time.sleep(0.4)

            status_text.write("🎹 Generating MIDI music...")
            progress_bar.progress(55)
            generate_music()

            status_text.write("🔊 Converting MIDI to playable audio...")
            progress_bar.progress(80)
            midi_to_wav(midi_file, wav_file)

            progress_bar.progress(100)
            status_text.write("✅ Music generation completed!")

            st.markdown("""
            <div class="result-box">
                ✅ Music generated successfully! You can play it below or download the MIDI file.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error while generating music: {e}")

# -----------------------------
# Play + Download Section
# -----------------------------
if os.path.exists(wav_file):
    st.markdown("""
    <div class="audio-card">
        <h3>🎧 Play Generated Music</h3>
        <p>This audio player uses a WAV version of the generated MIDI, so it can play directly in the browser.</p>
    </div>
    """, unsafe_allow_html=True)

    with open(wav_file, "rb") as audio:
        st.audio(audio.read(), format="audio/wav")

if os.path.exists(midi_file):
    with open(midi_file, "rb") as file:
        st.download_button(
            label="⬇️ Download Generated MIDI",
            data=file,
            file_name="generated_music.mid",
            mime="audio/midi",
            use_container_width=True
        )

# -----------------------------
# Simple Status
# -----------------------------
with st.expander("Check Project Files"):
    st.write("✅ music_model.keras" if model_exists else "❌ music_model.keras missing")
    st.write("✅ notes.pkl" if notes_exists else "❌ notes.pkl missing")
    st.write("✅ generated_music.mid" if os.path.exists(midi_file) else "❌ generated_music.mid not generated yet")
    st.write("✅ generated_music.wav" if os.path.exists(wav_file) else "❌ generated_music.wav not generated yet")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Developed for <b>CodeAlpha Artificial Intelligence Internship</b> | Task 3: Music Generation with AI
</div>
""", unsafe_allow_html=True)