import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os

st.set_page_config(
    page_title="Language Translation Tool",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Language Translation Tool")
st.write("Translate text from one language to another using an AI translation API.")

languages = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Turkish": "tr"
}

text = st.text_area("Enter text to translate:", height=150)

col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox("Select Source Language", list(languages.keys()))

with col2:
    target_language = st.selectbox("Select Target Language", list(languages.keys()), index=1)


def translate_text(text, source, target):
    try:
        url = "https://api.mymemory.translated.net/get"

        params = {
            "q": text,
            "langpair": f"{source}|{target}"
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            result = response.json()
            return result["responseData"]["translatedText"]
        else:
            return "Translation failed. Please try again."

    except requests.exceptions.ConnectionError:
        return "Connection error. Please check your internet connection."

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."

    except Exception as e:
        return f"Error: {e}"


if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text first.")

    elif source_language == target_language:
        st.warning("Source and target languages should be different.")

    else:
        with st.spinner("Translating..."):
            translated_text = translate_text(
                text,
                languages[source_language],
                languages[target_language]
            )

        st.subheader("Translated Text:")
        st.success(translated_text)

        st.text_area("Copy Translation:", translated_text, height=120)

        try:
            tts = gTTS(text=translated_text, lang=languages[target_language])
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)

            audio_file = open(temp_file.name, "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            audio_file.close()
            os.remove(temp_file.name)

        except Exception:
            st.info("Text-to-speech is not available for this language.")