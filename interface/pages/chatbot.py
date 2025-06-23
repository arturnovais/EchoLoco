import streamlit as st
from time import sleep
from utils.audio import transcribe_audio, tts_audio

st.set_page_config(page_title="EchoLoco – Chatbot de Voz",
                   page_icon=":microphone:")

st.title("🗣️ Converse com o EchoLoco")

# Estado de conversa
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system",
                                  "content":"Você é um assistente de voz amigável."}]

# ---------- 1. Entrada de voz ----------
audio_file = st.audio_input("Segure e fale...", key="mic")

if audio_file:                    # quando o usuário soltar o botão
    with st.spinner("🔎 Transcrevendo…"):
        text = transcribe_audio(audio_file)

    # Mostra bolha do usuário
    st.chat_message("user").write(text)
    st.session_state.messages.append({"role":"user", "content": text})

    # ---------- 2. Chamada ao LLM ----------
    with st.spinner("🤖 Pensando…"):
        sleep(5)

    st.session_state.messages.append({"role":"assistant", "content": "teste"})
    st.chat_message("assistant").write("teste")

    # ---------- 3. Síntese de voz ----------
    with st.spinner("🎙️ Gerando voz…"):
        speech_bytes = tts_audio("teste")

    st.audio(speech_bytes, format="audio/mpeg")
