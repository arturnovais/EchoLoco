import os, mimetypes
import streamlit as st
from utils.audio import transcribe_audio, tts_audio, chat_completion
from infra.storage.utils import _download_from_gcs

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

    # Mostra mensagem do usuário
    st.chat_message("user").write(text)
    st.session_state.messages.append({"role":"user", "content": text})

    # ---------- 2. Chamada ao LLM ----------
    with st.spinner("🤖 Pensando…"):
        reply = chat_completion(st.session_state.messages)

    st.session_state.messages.append({"role":"assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    # ---------- 3. TTS ----------
    with st.spinner("🎙️ Gerando voz…"):
        gs_uri = tts_audio(reply)
        audio_bytes = _download_from_gcs(gs_uri)
        mime, _ = mimetypes.guess_type(gs_uri)
        mime = mime or "audio/mpeg"

    st.audio(audio_bytes, format=mime)
