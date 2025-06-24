import streamlit as st
from utils.audio import transcribe_audio, tts_audio, chat_completion, verify_speaker, upload_audio_to_gcs
import os

st.set_page_config(page_title="Chatbot de Voz",
                   page_icon=":microphone:")

st.title("🗣️ Prova de conceito")

# Estado de conversa
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system",
                                  "content":"Você é um assistente de voz que se comporta diferente para cada usuário. Você deve exagerar nos detalhes e personalização de cada usuário."}]

# ---------- 1. Entrada de voz ----------
audio_file = st.audio_input("Segure e fale...", key="mic")

if audio_file:                    # quando o usuário soltar o botão
    # Upload do áudio uma única vez
    with st.spinner("📤 Enviando áudio..."):
        gs_uri, tmp_path = upload_audio_to_gcs(audio_file, dest_prefix="audio")
    
    try:
        speaker_info = None
        with st.spinner("🔎 Identificando usuário…"):
            # Usar o mesmo gs_uri para verificação do speaker
            speaker_info = verify_speaker(uploaded_file=audio_file, gs_uri=gs_uri)
            if speaker_info and speaker_info.get("found_in_bq"):
                st.chat_message("human").write(f"Usuário identificado: {speaker_info['speaker_name']}")
            else:
                st.chat_message("human").write("Usuário não identificado - usando perfil padrão")
        
        with st.spinner("🔎 Transcrevendo…"):
            # Usar o mesmo gs_uri para transcrição
            text = transcribe_audio(uploaded_file=None, gs_uri=gs_uri)

        # Mostra mensagem do usuário
        st.chat_message("user").write(text)
        
        # Formatar mensagem com informações do speaker
        if speaker_info and speaker_info.get("found_in_bq"):
            nome = speaker_info.get("speaker_name", "Usuário desconhecido")
            instrucoes = speaker_info.get("instructions", "Nenhuma instrução específica")
            formatted_message = f"A mensagem está sendo enviada pelo {nome}. Instruções: {instrucoes}. Mensagem: {text}"
        else:
            formatted_message = f"A mensagem está sendo enviada por um usuário não identificado. Instruções: Responda de forma amigável e genérica. Mensagem: {text}"
        
        st.session_state.messages.append({"role":"user", "content": formatted_message})

        # ---------- 2. Chamada ao LLM ----------
        with st.spinner("🤖 Pensando…"):
            reply = chat_completion(st.session_state.messages)

        st.session_state.messages.append({"role":"assistant", "content": reply})

        # ---------- 3. TTS ----------
        with st.spinner("🎙️ Gerando voz…"):
            audio_bytes, mime = tts_audio(reply)

        st.audio(audio_bytes, format=mime)
        
    finally:
        # Limpar arquivo temporário
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
