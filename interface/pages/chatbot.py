import os
import streamlit as st
from utils.audio import (
    transcribe_audio,
    tts_audio,
    chat_completion,
    verify_speaker,
    upload_audio_to_gcs,
)

st.set_page_config(page_title="Chatbot de Voz", page_icon=":microphone:")

st.title("🗣️ Chatbot de Voz Personalizado")

# -----------------------------------------------------------------------------
# Barra lateral – reiniciar conversa + ajustes avançados
# -----------------------------------------------------------------------------
if st.sidebar.button("🔄 Nova conversa"):
    st.session_state.clear()
    st.rerun()

# Expander discreto para ajustes avançados
with st.sidebar.expander("⚙️ Ajustes avançados", expanded=False):
    threshold_default = st.session_state.get("threshold", 0.1)
    threshold_str = st.text_input(
        "Threshold de verificação (0.1 – 0.9)",
        value=f"{threshold_default:.2f}",
        help="Aumente para exigir coincidência mais exata; diminua para ser mais permissivo.",
    )
    try:
        threshold_value = float(threshold_str)
        if not 0.1 <= threshold_value <= 0.9:
            raise ValueError
    except ValueError:
        st.warning("Por favor, digite um número entre 0.1 e 0.9")
        threshold_value = threshold_default

# Salva o threshold escolhido na sessão
st.session_state["threshold"] = threshold_value

# -----------------------------------------------------------------------------
# Estado da sessão – histórico + contador para widget de áudio
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente de voz que se adapta a cada usuário. "
                "Exagere nos detalhes e na personalização sempre que possível."
            ),
        }
    ]

# contador que gera chaves únicas para o widget de gravação
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0  # será incrementado após cada gravação

# -----------------------------------------------------------------------------
# Exibe todo o histórico já trocado
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            # Reexibe áudio da resposta, se existir (previne erros de midia)
            if msg.get("audio_bytes") is not None:
                st.audio(msg["audio_bytes"], format=msg.get("mime", "audio/wav"))
    elif role == "user":
        with st.chat_message("user"):
            if msg.get("speaker_name"):
                st.markdown(
                    f"*Usuário identificado: **{msg['speaker_name']}***",
                    unsafe_allow_html=False,
                )
            elif msg.get("identified") is False:
                st.markdown(
                    "*Usuário não identificado – usando perfil padrão*",
                    unsafe_allow_html=False,
                )
            st.write(msg["content"])

# -----------------------------------------------------------------------------
# Entrada de voz (rodapé) – usa chave única a cada ciclo
# -----------------------------------------------------------------------------
widget_key = f"mic_{st.session_state.audio_key}"
audio_file = st.audio_input(
    "Segure e fale…",
    key=widget_key,
    label_visibility="collapsed",
)

# -----------------------------------------------------------------------------
# Processa nova mensagem de voz (se houver)
# -----------------------------------------------------------------------------
if audio_file is not None:
    with st.spinner("📤 Enviando áudio…"):
        gs_uri, tmp_path = upload_audio_to_gcs(audio_file, dest_prefix="audio")

    try:
        # Identificação do locutor usando threshold configurável
        with st.spinner("🔎 Identificando usuário…"):
            speaker_info = verify_speaker(
                uploaded_file=audio_file,
                gs_uri=gs_uri,
                threshold=st.session_state["threshold"],
            )

        # Transcrição
        with st.spinner("🔎 Transcrevendo…"):
            user_text = transcribe_audio(uploaded_file=None, gs_uri=gs_uri)

        # Mostra o que o usuário disse
        with st.chat_message("user"):
            if speaker_info and speaker_info.get("found_in_bq"):
                st.markdown(
                    f"*Usuário identificado: **{speaker_info['speaker_name']}***",
                    unsafe_allow_html=False,
                )
            else:
                st.markdown(
                    "*Usuário não identificado – usando perfil padrão*",
                    unsafe_allow_html=False,
                )
            st.write(user_text)

        # Atualiza histórico (UI)
        if speaker_info and speaker_info.get("found_in_bq"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_text,
                    "speaker_name": speaker_info["speaker_name"],
                }
            )
            formatted_user_content = (
                f"A mensagem está sendo enviada pelo {speaker_info['speaker_name']}. "
                f"Instruções: {speaker_info.get('instructions', '')}. "
                f"Mensagem: {user_text}"
            )
        else:
            st.session_state.messages.append(
                {"role": "user", "content": user_text, "identified": False}
            )
            formatted_user_content = (
                "A mensagem está sendo enviada por um usuário não identificado. "
                "Instruções: Responda de forma amigável e genérica. "
                f"Mensagem: {user_text}"
            )

        # Prepara todo o histórico para o LLM (sem truncar)
        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        # Substitui o conteúdo da última mensagem do usuário por versão personalizada
        history_for_llm[-1]["content"] = formatted_user_content

        # Chama o LLM
        with st.spinner("🤖 Pensando…"):
            assistant_reply = chat_completion(history_for_llm)

        # Configurações de voz
        voice_settings = {
            "stability": 0.25,
            "similarity_boost": 0.9,
            "use_speaker_boost": True,
            "style": 0.4,
            "speed": 1,
        }

        # Guarda e mostra a resposta
        with st.spinner("🎙️ Gerando voz…"):
            audio_bytes, mime = tts_audio(
                assistant_reply,
                provider="elevenlabs",
                voice_id="ttLrPNfZdNBtVDeOUJsm",
                voice_settings=voice_settings,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_reply,
                "audio_bytes": audio_bytes,
                "mime": mime,
            }
        )

        # Exibe no chat
        with st.chat_message("assistant"):
            st.write(assistant_reply)
            st.audio(audio_bytes, format=mime)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # ------------------------------------------------------------------
        # Prepara a próxima gravação: incrementa a chave e faz rerun
        # ------------------------------------------------------------------
        st.session_state.audio_key += 1
        st.rerun()