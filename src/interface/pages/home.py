import streamlit as st

st.set_page_config(
    page_title="EchoLoco - Bem-vindo",
    page_icon=":wave:",
    layout="centered"
)

st.title("👋 Bem-vindo ao EchoLoco!")

st.markdown(
    """
    ## Sistema de Voz Inteligente

    Este sistema utiliza tecnologias de síntese e reconhecimento de voz para cadastro, identificação e manipulação de áudios.

    **Funcionalidades principais:**
    - 🔊 Conversão de texto em fala (TTS)
    - 🗣️ Cadastro e busca de locutores por similaridade de voz
    - 🎧 Player de áudio integrado

    ---
    Navegue pelo menu lateral para acessar as funcionalidades.
    """
)

st.image("https://cdn-icons-png.flaticon.com/512/943/943239.png", width=120)  # Exemplo de ícone, pode trocar

st.info("Selecione uma opção no menu lateral para começar.")
