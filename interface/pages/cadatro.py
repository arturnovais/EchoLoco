import streamlit as st
import tempfile, uuid, os, requests
import soundfile as sf
from infra.storage.utils import _upload_to_gcs
import numpy as np

API_URL = "http://0.0.0.0:8000/speaker_registration/"

st.set_page_config(page_title="Cadastro de Locutor",
                   page_icon=":bust_in_silhouette:")

st.title("🎤 Cadastro de novo locutor")

st.markdown(
    """
    Grave pelo menos **5 segundos** de áudio para cadastrar um novo locutor. Você pode gravar em partes.
    """
)

if "recordings" not in st.session_state:
    st.session_state.recordings = []

if "total_duration" not in st.session_state:
    st.session_state.total_duration = 0.0

col1, col2 = st.columns(2)
with col1:
    speaker_name = st.text_input("Nome do locutor", placeholder="Maria da Silva")
audio_chunk = st.audio_input("Grave uma amostra (aperte e fale)", key="mic")

if audio_chunk:
    # Salva temporário e calcula duração
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_chunk.getvalue())
        tmp_path = tmp.name

    data, sr = sf.read(tmp_path)
    duration_sec = len(data) / sr

    st.session_state.recordings.append(tmp_path)
    st.session_state.total_duration += duration_sec

    st.success(f"Amostra gravada: {duration_sec:.1f} s | Total acumulado: {st.session_state.total_duration/60:.2f} min")

    st.audio(audio_chunk)

# Exibe status
st.info(f"⏱️ Total gravado: {st.session_state.total_duration} segundos (mínimo 5 segundos)")

# Botão para enviar
if st.button("📡 Cadastrar locutor"):
    if not speaker_name:
        st.error("⚠️ Informe o nome do locutor.")
        st.stop()
    if st.session_state.total_duration < 5:
        st.error(f"⚠️ Áudio insuficiente: {st.session_state.total_duration} segundos (mínimo 5 segundos).")
        st.stop()

    # Junta os áudios
    combined = []
    for path in st.session_state.recordings:
        data, sr = sf.read(path)
        combined.append(data)

    combined_data = np.concatenate(combined)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as final_file:
        sf.write(final_file.name, combined_data, sr)
        final_path = final_file.name

    # Sobe para GCS
    with st.spinner("🔼 Enviando áudio ao GCS..."):
        gcs_uri = _upload_to_gcs(final_path, dest_prefix="speakers")

    # Limpeza dos temporários
    for p in st.session_state.recordings:
        os.remove(p)
    st.session_state.recordings = []
    st.session_state.total_duration = 0.0

    # Prepara payload
    if not speaker_id:
        speaker_id = uuid.uuid4().hex

    payload = {
        "speaker_name": speaker_name,
        "audio_path": gcs_uri,
        "speaker_id": speaker_id
    }

    with st.spinner("📡 Registrando locutor..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=90)
            resp.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Erro ao registrar locutor: {e}")
            st.stop()

    st.success(f"🎉 Locutor cadastrado com sucesso! ID: `{speaker_id}`")
    st.json(resp.json())
