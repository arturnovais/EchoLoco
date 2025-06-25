# interface/pages/verificar_locutor.py
import streamlit as st
import tempfile, os, requests, soundfile as sf
from infra.storage.utils import _upload_to_gcs
from infra.bq.bq_client import query

API_URL  = "http://0.0.0.0:8000/speaker/"
BQ_TABLE = "speech_chatbot.system_prompts"

st.set_page_config(page_title="Verificação de Locutor",
                   page_icon=":detective:")

st.title("🔍 Verificação de locutor")

st.markdown(
    """
    Grave um trecho de voz; o sistema identificará o locutor já cadastrado e mostrará 
    as informações correspondentes.
    """
)

# ---------- captura de áudio ----------
audio_chunk = st.audio_input("Grave um trecho (≥ 2 s)", key="verif_mic")

if audio_chunk:
    # salva temporário para medir tempo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_chunk.getvalue())
        tmp_path = tmp.name

    data, sr = sf.read(tmp_path)
    duration_sec = len(data) / sr

    if duration_sec < 2:
        os.remove(tmp_path)
        st.error("⚠️ Grave pelo menos 2 segundos.")
        st.stop()

    st.audio(audio_chunk, format="audio/wav")
    st.success(f"Áudio capturado ({duration_sec:.1f} s)")

    # ---------- upload para GCS ----------
    with st.spinner("🔼 Enviando áudio para o GCS…"):
        gcs_uri = _upload_to_gcs(tmp_path, dest_prefix="verify")

    # ---------- chamada à API de verificação ----------
    payload = {"audio_path": gcs_uri}

    with st.spinner("🔍 Verificando locutor…"):
        try:
            resp = requests.post(API_URL, json=payload, timeout=90)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            st.error(f"Erro na verificação: {e}")
            st.stop()

    speaker_id = result.get("speaker_id")
    confidence = result.get("score")

    # Verifica se speaker_id é válido antes de tentar usar replace
    if not speaker_id or not isinstance(speaker_id, str) or not speaker_id.strip():
        st.error("Esse locutor não está cadastrado. Eu não erro!")
        st.stop()

    # Remove hífens do speaker_id para consulta no BigQuery, com validação adicional
    try:
        speaker_id_clean = speaker_id.replace("-", "")
        if not speaker_id_clean:
            st.error("ID do locutor inválido após limpeza.")
            st.stop()
    except (AttributeError, TypeError):
        st.error("Erro ao processar ID do locutor.")
        st.stop()

    # ---------- consulta ao BigQuery ----------

    sql = f"""SELECT *
    FROM {BQ_TABLE}
    WHERE speaker_id = '{speaker_id_clean}'
    LIMIT 1;
    """

    bq_result = query(sql)
    row_list = list(bq_result)

    if not row_list:
        st.warning(f"Locutor identificado (ID: {speaker_id}), mas não encontrado no BigQuery.")
        st.stop()

    row = row_list[0]
    speaker_name = row.get("speaker_name", "Nome não disponível")
    instructions = row.get("instructions", "Instruções não disponíveis")

    # ---------- exibição ----------
    st.success("✅ Locutor identificado!")
    st.markdown(f"**ID:** `{speaker_id}`")
    st.markdown(f"**Nome:** {speaker_name}")
    st.markdown(f"**Instruções:** {instructions}")
    if confidence is not None:
        st.markdown(f"**Confiança:** {confidence}")
