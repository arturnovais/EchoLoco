import tempfile
from infra.storage.utils import _upload_to_gcs, _download_from_gcs
from infra.bq.bq_client import query
import requests
import mimetypes
import soundfile as sf
import os

API_URL = "http://0.0.0.0:8000"

def upload_audio_to_gcs(uploaded_file, dest_prefix="audio"):
    """
    Faz upload de um arquivo de áudio para o GCS.
    
    Args:
        uploaded_file: Arquivo de áudio carregado
        dest_prefix: Prefixo para o destino no GCS
        
    Returns:
        tuple: (gs_uri, tmp_path) - URI do GCS e caminho do arquivo temporário
    """
    # Salva temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Faz o upload para o GCS
    gs_uri = _upload_to_gcs(tmp_path, dest_prefix=dest_prefix)
    
    return gs_uri, tmp_path

def transcribe_audio(uploaded_file, gs_uri=None):
    """
    Transcreve um arquivo de áudio.
    
    Args:
        uploaded_file: Arquivo de áudio carregado (opcional se gs_uri fornecido)
        gs_uri: URI do GCS do áudio (opcional se uploaded_file fornecido)
        
    Returns:
        str: Texto transcrito
    """
    tmp_path = None
    
    try:
        if gs_uri is None:
            if uploaded_file is None:
                raise ValueError("É necessário fornecer uploaded_file ou gs_uri")
            gs_uri, tmp_path = upload_audio_to_gcs(uploaded_file, dest_prefix="stt")

        # Faz request para API de STT
        response = requests.post(
            f"{API_URL}/stt",
            json={"audio_path": gs_uri}
        )
        
        transcript = response.json()["text"]
        return transcript
        
    finally:
        # Remove arquivo temporário se foi criado aqui
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def chat_completion(history):
    response = requests.post(
        f"{API_URL}/assistant/reply_api",
        json={"messages": history}
    )
    return response.json()["assistant_text"]

def tts_audio(text):
    response = requests.post(
        f"{API_URL}/tts",
        json={"text": text}
    )
    audio_path = response.json()["audio_path"]
    audio_bytes = _download_from_gcs(audio_path)
    
    # Determina mime type pelo nome do arquivo
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/wav"
    
    return audio_bytes, mime

def verify_speaker(uploaded_file=None, gs_uri=None):
    """
    Verifica o speaker de um arquivo de áudio e retorna suas informações.
    
    Args:
        uploaded_file: Arquivo de áudio carregado (opcional se gs_uri fornecido)
        gs_uri: URI do GCS do áudio (opcional se uploaded_file fornecido)
        
    Returns:
        dict: Informações do speaker verificado ou None se não identificado
    """
    tmp_path = None
    
    try:
        if gs_uri is None:
            if uploaded_file is None:
                raise ValueError("É necessário fornecer uploaded_file ou gs_uri")
            
            # Salva temporariamente para verificar duração
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Verifica duração do áudio
            data, sr = sf.read(tmp_path)

            # Faz o upload para o GCS
            gs_uri = _upload_to_gcs(tmp_path, dest_prefix="verify")
        elif uploaded_file is not None:
            # Se gs_uri foi fornecido mas também temos uploaded_file, 
            # ainda precisamos verificar a duração
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            data, sr = sf.read(tmp_path)

        # Faz request para API de verificação
        response = requests.post(
            f"{API_URL}/speaker/",
            json={"audio_path": gs_uri},
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        speaker_id = result.get("speaker_id")
        confidence = result.get("score")

        if not speaker_id:
            return None

        # Remove hífens do speaker_id para consulta no BigQuery
        speaker_id_clean = speaker_id.replace("-", "")

        # Consulta ao BigQuery
        sql = f"""SELECT *
        FROM speech_chatbot.system_prompts
        WHERE speaker_id = '{speaker_id_clean}'
        LIMIT 1;
        """

        bq_result = query(sql)
        row_list = list(bq_result)

        if not row_list:
            return {
                "speaker_id": speaker_id,
                "confidence": confidence,
                "speaker_name": None,
                "instructions": None,
                "found_in_bq": False
            }

        row = row_list[0]
        return {
            "speaker_id": speaker_id,
            "confidence": confidence,
            "speaker_name": row["speaker_name"],
            "instructions": row["instructions"],
            "found_in_bq": True
        }

    finally:
        # Remove arquivo temporário se foi criado aqui
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)