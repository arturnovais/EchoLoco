import tempfile
from infra.storage.utils import _upload_to_gcs, _download_from_gcs
import requests
import mimetypes

def transcribe_audio(uploaded_file):
    # Salva temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Faz o upload para o GCS
    gs_uri = _upload_to_gcs(tmp_path, dest_prefix="stt")

    # Faz request para API de STT
    response = requests.post(
        "http://0.0.0.0:8000/stt",
        json={"audio_path": gs_uri}
    )
    
    transcript = response.json()["text"]

    return transcript

def chat_completion(history):
    response = requests.post(
        "http://0.0.0.0:8000/assistant/reply",
        #TODO: No momento, o history é uma lista de dicionários que está sendo passado como string.
        # É necessário passar isso da forma correta.
        json={"user_text": f"{history}"}
    )
    return response.json()["assistant_text"]

def tts_audio(text):
    response = requests.post(
        "http://0.0.0.0:8000/tts",
        json={"text": text}
    )
    audio_path = response.json()["audio_path"]
    audio_bytes = _download_from_gcs(audio_path)
    
    # Determina mime type pelo nome do arquivo
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/wav"
    
    return audio_bytes, mime