import tempfile
from infra.storage.utils import _upload_to_gcs
import requests

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
    return response.json()["audio_path"]