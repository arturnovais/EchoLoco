import tempfile
from infra.storage.utils import _upload_to_gcs, _download_from_gcs
from infra.bq.bq_client import query
import requests
import mimetypes
import soundfile as sf
import os

API_URL = "http://0.0.0.0:8000"

# Configuração global do TTS - altere aqui para mudar o provider padrão
DEFAULT_TTS_PROVIDER = "pretrained"  # "pretrained" ou "elevenlabs"
DEFAULT_VOICE_ID = "ttLrPNfZdNBtVDeOUJsm"  # ID da voz para ElevenLabs
DEFAULT_VOICE_SETTINGS = {}  # Configurações padrão para ElevenLabs

def set_default_tts_provider(provider, voice_id=None, voice_settings=None):
    """
    Função utilitária para alterar o provider padrão de TTS em tempo de execução.
    
    Args:
        provider (str): "pretrained" ou "elevenlabs"
        voice_id (str): ID da voz para ElevenLabs (opcional)
        voice_settings (dict): Configurações de voz para ElevenLabs (opcional)
    """
    global DEFAULT_TTS_PROVIDER, DEFAULT_VOICE_ID, DEFAULT_VOICE_SETTINGS
    
    if provider not in ["pretrained", "elevenlabs"]:
        raise ValueError(f"Provider '{provider}' não suportado. Use 'pretrained' ou 'elevenlabs'")
    
    DEFAULT_TTS_PROVIDER = provider
    if voice_id is not None:
        DEFAULT_VOICE_ID = voice_id
    if voice_settings is not None:
        DEFAULT_VOICE_SETTINGS = voice_settings

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
        response.raise_for_status()
        
        result = response.json()
        transcript = result.get("text", "")
        
        if not transcript:
            raise ValueError("Transcrição não retornou texto")
            
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
    response.raise_for_status()
    
    result = response.json()
    assistant_text = result.get("assistant_text", "")
    
    if not assistant_text:
        raise ValueError("Resposta do assistente está vazia")
        
    return assistant_text

def tts_audio(text, provider=None, voice_id=None, voice_settings=None):
    """
    Gera áudio a partir de texto usando TTS.
    
    Args:
        text (str): Texto para sintetizar
        provider (str): "pretrained" ou "elevenlabs" (usa DEFAULT_TTS_PROVIDER se None)
        voice_id (str): ID da voz para ElevenLabs (usa DEFAULT_VOICE_ID se None)
        voice_settings (dict): Configurações de voz para ElevenLabs (usa DEFAULT_VOICE_SETTINGS se None)
        
    Returns:
        tuple: (audio_bytes, mime_type)
    """
    if not text or not text.strip():
        raise ValueError("Texto para TTS não pode estar vazio")
    
    # Usa valores padrão se não fornecidos
    provider = provider or DEFAULT_TTS_PROVIDER
    voice_id = voice_id or DEFAULT_VOICE_ID
    voice_settings = voice_settings or DEFAULT_VOICE_SETTINGS
    
    # Define endpoint e payload baseado no provider
    if provider == "pretrained":
        endpoint = f"{API_URL}/tts/pretrained"
        payload = {"text": text}
    elif provider == "elevenlabs":
        endpoint = f"{API_URL}/tts/elevenlabs"
        payload = {
            "text": text,
            "voice_id": voice_id,
            "model_id": "21m00Tcm4TlvDq8ikWAM",
            "voice_settings": voice_settings or {}
        }
    else:
        raise ValueError(f"Provider '{provider}' não suportado. Use 'pretrained' ou 'elevenlabs'")
        
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    
    result = response.json()
    audio_path = result.get("audio_path")
    
    if not audio_path:
        raise ValueError("Caminho do áudio não foi retornado pela API")
        
    audio_bytes = _download_from_gcs(audio_path)
    
    # Determina mime type pelo nome do arquivo
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/wav"
    
    return audio_bytes, mime

def verify_speaker(uploaded_file=None, gs_uri=None, threshold=0.45):
    """
    Verifica o speaker de um arquivo de áudio e retorna suas informações.
    
    Args:
        uploaded_file: Arquivo de áudio carregado (opcional se gs_uri fornecido)
        gs_uri: URI do GCS do áudio (opcional se uploaded_file fornecido)
        threshold: Threshold para verificação do speaker (padrão: 0.45)
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
            json={"audio_path": gs_uri, "threshold": threshold},
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        speaker_id = result.get("speaker_id")
        confidence = result.get("score")

        # Verifica se speaker_id é válido (não None, não vazio e é string)
        if not speaker_id or not isinstance(speaker_id, str) or not speaker_id.strip():
            return None

        # Remove hífens do speaker_id para consulta no BigQuery, com validação adicional
        try:
            speaker_id_clean = speaker_id.replace("-", "")
            if not speaker_id_clean:
                return None
        except (AttributeError, TypeError):
            # Se speaker_id não for string válida
            return None

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
            "speaker_name": row.get("speaker_name"),
            "instructions": row.get("instructions"),
            "found_in_bq": True
        }

    except requests.RequestException as e:
        # Erro na requisição HTTP
        print(f"Erro na requisição para verificação de speaker: {e}")
        return None
    except Exception as e:
        # Outros erros
        print(f"Erro inesperado na verificação de speaker: {e}")
        return None
    finally:
        # Remove arquivo temporário se foi criado aqui
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)