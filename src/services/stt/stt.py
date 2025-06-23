import os
import tempfile
from typing import Dict, Tuple, Literal

from utils.load_config import load_config
from infra.storage import gcs_client

# openai só é importado se for necessário
try:
    import openai
except ModuleNotFoundError:
    openai = None

try:
    from transformers import pipeline
except ModuleNotFoundError:
    pipeline = None

def load_stt_pipeline(config_path: str | None = None) -> Tuple[Literal["huggingface", "openai"], object, Dict]:
    cfg = load_config(config_path) if config_path else load_config()
    stt_cfg = cfg["stt"]

    provider = stt_cfg.get("provider", "huggingface").lower()

    if provider == "huggingface":
        if pipeline is None:
            raise ImportError("transformers não instalado.")
        asr = pipeline(
            "automatic-speech-recognition",
            model=stt_cfg["model_checkpoint"],
            device=0 if stt_cfg.get("device", "cpu") == "cuda" else -1,
        )
        return "huggingface", asr, stt_cfg.get("transcription_kwargs", {})

    elif provider == "openai":
        if openai is None:
            raise ImportError("pip install openai")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        model_name = stt_cfg.get("model_checkpoint", "whisper-1")
        return "openai", {"model": model_name}, stt_cfg.get("transcription_kwargs", {})

    else:
        raise ValueError(f"Provider STT desconhecido: {provider}")

def stt_from_audio(
    audio_source: str | bytes,
    provider: Literal["huggingface", "openai"],
    asr_obj: object,
    transcription_kwargs: Dict | None = None,
) -> str:
    if transcription_kwargs is None:
        transcription_kwargs = {}

    # Se for gs://... baixa bytes primeiro
    if isinstance(audio_source, str) and audio_source.startswith("gs://"):
        audio_bytes = gcs_client.download_bytes(audio_source)
        audio_input = audio_bytes
        filename_hint = os.path.basename(audio_source)
    else:
        audio_input = audio_source
        filename_hint = "audio.wav" if isinstance(audio_source, (bytes, bytearray)) else os.path.basename(audio_source)

    # ------------------------------ HuggingFace ------------------------------ #
    if provider == "huggingface":
        asr_pipe = asr_obj  # transformers.pipeline
        try:
            result = asr_pipe(audio_input, **transcription_kwargs)
        except Exception:
            # fallback: escreve em disco se for bytes
            if not isinstance(audio_input, (bytes, bytearray)):
                raise
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                tmp.write(audio_input)
                tmp.flush()
                result = asr_pipe(tmp.name, **transcription_kwargs)
        return result["text"]

    # -------------------------------- OpenAI -------------------------------- #
    elif provider == "openai":
        if not isinstance(audio_input, (bytes, bytearray)):
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
        else:
            audio_bytes = audio_input

        response = openai.audio.transcriptions.create(
            model=asr_obj["model"],
            file=(filename_hint, audio_bytes),
            **transcription_kwargs,
        )
        return response.text 