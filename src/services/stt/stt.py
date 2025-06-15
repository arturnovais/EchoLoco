import os
from transformers import pipeline
from src.utils.load_config import load_config


def stt_from_audio(audio_path, config_path=None):
    config = load_config(config_path) if config_path else load_config()
    stt_config = config["stt"]
    model_checkpoint = stt_config["model_checkpoint"]
    provider = stt_config.get("provider", "huggingface")
    device = 0 if stt_config.get("device", "cpu") == "cuda" else -1
    transcription_kwargs = stt_config.get("transcription_kwargs", {})
    
    if provider != "huggingface":
        raise NotImplementedError("Por enquanto só HuggingFace.")
    
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_checkpoint,
        device=device
    )

    result = asr(audio_path, **transcription_kwargs)
    return result["text"]


# Verificar depois um jeito de não carregar o modelo dentro da função
    
