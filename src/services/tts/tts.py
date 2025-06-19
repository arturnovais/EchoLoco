import os
import numpy as np
from src.utils.load_config import load_config
from transformers import pipeline, VitsModel, AutoTokenizer
import torch

import soundfile as sf 


def load_tts_pipeline(config_path=None):
    config = load_config(config_path) if config_path else load_config()
    tts_config = config["tts"]
    provider = tts_config.get("provider", "huggingface")
    model_checkpoint = tts_config["model_checkpoint"]
    language = tts_config.get("language", "pt")
    output_dir = tts_config.get("output_dir", "./audios_sintetizados")
    audio_format = tts_config.get("audio_format", "wav")
    device = tts_config.get("device", "cpu")
    
    if "vits" in model_checkpoint or "mms-tts" in model_checkpoint:
        model = VitsModel.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return (model, tokenizer), language, output_dir, audio_format
    elif provider == "huggingface":
        tts_pipe = pipeline(
            "text-to-speech",
            model=model_checkpoint,
            device=0 if device == "cuda" else -1
        )
        return tts_pipe, language, output_dir, audio_format
    else:
        raise NotImplementedError(f"Provider {provider} ainda não implementado.")

def tts_from_text(
    text,
    tts_obj,
    language="pt",
    output_dir="./audios_sintetizados",
    file_name="tts_output",
    audio_format="wav"
):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, f"{file_name}.{audio_format}")

    if isinstance(tts_obj, tuple) and isinstance(tts_obj[0], VitsModel):
        model, tokenizer = tts_obj
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        sampling_rate = model.config.sampling_rate
        sf.write(audio_path, waveform, sampling_rate)
        return audio_path

    else:
        result = tts_obj(text)
        if isinstance(result, dict) and "audio" in result:
            with open(audio_path, "wb") as f:
                f.write(result["audio"])
            return audio_path

    raise RuntimeError("Formato de saída TTS não reconhecido.")