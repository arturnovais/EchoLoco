import os
import numpy as np
from src.utils.load_config import load_config

import torch
import soundfile as sf

def load_tts_pipeline(config_path=None):
    config = load_config(config_path) if config_path else load_config()
    tts_config = config["tts"]
    tts_type = tts_config.get("tts_type", "pipeline")
    provider = tts_config.get("provider", "huggingface")
    model_checkpoint = tts_config["model_checkpoint"]
    language = tts_config.get("language", "pt")
    output_dir = tts_config.get("output_dir", "./audios_sintetizados")
    audio_format = tts_config.get("audio_format", "wav")
    device = tts_config.get("device", "cpu")

    if tts_type in ["vits", "mms-tts"]:
        from transformers import VitsModel, AutoTokenizer
        model = VitsModel.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return ("vits", (model, tokenizer)), language, output_dir, audio_format

    elif tts_type == "pipeline":
        from transformers import pipeline
        tts_pipe = pipeline(
            "text-to-speech",
            model=model_checkpoint,
            device=0 if device == "cuda" else -1
        )
        return ("pipeline", tts_pipe), language, output_dir, audio_format

    elif tts_type == "parler-tts":
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return ("parler-tts", (model, tokenizer)), language, output_dir, audio_format

    else:
        raise NotImplementedError(f"TTS type '{tts_type}' não implementado.")

def tts_from_text(
    text,
    tts_tuple,
    language="pt",
    output_dir="./audios_sintetizados",
    file_name="tts_output",
    audio_format="wav"
):
    os.makedirs(output_dir, exist_ok=True)
    tts_type, tts_obj = tts_tuple
    audio_path = os.path.join(output_dir, f"{file_name}.{audio_format}")

    if tts_type == "vits":
        model, tokenizer = tts_obj
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        sampling_rate = model.config.sampling_rate
        sf.write(audio_path, waveform, sampling_rate)
        return audio_path

    elif tts_type == "pipeline":
        tts_pipe = tts_obj
        result = tts_pipe(text)
        if isinstance(result, dict) and "audio" in result:
            with open(audio_path, "wb") as f:
                f.write(result["audio"])
            return audio_path

    elif tts_type == "parler-tts":
        model, tokenizer = tts_obj
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            audio_arr = output.cpu().numpy().squeeze()
        sampling_rate = model.config.sampling_rate
        sf.write(audio_path, audio_arr, sampling_rate)
        return audio_path

    raise RuntimeError("Formato de saída TTS não reconhecido.")
