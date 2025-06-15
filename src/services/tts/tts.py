import os
from src.utils.load_config import load_config

def load_tts_pipeline(config_path=None):
    config = load_config(config_path) if config_path else load_config()
    tts_config = config["tts"]
    provider = tts_config.get("provider", "huggingface")
    model_checkpoint = tts_config["model_checkpoint"]
    language = tts_config.get("language", "pt")
    output_dir = tts_config.get("output_dir", "./audios_sintetizados")
    
    if provider == "huggingface":
        from transformers import pipeline
        tts_pipe = pipeline("text-to-speech", model=model_checkpoint, device=tts_config.get("device", "cpu").lower()
)
        return tts_pipe, language, output_dir
    else:
        raise NotImplementedError(f"Provider {provider} ainda não implementado.")

def tts_from_text(text, tts_pipe, language="pt", output_dir="./audios_sintetizados", file_name="tts_output.mp3"):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, file_name)
    out = tts_pipe(text)
    with open(audio_path, "wb") as f:
        f.write(out["audio"])
    return audio_path
