import os
import uuid
from io import BytesIO
import mimetypes
import os
import uuid
from io import BytesIO
import requests
from typing import Dict, Any

import numpy as np
import torch
import soundfile as sf
from utils.load_config import load_config
from infra.storage.utils import _upload_to_gcs
from infra.storage import gcs_client

ELEVEN_ENDPOINT_TMPL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

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
    output_dir="./audios_tmp",   
    file_name: str | None = None,
    audio_format="wav"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if file_name is None:
        file_name = str(uuid.uuid4())        

    tmp_audio_path = os.path.join(output_dir, f"{file_name}.{audio_format}")

    tts_type, tts_obj = tts_tuple

    if tts_type == "vits":
        model, tokenizer = tts_obj
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        sampling_rate = model.config.sampling_rate
        sf.write(tmp_audio_path, waveform, sampling_rate)

    elif tts_type == "pipeline":
        tts_pipe = tts_obj
        result = tts_pipe(text)
        if isinstance(result, dict) and "audio" in result:
            with open(tmp_audio_path, "wb") as f:
                f.write(result["audio"])

    elif tts_type == "parler-tts":
        model, tokenizer = tts_obj
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            audio_arr = output.cpu().numpy().squeeze()
        sampling_rate = model.config.sampling_rate
        sf.write(tmp_audio_path, audio_arr, sampling_rate)

    else:
        raise RuntimeError("Formato de saída TTS não reconhecido.")

    return _upload_to_gcs(tmp_audio_path, dest_prefix="tts_outputs")

def tts_eleven(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    voice_settings: Dict[str, Any] | None = None,
    dest_prefix: str = "tts_outputs",
) -> str:
    """
    Transforma texto em áudio via ElevenLabs e devolve caminho gs://...
    """
    api_key = os.getenv("XI_API_KEY")
    if not api_key:
        raise RuntimeError("XI_API_KEY não definido.")

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings
        or {
            "stability": 0.4,
            "similarity_boost": 0.8,
        },
    }

    resp = requests.post(
        ELEVEN_ENDPOINT_TMPL.format(voice_id=voice_id),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs API erro {resp.status_code}: {resp.text[:200]}"
        )

    # --- envia bytes direto para o GCS ------------------------------------ #
    file_id = f"{uuid.uuid4()}.mp3"
    dest_path = f"{dest_prefix}/{file_id}"

    gs_uri = gcs_client.upload_file(
        BytesIO(resp.content), dest_path, content_type="audio/mpeg"
    )
    return gs_uri